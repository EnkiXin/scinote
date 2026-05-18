"""train_notetaker_vl_v2.py — Multimodal LoRA SFT for the v2 split.

Differences from train_notetaker_vl.py:
  * Reads v2_split_train.jsonl / v2_split_val.jsonl which mix ExpVid AND
    SciVideoBench training items.
  * Video-path resolution dispatches on `benchmark` field of each row:
      - ExpVid          -> hf_hub_download from OpenGVLab/ExpVid
      - SciVideoBench   -> local jove_<vid>.mp4 / <vid>.mp4 lookup
  * Otherwise identical to v1: Qwen2.5-VL-7B base (bf16, vision frozen),
    LoRA r=32 α=64 attention-only, fp32 LoRA, warmup=0, NanGuard callback.
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import av
import torch
from PIL import Image
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoProcessor, Qwen2_5_VLForConditionalGeneration,
    Trainer, TrainingArguments, TrainerCallback,
)
from qwen_vl_utils import process_vision_info

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate_unified import REPO_ID, MAX_PIXELS
from huggingface_hub import hf_hub_download


SYSTEM = (
    "You are a careful, precise observer of scientific experiment videos. "
    "Given a video and a question, write structured visual notes that describe "
    "ONLY what is visible in the video and that are useful for answering the "
    "question. Output ONLY valid JSON."
)

SCIVB_VIDEO_DIR = Path("/home/yz0392@unt.ad.unt.edu/xin_ai/scivideobench/videos")


def build_user_text(item):
    q = item["question"]
    if item["task_type"] == "mc":
        opts = "\n".join(f"  {k}. {v}" for k, v in sorted(item["options"].items()))
        return (f"Question: {q}\n\nOptions:\n{opts}\n\nOutput ONLY a JSON object "
                f"describing the visual evidence relevant to the question.")
    return f"Question: {q}\n\nOutput ONLY a JSON describing relevant visual evidence."


def resolve_video_path(item) -> str:
    """Dispatch on benchmark to find on-disk video file."""
    benchmark = item.get("benchmark", "expvid")
    if benchmark == "scivideobench":
        vid = str(item.get("id") or "").split("|")[0]
        # item["video_path"] is virtual ("scivb_video_id:<vid>"); strip
        if "video_path" in item and ":" in str(item["video_path"]):
            vid = str(item["video_path"]).split(":")[-1]
        for pat in (f"jove_{vid}.mp4", f"{vid}.mp4"):
            p = SCIVB_VIDEO_DIR / pat
            if p.exists():
                return str(p)
        return ""
    # ExpVid: download from HF
    return hf_hub_download(repo_id=REPO_ID, filename=item["video_path"],
                            repo_type="dataset")


def extract_frames(video_path: str, max_frames: int = 16, max_pixels: int = MAX_PIXELS):
    if not video_path:
        return []
    container = av.open(video_path)
    stream = container.streams.video[0]
    total = stream.frames
    n = max_frames
    target_idx = set(int(i * total / n) for i in range(n)) if total > 0 else None
    out = []
    try:
        for i, f in enumerate(container.decode(video=0)):
            if target_idx is not None and i not in target_idx:
                continue
            img = f.to_image()
            w, h = img.size
            if w * h > max_pixels:
                scale = (max_pixels / (w * h)) ** 0.5
                img = img.resize((max(28, int(w * scale)), max(28, int(h * scale))),
                                   Image.BILINEAR)
            out.append(img)
            if len(out) >= n: break
    finally:
        container.close()
    while out and len(out) < max_frames:
        out.append(out[-1])
    return out


PAD_ID = 151643


class NanGuardCallback(TrainerCallback):
    def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
        if model is None: return
        n_nan = 0
        for p in model.parameters():
            if p.grad is None: continue
            mask = torch.isnan(p.grad) | torch.isinf(p.grad)
            if mask.any():
                n_nan += int(mask.sum().item())
                p.grad[mask] = 0.0
        if n_nan > 0 and state.global_step < 50:
            print(f"  [NanGuard] zeroed {n_nan} NaN/Inf grad elems at step {state.global_step}",
                  flush=True)


class VLNoteSFTDatasetV2(torch.utils.data.Dataset):
    def __init__(self, jsonl_path, processor, max_frames=16):
        self.items = [json.loads(l) for l in open(jsonl_path)]
        self.processor = processor
        self.max_frames = max_frames

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        for off in range(len(self.items)):
            i = (idx + off) % len(self.items)
            ret = self._try_one(i)
            if ret is not None: return ret
        raise RuntimeError("no valid items")

    def _try_one(self, idx):
        it = self.items[idx]
        try:
            vp = resolve_video_path(it)
            if not vp:
                return None
            frames = extract_frames(vp, max_frames=self.max_frames)
            if not frames or len(frames) < self.max_frames:
                return None
        except Exception:
            return None
        try:
            user_text = build_user_text(it)
            target = it["oracle_note"]
            messages = [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": [
                    {"type": "video", "video": frames, "max_pixels": MAX_PIXELS},
                    {"type": "text", "text": user_text},
                ]},
                {"role": "assistant", "content": target},
            ]
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            prompt_text = self.processor.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
            if "fps" in video_kwargs and isinstance(video_kwargs["fps"], list):
                video_kwargs["fps"] = video_kwargs["fps"][0] if video_kwargs["fps"] else 1.0
            full = self.processor(text=[text], images=image_inputs, videos=video_inputs,
                                     return_tensors="pt", **video_kwargs)
            prompt = self.processor(text=[prompt_text], images=image_inputs,
                                       videos=video_inputs, return_tensors="pt",
                                       **video_kwargs)
            input_ids = full["input_ids"][0]
            plen = min(prompt["input_ids"].shape[1], len(input_ids))
            labels = input_ids.clone()
            labels[:plen] = -100
            TEXT_FIELDS = {"input_ids", "attention_mask"}
            ret = {}
            for k, v in full.items():
                if k in TEXT_FIELDS and hasattr(v, "ndim") and v.ndim > 0:
                    ret[k] = v[0]
                else:
                    ret[k] = v
            ret["labels"] = labels
            return ret
        except Exception as e:
            print(f"  skip idx={idx}: {e}", flush=True)
            return None


def vl_collate(batch):
    batch = [b for b in batch if b is not None]
    if not batch: return None
    if len(batch) > 1:
        raise NotImplementedError("per_device_batch_size=1 only")
    b = batch[0]
    out = {}
    for k, v in b.items():
        if k in ("input_ids", "attention_mask", "labels") and isinstance(v, torch.Tensor):
            out[k] = v.unsqueeze(0)
        else:
            out[k] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--train_jsonl", default="train_data/v2_split_train.jsonl")
    ap.add_argument("--val_jsonl",   default="train_data/v2_split_val.jsonl")
    ap.add_argument("--output_dir",  default="checkpoints/notetaker_vl_lora_v2_split")
    ap.add_argument("--lora_r", type=int, default=32)
    ap.add_argument("--lora_alpha", type=int, default=64)
    ap.add_argument("--learning_rate", type=float, default=5e-6)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max_frames", type=int, default=16)
    args = ap.parse_args()

    print(f"Loading processor + model: {args.model}", flush=True)
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map=None)

    for n, p in model.named_parameters():
        if "visual" in n:
            p.requires_grad = False

    lora_cfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj","k_proj","v_proj","o_proj"],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    for n, p in model.named_parameters():
        if p.requires_grad: p.data = p.data.float()
    print("  cast trainable LoRA params to fp32", flush=True)

    train_ds = VLNoteSFTDatasetV2(args.train_jsonl, processor, args.max_frames)
    val_ds = VLNoteSFTDatasetV2(args.val_jsonl, processor, args.max_frames)
    print(f"train: {len(train_ds)}, val: {len(val_ds)}", flush=True)

    targs = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        warmup_ratio=0.0,
        max_grad_norm=1.0,
        weight_decay=0.0,
        bf16=False, fp16=False,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="no",
        report_to=[],
        gradient_checkpointing=False,
        ddp_find_unused_parameters=True,
        dataloader_drop_last=True,
        remove_unused_columns=False,
    )

    trainer = Trainer(model=model, args=targs,
                       train_dataset=train_ds, eval_dataset=val_ds,
                       data_collator=vl_collate,
                       callbacks=[NanGuardCallback()])
    trainer.train()
    trainer.save_model(args.output_dir + "/final")
    processor.save_pretrained(args.output_dir + "/final")
    print(f"saved -> {args.output_dir}/final", flush=True)


if __name__ == "__main__":
    main()
