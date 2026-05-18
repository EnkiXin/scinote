"""
train_notetaker_vl.py — Multimodal LoRA SFT for Qwen2.5-VL-7B.

This is the PROPER version of the trained-noter experiment:
  Input  : video frames + question + options (NO answer)
  Target : 72B oracle note text
  Train  : Qwen2.5-VL-7B + LoRA (attention only), bf16 base + fp32 LoRA,
            NanGuard callback (proven stable in text-only run).

After this trains, deploy the trained VLM noter on SciVideoBench:
  Input  : SciVideoBench video frames + question + options
  Output : trained-noter note
  Then Qwen-3B answers with (video + trained-noter-note + question + options).
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


def build_user_text(item):
    q = item["question"]
    if item["task_type"] == "mc":
        opts = "\n".join(f"  {k}. {v}" for k, v in sorted(item["options"].items()))
        return (f"Question: {q}\n\nOptions:\n{opts}\n\nOutput ONLY a JSON object describing the visual evidence relevant to the question.")
    return f"Question: {q}\n\nOutput ONLY a JSON describing relevant visual evidence."


def get_video_path(vp):
    return hf_hub_download(repo_id=REPO_ID, filename=vp, repo_type="dataset")


def extract_frames(video_path: str, max_frames: int = 16, max_pixels: int = MAX_PIXELS):
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
    """Same as text-only: zero NaN/Inf grads before optimizer step."""
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
            print(f"  [NanGuard] zeroed {n_nan} NaN/Inf grad elems at step {state.global_step}", flush=True)


class VLNoteSFTDataset(torch.utils.data.Dataset):
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
            vp = get_video_path(it["video_path"])
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
            # Only squeeze batch dim for text fields. Multimodal tensors
            # (pixel_values_videos, video_grid_thw, second_per_grid_ts) have
            # NO batch dim — they are [N_patches, dim] / [N_videos, 3]. Squeezing
            # them with v[0] corrupts to single-patch / single-video and triggers
            # the [0, 4, -1] reshape bug in the vision tower.
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
    """Re-add batch dim ONLY for text fields (which were squeezed in dataset).
    Multimodal fields pass through as-is — Qwen2.5-VL expects them un-batched."""
    batch = [b for b in batch if b is not None]
    if not batch: return None
    if len(batch) > 1:
        raise NotImplementedError("only per_device_batch_size=1 supported")
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
    ap.add_argument("--train_jsonl", default="train_data/expvid_oracle_sft_train.jsonl")
    ap.add_argument("--val_jsonl",   default="train_data/expvid_oracle_sft_val.jsonl")
    ap.add_argument("--output_dir",  default="checkpoints/notetaker_vl_lora")
    ap.add_argument("--lora_r", type=int, default=32)
    ap.add_argument("--lora_alpha", type=int, default=64)
    ap.add_argument("--learning_rate", type=float, default=5e-6)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--max_frames", type=int, default=16)
    args = ap.parse_args()

    print(f"Loading processor + model: {args.model}", flush=True)
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map=None)

    # Freeze vision tower (don't train ViT)
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

    # Cast LoRA params to fp32 for stability (lesson from text-only training)
    for n, p in model.named_parameters():
        if p.requires_grad: p.data = p.data.float()
    print("  cast trainable LoRA params to fp32", flush=True)

    train_ds = VLNoteSFTDataset(args.train_jsonl, processor, args.max_frames)
    val_ds = VLNoteSFTDataset(args.val_jsonl, processor, args.max_frames)
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
        logging_steps=5,
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
    print(f"✅ saved → {args.output_dir}/final", flush=True)


if __name__ == "__main__":
    main()
