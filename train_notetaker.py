"""
train_notetaker.py — LoRA fine-tune Qwen2.5-VL-7B as a note-taker.

Input  : video frames + question (+ options for MC)  — NO gold answer
Target : 72B oracle note text (from ExpVid L2+L3)

Strategy:
  • Freeze vision encoder; LoRA on language-model attention + MLP only.
  • bf16, gradient accumulation, multi-GPU DDP via accelerate.
"""

import argparse
import json
import os
import sys
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    Trainer, TrainingArguments,
)
from qwen_vl_utils import process_vision_info

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate_unified import REPO_ID, MAX_PIXELS

import av
from huggingface_hub import hf_hub_download


NOTETAKER_SYSTEM = (
    "You are a careful, precise observer of scientific experiment videos. "
    "Given a video and a question about it, you write structured visual notes "
    "(JSON) that describe ONLY what is visible in the video and that are useful "
    "for answering the question. Do not speculate beyond visible evidence. "
    "Output ONLY valid JSON."
)


def build_user_prompt(item):
    q = item["question"]
    if item["task_type"] == "mc":
        opts = "\n".join(f"  {k}. {v}" for k, v in sorted(item["options"].items()))
        return (f"Question: {q}\n\nOptions:\n{opts}\n\n"
                "Write a structured note about the video that supports answering "
                "this question. Output ONLY a JSON object.")
    elif item["task_type"] == "seqgen":
        return (f"Question: {q}\n\nWrite a structured note listing the procedural "
                "steps actually shown in the video. Output ONLY a JSON object.")
    elif item["task_type"] == "steppred":
        return (f"Question: {q}\n\nWrite a structured note describing the steps "
                "observed and the visible state at the end of the clip. JSON only.")
    elif item["task_type"] == "fitb":
        return (f"Question: {q}\n\nWrite a structured note describing the visible "
                "evidence needed to fill in the blanks. JSON only.")
    return q


def get_video_path(video_path):
    return hf_hub_download(repo_id=REPO_ID, filename=video_path, repo_type="dataset")


def extract_frames(video_path: str, fps: float = 1.0, max_frames: int = 32,
                   max_pixels: int = MAX_PIXELS, fixed_dim: int = 280):
    """Always returns exactly max_frames RGB frames at a FIXED 280x280 size
    (20 patches/side, 100 merged tokens/frame). Avoids size mismatches in
    Qwen2.5-VL's vision tower during multi-rank DDP.
    """
    container = av.open(video_path)
    stream = container.streams.video[0]
    total = stream.frames
    n = max_frames  # fixed
    target_idx = set(int(i * total / n) for i in range(n)) if total > 0 else None
    out = []
    try:
        for i, f in enumerate(container.decode(video=0)):
            if target_idx is not None and i not in target_idx:
                continue
            img = f.to_image().convert("RGB").resize((fixed_dim, fixed_dim), Image.BILINEAR)
            out.append(img)
            if len(out) >= n: break
    finally:
        container.close()
    # pad
    while out and len(out) < max_frames:
        out.append(out[-1])
    return out


class NoteSFTDataset(Dataset):
    def __init__(self, jsonl_path, processor, max_frames=16):
        self.items = [json.loads(l) for l in open(jsonl_path)]
        self.processor = processor
        self.max_frames = max_frames

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        # If anything goes wrong, return the next valid item to avoid None batches.
        for off in range(len(self.items)):
            i = (idx + off) % len(self.items)
            ret = self._try_one(i)
            if ret is not None:
                return ret
        # absolute fallback — should never hit
        raise RuntimeError("no valid items in dataset")

    def _try_one(self, idx):
        it = self.items[idx]
        try:
            vp = get_video_path(it["video_path"])
            frames = extract_frames(vp, fps=1.0, max_frames=self.max_frames)
            if not frames or len(frames) < self.max_frames:
                return None
        except Exception:
            return None
        try:
            user_text = build_user_prompt(it)
            target = it["oracle_note"]
            messages = [
                {"role": "system", "content": NOTETAKER_SYSTEM},
                {"role": "user", "content": [
                    {"type": "video", "video": frames, "max_pixels": MAX_PIXELS},
                    {"type": "text", "text": user_text},
                ]},
                {"role": "assistant", "content": target},
            ]
            text = self.processor.apply_chat_template(messages, tokenize=False,
                                                        add_generation_prompt=False)
            prompt_messages = messages[:-1]
            prompt_text = self.processor.apply_chat_template(prompt_messages,
                                                              tokenize=False,
                                                              add_generation_prompt=True)
            image_inputs, video_inputs, video_kwargs = process_vision_info(
                messages, return_video_kwargs=True)
            if "fps" in video_kwargs and isinstance(video_kwargs["fps"], list):
                video_kwargs["fps"] = video_kwargs["fps"][0] if video_kwargs["fps"] else 1.0
            full = self.processor(text=[text], images=image_inputs, videos=video_inputs,
                                    return_tensors="pt", **video_kwargs)
            prompt = self.processor(text=[prompt_text], images=image_inputs,
                                      videos=video_inputs, return_tensors="pt",
                                      **video_kwargs)
            input_ids = full["input_ids"][0]
            prompt_len = prompt["input_ids"].shape[1]
            labels = input_ids.clone()
            labels[:prompt_len] = -100
            ret = {k: v[0] if hasattr(v, "shape") and v.ndim > 0 else v for k, v in full.items()}
            ret["labels"] = labels
            return ret
        except Exception as e:
            print(f"  skip idx={idx}: {e}", flush=True)
            return None


def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch: return None
    # Right-pad input_ids / labels to longest
    max_len = max(b["input_ids"].shape[0] for b in batch)
    pad_id = 0  # Qwen pad token id
    out = {}
    for k in batch[0].keys():
        if k in ("input_ids", "attention_mask"):
            padded = []
            for b in batch:
                cur = b[k]
                pad_n = max_len - cur.shape[0]
                if pad_n > 0:
                    pad_val = pad_id if k == "input_ids" else 0
                    pad = torch.full((pad_n,), pad_val, dtype=cur.dtype)
                    padded.append(torch.cat([cur, pad]))
                else:
                    padded.append(cur)
            out[k] = torch.stack(padded)
        elif k == "labels":
            padded = []
            for b in batch:
                cur = b[k]
                pad_n = max_len - cur.shape[0]
                if pad_n > 0:
                    pad = torch.full((pad_n,), -100, dtype=cur.dtype)
                    padded.append(torch.cat([cur, pad]))
                else:
                    padded.append(cur)
            out[k] = torch.stack(padded)
        elif isinstance(batch[0][k], torch.Tensor):
            # stack vision tensors as is (they may already be batchable)
            try:
                out[k] = torch.stack([b[k] for b in batch])
            except Exception:
                # variable-shape (e.g. pixel values per video) — keep as list
                out[k] = [b[k] for b in batch]
        else:
            out[k] = [b[k] for b in batch]
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--train_jsonl", default="train_data/expvid_oracle_sft_train.jsonl")
    p.add_argument("--val_jsonl", default="train_data/expvid_oracle_sft_val.jsonl")
    p.add_argument("--output_dir", default="checkpoints/notetaker_lora")
    p.add_argument("--lora_r", type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=64)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--per_device_batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--max_frames", type=int, default=16)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    args = p.parse_args()

    print(f"Loading processor + model: {args.model}", flush=True)
    processor = AutoProcessor.from_pretrained(args.model)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map=None)

    # Freeze vision tower
    for n, p_ in model.named_parameters():
        if "visual" in n:
            p_.requires_grad = False

    # LoRA on language model
    lora_cfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    train_ds = NoteSFTDataset(args.train_jsonl, processor, args.max_frames)
    val_ds = NoteSFTDataset(args.val_jsonl, processor, args.max_frames)
    print(f"train: {len(train_ds)}, val: {len(val_ds)}", flush=True)

    targs = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        bf16=True,
        logging_steps=20,
        save_strategy="epoch",
        eval_strategy="epoch",
        report_to=[],
        remove_unused_columns=False,
        gradient_checkpointing=True,
        ddp_find_unused_parameters=False,
    )

    trainer = Trainer(
        model=model, args=targs,
        train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=collate_fn,
    )
    trainer.train()
    trainer.save_model(args.output_dir + "/final")
    print(f"✅ saved LoRA → {args.output_dir}/final")


if __name__ == "__main__":
    main()
