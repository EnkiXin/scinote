"""train_planner_sft.py — Step A / B SFT trainer for the ProtoNote planner.

Text-only LoRA training on Qwen2.5-VL-7B-Instruct (we keep the same backbone
as the answer model so the same VLMClient can load the adapter for the
planner-only call at inference). Input: `prompt` field. Target: `completion`
field. Both produced by prepare_planner_data.py.

Architecture choice — train the LANGUAGE side only:
  * The planner reads markdown notes + text question and outputs a short
    JSON action. No video frames involved.
  * We still load the VL processor + model so the eval pipeline can swap
    adapter weights into the existing VLMClient without re-loading base.
  * LoRA on q/k/v/o_proj in the language attention layers; vision encoder
    + LM head frozen.

LoRA recipe is the v5b-noter recipe: r=32, α=64, lr=5e-6, batch=1, ga=8,
1 epoch, fp32 LoRA on bf16 base, NanGuardCallback. Confirmed working on
this codebase (paper-1).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoProcessor, Qwen2_5_VLForConditionalGeneration,
    Trainer, TrainingArguments, TrainerCallback,
)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


_PLANNER_SYSTEM = (
    "You are a careful video-analysis agent. Given a question and your "
    "current notes about a scientific lab video, decide the next action."
)


class NanGuardCallback(TrainerCallback):
    def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
        if model is None:
            return
        n_nan = 0
        for p in model.parameters():
            if p.grad is None:
                continue
            mask = torch.isnan(p.grad) | torch.isinf(p.grad)
            if mask.any():
                n_nan += int(mask.sum().item())
                p.grad[mask] = 0.0
        if n_nan > 0 and state.global_step < 50:
            print(f"  [NanGuard] zeroed {n_nan} NaN/Inf grad elements at "
                  f"step {state.global_step}", flush=True)


def make_dataset(jsonl_path: Path, processor, max_len: int = 2048) -> Dataset:
    tokenizer = processor.tokenizer
    rows = [json.loads(l) for l in open(jsonl_path)]
    examples = []
    for r in rows:
        messages = [
            {"role": "system", "content": _PLANNER_SYSTEM},
            {"role": "user",   "content": r["prompt"]},
            {"role": "assistant", "content": r["completion"]},
        ]
        full = tokenizer.apply_chat_template(messages, tokenize=False,
                                               add_generation_prompt=False)
        prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=False,
                                                 add_generation_prompt=True)
        full_ids = tokenizer(full, return_tensors=None,
                              truncation=True, max_length=max_len).input_ids
        prompt_ids = tokenizer(prompt, return_tensors=None,
                                truncation=True, max_length=max_len).input_ids
        labels = list(full_ids)
        plen = min(len(prompt_ids), len(labels))
        for i in range(plen):
            labels[i] = -100
        examples.append({
            "input_ids":      full_ids,
            "attention_mask": [1] * len(full_ids),
            "labels":         labels,
        })
    return Dataset.from_list(examples)


PAD_TOKEN_ID = 151643  # Qwen2.5 <|endoftext|>


def collate(features):
    max_len = max(len(f["input_ids"]) for f in features)
    out = {"input_ids": [], "attention_mask": [], "labels": []}
    for f in features:
        n = max_len - len(f["input_ids"])
        out["input_ids"].append(f["input_ids"] + [PAD_TOKEN_ID] * n)
        out["attention_mask"].append(f["attention_mask"] + [0] * n)
        out["labels"].append(f["labels"] + [-100] * n)
    return {k: torch.tensor(v) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--train_jsonl", required=True)
    ap.add_argument("--val_jsonl", default="")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--lora_r", type=int, default=32)
    ap.add_argument("--lora_alpha", type=int, default=64)
    ap.add_argument("--learning_rate", type=float, default=5e-6)
    ap.add_argument("--per_device_batch_size", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max_len", type=int, default=2048)
    args = ap.parse_args()

    print(f"[planner_sft] loading {args.model}", flush=True)
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16)

    # Freeze the vision encoder. Planner is text-only — no need to train it.
    for n, p in model.named_parameters():
        if "visual" in n:
            p.requires_grad = False

    lora_cfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # Cast LoRA params to fp32 for DDP stability (paper-1 lesson).
    n_cast = 0
    for n, p in model.named_parameters():
        if p.requires_grad:
            p.data = p.data.float()
            n_cast += 1
    print(f"  cast {n_cast} trainable params to fp32 (base stays bf16)",
          flush=True)

    train_ds = make_dataset(Path(args.train_jsonl), processor, args.max_len)
    val_ds = (make_dataset(Path(args.val_jsonl), processor, args.max_len)
              if args.val_jsonl else None)
    print(f"  train: {len(train_ds)}  val: {len(val_ds) if val_ds else 0}",
          flush=True)

    targs = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        warmup_ratio=0.0,
        max_grad_norm=1.0,
        weight_decay=0.0,
        bf16=False, fp16=False,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch" if val_ds else "no",
        report_to=[],
        gradient_checkpointing=False,
        ddp_find_unused_parameters=True,
        dataloader_drop_last=True,
        dataloader_num_workers=4,
    )
    trainer = Trainer(model=model, args=targs,
                       train_dataset=train_ds, eval_dataset=val_ds,
                       data_collator=collate,
                       callbacks=[NanGuardCallback()])
    trainer.train()
    out_final = Path(args.output_dir) / "final"
    trainer.save_model(str(out_final))
    processor.save_pretrained(str(out_final))
    print(f"✅ saved → {out_final}", flush=True)


if __name__ == "__main__":
    main()
