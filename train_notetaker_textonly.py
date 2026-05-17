"""
train_notetaker_textonly.py — Text-only SFT of a small chat LLM to learn
the mapping (self_note + question + options) -> oracle_note.

Pure text-to-text training. No video processing. Trainer + LoRA on
Qwen2.5-7B-Instruct (or 3B).
"""
import argparse
import json
import os
import sys

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    Trainer, TrainingArguments, DataCollatorForLanguageModeling,
    TrainerCallback,
)


class NanGuardCallback(TrainerCallback):
    """After each backward, zero out any NaN/Inf gradients so they don't poison weights."""
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
            print(f"  [NanGuard] zeroed {n_nan} NaN/Inf grad elements at step {state.global_step}", flush=True)


SYSTEM = (
    "You are a careful, precise scientific-experiment-video note rewriter. "
    "You will be given (a) a raw structured description of a science experiment "
    "video, and (b) a multiple-choice question about it. Your task: rewrite the "
    "description into a focused note that selects and reorganises the visible "
    "evidence so the question is easier to answer. STRICT CONSTRAINTS:\n"
    "  • Use ONLY content present in the provided description.\n"
    "  • Do NOT invent new visual details.\n"
    "  • Do NOT mention any answer letter (A, B, ...).\n"
    "  • Output ONLY valid JSON."
)


def build_user(item):
    q = item["question"]
    if item.get("options"):
        opts = "\n".join(f"  {k}. {v}" for k, v in sorted(item["options"].items()))
        return (f"Original video description:\n{item['self_note']}\n\n---\n\n"
                f"Question: {q}\n\nOptions:\n{opts}\n\n"
                "Output ONLY a JSON object that selects and reorganises the "
                "relevant evidence from the description.")
    return (f"Original video description:\n{item['self_note']}\n\n---\n\n"
            f"Question: {q}\n\nOutput ONLY a JSON object that selects relevant evidence.")


def make_dataset(jsonl_path, tokenizer, max_len=8192):
    items = [json.loads(l) for l in open(jsonl_path)]
    examples = []
    for it in items:
        target = it["oracle_note"]
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": build_user(it)},
            {"role": "assistant", "content": target},
        ]
        full = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
        full_ids = tokenizer(full, return_tensors=None, truncation=True, max_length=max_len).input_ids
        prompt_ids = tokenizer(prompt, return_tensors=None, truncation=True, max_length=max_len).input_ids
        labels = list(full_ids)
        plen = min(len(prompt_ids), len(labels))
        for i in range(plen): labels[i] = -100
        examples.append({"input_ids": full_ids, "attention_mask": [1]*len(full_ids), "labels": labels})
    return Dataset.from_list(examples)


PAD_TOKEN_ID = 151643  # Qwen2.5 <|endoftext|>


def collate(features):
    # pad right to longest with proper pad token
    max_len = max(len(f["input_ids"]) for f in features)
    out = {"input_ids": [], "attention_mask": [], "labels": []}
    for f in features:
        n = max_len - len(f["input_ids"])
        out["input_ids"].append(f["input_ids"] + [PAD_TOKEN_ID]*n)
        out["attention_mask"].append(f["attention_mask"] + [0]*n)
        out["labels"].append(f["labels"] + [-100]*n)
    return {k: torch.tensor(v) for k, v in out.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--train_jsonl", default="train_data/textonly_sft_train.jsonl")
    p.add_argument("--val_jsonl",   default="train_data/textonly_sft_val.jsonl")
    p.add_argument("--output_dir",  default="checkpoints/notetaker_text_lora")
    p.add_argument("--lora_r", type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=64)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--per_device_batch_size", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--epochs", type=int, default=2)
    args = p.parse_args()

    print(f"Loading {args.model} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)

    # Attention-only LoRA (FFN LoRA in bf16 can overflow on long sequences)
    lora_cfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # Convert LoRA params to fp32 for numerical stability under DDP all-reduce.
    # bf16 LoRA grads summed across ranks easily overflow → NaN.
    n_cast = 0
    for n, p in model.named_parameters():
        if p.requires_grad:
            p.data = p.data.float()
            n_cast += 1
    print(f"  cast {n_cast} trainable params to fp32 (base stays bf16)", flush=True)

    train_ds = make_dataset(args.train_jsonl, tokenizer)
    val_ds = make_dataset(args.val_jsonl, tokenizer)
    print(f"train: {len(train_ds)}, val: {len(val_ds)}", flush=True)

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
        # Trainer's bf16=True forces autocast which converts LoRA grads back
        # to bf16 → DDP all-reduce overflow → NaN. Leave Trainer in fp32;
        # base model is already bf16, LoRA is fp32 — mixed precision but
        # numerically stable.
        bf16=False,
        fp16=False,
        logging_steps=5,
        save_strategy="epoch",
        eval_strategy="epoch",
        report_to=[],
        gradient_checkpointing=False,
        ddp_find_unused_parameters=True,
        dataloader_drop_last=True,
    )
    trainer = Trainer(model=model, args=targs,
                       train_dataset=train_ds, eval_dataset=val_ds,
                       data_collator=collate,
                       callbacks=[NanGuardCallback()])
    trainer.train()
    trainer.save_model(args.output_dir + "/final")
    tokenizer.save_pretrained(args.output_dir + "/final")
    print(f"✅ saved → {args.output_dir}/final")


if __name__ == "__main__":
    main()
