"""Stage 3 — train the per-segment relevance ranker.

Base model: Qwen2.5-VL-3B-Instruct (we use only its LLM half — vision tower
frozen, never invoked).
Adapter:   LoRA on `q_proj, k_proj, v_proj, o_proj` of the language model.

Reuses paper 1's stability tricks:
  * LoRA params cast to fp32
  * NanGuard callback (zero NaN/Inf grads at the source)
  * warmup_ratio=0.0, max_grad_norm=1.0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoTokenizer,
    Qwen2_5_VLForConditionalGeneration,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ranker_pipeline"))

from ranker_pipeline.stage3_train_ranker.ranker_dataset import (  # noqa: E402
    RankerDataset,
    ranker_collator,
)


class NanGuardCallback(TrainerCallback):
    """Zero NaN/Inf gradient elements before the optimizer step.

    Same callback used in paper 1's stable VL run — necessary at small
    learning rates / fp32 LoRA where the occasional spike would otherwise
    inject NaNs into the optimizer state.
    """

    def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
        if model is None:
            return
        n = 0
        for p in model.parameters():
            if p.grad is None:
                continue
            mask = torch.isnan(p.grad) | torch.isinf(p.grad)
            if mask.any():
                n += int(mask.sum().item())
                p.grad[mask] = 0.0
        if n > 0 and state.global_step < 50:
            print(f"  [NanGuard] zeroed {n} NaN/Inf grad elems at step {state.global_step}",
                  flush=True)


def build_model(model_name: str, lora_r: int, lora_alpha: int) -> Qwen2_5_VLForConditionalGeneration:
    print(f"Loading {model_name} ...", flush=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name, dtype=torch.bfloat16, device_map=None,
    )

    # Freeze vision tower — the ranker never sees frames
    for n, p in model.named_parameters():
        if "visual" in n:
            p.requires_grad = False

    cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, cfg)

    # fp32 LoRA — paper 1's stability lesson
    for n, p in model.named_parameters():
        if p.requires_grad:
            p.data = p.data.float()
    model.print_trainable_parameters()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--labels_glob", default="ranker_pipeline/stage2_counterfactual_labels/labels*.jsonl")
    ap.add_argument("--output_dir", default="ranker_pipeline/stage3_train_ranker/checkpoints/v1")
    ap.add_argument("--lora_r", type=int, default=32)
    ap.add_argument("--lora_alpha", type=int, default=64)
    ap.add_argument("--learning_rate", type=float, default=5e-6)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--val_frac", type=float, default=0.05)
    ap.add_argument("--max_length", type=int, default=2048)
    args = ap.parse_args()

    import glob
    labels_paths = sorted(glob.glob(args.labels_glob))
    print(f"label files: {labels_paths}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_ds = RankerDataset(labels_paths, tokenizer, split="train",
                              val_frac=args.val_frac, max_length=args.max_length)
    val_ds = RankerDataset(labels_paths, tokenizer, split="val",
                            val_frac=args.val_frac, max_length=args.max_length)

    model = build_model(args.model, args.lora_r, args.lora_alpha)
    model.enable_input_require_grads()

    targs = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        warmup_ratio=0.0,
        max_grad_norm=1.0,
        weight_decay=0.0,
        bf16=False, fp16=False,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="steps", eval_steps=200,
        save_total_limit=3,
        report_to=[],
        gradient_checkpointing=False,
        dataloader_drop_last=True,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model, args=targs,
        train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=ranker_collator,
        tokenizer=tokenizer,
        callbacks=[NanGuardCallback()],
    )
    trainer.train()
    final_dir = Path(args.output_dir) / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"saved -> {final_dir}", flush=True)


if __name__ == "__main__":
    main()
