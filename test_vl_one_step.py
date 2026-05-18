"""Quickly verify that the fixed collator yields a working forward+backward."""
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_notetaker_vl import VLNoteSFTDataset, vl_collate
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import LoraConfig, get_peft_model

MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
print(f"Loading {MODEL} ...", flush=True)
proc = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL, dtype=torch.bfloat16, device_map="cuda:0")

for n, p in model.named_parameters():
    if "visual" in n: p.requires_grad = False
lora = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0, bias="none",
                  task_type="CAUSAL_LM",
                  target_modules=["q_proj","k_proj","v_proj","o_proj"])
model = get_peft_model(model, lora)
for n, p in model.named_parameters():
    if p.requires_grad: p.data = p.data.float()
model.print_trainable_parameters()

ds = VLNoteSFTDataset("train_data/expvid_oracle_sft_train.jsonl", proc, max_frames=16)
print(f"dataset: {len(ds)} items", flush=True)

n_ok = 0; n_fail = 0
for i in range(5):
    item = ds[i]
    batch = vl_collate([item])
    if batch is None:
        print(f"[{i}] None batch"); n_fail += 1; continue
    batch = {k: v.to("cuda:0") if hasattr(v, "to") else v for k, v in batch.items()}
    try:
        out = model(**batch)
        loss = out.loss
        loss.backward()
        # Check trainable grads for NaN
        g_nan = 0
        for p in model.parameters():
            if p.grad is None: continue
            if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                g_nan += 1
        print(f"[{i}] ✅ loss={loss.item():.3f}  grad_nan_params={g_nan}", flush=True)
        n_ok += 1
        model.zero_grad()
    except Exception as e:
        print(f"[{i}] ❌ {type(e).__name__}: {str(e)[:300]}", flush=True)
        n_fail += 1
    torch.cuda.empty_cache()

print(f"\nSUMMARY: ok={n_ok} fail={n_fail}")
