"""
Text-only inference using the trained Qwen-7B-Instruct LoRA noter.

Input  : SciVideoBench item's 3B self-note (already cached) + question + options
Output : trained_noter_note (saved to results_scivideobench/trained_noter_notes/)
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_notetaker_textonly import SYSTEM, build_user, PAD_TOKEN_ID

SCIVB = "/home/yz0392@unt.ad.unt.edu/xin_ai/scivideobench"
ANN_PATH = f"{SCIVB}/scivideobench_1k.jsonl"
SCIVB_SELFNOTE = f"{SCIVB}/results_scivideobench/notes_cache"


def load_selfnote(video_id):
    """SciVideoBench self-notes (3B-generated, no Q/A) keyed by video_id."""
    p = Path(SCIVB_SELFNOTE) / (hashlib.md5(video_id.encode()).hexdigest()[:16] + ".json")
    if not p.exists(): return None
    try: return json.load(open(p)).get("note", None)
    except: return None


def save_note(output_dir, vid, qid, note):
    sub = Path(output_dir) / "trained_noter_notes"
    sub.mkdir(parents=True, exist_ok=True)
    key = f"{vid}|{qid}"
    fp = sub / (hashlib.md5(key.encode()).hexdigest()[:16] + ".json")
    json.dump({"video_id": vid, "question_id": qid, "note": note}, open(fp, "w"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--lora_path", required=True)
    p.add_argument("--output_dir", default=f"{SCIVB}/results_scivideobench")
    p.add_argument("--max_tokens", type=int, default=512)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--chunk_id", type=int, default=0)
    p.add_argument("--num_chunks", type=int, default=1)
    args = p.parse_args()

    items = [json.loads(l) for l in open(ANN_PATH) if l.strip()]
    if args.num_chunks > 1:
        items = [it for i, it in enumerate(items) if i % args.num_chunks == args.chunk_id]
    if args.limit:
        items = items[:args.limit]
    print(f"n={len(items)}{f' chunk {args.chunk_id}/{args.num_chunks}' if args.num_chunks > 1 else ''}", flush=True)

    print(f"Loading {args.base_model} + LoRA from {args.lora_path}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.base_model)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=torch.bfloat16, device_map="cuda")
    model = PeftModel.from_pretrained(model, args.lora_path)
    model.eval()
    print("loaded", flush=True)

    n_done = 0; n_skip = 0
    for i, it in enumerate(items):
        out_p = Path(args.output_dir) / "trained_noter_notes" / (hashlib.md5(f"{it['video_id']}|{it['question_id']}".encode()).hexdigest()[:16] + ".json")
        if out_p.exists():
            n_skip += 1; continue
        sn = load_selfnote(it["video_id"])
        if sn is None:
            n_skip += 1; continue

        item_for_prompt = {
            "task_type": "mc",
            "question": it["question"],
            "options": it["options"],
            "self_note": sn,
        }
        msgs = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": build_user(item_for_prompt)},
        ]
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=8192)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=args.max_tokens, do_sample=False)
        note = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        save_note(args.output_dir, it["video_id"], it["question_id"], note)
        n_done += 1
        if i % 20 == 0:
            print(f"  [{n_done}/{len(items)}] {it['video_id']}|{it['question_id']}", flush=True)
    print(f"\n✅ Done. {n_done} notes generated, {n_skip} skipped", flush=True)


if __name__ == "__main__":
    main()
