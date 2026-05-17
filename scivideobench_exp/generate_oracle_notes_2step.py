"""
generate_oracle_notes_2step.py — Two-step oracle note generation, leak-proofed.

Step 1 (per video, video-only):
    Input  : video frames ONLY (no question, no options, no answer)
    Output : comprehensive visual description JSON
    Why    : the model has no idea what's being asked, so any
             "answer-shaped emphasis" is structurally impossible.

Step 2 (per question, text-only):
    Input  : Step-1 description (text) + question + options + gold answer
    Output : filtered note that pulls from Step-1 text only
    Strict : "Do not add any visual detail not present in the provided
             description. Only select / paraphrase from the description text."
    Why    : answer-conditioning is allowed but bounded — model can choose
             what to keep, cannot fabricate new visual evidence.

Cache:
    step1_descriptions/<md5(video_id)>.json   — keyed by video_id only
    step2_oracle_notes/<md5(video_id|qid)>.json — per question
"""

import argparse
import hashlib
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Queue
from typing import Optional

import av
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate_scivideobench import ANN_PATH, MAX_PIXELS, get_video_path, extract_frames


# ───── Prompts ─────────────────────────────────────────────────────────
STEP1_SYSTEM = (
    "You are a careful, precise observer of scientific experiment videos. "
    "Your task is to produce a COMPREHENSIVE, structured description of "
    "everything that is visible in this video, with no knowledge of any "
    "downstream question. Describe procedures, materials, tools, "
    "quantitative readings, transitions, anything visible. "
    "Do NOT speculate beyond visible evidence. Output ONLY valid JSON."
)

STEP1_PROMPT = (
    "Watch this scientific experiment video and produce a COMPREHENSIVE "
    "structured description of every observable thing.\n\n"
    "Output ONLY this JSON (you do not need to fill every field if not "
    "applicable, but be exhaustive about what IS visible):\n"
    "{\n"
    '  "experiment_overview": "what the experiment is about based on what is shown",\n'
    '  "procedures_observed": ["all procedures performed, in temporal order"],\n'
    '  "materials_and_subjects": ["samples, animals, materials, tissues used"],\n'
    '  "tools_and_setup": ["specific equipment, instruments, chambers, apparatus"],\n'
    '  "quantitative_observations": ["all numbers, volumes, times, temperatures, concentrations visible"],\n'
    '  "key_transitions": ["important state/process transitions"],\n'
    '  "outcomes_or_indicators": ["all results, signals, color changes, readings"],\n'
    '  "readable_text_or_labels": ["any text actually visible on screen"],\n'
    '  "salient_visual_details": ["distinctive visual features that may matter"]\n'
    "}"
)


STEP2_SYSTEM = (
    "You are filtering an existing video description to extract evidence "
    "relevant to a question. STRICT CONSTRAINTS:\n"
    "  • You CAN see the question and correct answer.\n"
    "  • You MUST only select / paraphrase content that is present in the "
    "provided video description. Do NOT invent any new visual detail.\n"
    "  • Do NOT mention the answer letter (A, B, ..., J).\n"
    "  • Do NOT copy option text verbatim.\n"
    "  • Output ONLY valid JSON, no extra text."
)


def build_step2_prompt(description_text: str, item: dict) -> str:
    opts = item["options"]
    options_text = "\n".join(f"  {k}. {v}" for k, v in sorted(opts.items()))
    return (
        f"Original full video description:\n{description_text}\n\n"
        f"---\n\n"
        f"Question: {item['question']}\n\n"
        f"Options:\n{options_text}\n\n"
        f"Correct answer: {item['answer']} "
        f"(use this only to know what evidence is relevant; do NOT reveal the letter)\n\n"
        f"Output ONLY this JSON (selecting / paraphrasing ONLY from the description above):\n"
        f"{{\n"
        f'  "key_evidence": ["evidence drawn from the description above that supports the correct answer"],\n'
        f'  "context_observations": ["additional context from the description that helps reasoning"],\n'
        f'  "salient_objects_or_text": ["distinctive items mentioned in the description"]\n'
        f"}}"
    )


# ───── Cache paths ─────────────────────────────────────────────────────
def step1_cache_path(output_dir: str, video_id: str) -> Path:
    sub = Path(output_dir) / "step1_descriptions"
    sub.mkdir(parents=True, exist_ok=True)
    return sub / (hashlib.md5(video_id.encode()).hexdigest()[:16] + ".json")


def step2_cache_path(output_dir: str, video_id: str, question_id: str) -> Path:
    sub = Path(output_dir) / "step2_oracle_notes"
    sub.mkdir(parents=True, exist_ok=True)
    key = f"{video_id}|{question_id}"
    return sub / (hashlib.md5(key.encode()).hexdigest()[:16] + ".json")


def load_step1(output_dir, video_id):
    """Returns the description string. Stored under 'note' key for eval compatibility."""
    p = step1_cache_path(output_dir, video_id)
    if not p.exists(): return None
    try: return json.load(open(p)).get("note", None)
    except Exception: return None


def save_step1(output_dir, video_id, description):
    p = step1_cache_path(output_dir, video_id)
    # save under "note" key so the same evaluate_oracle / note-only scripts can read it
    json.dump({"video_id": video_id, "note": description}, open(p, "w"))


def load_step2(output_dir, video_id, qid):
    p = step2_cache_path(output_dir, video_id, qid)
    if not p.exists(): return None
    try: return json.load(open(p)).get("note", None)
    except Exception: return None


def save_step2(output_dir, video_id, qid, note, item):
    p = step2_cache_path(output_dir, video_id, qid)
    json.dump({"video_id": video_id, "question_id": qid,
               "gold": item["answer"],
               "question_type": item.get("question_type", ""),
               "discipline": item.get("discipline", ""),
               "note": note}, open(p, "w"))


# ───── Step 1 driver (vLLM + video) ────────────────────────────────────
def run_step1(args):
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    with open(ANN_PATH) as f:
        items = [json.loads(l) for l in f if l.strip()]
    unique_vids = list(dict.fromkeys(it["video_id"] for it in items))
    print(f"{len(unique_vids)} unique videos", flush=True)

    todo = [v for v in unique_vids if load_step1(args.output, v) is None]
    print(f"todo: {len(todo)}", flush=True)
    if not todo: print("✅ step 1 all cached"); return

    print(f"Loading processor {args.model} ...", flush=True)
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    print(f"Loading vLLM TP={args.tensor_parallel_size} ...", flush=True)
    llm = LLM(model=args.model, tensor_parallel_size=args.tensor_parallel_size,
              gpu_memory_utilization=args.gpu_memory_utilization,
              max_model_len=args.max_model_len, dtype="bfloat16",
              limit_mm_per_prompt={"image": 0, "video": 1}, trust_remote_code=True)
    sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=args.max_tokens)

    def prepare_one(vid):
        try:
            vp = get_video_path(vid)
            frames = extract_frames(vp, fps=args.fps, max_frames=args.max_frames)
        except Exception as e:
            return ("err", vid, f"video err: {e}")
        if not frames: return ("err", vid, "empty frames")
        messages = [
            {"role": "system", "content": STEP1_SYSTEM},
            {"role": "user", "content": [
                {"type": "video", "video": frames},
                {"type": "text", "text": STEP1_PROMPT},
            ]},
        ]
        try:
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception as e:
            return ("err", vid, f"template err: {e}")
        frames_np = np.stack([np.array(f.convert("RGB")) for f in frames])
        return ("ok", vid, {"prompt": text, "multi_modal_data": {"video": frames_np}})

    def chunked(seq, n):
        for i in range(0, len(seq), n): yield seq[i: i + n]

    pool = ThreadPoolExecutor(max_workers=args.decode_workers)
    batch_iter = list(chunked(todo, args.batch_size))
    pending: Queue = Queue()
    next_idx = 0
    def schedule(idx):
        batch = batch_iter[idx]
        pending.put((idx, [pool.submit(prepare_one, x) for x in batch]))
    for _ in range(min(args.prefetch_batches + 1, len(batch_iter))):
        schedule(next_idx); next_idx += 1

    from tqdm import tqdm
    n_done = n_err = 0
    pbar = tqdm(total=len(batch_iter), desc="step1 batches")
    while not pending.empty():
        idx, futures = pending.get()
        inputs, meta = [], []
        for fut in futures:
            st, vid, payload = fut.result()
            if st == "err":
                n_err += 1; continue
            inputs.append(payload); meta.append(vid)
        if next_idx < len(batch_iter):
            schedule(next_idx); next_idx += 1
        if not inputs:
            pbar.update(1); continue
        try:
            outs = llm.generate(inputs, sp)
        except Exception as e:
            print(f"  batch fail: {str(e)[:200]}", flush=True)
            outs, ok = [], []
            for vid, inp in zip(meta, inputs):
                try:
                    o = llm.generate([inp], sp); outs.extend(o); ok.append(vid)
                except Exception:
                    n_err += 1
            meta = ok
        for vid, o in zip(meta, outs):
            save_step1(args.output, vid, o.outputs[0].text.strip())
            n_done += 1
        pbar.update(1); pbar.set_postfix(done=n_done, err=n_err)
    pbar.close(); pool.shutdown(wait=True)
    print(f"\n✅ Step 1 done. {n_done} descriptions, {n_err} errors.", flush=True)


# ───── Step 2 driver (text-only) ───────────────────────────────────────
def run_step2(args):
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    with open(ANN_PATH) as f:
        items = [json.loads(l) for l in f if l.strip()]

    todo = []
    for it in items:
        desc = load_step1(args.output, it["video_id"])
        if desc is None:
            continue  # need step 1 first
        if load_step2(args.output, it["video_id"], it["question_id"]) is not None:
            continue
        todo.append((it, desc))
    print(f"step 2 todo: {len(todo)}", flush=True)
    if not todo: print("✅ step 2 all cached"); return

    print(f"Loading processor {args.model} ...", flush=True)
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    print(f"Loading vLLM TP={args.tensor_parallel_size} text-only ...", flush=True)
    llm = LLM(model=args.model, tensor_parallel_size=args.tensor_parallel_size,
              gpu_memory_utilization=args.gpu_memory_utilization,
              max_model_len=args.max_model_len, dtype="bfloat16",
              limit_mm_per_prompt={"image": 0, "video": 0}, trust_remote_code=True)
    sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=args.max_tokens)

    # Build all chat prompts (text-only)
    def build_prompt(item, desc):
        messages = [
            {"role": "system", "content": STEP2_SYSTEM},
            {"role": "user", "content": build_step2_prompt(desc, item)},
        ]
        return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    BS = args.batch_size
    from tqdm import tqdm
    n_done = 0
    for i in tqdm(range(0, len(todo), BS), desc="step2 batches"):
        batch = todo[i: i + BS]
        prompts = [build_prompt(it, desc) for it, desc in batch]
        try:
            outs = llm.generate(prompts, sp)
        except Exception as e:
            print(f"  step2 batch fail: {str(e)[:200]}", flush=True); continue
        for (it, _), o in zip(batch, outs):
            save_step2(args.output, it["video_id"], it["question_id"],
                        o.outputs[0].text.strip(), it)
            n_done += 1
    print(f"\n✅ Step 2 done. {n_done} oracle notes.", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--step", required=True, choices=["1", "2", "all"])
    p.add_argument("--model", default="Qwen/Qwen2.5-VL-72B-Instruct")
    p.add_argument("--output", default="results_scivideobench")
    p.add_argument("--tensor_parallel_size", type=int, default=4)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    p.add_argument("--max_model_len", type=int, default=32768)
    p.add_argument("--max_tokens", type=int, default=800)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--decode_workers", type=int, default=16)
    p.add_argument("--prefetch_batches", type=int, default=2)
    p.add_argument("--fps", type=float, default=1.0)
    p.add_argument("--max_frames", type=int, default=32)
    args = p.parse_args()

    if args.step in ("1", "all"):
        run_step1(args)
    if args.step in ("2", "all"):
        run_step2(args)


if __name__ == "__main__":
    main()
