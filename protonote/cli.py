"""protonote/cli.py — entry-point for the ProtoNote agent.

Phase 0: agent is a single Qwen2.5-VL-7B call (no tools, no notes). Reproduces
the fresh-pipeline C0 baseline (ExpVid 26.73% overall).

Phase 3+ will swap this for a ReAct loop that uses tools (Phase 2) and the
NoteBuffer (Phase 1).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from evaluate_unified import (  # noqa: E402
    MC_SYSTEM, FITB_SYSTEM, SEQGEN_SYSTEM, STEPPRED_SYSTEM,
    SCORERS, MAX_PIXELS,
)
# Reuse the C0 evaluator's prompt builders + parsers (same fresh-pipeline behavior)
from evaluate_c0_test_split import (  # noqa: E402
    BUILDERS, parse_for_task, gold_for, extract_frames,
)

from protonote.data.loaders import load_test_split, resolve_video_path  # noqa: E402


# ── LLM client (HF transformers in-process for Phase 0) ─────────────────────

class VLMClient:
    """Wraps a Qwen2.5-VL model. Single .generate() call per question.

    Phase 2 (tools) will share this client across the visual / OCR tools too.
    For Phase 0 it is only used by the agent's answer call.
    """

    def __init__(self, model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
                 device: str = "cuda:0", dtype=torch.bfloat16):
        self.model_name = model_name
        self.device = device
        print(f"[VLMClient] loading {model_name} on {device}", flush=True)
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name, dtype=dtype, device_map=device)
        self.model.eval()
        print(f"[VLMClient] loaded", flush=True)

    @torch.no_grad()
    def generate(self, messages: list, max_new_tokens: int = 64) -> str:
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages, return_video_kwargs=True)
        if "fps" in video_kwargs and isinstance(video_kwargs["fps"], list):
            video_kwargs["fps"] = video_kwargs["fps"][0] if video_kwargs["fps"] else 1.0
        inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs,
                                 return_tensors="pt", **video_kwargs)
        inputs = {k: v.to(self.model.device) if hasattr(v, "to") else v
                  for k, v in inputs.items()}
        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        raw = self.processor.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        return raw


# ── Agent ───────────────────────────────────────────────────────────────────

class ProtoNoteAgent:
    """Phase 0 implementation: single-call C0 baseline agent.

    The trajectory format is generic (list of steps) so Phase 3+ can extend
    without changing the return type.
    """

    def __init__(self, vlm: VLMClient, condition: str = "C0"):
        self.vlm = vlm
        self.condition = condition  # Phase 0 only supports "C0"

    def answer(self, item: dict, max_frames: int = 32) -> dict:
        """Returns: dict with sample_id / pred / gold / score / trajectory."""
        out = {
            "sample_id": item["sample_id"],
            "benchmark": item["benchmark"], "task": item.get("task"),
            "task_type": item.get("task_type", "mc"),
            "gold": gold_for(item),
            "condition": self.condition,
            "trajectory": [],
        }
        try:
            vp = resolve_video_path(item)
            if not vp:
                return {**out, "error": "no_video"}
            frames = extract_frames(vp, max_frames=max_frames)
            if not frames:
                return {**out, "error": "no_frames"}
        except Exception as e:
            return {**out, "error": f"video err: {str(e)[:120]}"}

        task_type = item.get("task_type", "mc")
        builder = BUILDERS[task_type]
        # C0: no note context (None)
        if task_type == "mc":
            messages = builder(item, frames, None, item["benchmark"])
        else:
            messages = builder(item, frames, None)

        max_new = 8 if task_type == "mc" else 64
        try:
            t0 = time.time()
            raw = self.vlm.generate(messages, max_new_tokens=max_new)
            elapsed = time.time() - t0
        except Exception as e:
            return {**out, "error": f"gen err: {str(e)[:120]}"}

        pred = parse_for_task(raw, task_type, item)
        scorer = SCORERS[task_type]
        sc = float(scorer(pred, out["gold"]))

        out["trajectory"].append({
            "step": 0,
            "action": "answer",  # Phase 0: single answer step
            "raw": raw[:200],
            "elapsed_s": round(elapsed, 3),
        })
        out["pred"] = pred
        out["raw"] = raw[:120]
        out["score"] = sc
        return out


# ── CLI ─────────────────────────────────────────────────────────────────────

def _build_agent(condition: str, vlm, notes_cache_dir: str):
    """Factory: return either the C0 single-call agent or the Phase 3
    fixed-schedule tool agent."""
    if condition == "C0":
        return ProtoNoteAgent(vlm=vlm, condition="C0")
    if condition == "C1_fixed":
        from protonote.notes.note_buffer import NoteBuffer
        from protonote.planner.controller import FixedScheduleAgent
        from protonote.tools import build_default_tools
        buf = NoteBuffer(cache_dir=notes_cache_dir)
        tools = build_default_tools(vlm=vlm, note_buffer=buf)
        return FixedScheduleAgent(vlm=vlm, tools=tools, note_buffer=buf)
    raise ValueError(f"unknown condition: {condition!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--benchmark", default="expvid", choices=["expvid", "scivideobench", "all"])
    ap.add_argument("--limit", type=int, default=50, help="Pilot limit; 0 = full split")
    ap.add_argument("--max_frames", type=int, default=32)
    ap.add_argument("--output_dir", default="results_protonote/pilot")
    ap.add_argument("--chunk_id", type=int, default=0)
    ap.add_argument("--num_chunks", type=int, default=1)
    ap.add_argument("--condition", default="C0", choices=["C0", "C1_fixed"],
                     help="C0 = single VLM call (baseline). C1_fixed = "
                          "task-routed tools → NoteBuffer → answer-with-notes.")
    ap.add_argument("--notes_cache", default="",
                     help="NoteBuffer cache dir (C1_fixed only). Defaults to "
                          "<output_dir>/notes_cache.")
    args = ap.parse_args()

    benchmark = None if args.benchmark == "all" else args.benchmark
    items = load_test_split(benchmark=benchmark, limit=args.limit if args.limit > 0 else None)
    if args.num_chunks > 1:
        items = [it for i, it in enumerate(items) if i % args.num_chunks == args.chunk_id]
    print(f"[cli] {len(items)} items (benchmark={benchmark}, "
          f"chunk={args.chunk_id}/{args.num_chunks}, "
          f"condition={args.condition})", flush=True)

    vlm = VLMClient(model_name=args.model, device=args.device)

    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    notes_cache_dir = args.notes_cache or str(out_dir / "notes_cache")
    agent = _build_agent(args.condition, vlm=vlm, notes_cache_dir=notes_cache_dir)

    out_path = out_dir / (
        f"trajectory_{args.benchmark}"
        + (f"_chunk{args.chunk_id}of{args.num_chunks}" if args.num_chunks > 1 else "")
        + ".jsonl"
    )

    results = []
    with open(out_path, "w") as fout:
        for i, item in enumerate(items):
            r = agent.answer(item, max_frames=args.max_frames)
            fout.write(json.dumps(r, default=str) + "\n")
            fout.flush()
            results.append(r)
            if i % 25 == 0:
                valid = [x for x in results if "score" in x]
                acc = sum(x["score"] for x in valid) / max(len(valid), 1) * 100
                print(f"  [{i+1}/{len(items)}] running acc={acc:.2f}% n={len(valid)}", flush=True)

    # Per-task summary
    by_task = {}
    n_err = 0
    for r in results:
        if "score" not in r:
            n_err += 1; continue
        t = r.get("task", "?")
        by_task.setdefault(t, []).append(r["score"])
    print()
    print(f"=== ProtoNote {args.benchmark} {args.condition} (limit={args.limit}) ===")
    for t, s in sorted(by_task.items()):
        print(f"  {t:<30} acc={100*sum(s)/len(s):.2f}%  n={len(s)}")
    all_scores = [r["score"] for r in results if "score" in r]
    if all_scores:
        print(f"  overall acc={100*sum(all_scores)/len(all_scores):.2f}%  "
              f"n_valid={len(all_scores)}  n_err={n_err}")
    print(f"\nTrajectory: {out_path}")


if __name__ == "__main__":
    main()
