"""Unified fair-comparison harness — ONE measurement contract for every condition.

Motivated by the 2026-06-11 harness audit: historical conclusions mixed five
different "C0" baselines (answer-only / CoT / direct-unified, 16 vs 32 frames,
different parsers), so condition deltas of ±5pp were not interpretable. Here
EVERYTHING is pinned: same items (uid-keyed, SciVB sample_id collisions
disambiguated), same 32-frame extraction (once per item), same task BUILDERS,
same greedy decode, same FIXED parse/score path. Conditions differ in exactly
one declared variable:

  c0   : BUILDERS answer-only, note=None                       (baseline)
  cot  : c0 prompt + step-by-step suffix, FINAL ANSWER line,   (vary: reasoning)
         marker-aware extraction, cot_tokens budget
  c1   : BUILDERS answer-only, note=C1_fixed prose NoteBuffer  (vary: injected note)
         built INLINE by the same model (classify_task -> tools_for_task ->
         visual_inspect/ocr -> NoteBuffer.render_for_llm), cached per uid

Repeatability noise floor: re-run with `--conditions c0 --tag rep2` in a fresh
process; rep2-vs-main flip rate on identical config bounds what delta sizes
are readable.

Usage (72B, 2 GPUs):
  CUDA_VISIBLE_DEVICES=0,1 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
  python -m scripts.unified_harness --model Qwen/Qwen2.5-VL-72B-Instruct \
    --benchmark expvid --num_chunks 2 --chunk_id 0 --tag main
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_VIS_QUERY = ("In 1-2 sentences, describe the key actions, materials, and any "
              "visible labels/quantities in this clip.")

_MARKER = re.compile(r"(?:FINAL|EXACT)?\s*ANSWER\s*[:：]", re.I)

ANSWER_TOKENS = {"mc": 8, "seqgen": 96, "steppred": 16, "fitb": 96}


def fmt_hint(item) -> str:
    tt = item.get("task_type", "mc")
    if tt == "mc":
        return "the SINGLE correct option letter (A, B, C, ...)"
    if tt == "seqgen":
        return "the space-separated step numbers (e.g. '3 4 5')"
    if tt == "steppred":
        return "ONLY the step NUMBER of the next step (a single integer)"
    if tt == "fitb":
        return "the value for each blank, separated by ' | '"
    return "the answer in the format the question requests"


def add_cot_suffix(messages, item):
    """Append the reasoning instruction to the user text of a BUILDERS prompt
    without touching anything else (frames, system prompt, options)."""
    import copy
    msgs = copy.deepcopy(messages)
    for part in reversed(msgs[-1]["content"]):
        if isinstance(part, dict) and part.get("type") == "text":
            part["text"] += (
                "\n\nFirst reason step by step about what the video actually "
                "shows and how it bears on the question. Keep the reasoning "
                "under 150 words — do NOT enumerate every protocol step. Then "
                "end with ONE line exactly of the form:\n"
                f"FINAL ANSWER: <{fmt_hint(item)}>"
            )
            break
    return msgs


def parse_cot(raw: str, task_type: str, item: dict):
    """Marker-aware extraction for verbose outputs (audit fix: never take the
    first stray letter of the reasoning)."""
    from evaluate_c0_test_split import parse_for_task
    from scripts.cot_ablation import extract_final
    raw = raw or ""
    if task_type == "mc":
        tail = extract_final(raw) if _MARKER.search(raw) else raw
        return parse_for_task(tail, task_type, item)
    if _MARKER.search(raw):
        return parse_for_task(extract_final(raw), task_type, item)
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    return parse_for_task(lines[-1] if lines else "", task_type, item)


class _V9VLMAdapter:
    """v9 stages expect vlm.generate_video(prompt, frames, max_tokens=, temperature=);
    VLMClient exposes generate(messages, max_new_tokens). Bridge the two."""

    def __init__(self, vlm, max_pixels):
        self._vlm = vlm
        self._max_pixels = max_pixels

    def generate_video(self, prompt, frames, max_tokens=512, temperature=0.0):
        msgs = [{"role": "user", "content": [
            {"type": "video", "video": frames, "max_pixels": self._max_pixels},
            {"type": "text", "text": prompt},
        ]}]
        return self._vlm.generate(msgs, max_new_tokens=max_tokens)

    def generate_image(self, prompt, image, max_tokens=256, temperature=0.0):
        # OCRLedgerBuilder contract (per-frame OCR calls)
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": image, "max_pixels": self._max_pixels},
            {"type": "text", "text": prompt},
        ]}]
        return self._vlm.generate(msgs, max_new_tokens=max_tokens)


def _kg_sparse_facts(kg, question: str, k: int = 2) -> str:
    """question-conditioned sparse injection (the 2026-06-01 pivot, never run):
    pick the k KG facts with the highest lexical overlap with the question.
    Deterministic, zero extra model calls."""
    qtok = set(re.findall(r"[a-z0-9]+", (question or "").lower()))
    facts = []
    for op in getattr(kg, "operations", []) or []:
        # v9 field names: action (verb phrase), action_category, timestamp
        d = getattr(op, "action", None) or getattr(op, "description", None) or ""
        if not d:
            continue
        ts = getattr(op, "timestamp", None)
        facts.append(f"Operation: {d}" + (f" (at ~{ts:.0f}s)" if isinstance(ts, (int, float)) else ""))
    for e in (getattr(kg, "entities", {}) or {}).values():
        name = getattr(e, "canonical_name", None) or ""
        if not name:
            continue
        feats = [getattr(s, "visual_features", None) or ""
                 for s in (getattr(e, "states", []) or [])]
        feats = [f for f in feats if f]
        facts.append(f"Entity: {name} ({getattr(e, 'type', '')})"
                     + (f" — {feats[0]}" if feats else ""))
    def score(t):
        return len(qtok & set(re.findall(r"[a-z0-9]+", t.lower())))
    top = sorted(facts, key=score, reverse=True)[:k]
    top = [t for t in top if score(t) > 0] or top[:1]
    return "\n".join(f"- {t}" for t in top) if top else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-72B-Instruct")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--benchmark", default="expvid", choices=["expvid", "scivideobench"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--num_chunks", type=int, default=1)
    ap.add_argument("--chunk_id", type=int, default=0)
    ap.add_argument("--max_frames", type=int, default=32)
    ap.add_argument("--cot_tokens", type=int, default=1536)
    ap.add_argument("--conditions", default="c0,cot,c1")
    ap.add_argument("--tag", default="main")
    ap.add_argument("--out_dir", default="results_unified")
    args = ap.parse_args()
    conds = [c.strip() for c in args.conditions.split(",") if c.strip()]

    from evaluate_c0_test_split import BUILDERS, parse_for_task, gold_for, extract_frames
    from evaluate_unified import SCORERS
    from protonote.data.loaders import load_test_split, resolve_video_path
    from protonote.planner.task_classifier import classify_task
    from protonote.planner.tool_policy import tools_for_task
    from protonote.notes.note_buffer import NoteBuffer
    from protonote.notes.note_schema import NoteEntry
    from protonote.tools import build_default_tools
    from protonote.cli import VLMClient

    items = load_test_split(benchmark=args.benchmark, limit=None)
    if args.limit > 0:
        items = items[:args.limit]
    if args.num_chunks > 1:
        items = [it for i, it in enumerate(items) if i % args.num_chunks == args.chunk_id]

    short = args.model.split("/")[-1].replace("Qwen2.5-VL-", "").replace("-Instruct", "").lower()
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{short}_{args.benchmark}_{args.tag}_chunk{args.chunk_id}of{args.num_chunks}.jsonl"
    notes_dir = out_dir / f"notes_cache_{short}_{args.benchmark}"
    notes_dir.mkdir(exist_ok=True)

    done = set()
    if out_path.exists():
        for line in open(out_path):
            try:
                r = json.loads(line)
                if "uid" in r:
                    done.add(r["uid"])
            except Exception:
                pass
    todo = [it for it in items if it["uid"] not in done]
    print(f"[unified] {args.benchmark} chunk {args.chunk_id}/{args.num_chunks} "
          f"model={short} conds={conds} tag={args.tag}: {len(todo)}/{len(items)} to run "
          f"(resume skips {len(done)})", flush=True)

    config = {"_config": {"model": args.model, "benchmark": args.benchmark,
                          "max_frames": args.max_frames, "cot_tokens": args.cot_tokens,
                          "answer_tokens": ANSWER_TOKENS, "conditions": conds,
                          "tag": args.tag, "parser": "fixed_2026-06-11"}}
    if not out_path.exists():
        with open(out_path, "w") as f:
            f.write(json.dumps(config) + "\n")

    vlm = VLMClient(model_name=args.model, device=args.device)
    nb = NoteBuffer(cache_dir=str(out_dir / "_notebuffer_cache"))
    tools = build_default_tools(vlm, nb) if "c1" in conds else None

    kg_machinery = None
    if "kg" in conds or "kgs" in conds:
        from evaluate_unified import MAX_PIXELS
        from protonote.v9.stages.stage1_1_core_entities import run_stage1_1
        from protonote.v9.stages.stage1_2_state_tracking import run_stage1_2
        from protonote.v9.stages.stage4_multi_view_strategist import render_multi_view_kg
        from protonote.v9.kg.state_machine_kg import StateMachineKG
        kg_dir = out_dir / f"kg_cache_{short}_{args.benchmark}"
        kg_dir.mkdir(exist_ok=True)
        v9vlm = _V9VLMAdapter(vlm, MAX_PIXELS)

        def build_kg(frames, uid):
            cache_f = kg_dir / (re.sub(r"[^\w#-]", "_", uid) + ".json")
            if cache_f.exists():
                return StateMachineKG.from_dict(json.loads(cache_f.read_text()))
            s1_1 = run_stage1_1(frames, v9vlm)
            s1_2 = run_stage1_2(frames, s1_1, [], v9vlm)  # ledger=[] — no OCR pass
            kg = s1_2.kg
            cache_f.write_text(json.dumps(kg.to_dict()))
            return kg

        KG_VIEWS = ["procedural", "conceptual", "quantitative", "hypothetical"]
        kg_machinery = (build_kg, render_multi_view_kg, KG_VIEWS)
    print("[unified] model ready", flush=True)

    def answer(item, frames, note, task_type):
        if task_type == "mc":
            messages = BUILDERS[task_type](item, frames, note, item["benchmark"])
        else:
            messages = BUILDERS[task_type](item, frames, note)
        raw = vlm.generate(messages, max_new_tokens=ANSWER_TOKENS.get(task_type, 64))
        return parse_for_task(raw, task_type, item), raw

    def build_c1_note(item, vp, q, uid):
        cache_f = notes_dir / (re.sub(r"[^\w#-]", "_", uid) + ".md")
        if cache_f.exists():
            note = cache_f.read_text()
            return note if note.strip() else None
        task = classify_task(item)
        nb.reset_for_video(uid)
        for tn in tools_for_task(task):
            if tn == "visual_inspect":
                res = tools["visual_inspect"](video_path=vp, query=_VIS_QUERY)
                if res.success and res.content:
                    nb.append_entry(uid, NoteEntry(section="Visual", content=res.content,
                                                   evidence=[res.evidence]))
            elif tn == "ocr":
                res = tools["ocr"](video_path=vp, focus_query=q[:160])
                if res.success and res.content:
                    nb.append_entry(uid, NoteEntry(section="OCR", content=res.content,
                                                   evidence=[res.evidence]))
        note = nb.render_for_llm(uid, question_context=q) if nb.num_entries(uid) > 0 else ""
        cache_f.write_text(note or "")
        return note or None

    agg = {}  # (cond, task_type) -> [n, sum]
    t0 = time.time()
    with open(out_path, "a") as fout:
        for i, item in enumerate(todo):
            tt = item.get("task_type", "mc")
            rec = {"uid": item["uid"], "sample_id": item.get("sample_id"),
                   "benchmark": item.get("benchmark"), "task": item.get("task"),
                   "task_type": tt, "gold": gold_for(item)}
            try:
                vp = resolve_video_path(item)
                if not vp:
                    raise RuntimeError("no_video")
                frames = extract_frames(vp, max_frames=args.max_frames)
                if not frames:
                    raise RuntimeError("no_frames")
                q = item.get("question", "")
                res = {}
                if "c0" in conds:
                    pred, raw = answer(item, frames, None, tt)
                    res["c0"] = {"pred": pred, "score": float(SCORERS[tt](pred, rec["gold"])),
                                 "raw": raw[:300]}
                if "cot" in conds:
                    msgs = (BUILDERS[tt](item, frames, None, item["benchmark"])
                            if tt == "mc" else BUILDERS[tt](item, frames, None))
                    raw = vlm.generate(add_cot_suffix(msgs, item), max_new_tokens=args.cot_tokens)
                    pred = parse_cot(raw, tt, item)
                    res["cot"] = {"pred": pred, "score": float(SCORERS[tt](pred, rec["gold"])),
                                  "raw": raw[:4000],
                                  "has_marker": bool(_MARKER.search(raw or ""))}
                if "c1" in conds:
                    note = build_c1_note(item, vp, q, item["uid"])
                    pred, raw = answer(item, frames, note, tt)
                    res["c1"] = {"pred": pred, "score": float(SCORERS[tt](pred, rec["gold"])),
                                 "raw": raw[:300], "note_chars": len(note or ""),
                                 "note_used": note is not None}
                if kg_machinery is not None:
                    build_kg, render_kg, KG_VIEWS = kg_machinery
                    kg_obj = build_kg(frames, item["uid"])
                    rec["kg_summary"] = {"n_entities": kg_obj.metadata.n_entities,
                                         "n_operations": kg_obj.metadata.n_operations}
                    if "kg" in conds:
                        note = render_kg(kg_obj, KG_VIEWS, include_edges=True)[:4000] or None
                        pred, raw = answer(item, frames, note, tt)
                        res["kg"] = {"pred": pred, "score": float(SCORERS[tt](pred, rec["gold"])),
                                     "raw": raw[:300], "note_chars": len(note or ""),
                                     "note_used": note is not None}
                    if "kgs" in conds:
                        note = _kg_sparse_facts(kg_obj, q) or None
                        pred, raw = answer(item, frames, note, tt)
                        res["kgs"] = {"pred": pred, "score": float(SCORERS[tt](pred, rec["gold"])),
                                      "raw": raw[:300], "note_chars": len(note or ""),
                                      "note_used": note is not None}
                rec["results"] = res
                for c, sub in res.items():
                    k = (c, tt)
                    agg.setdefault(k, [0, 0.0])
                    agg[k][0] += 1
                    agg[k][1] += sub["score"]
            except Exception as e:
                rec["error"] = str(e)[:200]
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            if (i + 1) % 10 == 0 or i + 1 == len(todo):
                parts = []
                for c in conds:
                    n = sum(v[0] for (cc, _), v in agg.items() if cc == c)
                    s = sum(v[1] for (cc, _), v in agg.items() if cc == c)
                    parts.append(f"{c}~{s / n:.3f}" if n else f"{c}~-")
                print(f"  [{i + 1}/{len(todo)}] " + " ".join(parts) +
                      f" {time.time() - t0:.0f}s", flush=True)

    print(f"[unified] done -> {out_path}", flush=True)
    for (c, tt), (n, s) in sorted(agg.items()):
        print(f"  {c:4s} {tt:9s} n={n:4d} mean={s / n:.4f}", flush=True)


if __name__ == "__main__":
    main()
