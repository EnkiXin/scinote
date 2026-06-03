"""Precision-Gate experiment (controlled ablation) — see the plan in chat /
EXPERIMENTS_C0-C9.md §6,§11 style.

Per item: extract frames ONCE, run tools ONCE, cache (precision ATOMS + C1
prose NoteBuffer + a 0-5 confidence). Then answer 5 variants that differ ONLY
in the `note` text passed to BUILDERS (same model/frames/decoding):

  C0        note=None
  C1_fixed  note=full prose NoteBuffer (tools_for_task: visual_inspect[+ocr])
  P0        note=precision atoms only (OCR numbers/labels + existence{present,frame})
  P1        note=atoms if classify_task(item) in GATE_T else None   (Gate-T, headline)
  P2        note=atoms if conf<tau else None                        (Gate-C, stretch)

Precision = atoms only, NO prose: OCR is filtered to short/numeric tokens;
existence_verify is a constrained 2-line VLM call coerced to {present,frame_idx},
the prose discarded. Enforced by _assert_no_prose.

Usage:
  CUDA_VISIBLE_DEVICES=4,5 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 python -m scripts.precision_gate \
    --model Qwen/Qwen2.5-VL-72B-Instruct --device auto \
    --benchmark expvid --fixed_set tools/fixed_small_set_expvid.json \
    --num_chunks 2 --chunk_id 0 --out results_precision_gate/pilot/expvid_72b_c0.jsonl
  python -m scripts.precision_gate --selftest   # no-prose unit check, no GPU
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

GATE_T = {"sequence_ordering", "step_prediction", "video_verification", "sequence_generation"}
TAU = 3.0
_VIS_QUERY = ("In 1-2 sentences, describe the key actions, materials, and any "
              "visible labels/quantities in this clip.")


# ── precision atom extraction (prose -> atoms; prose discarded) ────────
def ocr_prose_to_atoms(content: str) -> list[str]:
    """Keep ONLY short/numeric tokens from the OCR bullet list; drop prose."""
    out, seen = [], set()
    for ln in (content or "").splitlines():
        s = re.sub(r"^[\s\-•*]+", "", ln)        # strip bullet markers + leading ws
        s = re.sub(r"^\d+[.)]\s+", "", s).strip()  # strip list-number "1. "/"1) " (not "2.0")
        if not s or len(s) > 60:
            continue
        toks = s.split()
        if any(ch.isdigit() for ch in s) or len(toks) <= 4:
            if s.lower() not in seen:
                seen.add(s.lower())
                out.append(s)
    return out[:12]


def existence_verify(vlm, frames, question: str) -> dict:
    """Constrained visual_inspect -> {present, frame_idx}. Prose DISCARDED."""
    sub = frames if len(frames) <= 8 else [frames[int(i * len(frames) / 8)] for i in range(8)]
    probe = (
        "Frames are sampled in temporal order. Is the specific activity/object "
        f"the question is about visibly present in this clip?\nQuestion: {question[:160]}\n"
        "Reply EXACTLY two lines and nothing else:\nPRESENT: yes|no\n"
        "FRAME: <0-based index of the frame where it first appears, or -1>")
    messages = [
        {"role": "system", "content": "Answer ONLY in the requested two-line format. No prose."},
        {"role": "user", "content": [
            {"type": "video", "video": sub, "max_pixels": 360 * 420},
            {"type": "text", "text": probe}]},
    ]
    raw = vlm.generate(messages, max_new_tokens=16)
    present = bool(re.search(r"PRESENT\s*[:=]\s*y", raw, re.I))
    m = re.search(r"FRAME\s*[:=]\s*(-?\d+)", raw, re.I)
    fi = int(m.group(1)) if m else -1
    return {"present": present, "frame_idx": (fi if 0 <= fi < len(sub) else None)}


def render_atoms(atoms: dict) -> str:
    lines = ["Extracted facts (verified, no description):"]
    for s in atoms.get("ocr_strings", []):
        lines.append(f"- text: {s}")
    ev = atoms.get("existence")
    if ev is not None:
        lines.append(f"- present: {'yes' if ev['present'] else 'no'}; frame_idx: {ev['frame_idx']}")
    for t in atoms.get("temporal", []):
        lines.append(f"- temporal: {t}")
    return "\n".join(lines) if len(lines) > 1 else ""


_ATOM_LINE = re.compile(r"^- (text|present|temporal|frame_idx): ")


def _assert_no_prose(block: str, prose_samples: list[str]) -> None:
    """Fail-fast: every bullet is a SHORT atom line; no tool prose leaked in."""
    if not block:
        return
    for ln in block.splitlines()[1:]:
        ln = ln.strip()
        if not ln:
            continue
        if not _ATOM_LINE.match(ln):
            raise AssertionError(f"non-atom line leaked into precision block: {ln!r}")
        m = re.match(r"^- \w+:\s*(.*)$", ln)
        val = m.group(1) if m else ln
        if len(val) > 80 or ". " in val:
            raise AssertionError(f"prose-like value in atom line: {ln!r}")
    for p in prose_samples:
        p = (p or "").strip()
        if len(p) >= 40 and p[:40] in block:
            raise AssertionError("tool prose leaked into precision block")


# ── confidence ask (Gate-C source) ────────────────────────────────────
def _append_conf_suffix(messages: list) -> list:
    import copy
    m = copy.deepcopy(messages)
    suffix = "\nAfter your answer, output a new line exactly: CONFIDENCE: <0-5> (5 = certain)."
    for msg in m:
        if msg.get("role") == "user":
            c = msg["content"]
            if isinstance(c, str):
                msg["content"] = c + suffix
            elif isinstance(c, list):
                for part in c:
                    if part.get("type") == "text":
                        part["text"] = part["text"] + suffix
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--benchmark", default="expvid")
    ap.add_argument("--fixed_set", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--num_chunks", type=int, default=1)
    ap.add_argument("--chunk_id", type=int, default=0)
    ap.add_argument("--max_frames", type=int, default=16)
    ap.add_argument("--out", default="results_precision_gate/run.jsonl")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        _run_selftest()
        return

    from evaluate_c0_test_split import (BUILDERS, parse_for_task, gold_for, extract_frames)
    from evaluate_unified import SCORERS
    from protonote.data.loaders import load_test_split, resolve_video_path
    from protonote.planner.task_classifier import classify_task
    from protonote.planner.tool_policy import tools_for_task
    from protonote.notes.note_buffer import NoteBuffer
    from protonote.notes.note_schema import NoteEntry
    from protonote.tools import build_default_tools
    from protonote.videoagent2.agent import _parse_assessment
    from protonote.cli import VLMClient

    items = load_test_split(benchmark=args.benchmark, limit=None)
    if args.fixed_set:
        ids = set(json.load(open(ROOT / args.fixed_set))["ids"])
        items = [it for it in items if it.get("sample_id") in ids]
    if args.limit > 0:
        items = items[:args.limit]
    if args.num_chunks > 1:
        items = [it for i, it in enumerate(items) if i % args.num_chunks == args.chunk_id]
    print(f"[pgate] {len(items)} items {args.benchmark} chunk={args.chunk_id}/{args.num_chunks} model={args.model}", flush=True)

    vlm = VLMClient(model_name=args.model, device=args.device)
    nb = NoteBuffer(cache_dir=str(ROOT / "results_precision_gate" / "_notecache"))
    tools = build_default_tools(vlm, nb)
    print("[pgate] model + tools ready", flush=True)

    def answer_for(item, frames, note, task_type):
        if task_type == "mc":
            messages = BUILDERS[task_type](item, frames, note, item["benchmark"])
        else:
            messages = BUILDERS[task_type](item, frames, note)
        max_new = 8 if task_type == "mc" else 64
        raw = vlm.generate(messages, max_new_tokens=max_new)
        pred = parse_for_task(raw, task_type, item)
        return pred, raw, messages

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    agg = {}  # (cond, task_type) -> [n, sum]
    t0 = time.time()
    with open(out_path, "w") as fout:
        for i, item in enumerate(items):
            sid = item.get("sample_id", f"?_{i}")
            tt = item.get("task_type", "mc")
            task = classify_task(item)
            rec = {"sample_id": sid, "benchmark": item.get("benchmark"), "task": task, "task_type": tt}
            try:
                vp = resolve_video_path(item)
                if not vp:
                    raise RuntimeError("no_video")
                frames = extract_frames(vp, max_frames=args.max_frames)
                if not frames:
                    raise RuntimeError("no_frames")
                gold = gold_for(item)
                q = item.get("question", "")

                # ---- extract once: tools + atoms + C1 prose ----
                routed = tools_for_task(task)
                vis_prose = ocr_prose = ""
                atoms = {"ocr_strings": [], "existence": None, "temporal": []}
                # OCR always (precision atom); also feeds C1 if routed
                ocr_res = tools["ocr"](video_path=vp, focus_query=q[:160])
                if ocr_res.success and ocr_res.content:
                    ocr_prose = ocr_res.content
                    atoms["ocr_strings"] = ocr_prose_to_atoms(ocr_prose)
                # visual_inspect prose (for C1)
                vis_res = tools["visual_inspect"](video_path=vp, query=_VIS_QUERY)
                if vis_res.success and vis_res.content:
                    vis_prose = vis_res.content
                # existence atom (precision)
                atoms["existence"] = existence_verify(vlm, frames, q)

                # C1_fixed note = render of exactly tools_for_task(task) outputs
                nb.reset_for_video(sid)  # isolate per item (no cross-question accumulation)
                for tn in routed:
                    if tn == "visual_inspect" and vis_prose:
                        nb.append_entry(sid, NoteEntry(section="Visual", content=vis_prose, evidence=[vis_res.evidence]))
                    elif tn == "ocr" and ocr_prose:
                        nb.append_entry(sid, NoteEntry(section="OCR", content=ocr_prose, evidence=[ocr_res.evidence]))
                c1_note = nb.render_for_llm(sid, question_context=q) if nb.num_entries(sid) > 0 else None

                atoms_note = render_atoms(atoms) or None
                if atoms_note:
                    _assert_no_prose(atoms_note, [vis_prose, ocr_prose])

                # ---- confidence (separate call; do NOT use for C0 pred) ----
                c0_msgs = BUILDERS[tt](item, frames, None, item["benchmark"]) if tt == "mc" else BUILDERS[tt](item, frames, None)
                conf_raw = vlm.generate(_append_conf_suffix(c0_msgs), max_new_tokens=(24 if tt == "mc" else 96))
                _, conf = _parse_assessment(conf_raw)

                # ---- 3 distinct answer calls; P1/P2 reuse by gate ----
                p_none = answer_for(item, frames, None, tt)
                p_c1 = answer_for(item, frames, c1_note, tt) if c1_note else p_none
                p_atoms = answer_for(item, frames, atoms_note, tt) if atoms_note else p_none

                def sc(pred):
                    return float(SCORERS[tt](pred, gold))
                s_c0 = sc(p_none[0]); s_c1 = sc(p_c1[0]); s_p0 = sc(p_atoms[0])
                p1_on = task in GATE_T
                p2_on = conf < TAU
                res = {
                    "C0":       {"pred": p_none[0], "score": s_c0, "note_used": False},
                    "C1_fixed": {"pred": p_c1[0],   "score": s_c1, "note_used": c1_note is not None},
                    "P0":       {"pred": p_atoms[0], "score": s_p0, "note_used": atoms_note is not None},
                    "P1":       {"pred": (p_atoms[0] if p1_on and atoms_note else p_none[0]),
                                 "score": (s_p0 if p1_on and atoms_note else s_c0), "note_used": bool(p1_on and atoms_note)},
                    "P2":       {"pred": (p_atoms[0] if p2_on and atoms_note else p_none[0]),
                                 "score": (s_p0 if p2_on and atoms_note else s_c0), "note_used": bool(p2_on and atoms_note)},
                }
                rec.update({"gold": gold, "conf": conf, "p1_gate_on": p1_on, "p2_gate_on": p2_on,
                            "n_ocr_atoms": len(atoms["ocr_strings"]), "existence": atoms["existence"],
                            "c1_note_chars": len(c1_note) if c1_note else 0,
                            "atoms_note": atoms_note, "results": res,
                            "raw_c0": p_none[1][:160]})
                for cond, sub in res.items():
                    k = (cond, tt); a = agg.setdefault(k, [0, 0.0]); a[0] += 1; a[1] += sub["score"]
            except Exception as e:
                import traceback
                rec["error"] = f"{type(e).__name__}: {e}"; rec["trace"] = traceback.format_exc()[-400:]
            fout.write(json.dumps(rec, default=str, ensure_ascii=False) + "\n"); fout.flush()
            if (i + 1) % 5 == 0:
                acc = {c: round(100 * v[1] / max(1, v[0]), 1) for (c, _t), v in
                       {(c, "_"): [sum(a[0] for (cc, _), a in agg.items() if cc == c),
                                    sum(a[1] for (cc, _), a in agg.items() if cc == c)] for c in
                        ["C0", "C1_fixed", "P0", "P1", "P2"]}.items()}
                print(f"  [{i+1}/{len(items)}] {acc} {time.time()-t0:.0f}s", flush=True)
    # summary
    summ = {}
    for (c, ttp), (n, s) in agg.items():
        summ.setdefault(c, {})[ttp] = {"n": n, "acc": round(100 * s / max(1, n), 2)}
    json.dump(summ, open(str(out_path).replace(".jsonl", "_summary.json"), "w"), indent=2)
    print(f"[pgate] done -> {out_path}\n{json.dumps(summ, indent=1)}", flush=True)


def _run_selftest():
    # no-GPU unit check that precision render never carries prose
    atoms = {"ocr_strings": ["2.0 mL", "METTLER", "0/10"], "existence": {"present": True, "frame_idx": 4}, "temporal": ["before(a,b)=yes"]}
    block = render_atoms(atoms)
    print("render_atoms output:\n" + block)
    _assert_no_prose(block, ["The researcher carefully pipettes the buffer into the tube while observing."])
    # leak attempt must fail
    bad = block + "\n- text: The researcher carefully pipettes the buffer into the tube while observing the reaction."
    try:
        _assert_no_prose(bad, [])
        raise SystemExit("SELFTEST FAILED: prose not caught")
    except AssertionError:
        pass
    # ocr filter drops prose lines, keeps atoms
    a = ocr_prose_to_atoms("- 2.0 mL\n- The person adds buffer to the centrifuge tube carefully and waits\n- pH 7.4\n- METTLER TOLEDO")
    assert "2.0 mL" in a and "pH 7.4" in a and all("carefully" not in x for x in a), a
    print("ocr atoms:", a)
    print("SELFTEST PASSED: no prose leaks; ocr filter ok")


if __name__ == "__main__":
    main()
