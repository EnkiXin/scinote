"""P2 wave-1 — uncertainty-triggered REWATCH (text-free), 3 arms.

Verified spec (workflow wf_0543a2eb). All arms deliver RAW FRAMES only —
nothing enters the Visual-notes slot (note=None) — because every text
injection harms or is null at 72B. Conditions differ from their paired
baseline by exactly one variable: which frames the model sees.

  arm-H   (ExpVid mc) : hedge-language fired items -> 64-frame whole-video
                        dense re-decode; paired vs same-process 32-frame cot.
  arm-SV  (SciVB mc)  : mm:ss-window items (guarded) -> 8 global + 24 in-window
                        frames; paired vs c0 (NOT cot — SciVB cot is -8.72).
  arm-Step(steppred)  : all 145 -> 64-frame dense + frontier head (frontier+1);
                        paired vs p0 32-frame frontier (same OFFSET).

GATE (per arm): McNemar on the discordant trigger subset, corrected>broken,
p<.05, net correction above the measured noise floor (ExpVid 1.61% / SciVB 0).

Usage:
  CUDA_VISIBLE_DEVICES=0,1 python -m scripts.p2_rewatch \
    --model Qwen/Qwen2.5-VL-72B-Instruct --arm H --out results_unified/p2_rwH.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

HEDGE_RE = re.compile(
    r"not shown|cannot determine|not visible|unclear|insufficient|not clearly", re.I)


def uid_of(sample_id: str, question: str) -> str:
    return f"{sample_id}#{hashlib.sha1((question or '').encode()).hexdigest()[:8]}"


def load_hedge_fired() -> set[str]:
    """Reconstruct the hedge-fired uid set from the 72B CoT-full ExpVid run."""
    fired = set()
    for c in (0, 1):
        p = ROOT / "results_72b_cot_full" / "expvid" / f"chunk{c}.jsonl"
        for line in open(p):
            r = json.loads(line)
            if r.get("task_type") != "mc":
                continue
            blob = (r.get("reasoning") or "") + " " + (r.get("final_answer") or "")
            if HEDGE_RE.search(blob):
                fired.add(uid_of(r["sample_id"], r["question"]))
    return fired


def load_c0(benchmark: str) -> dict:
    """uid -> c0 score from the frozen main matrix."""
    out = {}
    for c in (0, 1):
        p = ROOT / "results_unified" / f"72b_{benchmark}_main_chunk{c}of2.jsonl"
        for line in open(p):
            r = json.loads(line)
            if "_config" in r or "results" not in r or "c0" not in r.get("results", {}):
                continue
            out[r["uid"]] = r["results"]["c0"]["score"]
    return out


def load_p0_frontier() -> dict:
    out = {}
    p = ROOT / "results_unified" / "p0_frontier.jsonl"
    if p.exists():
        for line in open(p):
            r = json.loads(line)
            if r.get("cot_frontier") is not None:
                out[r["uid"]] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-72B-Instruct")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--arm", required=True, choices=["H", "SV", "Step"])
    ap.add_argument("--offset", type=int, default=1, help="frontier+OFFSET (steppred)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    from evaluate_c0_test_split import BUILDERS, parse_for_task, gold_for, extract_frames
    from evaluate_unified import SCORERS, MAX_PIXELS
    from ranker_pipeline.common.video_utils import extract_segment_frames, get_video_duration
    from scripts.probe_b_scivb_timestamp import parse_timestamps
    from scripts.p0_frontier_calib import FRONTIER_PROMPT, parse_last_observed
    from scripts.unified_harness import add_cot_suffix, parse_cot
    from protonote.cli import VLMClient
    from protonote.data.loaders import load_test_split, resolve_video_path

    bench = "scivideobench" if args.arm == "SV" else "expvid"
    items = load_test_split(benchmark=bench)
    tt_filter = {"H": "mc", "SV": "mc", "Step": "steppred"}[args.arm]
    items = [it for it in items if it.get("task_type") == tt_filter]

    # arm-specific trigger filtering + paired baseline source
    if args.arm == "H":
        fired = load_hedge_fired()
        items = [it for it in items if it["uid"] in fired]
        base_label = "cot (same-process 32f)"
    elif args.arm == "SV":
        c0map = load_c0("scivideobench")
        keep = []
        for it in items:
            ts = parse_timestamps(it["question"])
            if ts["kind"] in ("range", "multi_single"):
                it["_ts"] = ts
                keep.append(it)
        items = keep
        base_label = "c0 (frozen)"
    else:  # Step
        p0 = load_p0_frontier()
        base_label = "p0 32f frontier"

    if args.limit:
        items = items[:args.limit]
    out_path = ROOT / (args.out or f"results_unified/p2_rw{args.arm}.jsonl")
    done = set()
    if out_path.exists():
        for line in open(out_path):
            try:
                done.add(json.loads(line)["uid"])
            except Exception:
                pass
    todo = [it for it in items if it["uid"] not in done]
    print(f"[p2-{args.arm}] {len(todo)}/{len(items)} items, baseline={base_label}", flush=True)

    vlm = VLMClient(model_name=args.model, device=args.device)

    def answer(item, frames, tt):
        msgs = (BUILDERS["mc"](item, frames, None, item["benchmark"])
                if tt == "mc" else BUILDERS[tt](item, frames, None))
        from scripts.unified_harness import ANSWER_TOKENS
        raw = vlm.generate(msgs, max_new_tokens=ANSWER_TOKENS.get(tt, 64))
        return parse_for_task(raw, tt, item), raw

    print("[p2] model ready", flush=True)
    agg = []  # (rewatch_score, base_score)
    t0 = time.time()
    with open(out_path, "a") as fout:
        for i, item in enumerate(todo):
            uid = item["uid"]
            tt = item["task_type"]
            rec = {"uid": uid, "task_type": tt, "task": item.get("task"),
                   "gold": gold_for(item)}
            try:
                vp = resolve_video_path(item)
                if not vp:
                    raise RuntimeError("no video")

                if args.arm == "H":
                    frames = extract_frames(vp, max_frames=64)
                    rw_pred, rw_raw = answer(item, frames, "mc")
                    # paired base: same-process 32f cot
                    f32 = extract_frames(vp, max_frames=32)
                    cmsgs = BUILDERS["mc"](item, f32, None, item["benchmark"])
                    craw = vlm.generate(add_cot_suffix(cmsgs, item), max_new_tokens=1536)
                    base_pred = parse_cot(craw, "mc", item)
                    rw_s = float(SCORERS["mc"](rw_pred, rec["gold"]))
                    base_s = float(SCORERS["mc"](base_pred, rec["gold"]))
                    rec.update({"rw_pred": rw_pred, "base_pred": base_pred,
                                "rw_score": rw_s, "base_score": base_s, "n_frames": 64})

                elif args.arm == "SV":
                    ts = item["_ts"]
                    dur = get_video_duration(vp)
                    fb = False
                    if dur <= 0 or ts["t_hi"] > dur or (ts["t_hi"] - max(0.0, ts["t_lo"])) > 40:
                        rec["dropped"] = True
                        fout.write(json.dumps(rec, ensure_ascii=False) + "\n"); fout.flush()
                        continue
                    t_lo = max(0.0, ts["t_lo"])
                    win = extract_segment_frames(vp, t_lo, ts["t_hi"], n_frames=24, max_pixels=MAX_PIXELS)
                    glob = extract_segment_frames(vp, 0.0, dur, n_frames=8, max_pixels=MAX_PIXELS)
                    frames = (glob or []) + (win or [])
                    if len(frames) < 8:
                        fb = True
                    rw_pred, rw_raw = answer(item, frames, "mc")
                    rw_s = float(SCORERS["mc"](rw_pred, rec["gold"]))
                    base_s = load_c0("scivideobench").get(uid)
                    rec.update({"rw_pred": rw_pred, "rw_score": rw_s,
                                "base_score": base_s, "window": [t_lo, ts["t_hi"]],
                                "window_fallback": fb, "n_frames": len(frames)})
                    if base_s is None:
                        continue

                else:  # Step
                    frames = extract_frames(vp, max_frames=64)
                    fmsgs = [{"role": "user", "content": [
                        {"type": "video", "video": frames, "max_pixels": MAX_PIXELS},
                        {"type": "text", "text": FRONTIER_PROMPT.format(q=item["question"])}]}]
                    fraw = vlm.generate(fmsgs, max_new_tokens=768)
                    fc = parse_last_observed(fraw)
                    if fc is not None:
                        pred_next = str(fc + args.offset)
                    else:
                        pred_next, _ = answer(item, frames, "steppred")
                    rw_s = float(SCORERS["steppred"](pred_next, rec["gold"]))
                    # paired base: p0 32f frontier, same offset
                    pr = p0.get(uid)
                    base_s = None
                    if pr is not None:
                        bn = str(pr["cot_frontier"] + args.offset)
                        base_s = float(SCORERS["steppred"](bn, rec["gold"]))
                    true_front = int(rec["gold"]) - 1
                    rec.update({"rw_frontier": fc, "rw_pred_next": pred_next, "rw_score": rw_s,
                                "base_score": base_s, "n_frames": 64,
                                "undershoot_tail": pr is not None and (pr["cot_frontier"] - true_front) <= -6})

                if rec.get("base_score") is not None:
                    agg.append((rec["rw_score"], rec["base_score"]))
            except Exception as e:
                rec["error"] = str(e)[:200]
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n"); fout.flush()
            if (i + 1) % 10 == 0 or i + 1 == len(todo):
                rw = sum(a for a, _ in agg); bs = sum(b for _, b in agg)
                disc = [(a, b) for a, b in agg if a != b]
                cor = sum(1 for a, b in disc if a > b); brk = sum(1 for a, b in disc if a < b)
                n = max(1, len(agg))
                print(f"  [{i+1}/{len(todo)}] rw={rw/n:.3f} base={bs/n:.3f} "
                      f"discordant +{cor}/-{brk} (n={len(agg)}) {time.time()-t0:.0f}s", flush=True)

    # final McNemar
    import math
    disc = [(a, b) for a, b in agg if a != b]
    cor = sum(1 for a, b in disc if a > b); brk = sum(1 for a, b in disc if a < b)
    m = cor + brk
    p = 1.0 if m == 0 else min(1.0, sum(math.comb(m, i) for i in range(min(cor, brk) + 1)) / 2 ** m * 2)
    rw = sum(a for a, _ in agg) / max(1, len(agg)); bs = sum(b for _, b in agg) / max(1, len(agg))
    print(f"[p2-{args.arm}] FINAL n={len(agg)} rw={rw:.4f} base={bs:.4f} "
          f"net={(rw-bs)*100:+.2f}pp McNemar +{cor}/-{brk} p={p:.4f}", flush=True)


if __name__ == "__main__":
    main()
