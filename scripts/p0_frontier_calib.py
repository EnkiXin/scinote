"""P0 — frontier-localization calibration (V10 plan §4, gate P0).

R3's steppred head overrides the answer to frontier+1, where frontier = the
last step actually shown in the clip. Its entire value rests on ONE unmeasured
quantity: how accurately can the model localize that frontier? This pilot
measures it directly, in two prompt modes (the critique found answer-only
overshoots, cot-mode is calibrated — verify).

steppred gold = the NEXT step number, so the true frontier = gold - 1.

Per steppred item (n=145), 32 frames shared:
  - cot mode    : "reason about which steps are visible, then output JSON
                  {last_observed_step: N, evidence: ...}"  (max 768 tok)
  - answer mode : the plain steppred prompt -> next step -> frontier = next-1

Report per mode: frontier-exact (last_observed == gold-1), frontier within +-1,
median signed offset (pred_frontier - true_frontier), and the IMPLIED steppred
accuracy if we answered frontier+1.

GATE (plan): frontier-exact < 40% in BOTH modes -> R3 steppred head demoted.

Usage:
  CUDA_VISIBLE_DEVICES=0,1 python -m scripts.p0_frontier_calib \
    --model Qwen/Qwen2.5-VL-72B-Instruct
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

FRONTIER_PROMPT = (
    "{q}\n\n"
    "Before answering, determine the FRONTIER: the number of the LAST step "
    "from the list that is actually performed/visible in this video clip. "
    "Reason briefly about which steps you can see, then end with ONE line "
    "of exactly this form:\n"
    'LAST OBSERVED STEP: <integer>'
)


def parse_last_observed(raw: str) -> int | None:
    m = None
    for m_ in re.finditer(r"LAST\s+OBSERVED\s+STEP\s*[:=]?\s*(\d+)", raw or "", re.I):
        m = m_
    if m:
        return int(m.group(1))
    return None  # no marker -> uncounted (don't grab stray step-list integers)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-72B-Instruct")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="results_unified/p0_frontier.jsonl")
    args = ap.parse_args()

    from evaluate_c0_test_split import extract_frames, BUILDERS, parse_for_task
    from protonote.data.loaders import load_test_split, resolve_video_path
    from protonote.cli import VLMClient

    items = [it for it in load_test_split(benchmark="expvid")
             if it["task_type"] == "steppred"]
    if args.limit:
        items = items[:args.limit]
    out_path = ROOT / args.out
    done = set()
    if out_path.exists():
        for line in open(out_path):
            try:
                done.add(json.loads(line)["uid"])
            except Exception:
                pass
    todo = [it for it in items if it["uid"] not in done]
    print(f"[p0] {len(todo)}/{len(items)} steppred items to run", flush=True)

    vlm = VLMClient(model_name=args.model, device=args.device)
    print("[p0] model ready", flush=True)

    agg = {"cot": [], "ans": []}  # signed offsets (pred_frontier - true_frontier)
    t0 = time.time()
    with open(out_path, "a") as fout:
        for i, item in enumerate(todo):
            uid = item["uid"]
            rec = {"uid": uid, "gold_next": item["gold"]}
            try:
                true_frontier = int(item["gold"]) - 1
                vp = resolve_video_path(item)
                frames = extract_frames(vp, max_frames=32)
                if not frames:
                    raise RuntimeError("no frames")
                q = item["question"]

                # cot frontier mode
                msgs = [{"role": "user", "content": [
                    {"type": "video", "video": frames, "max_pixels": 360 * 420},
                    {"type": "text", "text": FRONTIER_PROMPT.format(q=q)},
                ]}]
                raw_c = vlm.generate(msgs, max_new_tokens=768)
                fc = parse_last_observed(raw_c)

                # answer-only steppred (frontier = next - 1)
                amsgs = BUILDERS["steppred"](item, frames, None)
                raw_a = vlm.generate(amsgs, max_new_tokens=16)
                pa = parse_for_task(raw_a, "steppred", item)
                na = int(re.findall(r"\d+", pa)[0]) if re.findall(r"\d+", pa) else None
                fa = (na - 1) if na is not None else None

                rec.update({"true_frontier": true_frontier,
                            "cot_frontier": fc, "ans_next": na, "ans_frontier": fa,
                            "raw_cot": raw_c[:300]})
                if fc is not None:
                    agg["cot"].append(fc - true_frontier)
                if fa is not None:
                    agg["ans"].append(fa - true_frontier)
            except Exception as e:
                rec["error"] = str(e)[:200]
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            if (i + 1) % 10 == 0 or i + 1 == len(todo):
                def stat(offs):
                    if not offs:
                        return "n=0"
                    ex = sum(1 for o in offs if o == 0) / len(offs)
                    w1 = sum(1 for o in offs if abs(o) <= 1) / len(offs)
                    med = sorted(offs)[len(offs) // 2]
                    return f"exact={ex*100:.0f}% ±1={w1*100:.0f}% med={med:+d} (n={len(offs)})"
                print(f"  [{i+1}/{len(todo)}] cot[{stat(agg['cot'])}] "
                      f"ans[{stat(agg['ans'])}] {time.time()-t0:.0f}s", flush=True)

    def final(offs, label):
        if not offs:
            print(f"[p0] {label}: no data")
            return
        ex = sum(1 for o in offs if o == 0) / len(offs)
        impl = ex  # frontier+1 exact == frontier exact
        print(f"[p0] {label}: frontier-exact={ex*100:.1f}% "
              f"-> implied steppred acc={impl*100:.1f}% "
              f"(current cot steppred ~12.4%); "
              f"{'PASS' if ex >= 0.40 else 'below 40%'}", flush=True)
    final(agg["cot"], "COT-mode")
    final(agg["ans"], "ANSWER-mode")


if __name__ == "__main__":
    main()
