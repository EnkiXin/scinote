"""STEP 2 PROBE (controlled): does adding explicit temporal edges to the
V9 Stage-4 prompt change the answer score?

Per item, extract the KG ONCE (Stage1.1 + 1.2 + router), then answer the
SAME question under 3 prompt conditions (greedy decode, deterministic):
  C0  vanilla         — no KG at all (answer-model-only baseline)
  C1  KG, no edges    — current multi-view KG, gate_kg=False (force shown)
  C2  KG + temporal   — C1 + explicit TEMPORAL EDGES block + instruction
Only the prompt differs between conditions (same frames/KG), so any score
delta is 100% attributable to the KG / edges. gate_kg is forced False so
the KG block is never skipped (otherwise ~40% of questions get no KG and
the probe is uninterpretable).

Also records STEP-1 VERIFY stats (non-empty input/output op ratio) and
temporal-edge yield.

Usage (4-way shard):
  for c in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$c python -m scripts.v9_step2_probe \
    --num_chunks 4 --chunk_id $c \
    --out results_protonote_v9/step2_probe/chunk$c.jsonl & done
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FIXED_SET = ROOT / "tools" / "fixed_small_set_expvid.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--benchmark", default="expvid")
    ap.add_argument("--num_chunks", type=int, default=1)
    ap.add_argument("--chunk_id", type=int, default=0)
    ap.add_argument("--max_frames", type=int, default=16)
    ap.add_argument("--stage1_2_max_tokens", type=int, default=4096)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dump_prompt", action="store_true",
                    help="dump C1/C2 prompts of first item for eyeballing")
    ap.add_argument("--capture", action="store_true",
                    help="store raw VLM outputs + rendered KG/edge markdown "
                         "per item (for the 'why does KG hurt' failure analysis)")
    args = ap.parse_args()

    from evaluate_c0_test_split import extract_frames, parse_for_task, gold_for
    from evaluate_unified import SCORERS
    from protonote.data.loaders import load_test_split, resolve_video_path
    from protonote.v6.llm_client import QwenVL72BClient
    from protonote.v9.preprocessing.ocr_preprocessor import OCRLedgerBuilder
    from protonote.v9.stages.stage1_1_core_entities import run_stage1_1
    from protonote.v9.stages.stage1_2_state_tracking import run_stage1_2
    from protonote.v9.stages.stage3_multi_label_router import determine_active_views
    from protonote.v9.stages.stage4_multi_view_strategist import (
        build_stage4_prompt, _build_vanilla_prompt, should_skip_kg,
        _temporal_edges, render_multi_view_kg, render_temporal_edges,
    )

    fixed = json.load(open(FIXED_SET))
    id_set = set(fixed["ids"])
    items = [it for it in load_test_split(benchmark=args.benchmark, limit=None)
             if it.get("sample_id") in id_set]
    if args.num_chunks > 1:
        items = [it for i, it in enumerate(items)
                 if i % args.num_chunks == args.chunk_id]
    print(f"[probe] {len(items)} items chunk={args.chunk_id}/{args.num_chunks}",
          flush=True)

    vlm = QwenVL72BClient(model_name=args.model, device=args.device)
    ocr_builder = OCRLedgerBuilder(vlm=vlm)
    print("[probe] model ready", flush=True)

    def _frame_ts(n, dur):
        if n <= 1 or dur <= 0:
            return [0.0] * max(1, n)
        step = dur / max(1, n - 1)
        return [round(i * step, 3) for i in range(n)]

    def answer(prompt, frames, item):
        raw = vlm.generate_video(prompt, frames, max_tokens=400, temperature=0.0)
        pred = parse_for_task(raw, item.get("task_type", "mc"), item)
        scorer = SCORERS.get(item.get("task_type", "mc"))
        gold = gold_for(item)
        return (float(scorer(pred, gold)) if scorer else 0.0), pred, gold, raw

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(out_path, "w") as fout:
        for i, item in enumerate(items):
            sid = item.get("sample_id", f"?_{i}")
            task_type = item.get("task_type", "mc")
            rec = {"sample_id": sid, "task_type": task_type}
            try:
                vp = resolve_video_path(item)
                if not vp or not Path(vp).exists():
                    rec["error"] = "no_video"
                    fout.write(json.dumps(rec, default=str) + "\n"); fout.flush(); continue
                frames = extract_frames(vp, max_frames=args.max_frames)
                if not frames:
                    rec["error"] = "no_frames"
                    fout.write(json.dumps(rec, default=str) + "\n"); fout.flush(); continue
                dur = item.get("duration_sec") or (args.max_frames * 1.0)
                ts = _frame_ts(len(frames), dur)
                try:
                    ledger = ocr_builder.build_ledger(frames, ts)
                except Exception:
                    ledger = []

                s1_1 = run_stage1_1(frames, vlm)
                s1_2 = run_stage1_2(frames, s1_1, ledger, vlm,
                                    max_tokens=args.stage1_2_max_tokens)
                kg = s1_2.kg
                q = item.get("question", "")
                opts = item.get("options", {})
                active_views, routing = determine_active_views(q, opts, vlm)

                edges = _temporal_edges(kg)
                rec.update({
                    "active_views": active_views,
                    "would_skip_kg": should_skip_kg(active_views),
                    "n_entities": len(kg.entities),
                    "n_operations": len(kg.operations),
                    "n_states": kg.metadata.n_states_total,
                    "n_transmutations": kg.metadata.n_transmutations,
                    "n_ops_nonempty_in": sum(1 for o in kg.operations if o.input_states),
                    "n_ops_nonempty_out": sum(1 for o in kg.operations if o.output_states),
                    "n_temporal_edges": len(edges),
                })

                # 3 conditions — same frames/KG, only prompt differs
                p0 = _build_vanilla_prompt(question=q, options=opts, task_type=task_type)
                p1 = build_stage4_prompt(question=q, options=opts, kg=kg,
                                         active_views=active_views, task_type=task_type,
                                         gate_kg=False, include_edges=False)
                p2 = build_stage4_prompt(question=q, options=opts, kg=kg,
                                         active_views=active_views, task_type=task_type,
                                         gate_kg=False, include_edges=True)
                if args.dump_prompt and i == 0:
                    (out_path.parent / f"PROMPTS_chunk{args.chunk_id}.txt").write_text(
                        f"==== C1 (no edges) ====\n{p1}\n\n==== C2 (edges) ====\n{p2}")

                s0, pr0, gold, raw0 = answer(p0, frames, item)
                s1, pr1, _, raw1 = answer(p1, frames, item)
                s2, pr2, _, raw2 = answer(p2, frames, item)
                rec.update({
                    "score_c0_vanilla": s0, "score_c1_kg": s1, "score_c2_kg_edges": s2,
                    "pred_c0": pr0, "pred_c1": pr1, "pred_c2": pr2, "gold": gold,
                })
                if args.capture:
                    rec["question"] = q
                    rec["raw_c0"] = raw0
                    rec["raw_c1"] = raw1
                    rec["raw_c2"] = raw2
                    rec["kg_md"] = render_multi_view_kg(kg, active_views, include_edges=False)
                    rec["edge_md"] = render_temporal_edges(kg)
            except Exception as e:
                import traceback
                rec["error"] = f"{type(e).__name__}: {e}"
                rec["trace"] = traceback.format_exc()[-700:]
            fout.write(json.dumps(rec, default=str) + "\n"); fout.flush()
            if (i + 1) % 5 == 0:
                print(f"  [{i+1}/{len(items)}] {time.time()-t0:.0f}s", flush=True)
    print(f"[probe] done chunk {args.chunk_id}: {out_path} ({time.time()-t0:.0f}s)",
          flush=True)


if __name__ == "__main__":
    main()
