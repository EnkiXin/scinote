"""Stage B of the oracle-KG experiment: answer the 10 selected samples
under THREE conditions with the SAME 7B model + SAME frames + SAME renderer:
  C0       vanilla (no KG)
  C2       current AUTO KG (Stage1.1+1.2) + temporal edges  [gate_kg=False]
  C_oracle 72B-built ORACLE KG (from ground-truth note) + temporal edges
Only the KG CONTENT differs between C2 and C_oracle (form held constant), so
the comparison isolates KG-quality. C_oracle>C0 => harm is a quality problem
(fix upstream); C_oracle<=C0 => KG-as-text is intrinsically lossy for a
video-capable 7B (form problem, change approach).

Usage:
  CUDA_VISIBLE_DEVICES=4 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
    python -m scripts.v9_oracle_answer --device cuda:0 \
    --out results_protonote_v9/oracle_kg/answers.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SEL = ROOT / "tools" / "oracle_kg_sample_ids.json"
ORACLE = ROOT / "tools" / "oracle_kgs.json"

VALID_T = {"Operator", "Instrument", "Container", "Material", "Display", "Measurement"}
VALID_L = {"active", "consumed", "transformed", "merged", "split"}


def build_oracle_kg(odict):
    from protonote.v9.kg.state_machine_kg import StateMachineKG
    from protonote.v9.kg.state_entity import StateMachineEntity, EntityState
    from protonote.v9.kg.state_operation import StateTransitionOperation
    kg = StateMachineKG()
    for e in odict.get("entities", []):
        try:
            t = e.get("type") if e.get("type") in VALID_T else "Material"
            cr = e.get("core_role")
            ent = StateMachineEntity(
                entity_id=str(e["entity_id"]),
                canonical_name=str(e.get("canonical_name") or e["entity_id"]),
                type=t, core_role=(cr if cr not in (None, "null", "") else None))
            for s in e.get("states", []) or []:
                ti = s.get("time_interval") or [0, 0]
                try:
                    ti = (float(ti[0]), float(ti[1]))
                except Exception:
                    ti = (0.0, 0.0)
                ls = s.get("lifecycle_status")
                ent.add_state(EntityState(
                    state_id=str(s["state_id"]), time_interval=ti,
                    visual_features=str(s.get("visual_features") or ""),
                    lifecycle_status=ls if ls in VALID_L else "active",
                    transmuted_to_entity_ids=list(s.get("transmuted_to_entity_ids") or []),
                    transmuted_from_entity_ids=list(s.get("transmuted_from_entity_ids") or [])))
            kg.add_entity(ent)
        except Exception:
            continue
    for o in odict.get("operations", []):
        try:
            d = o.get("duration")
            kg.add_operation(StateTransitionOperation(
                operation_id=str(o["operation_id"]), action=str(o.get("action") or ""),
                timestamp=float(o.get("timestamp") or 0),
                duration=(float(d) if d not in (None, "null", "") else None),
                input_states=list(o.get("input_states") or []),
                output_states=list(o.get("output_states") or []),
                action_category=o.get("action_category"),
                description=o.get("description")))
        except Exception:
            continue
    return kg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max_frames", type=int, default=16)
    ap.add_argument("--stage1_2_max_tokens", type=int, default=4096)
    ap.add_argument("--out", default="results_protonote_v9/oracle_kg/answers.jsonl")
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
        build_stage4_prompt, _build_vanilla_prompt, render_multi_view_kg,
        render_temporal_edges, _temporal_edges,
    )

    sel = json.load(open(SEL))
    oracle = json.load(open(ORACLE))
    items = {it["sample_id"]: it
             for it in load_test_split(benchmark="expvid", limit=None)
             if it.get("sample_id") in set(sel["ids"])}

    vlm = QwenVL72BClient(model_name="Qwen/Qwen2.5-VL-7B-Instruct", device=args.device)
    ocr_builder = OCRLedgerBuilder(vlm=vlm)
    print("[oracle-ans] 7B ready", flush=True)

    def _ts(n, dur):
        if n <= 1 or dur <= 0:
            return [0.0] * max(1, n)
        step = dur / max(1, n - 1)
        return [round(i * step, 3) for i in range(n)]

    def answer(prompt, frames, item):
        raw = vlm.generate_video(prompt, frames, max_tokens=400, temperature=0.0)
        tt = item.get("task_type", "mc")
        pred = parse_for_task(raw, tt, item)
        sc = SCORERS.get(tt)
        gold = gold_for(item)
        return (float(sc(pred, gold)) if sc else 0.0), pred, gold, raw

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dump = []
    t0 = time.time()
    with open(out_path, "w") as fout:
        for k, sid in enumerate(sel["ids"]):
            item = items.get(sid)
            rec = {"sample_id": sid, "task_type": item.get("task_type", "mc") if item else "?"}
            try:
                od = oracle.get(sid, {})
                if "error" in od or not od.get("entities"):
                    rec["error"] = f"oracle KG unavailable: {od.get('error','empty')}"
                    fout.write(json.dumps(rec, default=str) + "\n"); fout.flush(); continue
                vp = resolve_video_path(item)
                frames = extract_frames(vp, max_frames=args.max_frames)
                dur = item.get("duration_sec") or (args.max_frames * 1.0)
                try:
                    ledger = ocr_builder.build_ledger(frames, _ts(len(frames), dur))
                except Exception:
                    ledger = []
                q = item.get("question", "")
                opts = item.get("options", {})
                # router once -> same active_views for C2 and C_oracle
                active_views, _ = determine_active_views(q, opts, vlm)
                tt = item.get("task_type", "mc")

                # auto KG (C2)
                s1_1 = run_stage1_1(frames, vlm)
                s1_2 = run_stage1_2(frames, s1_1, ledger, vlm,
                                    max_tokens=args.stage1_2_max_tokens)
                auto_kg = s1_2.kg
                # oracle KG (C_oracle)
                orc_kg = build_oracle_kg(od)

                p0 = _build_vanilla_prompt(question=q, options=opts, task_type=tt)
                p2 = build_stage4_prompt(question=q, options=opts, kg=auto_kg,
                                         active_views=active_views, task_type=tt,
                                         gate_kg=False, include_edges=True)
                porc = build_stage4_prompt(question=q, options=opts, kg=orc_kg,
                                           active_views=active_views, task_type=tt,
                                           gate_kg=False, include_edges=True)

                s0, pr0, gold, r0 = answer(p0, frames, item)
                s2, pr2, _, r2 = answer(p2, frames, item)
                so, pro, _, ro = answer(porc, frames, item)

                rec.update({
                    "active_views": active_views, "gold": gold,
                    "score_c0_vanilla": s0, "score_c2_auto_kg": s2,
                    "score_c_oracle": so,
                    "pred_c0": pr0, "pred_c2": pr2, "pred_oracle": pro,
                    "auto_n_entities": len(auto_kg.entities),
                    "auto_n_ops": len(auto_kg.operations),
                    "auto_n_edges": len(_temporal_edges(auto_kg)),
                    "oracle_n_entities": len(orc_kg.entities),
                    "oracle_n_ops": len(orc_kg.operations),
                    "oracle_n_edges": len(_temporal_edges(orc_kg)),
                })
                dump.append({
                    "sample_id": sid, "task_type": tt, "question": q,
                    "gold": gold, "scores": {"C0": s0, "C2": s2, "C_oracle": so},
                    "auto_kg_md": render_multi_view_kg(auto_kg, active_views, include_edges=True),
                    "oracle_kg_md": render_multi_view_kg(orc_kg, active_views, include_edges=True),
                    "raw_c0": r0, "raw_c2": r2, "raw_oracle": ro,
                })
                print(f"  [{k+1}/{len(sel['ids'])}] {tt:8s} "
                      f"C0={s0:.2f} C2={s2:.2f} Coracle={so:.2f}  "
                      f"auto(e{len(auto_kg.entities)}/o{len(auto_kg.operations)}) "
                      f"oracle(e{len(orc_kg.entities)}/o{len(orc_kg.operations)})",
                      flush=True)
            except Exception as e:
                import traceback
                rec["error"] = f"{type(e).__name__}: {e}"
                rec["trace"] = traceback.format_exc()[-700:]
                print(f"  [{k+1}] ERROR {e}", flush=True)
            fout.write(json.dumps(rec, default=str) + "\n"); fout.flush()
    json.dump(dump, open(out_path.parent / "dump.json", "w"), indent=2, default=str)
    print(f"[oracle-ans] done ({time.time()-t0:.0f}s) -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
