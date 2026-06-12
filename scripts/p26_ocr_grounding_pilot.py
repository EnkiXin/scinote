"""P2.6-A — OCR-grounding yield pilot (V10 plan §3.6, gate A).

Question: if we ENABLE the (never-used) V9 OCR ledger during KG construction,
how much of the graph actually gets grounded in on-screen text?

Per item (stratified 150 ExpVid slice):
  1. frames (32, shared policy) + timestamps from video duration
  2. OCR ledger over 8 keyframes (VLM per-frame OCR, V9 OCRLedgerBuilder)
  3. rebuild Stage1.1 -> Stage1.2 WITH the ledger -> KG_ocr
  4. deterministic grounding marks (lexical, must-be-able-to-fail):
       entity.grounded_name  : a distinctive token of canonical_name appears in ledger
       entity.grounded_qty   : estimated_quantity matches a ledger NUMBER token
  5. diff vs the cached no-OCR KG (entity count / quantity changes)

GATES (plan §3.6-A): on items WITH any readable text, >=20% entities grounded,
else G1 is declared yield-starved and closed.

Usage:
  CUDA_VISIBLE_DEVICES=0,1 python -m scripts.p26_ocr_grounding_pilot \
    --model Qwen/Qwen2.5-VL-72B-Instruct --n_per_task 50
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

_GENERIC = {"the", "and", "with", "for", "from", "into", "solution", "sample",
            "samples", "tube", "tubes", "water", "operator", "researcher"}


def distinctive_tokens(name: str) -> set[str]:
    toks = re.findall(r"[a-z0-9µ%°\.-]+", (name or "").lower())
    return {t for t in toks if len(t) > 3 and t not in _GENERIC}


def numbers_in(texts: list[str]) -> set[str]:
    out = set()
    for t in texts:
        out |= set(re.findall(r"\d+(?:\.\d+)?", t))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-72B-Instruct")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--n_per_task", type=int, default=50)
    ap.add_argument("--ocr_frames", type=int, default=8)
    ap.add_argument("--out", default="results_unified/p26_ocr_grounding.jsonl")
    args = ap.parse_args()

    from evaluate_c0_test_split import extract_frames
    from evaluate_unified import MAX_PIXELS
    from protonote.data.loaders import load_test_split, resolve_video_path
    from protonote.cli import VLMClient
    from protonote.v9.preprocessing.ocr_preprocessor import OCRLedgerBuilder
    from protonote.v9.stages.stage1_1_core_entities import run_stage1_1
    from protonote.v9.stages.stage1_2_state_tracking import run_stage1_2
    from protonote.v9.kg.state_machine_kg import StateMachineKG
    from scripts.unified_harness import _V9VLMAdapter
    from ranker_pipeline.common.video_utils import get_video_duration

    items = load_test_split(benchmark="expvid", limit=None)
    by_task = {}
    for it in items:
        by_task.setdefault(it["task_type"], []).append(it)
    sel = []
    for tt in ("fitb", "steppred", "seqgen", "mc"):
        sel += by_task.get(tt, [])[: (args.n_per_task if tt != "mc" else args.n_per_task // 2)]
    print(f"[p26] {len(sel)} items "
          f"({ {tt: sum(1 for s in sel if s['task_type']==tt) for tt in by_task} })", flush=True)

    out_path = ROOT / args.out
    done = set()
    if out_path.exists():
        for line in open(out_path):
            try:
                done.add(json.loads(line)["uid"])
            except Exception:
                pass
    todo = [it for it in sel if it["uid"] not in done]
    print(f"[p26] {len(todo)} to run (resume skips {len(done)})", flush=True)

    vlm = VLMClient(model_name=args.model, device=args.device)
    v9vlm = _V9VLMAdapter(vlm, MAX_PIXELS)
    ocr_builder = OCRLedgerBuilder(v9vlm)
    cache_old = ROOT / "results_unified" / "kg_cache_72b_expvid"
    kg_dir = ROOT / "results_unified" / "kg_cache_72b_expvid_ocr"
    kg_dir.mkdir(exist_ok=True)
    print("[p26] model ready", flush=True)

    agg = {"items": 0, "items_with_text": 0, "ents": 0, "ents_on_text_items": 0,
           "grounded_name": 0, "grounded_qty": 0, "qty_changed_vs_blind": 0}
    t0 = time.time()
    with open(out_path, "a") as fout:
        for i, item in enumerate(todo):
            uid = item["uid"]
            rec = {"uid": uid, "task_type": item["task_type"]}
            try:
                vp = resolve_video_path(item)
                frames = extract_frames(vp, max_frames=32)
                if not frames:
                    raise RuntimeError("no frames")
                try:
                    dur = float(get_video_duration(vp)) or float(len(frames))
                except Exception:
                    dur = float(len(frames))
                step = max(1, len(frames) // args.ocr_frames)
                kf_idx = list(range(0, len(frames), step))[: args.ocr_frames]
                ledger = ocr_builder.build_ledger(
                    [frames[j] for j in kf_idx],
                    [dur * j / len(frames) for j in kf_idx])
                texts = [e.text for e in ledger]
                rec["ocr_tokens"] = len(texts)
                rec["ocr_sample"] = texts[:12]

                s1_1 = run_stage1_1(frames, v9vlm)
                s1_2 = run_stage1_2(frames, s1_1, ledger, v9vlm)
                kg = s1_2.kg
                cache_f = kg_dir / (re.sub(r"[^\w#-]", "_", uid) + ".json")
                cache_f.write_text(json.dumps(kg.to_dict()))
                rec["ocr_alignment_warnings"] = s1_2.ocr_alignment_warnings

                ledger_blob = " ".join(texts).lower()
                ledger_nums = numbers_in(texts)
                ents = []
                for e in kg.entities.values():
                    name = getattr(e, "canonical_name", "") or ""
                    gname = any(t in ledger_blob for t in distinctive_tokens(name))
                    qty = getattr(e, "estimated_quantity", None)
                    gqty = qty is not None and str(qty) in ledger_nums
                    ents.append({"name": name[:60], "grounded_name": gname,
                                 "qty": qty, "grounded_qty": gqty})
                rec["entities"] = ents

                old_f = cache_old / (re.sub(r"[^\w#-]", "_", uid) + ".json")
                if old_f.exists():
                    old = StateMachineKG.from_dict(json.loads(old_f.read_text()))
                    old_q = {getattr(e, "canonical_name", ""): getattr(e, "estimated_quantity", None)
                             for e in old.entities.values()}
                    rec["qty_changed_vs_blind"] = sum(
                        1 for e in ents
                        if e["name"][:60] in {k[:60] for k in old_q}
                        and old_q.get(e["name"]) not in (None, e["qty"]))
                    rec["n_entities_blind"] = len(old.entities)

                agg["items"] += 1
                agg["ents"] += len(ents)
                if texts:
                    agg["items_with_text"] += 1
                    agg["ents_on_text_items"] += len(ents)
                    agg["grounded_name"] += sum(1 for e in ents if e["grounded_name"])
                    agg["grounded_qty"] += sum(1 for e in ents if e["grounded_qty"])
                agg["qty_changed_vs_blind"] += rec.get("qty_changed_vs_blind", 0)
            except Exception as e:
                rec["error"] = str(e)[:200]
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            if (i + 1) % 10 == 0 or i + 1 == len(todo):
                g = agg["grounded_name"] + agg["grounded_qty"]
                base = max(1, agg["ents_on_text_items"])
                print(f"  [{i+1}/{len(todo)}] text-items={agg['items_with_text']}/{agg['items']} "
                      f"grounded={g}/{base} ({g/base*100:.1f}%) "
                      f"qty-fix={agg['qty_changed_vs_blind']} {time.time()-t0:.0f}s", flush=True)

    base = max(1, agg["ents_on_text_items"])
    rate = (agg["grounded_name"] + agg["grounded_qty"]) / base
    print(f"[p26] GATE-A: grounding rate on text-bearing items = {rate*100:.1f}% "
          f"({'PASS >=20%' if rate >= 0.20 else 'FAIL — G1 yield-starved'})", flush=True)


if __name__ == "__main__":
    main()
