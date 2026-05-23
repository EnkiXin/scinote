"""smoke_rewrite.py — Phase 0 gate criterion: does LLM query rewriting
lift KB fire rate by ≥30 pp vs raw question?

Picks 100 SciVB items, runs each question both ways:
  raw_q  → KB.search(raw_question)             [baseline]
  rewrite→ KB.search(rewriter.rewrite(raw_q))  [v5 strategy]

Reports per-method fire rate (= % items where ≥1 passage passes the
0.2 threshold) overall and per-discipline.

Usage:
    CUDA_VISIBLE_DEVICES=0 python -m protonote.v5.smoke_rewrite \
        --rewriter Qwen/Qwen2.5-VL-7B-Instruct --n 100
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from protonote.cli import VLMClient                              # noqa: E402
from protonote.data.loaders import load_test_split               # noqa: E402
from protonote.v5.kb.kb_tool import KBSearchToolV5               # noqa: E402
from protonote.v5.kb.query_rewriter import QueryRewriter, is_protocol_style  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rewriter", default="Qwen/Qwen2.5-VL-7B-Instruct",
                     help="Model to use for query rewriting (text-only).")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--kb_dir", default="data/bioprobench")
    ap.add_argument("--n", type=int, default=100,
                     help="number of SciVB items to test (0 = all)")
    ap.add_argument("--output", default="results_protonote_v5/smoke_rewrite.json")
    args = ap.parse_args()

    # Load SciVB items
    items = load_test_split(benchmark="scivideobench", limit=None)
    if args.n > 0:
        items = items[:args.n]

    # Join discipline metadata
    disc_map = {}
    sci_jsonl = ("/home/yz0392@unt.ad.unt.edu/xin_ai/scivideobench/"
                  "scivideobench_1k.jsonl")
    for l in open(sci_jsonl):
        d = json.loads(l)
        disc_map[(str(d["video_id"]), int(d["question_id"]))] = d["discipline"]
    def disc_of(it):
        vid = it["video_path"].split(":")[-1]
        return disc_map.get((vid, int(it["id"])), "?")

    print(f"[smoke-rewrite] {len(items)} SciVB items", flush=True)

    # Load components
    vlm = VLMClient(model_name=args.rewriter, device=args.device)
    kb  = KBSearchToolV5.from_dir(args.kb_dir, device=args.device)
    rewriter = QueryRewriter(vlm=vlm)

    rows = []
    t0 = time.time()
    for i, it in enumerate(items):
        raw = it.get("question", "")
        rewritten = rewriter.rewrite(raw)
        # Two retrievals
        r_raw      = kb.search(raw)
        r_rewrite  = kb.search(rewritten)
        rows.append({
            "sample_id":  it["sample_id"],
            "discipline": disc_of(it),
            "question":   raw,
            "rewritten":  rewritten,
            "is_protocol_style": is_protocol_style(rewritten),
            "raw_n_kept":      len(r_raw["passages"]),
            "rewrite_n_kept":  len(r_rewrite["passages"]),
            "raw_top_score":      max(r_raw["scores"]) if r_raw["scores"] else 0.0,
            "rewrite_top_score":  max(r_rewrite["scores"]) if r_rewrite["scores"] else 0.0,
        })
        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            raw_fire   = sum(1 for r in rows if r["raw_n_kept"] > 0)
            rew_fire   = sum(1 for r in rows if r["rewrite_n_kept"] > 0)
            print(f"  [{i+1}/{len(items)}]  raw_fire={raw_fire}/{i+1}={100*raw_fire/(i+1):.0f}%  "
                  f"rewrite_fire={rew_fire}/{i+1}={100*rew_fire/(i+1):.0f}%  "
                  f"elapsed={elapsed:.0f}s", flush=True)

    out_path = ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "n":          len(rows),
        "raw_fire_rate":      sum(1 for r in rows if r["raw_n_kept"] > 0) / len(rows),
        "rewrite_fire_rate":  sum(1 for r in rows if r["rewrite_n_kept"] > 0) / len(rows),
        "raw_avg_passages":   sum(r["raw_n_kept"] for r in rows) / len(rows),
        "rewrite_avg_passages": sum(r["rewrite_n_kept"] for r in rows) / len(rows),
        "protocol_style_rate": sum(1 for r in rows if r["is_protocol_style"]) / len(rows),
    }

    # Per-discipline
    from collections import defaultdict
    by_disc = defaultdict(list)
    for r in rows:
        by_disc[r["discipline"]].append(r)
    per_disc = {}
    for d, rs in by_disc.items():
        per_disc[d] = {
            "n":             len(rs),
            "raw_fire":      sum(1 for r in rs if r["raw_n_kept"] > 0) / len(rs),
            "rewrite_fire":  sum(1 for r in rs if r["rewrite_n_kept"] > 0) / len(rs),
        }

    summary["per_discipline"] = per_disc

    with open(out_path, "w") as f:
        json.dump({"summary": summary, "rows": rows[:30]}, f, indent=2, default=str)

    print()
    print("=" * 60)
    print("SMOKE-REWRITE SUMMARY")
    print("=" * 60)
    print(f"  raw fire rate     : {100*summary['raw_fire_rate']:.1f}%  "
          f"(avg passages={summary['raw_avg_passages']:.2f})")
    print(f"  rewrite fire rate : {100*summary['rewrite_fire_rate']:.1f}%  "
          f"(avg passages={summary['rewrite_avg_passages']:.2f})")
    delta = 100 * (summary["rewrite_fire_rate"] - summary["raw_fire_rate"])
    print(f"  Δ fire rate       : {delta:+.1f} pp")
    print(f"  protocol-style rewrite output rate: "
          f"{100*summary['protocol_style_rate']:.1f}%")
    print()
    print(f"  Per-discipline:")
    for d in ["Biology","Biochemistry","Medicine","Bioengineering",
                "Engineering","Chemistry","Physics"]:
        if d not in per_disc: continue
        pd = per_disc[d]
        print(f"    {d:<16} n={pd['n']:>3}  "
              f"raw={100*pd['raw_fire']:.0f}%  rewrite={100*pd['rewrite_fire']:.0f}%  "
              f"Δ={100*(pd['rewrite_fire']-pd['raw_fire']):+.0f}pp")
    print()
    print(f"  GATE (Δ ≥ +30 pp): "
          f"{'PASS' if delta >= 30.0 else 'FAIL'}")
    print(f"\nOutput: {out_path}")


if __name__ == "__main__":
    main()
