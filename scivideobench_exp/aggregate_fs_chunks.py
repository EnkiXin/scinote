"""Aggregate chunked frame-selection results → single accuracy + breakdown."""
import argparse, json, glob
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--results_dir", default="results_scivideobench")
    args = ap.parse_args()

    paths = sorted(glob.glob(f"{args.results_dir}/{args.tag}/eval_scivideobench_chunk*.json"))
    if not paths:
        paths = [f"{args.results_dir}/{args.tag}/eval_scivideobench.json"]
    print(f"merging {len(paths)} files for {args.tag}", flush=True)
    all_results = []
    for p in paths:
        try:
            j = json.load(open(p))
            all_results.extend(j.get("results", []))
        except FileNotFoundError:
            continue
    valid = [r for r in all_results if "error" not in r]
    n_err = sum(1 for r in all_results if "error" in r)
    acc = sum(r["score"] for r in valid) / max(len(valid), 1) * 100
    print(f"  total: {len(all_results)} (valid={len(valid)} err={n_err})")
    print(f"  acc:   {acc:.2f}%")
    # Per question_type
    by_qt = defaultdict(list)
    for r in valid: by_qt[r.get("question_type", "?")].append(r["score"])
    print("  by question_type:")
    for k, v in sorted(by_qt.items()):
        print(f"    {k}: {100*sum(v)/len(v):.2f}% (n={len(v)})")
    # Per discipline
    by_d = defaultdict(list)
    for r in valid: by_d[r.get("discipline", "?")].append(r["score"])
    print("  by discipline:")
    for k, v in sorted(by_d.items()):
        print(f"    {k}: {100*sum(v)/len(v):.2f}% (n={len(v)})")
    # Write merged
    out = f"{args.results_dir}/{args.tag}/merged.json"
    json.dump({"tag": args.tag, "accuracy": round(acc, 2),
                "n_total": len(all_results), "n_valid": len(valid), "n_err": n_err,
                "by_question_type": {k: {"acc": round(100*sum(v)/len(v),2), "n": len(v)} for k,v in by_qt.items()},
                "by_discipline": {k: {"acc": round(100*sum(v)/len(v),2), "n": len(v)} for k,v in by_d.items()},
                "results": all_results,
              }, open(out, "w"))
    print(f"  saved → {out}")


if __name__ == "__main__":
    main()
