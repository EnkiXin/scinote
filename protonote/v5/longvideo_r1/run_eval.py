"""run_eval.py — LongVideo-R1 eval on ExpVid + SciVB.

Wraps `LongVideoDemo` from /tmp/LongVideo-R1/cli.py with our test split.

Requires 3 vLLM serves running:
  1. Reasoning  (ChurchillQAQ/LongVideo-R1-Qwen3)  on port 25600
  2. Caption    (Qwen/Qwen3-VL-32B-Instruct)       on port 9081
  3. Video QA   (same instance as caption)         on port 9081

Usage:
    python -m protonote.v5.longvideo_r1.run_eval \
        --benchmark scivideobench \
        --output_dir results/longvideo_r1
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
# Make cli.py from cloned repo importable
sys.path.insert(0, "/tmp/LongVideo-R1")

from protonote.data.loaders import load_test_split, resolve_video_path  # noqa: E402

# Import LongVideoDemo from the cloned repo
from cli import LongVideoDemo                                            # noqa: E402

# For paper-1 baselines (gold answer extraction)
from evaluate_c0_test_split import gold_for                              # noqa: E402
from evaluate_unified import SCORERS                                     # noqa: E402


def _build_question(item: dict) -> str:
    """Build the LongVideo-R1 question string from a test-split item.

    For MC tasks: append options as letter list, ask for the letter.
    For open-ended (sequence/fitb): use raw question.
    """
    q = item.get("question", "").strip()
    task_type = item.get("task_type", "mc")
    if task_type == "mc" and isinstance(item.get("options"), dict):
        opts = item["options"]
        opt_lines = "\n".join(f"  {k}. {v}" for k, v in sorted(opts.items()))
        q = (
            f"{q}\n\n"
            f"## Options\n{opt_lines}\n\n"
            f"Answer with ONLY the letter (e.g., 'A')."
        )
    return q


class _Args:
    """Mimic argparse.Namespace for LongVideoDemo constructor."""
    def __init__(self, video_path, **kwargs):
        self.video_path = video_path
        self.cache_dir = kwargs.get("cache_dir",
                                       "results/longvideo_r1/.caption_cache")
        self.max_rounds = kwargs.get("max_rounds", 8)
        self.api_key = kwargs.get("api_key", "111111")
        self.reasoning_base_url = kwargs.get("reasoning_base_url",
                                               "http://127.0.0.1:25600/v1")
        self.reasoning_model = kwargs.get("reasoning_model", "longvideor1")
        self.caption_base_url = kwargs.get("caption_base_url",
                                              "http://127.0.0.1:9081/v1")
        self.caption_model = kwargs.get("caption_model", "Qwen3-VL-32B")
        self.videoqa_base_url = kwargs.get("videoqa_base_url",
                                              "http://127.0.0.1:9081/v1")
        self.videoqa_model = kwargs.get("videoqa_model", "Qwen3-VL-32B")
        self.decode_threads = kwargs.get("decode_threads", 8)


def _normalize_pred(pred_text: str, task_type: str) -> str:
    """Convert LongVideo-R1 freeform answer to evaluator-expected format."""
    if not pred_text:
        return ""
    s = pred_text.strip()
    if task_type == "mc":
        # MC tasks score on exact letter (A-J typically).
        # Try to find the first uppercase letter inside the response.
        import re
        m = re.search(r"\b([A-J])\b", s)
        if m: return m.group(1)
        m = re.match(r"^([A-Za-z])", s)
        if m: return m.group(1).upper()
        return s[:1].upper()
    return s


def run_one(item: dict, **demo_kwargs) -> dict:
    """Run a single item through LongVideoDemo and return result dict."""
    out = {
        "sample_id":  item["sample_id"],
        "benchmark":  item["benchmark"],
        "task":       item.get("task"),
        "task_type":  item.get("task_type", "mc"),
        "gold":       gold_for(item),
        "discipline": item.get("discipline"),
    }
    try:
        vp = resolve_video_path(item)
        if not vp:
            out["error"] = "no_video"
            return out
    except Exception as e:
        out["error"] = f"video err: {str(e)[:120]}"
        return out

    question = _build_question(item)
    out["question_full"] = question[:300]

    args = _Args(video_path=vp, **demo_kwargs)
    t0 = time.time()
    try:
        demo = LongVideoDemo(args)
        result = demo.answer_question(question)
    except Exception as e:
        out["error"] = f"demo err: {str(e)[:240]}"
        out["elapsed_s"] = time.time() - t0
        return out

    out["elapsed_s"] = round(time.time() - t0, 2)
    out["raw_answer"] = (result.get("answer") or "")[:300]
    out["n_rounds"] = len(result.get("history", []))
    out["timing"] = result.get("timing", {})

    # Tool call accounting (post-hoc by scanning history)
    import re
    tool_calls = []
    for content in result.get("history", []):
        m = re.search(r"<tool>(.*?)</tool>", content, re.DOTALL)
        if m:
            call = m.group(1).strip()
            if call.startswith("get_caption"):
                tool_calls.append("get_caption")
            elif call.startswith("video_qa"):
                tool_calls.append("video_qa")
            else:
                tool_calls.append("other")
    out["tool_calls"] = tool_calls

    # Normalize prediction
    pred = _normalize_pred(out["raw_answer"], out["task_type"])
    out["pred"] = pred
    # Score
    try:
        out["score"] = float(SCORERS[out["task_type"]](pred, out["gold"]))
    except Exception as e:
        out["score"] = 0.0
        out["score_err"] = str(e)[:120]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default="scivideobench",
                     choices=["expvid", "scivideobench"])
    ap.add_argument("--limit", type=int, default=0,
                     help="cap items for smoke (0 = all)")
    ap.add_argument("--max_rounds", type=int, default=8)
    ap.add_argument("--output_dir", default="results/longvideo_r1")
    ap.add_argument("--reasoning_base_url",
                     default="http://127.0.0.1:25600/v1")
    ap.add_argument("--caption_base_url",
                     default="http://127.0.0.1:9081/v1")
    ap.add_argument("--videoqa_base_url",
                     default="http://127.0.0.1:9081/v1")
    ap.add_argument("--reasoning_model", default="longvideor1")
    ap.add_argument("--caption_model", default="Qwen3-VL-32B")
    ap.add_argument("--videoqa_model", default="Qwen3-VL-32B")
    args = ap.parse_args()

    items = load_test_split(benchmark=args.benchmark, limit=None)
    if args.limit > 0: items = items[:args.limit]
    print(f"[longvideor1-eval] {len(items)} items "
          f"(benchmark={args.benchmark})", flush=True)

    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    traj_path = out_dir / f"{args.benchmark}_results.jsonl"
    fail_path = out_dir / f"{args.benchmark}_failures.jsonl"

    demo_kwargs = {
        "max_rounds":          args.max_rounds,
        "reasoning_base_url":  args.reasoning_base_url,
        "caption_base_url":    args.caption_base_url,
        "videoqa_base_url":    args.videoqa_base_url,
        "reasoning_model":     args.reasoning_model,
        "caption_model":       args.caption_model,
        "videoqa_model":       args.videoqa_model,
    }

    from collections import Counter
    results = []
    action_count = Counter()
    n_rounds_acc = []
    elapsed_acc = []
    t0 = time.time()
    with open(traj_path, "w") as tf, open(fail_path, "w") as ff:
        for i, item in enumerate(items):
            r = run_one(item, **demo_kwargs)
            tf.write(json.dumps(r, default=str) + "\n")
            tf.flush()
            if "error" in r:
                ff.write(json.dumps({**r, "i": i}, default=str) + "\n")
                ff.flush()
                continue
            results.append(r)
            for tc in r.get("tool_calls", []):
                action_count[tc] += 1
            n_rounds_acc.append(r.get("n_rounds", 0))
            elapsed_acc.append(r.get("elapsed_s", 0))

            if (i + 1) % 5 == 0 or i == len(items) - 1:
                valid = [x for x in results if "score" in x]
                acc = (sum(x["score"] for x in valid) / max(len(valid), 1)) * 100
                avg_rounds = (sum(n_rounds_acc) / max(len(n_rounds_acc), 1))
                avg_t = (sum(elapsed_acc) / max(len(elapsed_acc), 1))
                print(f"  [{i+1}/{len(items)}] acc={acc:.2f}%  "
                      f"avg_rounds={avg_rounds:.1f}  avg_time={avg_t:.1f}s  "
                      f"elapsed={time.time()-t0:.0f}s", flush=True)

    # Summary
    valid = [x for x in results if "score" in x]
    acc = 100 * sum(x["score"] for x in valid) / max(len(valid), 1)
    summary = {
        "benchmark":   args.benchmark,
        "n":           len(valid),
        "n_failed":    len(items) - len(valid),
        "acc":         acc,
        "avg_rounds":  sum(n_rounds_acc) / max(len(n_rounds_acc), 1),
        "avg_time_s":  sum(elapsed_acc) / max(len(elapsed_acc), 1),
        "tool_call_distribution": dict(action_count),
        "max_rounds_used": max(n_rounds_acc) if n_rounds_acc else 0,
    }
    sum_path = out_dir / f"{args.benchmark}_summary.json"
    with open(sum_path, "w") as f: json.dump(summary, f, indent=2)

    print()
    print("=" * 60)
    print(f"LongVideo-R1 SUMMARY ({args.benchmark}, n={len(valid)})")
    print("=" * 60)
    print(f"  acc          : {acc:.2f}%")
    print(f"  n_failed     : {len(items) - len(valid)}")
    print(f"  avg_rounds   : {summary['avg_rounds']:.2f}")
    print(f"  avg_time_s   : {summary['avg_time_s']:.2f}")
    print(f"  tool_calls   : {dict(action_count)}")
    print(f"  Output       : {traj_path}")


if __name__ == "__main__":
    main()
