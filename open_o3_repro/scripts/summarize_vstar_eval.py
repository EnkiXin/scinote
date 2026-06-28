#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


METRIC_PATTERNS = {
    "vqa_accuracy": r"VQA:.*?Accuracy:\s*([0-9.]+)",
    "chain1_mean_temporal_iou": r"Chain 1:.*?Temporal Answer:.*?Mean IoU:\s*([0-9.]+)",
    "chain1_mean_spatial_miou": r"Chain 1:.*?Spatial Answer:.*?Mean mIoU:\s*([0-9.]+)",
    "chain2_mean_temporal_iou": r"Chain 2:.*?Temporal Answer:.*?Mean IoU:\s*([0-9.]+)",
    "chain2_mean_spatial_miou": r"Chain 2:.*?Spatial Answer:.*?Mean mIoU:\s*([0-9.]+)",
    "am1": r"AM1:([0-9.]+)",
    "am2": r"AM2:([0-9.]+)",
    "mam": r"mAM:([0-9.]+)",
    "lgm1": r"LGM1:([0-9.]+)",
    "lgm2": r"LGM2:([0-9.]+)",
    "mlgm": r"mLGM:([0-9.]+)",
}


def parse_metrics(text: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for key, pattern in METRIC_PATTERNS.items():
        match = re.search(pattern, text, re.DOTALL)
        if match:
            metrics[key] = float(match.group(1))
    return metrics


def maybe_count_json(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return len(data) if isinstance(data, list) else None
    except Exception:
        return None


def write_markdown(path: Path, summary: dict[str, object]) -> None:
    metrics = summary.get("metrics", {})
    lines = [
        "# V-STAR Evaluation Summary",
        "",
        f"- eval_log: `{summary['eval_log']}`",
        f"- test_log: `{summary['test_log']}`",
        f"- result_json: `{summary['result_json']}`",
        f"- result_count: {summary.get('result_count')}",
        "",
        "## Metrics",
        "",
    ]
    if metrics:
        for key, value in metrics.items():
            lines.append(f"- {key}: {value}")
    else:
        lines.append("- No metrics parsed.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-log", type=Path, required=True)
    parser.add_argument("--test-log", type=Path, required=True)
    parser.add_argument("--result-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    text = args.eval_log.read_text(encoding="utf-8", errors="replace") if args.eval_log.exists() else ""
    summary = {
        "eval_log": str(args.eval_log),
        "test_log": str(args.test_log),
        "result_json": str(args.result_json),
        "result_count": maybe_count_json(args.result_json),
        "metrics": parse_metrics(text),
    }
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(args.output_md, summary)
    print(f"[summarize_vstar_eval] wrote {args.output_md}")
    print(f"[summarize_vstar_eval] wrote {args.output_json}")


if __name__ == "__main__":
    main()
