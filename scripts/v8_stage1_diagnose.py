"""Diagnose Stage 1 failures by dumping raw VLM output to disk.

For the 5 sample videos used by v8_stage1_smoke.py, re-runs Stage 1
but ALSO saves the raw VLM response (before parsing) so we can see
what the 7B model actually emitted on the 0-entity cases.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluate_c0_test_split import extract_frames                      # noqa: E402
from ranker_pipeline.common.video_utils import get_video_duration       # noqa: E402

from protonote.data.loaders import load_test_split, resolve_video_path  # noqa: E402
from protonote.v6.llm_client import QwenVL72BClient                    # noqa: E402
from protonote.v8.kg.stoa import (
    STAGE1_SYSTEM_PROMPT, build_extraction_prompt,
)
from protonote.v8.stages.stage1_extract import parse_kg_from_response


def pick(b, n):
    items = load_test_split(benchmark=b, limit=None)
    return items[::max(1, len(items)//n)][:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-frames", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=2048)
    args = ap.parse_args()

    vlm = QwenVL72BClient(model_name=args.model, device=args.device)

    samples = (
        [(it, "scivideobench") for it in pick("scivideobench", 2)] +
        [(it, "expvid") for it in pick("expvid", 3)]
    )
    out = ROOT / "V8_STAGE1_RAW_7B.md"
    lines = [f"# V8 Stage 1 raw VLM output ({args.model.split('/')[-1]})\n"]
    for i, (it, bench) in enumerate(samples, 1):
        sid = it["sample_id"]
        vp = resolve_video_path(it)
        q = (it.get("question") or "").strip()
        if not vp or not Path(vp).exists():
            continue
        duration = float(get_video_duration(vp) or 60.0)
        frames = extract_frames(vp, max_frames=args.max_frames)
        prompt = build_extraction_prompt(
            n_frames=len(frames), duration_sec=duration, question=q,
        )
        t0 = time.time()
        raw = vlm.generate_video(
            prompt, frames,
            system=STAGE1_SYSTEM_PROMPT,
            max_tokens=args.max_tokens,
            temperature=0.0,
        )
        elapsed = time.time() - t0
        kg = parse_kg_from_response(raw)
        print(f"[{i}] {sid[:50]} elapsed {elapsed:.1f}s → "
              f"{len(kg.entities)} entities, {len(kg.operations)} ops, "
              f"raw len {len(raw)}")
        lines.append(f"## {i}. `{sid}`")
        lines.append(f"- bench={bench}, elapsed={elapsed:.1f}s, "
                       f"raw len={len(raw)}, parsed: "
                       f"{len(kg.entities)} ents / {len(kg.operations)} ops")
        lines.append(f"- question: `{q[:200]}`")
        lines.append("")
        lines.append("```")
        # Cap to avoid huge file
        lines.append(raw[:4000])
        if len(raw) > 4000:
            lines.append(f"\n... [{len(raw) - 4000} more chars truncated]")
        lines.append("```")
        lines.append("")
    out.write_text("\n".join(lines))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
