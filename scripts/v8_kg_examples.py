"""Re-run Stage 1 on selected sample_ids to dump rendered KG markdown.

Used to capture the actual KG content that Stage 4 sees as notes_md
in V8 runs (trajectories only store kg_summary counts, not the full
KG). Picks sample_ids representing the V8_SAVED / V8_HURT / BOTH_WRONG
cases from the live no_grounding run, re-runs Stage 1 (deterministic
at temperature=0), and writes a side-by-side markdown.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluate_c0_test_split import extract_frames                      # noqa: E402
from ranker_pipeline.common.video_utils import get_video_duration       # noqa: E402

from protonote.data.loaders import load_test_split, resolve_video_path  # noqa: E402
from protonote.v6.llm_client import QwenVL72BClient                    # noqa: E402
from protonote.v8.stages.stage1_extract import extract_kg              # noqa: E402


# Curated sample_ids covering 3 patterns (SAVED / HURT / BOTH_WRONG) per bench.
SCIVB_PICK = [
    ("scivideobench_mc_60167_3", "V8_SAVED"),    # V8 ✓, C0 ✗
    ("scivideobench_mc_67263_1", "V8_HURT"),     # V8 ✗, C0 ✓
    ("scivideobench_mc_58827_1", "BOTH_WRONG"),  # both ✗ in V8 7B run
]
EXPVID_PICK = [
    # All sequence_generation / first few items
    ("expvid_sequence_generation_videos_level_2_video_segments_53800_clip_1.mp4_53800_clip1_sequence_generation",
       "(partial-credit example)"),
]


def pick_items(bench: str, sids: list[tuple[str, str]]) -> list[tuple[dict, str]]:
    items = load_test_split(benchmark=bench, limit=None)
    by_sid = {it["sample_id"]: it for it in items}
    out = []
    for sid, label in sids:
        if sid in by_sid:
            out.append((by_sid[sid], label))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-frames", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--out", type=Path, default=ROOT / "V8_KG_EXAMPLES.md")
    args = ap.parse_args()

    print(f"loading {args.model} on {args.device}")
    vlm = QwenVL72BClient(model_name=args.model, device=args.device)

    lines = [
        "# V8 — example KGs (Stage 1 output, rendered)",
        "",
        f"VLM: {args.model}  | {args.max_frames} frames/video  "
        f"| max_tokens={args.max_tokens}.",
        "",
        "Each example shows: (a) item metadata, (b) the rendered KG that ",
        "Stage 4 sees in `notes_md`, (c) the gold answer.",
        "",
        "These KGs are from the **no_grounding** pipeline (Stage 1 only). ",
        "For the W/ grounding run the markdown additionally contains ",
        "`grounded via vlm_direct` / `image_match` / `retrieve_plus_image` ",
        "annotations replacing the `(ungrounded)` hedge — see commit history",
        "for the routing-bug analysis.",
        "",
    ]

    # Process SciVB picks
    lines.append("## SciVB examples")
    lines.append("")
    sci = pick_items("scivideobench", SCIVB_PICK)
    for i, (item, label) in enumerate(sci, 1):
        sid = item["sample_id"]
        q = (item.get("question") or "").strip()
        gold = item.get("gold")
        opts = item.get("options") or {}
        opt_block = " · ".join(f"({k}) {str(v)[:80]}"
                                    for k, v in sorted(opts.items()))[:500] \
                      if isinstance(opts, dict) else ""

        vp = resolve_video_path(item)
        duration = float(get_video_duration(vp) or 60.0)
        frames = extract_frames(vp, max_frames=args.max_frames)

        print(f"[SciVB {i}] {sid[:40]}  [{label}]  extracting KG...")
        t0 = time.time()
        kg = extract_kg(frames, vlm,
                              question=q,
                              duration_sec=duration,
                              max_tokens=args.max_tokens)
        elapsed = time.time() - t0
        print(f"  elapsed {elapsed:.1f}s, entities={len(kg.entities)}, "
                f"ops={len(kg.operations)}")

        lines.append(f"### Example {i}: `{sid}`  ({label})")
        lines.append("")
        lines.append(f"- **Question**: {q[:300]}")
        lines.append(f"  - Options: {opt_block}" if opt_block else "")
        lines.append(f"- **Gold answer**: `{gold}`")
        lines.append(f"- **video duration**: {duration:.0f}s")
        lines.append(f"- **Stage 1 elapsed**: {elapsed:.1f}s")
        lines.append(f"- **KG**: {len(kg.entities)} entities, "
                          f"{len(kg.operations)} operations")
        lines.append("")
        lines.append("**Rendered KG (the `notes_md` Stage 4 sees):**")
        lines.append("")
        lines.append("```markdown")
        lines.append(kg.render())
        lines.append("```")
        lines.append("")

    # ExpVid
    lines.append("## ExpVid examples")
    lines.append("")
    exp = pick_items("expvid", EXPVID_PICK)
    for i, (item, label) in enumerate(exp, 1):
        sid = item["sample_id"]
        q = (item.get("question") or "").strip()
        gold = item.get("gold")

        vp = resolve_video_path(item)
        duration = float(get_video_duration(vp) or 60.0)
        frames = extract_frames(vp, max_frames=args.max_frames)
        print(f"[ExpVid {i}] {sid[:40]}  extracting KG...")
        t0 = time.time()
        kg = extract_kg(frames, vlm,
                              question=q,
                              duration_sec=duration,
                              max_tokens=args.max_tokens)
        elapsed = time.time() - t0
        print(f"  elapsed {elapsed:.1f}s, entities={len(kg.entities)}, "
                f"ops={len(kg.operations)}")

        lines.append(f"### Example {i}: `{sid[:60]}`")
        lines.append("")
        lines.append(f"- **task**: `{item.get('task','?')}`")
        lines.append(f"- **Question**: {q[:400]}")
        lines.append(f"- **Gold answer**: {str(gold)[:200]}")
        lines.append(f"- **video duration**: {duration:.0f}s")
        lines.append(f"- **Stage 1 elapsed**: {elapsed:.1f}s")
        lines.append(f"- **KG**: {len(kg.entities)} entities, "
                          f"{len(kg.operations)} operations")
        lines.append("")
        lines.append("**Rendered KG:**")
        lines.append("")
        lines.append("```markdown")
        lines.append(kg.render())
        lines.append("```")
        lines.append("")

    args.out.write_text("\n".join(lines))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
