"""V8 benchmark runner.

Drives `answer_item` over a SciVB or ExpVid split. Writes per-item
JSONL + a JSON summary, mirroring the V6/V7 run_react conventions
so the existing comparison scripts pick it up.

Usage (defaults to 7B, GPU 4):

    CUDA_VISIBLE_DEVICES=4 python -m protonote.v8.run_v8 \
        --benchmark scivideobench --limit 5 \
        --model Qwen/Qwen2.5-VL-7B-Instruct \
        --output_dir results_protonote_v8/smoke_scivb \
        --condition_label v8_7b
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evaluate_c0_test_split import extract_frames                       # noqa: E402
from ranker_pipeline.common.video_utils import get_video_duration        # noqa: E402

from protonote.data.loaders import load_test_split, resolve_video_path  # noqa: E402
from protonote.v6.llm_client import QwenVL72BClient                    # noqa: E402
from protonote.v8.kg_pipeline import answer_item                       # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",
                     default="Qwen/Qwen2.5-VL-7B-Instruct",
                     help="HF model name (default: 7B)")
    ap.add_argument("--device", default="cuda",
                     help="cuda / auto / cpu / cuda:N")
    ap.add_argument("--benchmark",
                     default="scivideobench",
                     choices=["scivideobench", "expvid"])
    ap.add_argument("--limit", type=int, default=5,
                     help="0 = all items, default 5 for smoke")
    ap.add_argument("--num_chunks", type=int, default=1)
    ap.add_argument("--chunk_id", type=int, default=0)
    ap.add_argument("--max_frames", type=int, default=16)
    ap.add_argument("--max_extract_tokens", type=int, default=2048)

    ap.add_argument("--no_grounding", action="store_true",
                     help="skip Stages 2+3 (extraction-only ablation)")
    ap.add_argument("--image_index_dir",
                     default="cache/image_library/index")
    ap.add_argument("--kb_dir", default="data/bioprobench")
    ap.add_argument("--kb_device", default="cuda:0")

    ap.add_argument("--output_dir",
                     default="results_protonote_v8/smoke")
    ap.add_argument("--condition_label", default="v8")
    args = ap.parse_args()

    items = load_test_split(benchmark=args.benchmark, limit=None)
    if args.limit > 0:
        items = items[:args.limit]
    if args.num_chunks > 1:
        items = [it for i, it in enumerate(items)
                  if i % args.num_chunks == args.chunk_id]

    print(f"[v8] {len(items)} items "
          f"benchmark={args.benchmark} chunk={args.chunk_id}/{args.num_chunks}",
          flush=True)
    print(f"[v8] loading {args.model} on {args.device}", flush=True)
    vlm = QwenVL72BClient(model_name=args.model, device=args.device)

    image_library = None
    retrieve_tool = None
    if not args.no_grounding:
        try:
            from protonote.v8.grounding.faiss_index import FaissIndex
            from protonote.v8.grounding.image_library import (
                IndexedImageLibrary,
            )
            from protonote.v8.grounding.siglip2_embedder import (
                SigLIP2Embedder,
            )
            embedder = SigLIP2Embedder(
                device="cuda" if args.kb_device.startswith("cuda")
                       else args.kb_device,
            )
            image_library = IndexedImageLibrary.load(
                args.image_index_dir, embedder,
            )
            print(f"[v8] image_library loaded: {image_library}", flush=True)
        except Exception as e:
            print(f"[v8] image library disabled: {e}", flush=True)
            image_library = None

        try:
            from protonote.v6.tools import make_kb_tool
            from protonote.v8.tools.retrieve_tool import RetrieveToolV8
            kb = make_kb_tool(args.kb_dir, device=args.kb_device)
            retrieve_tool = RetrieveToolV8(kb_tool=kb, llm_client=vlm)
            print(f"[v8] retrieve_tool initialized", flush=True)
        except Exception as e:
            print(f"[v8] retrieve_tool disabled: {e}", flush=True)
            retrieve_tool = None
    else:
        print("[v8] --no_grounding: skipping Stages 2+3", flush=True)

    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = (f"_chunk{args.chunk_id}of{args.num_chunks}"
              if args.num_chunks > 1 else "")
    out_path = (out_dir /
                  f"trajectory_{args.benchmark}_{args.condition_label}{suffix}.jsonl")
    sum_path = out_dir / f"summary_{args.benchmark}_{args.condition_label}{suffix}.json"

    results: list[dict] = []
    t0 = time.time()
    with open(out_path, "w") as fout:
        for i, item in enumerate(items):
            sid = item.get("sample_id", f"?_{i}")
            try:
                vp = resolve_video_path(item)
                if not vp or not Path(vp).exists():
                    rec = {"sample_id": sid, "error": "no_video",
                            "score": 0.0}
                    fout.write(json.dumps(rec, default=str) + "\n")
                    fout.flush()
                    results.append(rec)
                    continue
                duration = float(get_video_duration(vp) or 60.0)
                frames = extract_frames(vp, max_frames=args.max_frames)
                if not frames:
                    rec = {"sample_id": sid, "error": "no_frames",
                            "score": 0.0}
                    fout.write(json.dumps(rec, default=str) + "\n")
                    fout.flush()
                    results.append(rec)
                    continue
                t_item = time.time()
                rec = answer_item(
                    item, frames, vlm,
                    image_library=image_library,
                    retrieve_tool=retrieve_tool,
                    duration_sec=duration,
                    max_extract_tokens=args.max_extract_tokens,
                )
                rec["item_elapsed_s"] = round(time.time() - t_item, 2)
            except Exception as e:
                rec = {"sample_id": sid, "error": f"{type(e).__name__}: {e}",
                        "score": 0.0}
            fout.write(json.dumps(rec, default=str) + "\n")
            fout.flush()
            results.append(rec)
            if (i + 1) % 5 == 0 or i == len(items) - 1:
                valid = [r for r in results if "score" in r and "error" not in r]
                acc = (100*sum(r["score"] for r in valid)
                          / max(len(valid), 1))
                print(f"  [{i+1}/{len(items)}] acc={acc:.2f}%  "
                      f"item_s={rec.get('item_elapsed_s', 0):.1f}  "
                      f"total={time.time()-t0:.0f}s", flush=True)

    valid = [r for r in results if "score" in r and "error" not in r]
    acc = 100*sum(r["score"] for r in valid) / max(len(valid), 1)
    summary = {
        "benchmark":  args.benchmark,
        "condition":  args.condition_label,
        "model":      args.model,
        "n_items":    len(valid),
        "n_failed":   len(results) - len(valid),
        "acc":        acc,
        "max_frames":   args.max_frames,
        "image_library": image_library is not None,
        "retrieve_tool": retrieve_tool is not None,
    }
    with open(sum_path, "w") as f:
        json.dump(summary, f, indent=2)
    print()
    print(f"v8 SUMMARY ({args.condition_label}, {args.benchmark}, "
            f"n={len(valid)})")
    print(f"  acc      : {acc:.2f}%")
    print(f"  n_failed : {len(results) - len(valid)}")
    print(f"  Output   : {out_path}")


if __name__ == "__main__":
    main()
