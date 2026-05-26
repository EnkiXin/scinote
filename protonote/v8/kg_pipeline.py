"""V8 end-to-end pipeline: video + question → answer.

Single entry-point that wires Stages 1-4 together:

    Stage 1  extract_kg(frames, vlm, question, duration_sec)
       ↓
    Stage 2+3 ground_kg(kg, frames, image_library, retrieve_tool, vlm)
       ↓
    Stage 4  answer_from_kg(kg, item, frames, vlm)

Returns a single dict trajectory with sample_id, KG counts at each
stage, final answer, score, and stage-by-stage timing for diagnosis.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

from PIL import Image

logger = logging.getLogger(__name__)


def answer_item(
    item: dict,
    frames: list[Image.Image],
    vlm,
    *,
    image_library=None,
    retrieve_tool=None,
    duration_sec: Optional[float] = None,
    max_extract_tokens: int = 2048,
    skip_grounding: bool = False,
    mc_max_tokens: int = 8,
    gen_max_tokens: int = 64,
) -> dict:
    """Run V8 Stages 1-4 on a single benchmark item.

    Args:
        item: benchmark item (with sample_id, question, task_type, gold, …).
        frames: list[PIL.Image] uniformly sampled from the source video.
        vlm: VLM client (production: V6 QwenVL72BClient, but works with
              Qwen2.5-VL-7B-Instruct via the same wrapper).
        image_library, retrieve_tool: dependencies for Stage 3. If either
              is None, Stage 2+3 grounding is skipped (USE_AS_IS only).
        duration_sec: optional video duration to thread into the Stage 1
              prompt's appearance_intervals scale.
        skip_grounding: if True, skip Stages 2+3 entirely. Useful for
              extraction-only ablations.
        mc_max_tokens / gen_max_tokens: Stage-4 answer caps.

    Returns:
        dict with sample_id, scoring fields, per-stage timing, KG sizes,
        ground counts, abstain flag, and any error encountered.
    """
    # Lazy imports so this module can be loaded without the full V6 stack
    from protonote.v8.stages.stage1_extract import extract_kg
    from protonote.v8.stages.stage3_ground import ground_kg
    from protonote.v8.stages.stage4_reason import answer_from_kg

    out: dict[str, Any] = {
        "sample_id":     item.get("sample_id"),
        "benchmark":     item.get("benchmark"),
        "task":          item.get("task"),
        "task_type":     item.get("task_type", "mc"),
        "n_frames":      len(frames),
        "duration_sec":  duration_sec,
        "stage_timings": {},
        "kg_counts":     {},
    }

    question = (item.get("question") or "").strip() or None

    # ---- Stage 1: extract KG ----
    t0 = time.time()
    try:
        kg = extract_kg(
            frames, vlm,
            question=question,
            duration_sec=duration_sec,
            max_tokens=max_extract_tokens,
        )
    except Exception as e:
        logger.exception("Stage 1 failed: %s", e)
        return {**out, "error": f"stage1: {e}", "score": 0.0,
                  "pred": None, "raw": ""}
    out["stage_timings"]["stage1"] = round(time.time() - t0, 2)
    out["kg_counts"]["after_stage1"] = {
        "entities":   len(kg.entities),
        "operations": len(kg.operations),
    }

    # ---- Stage 2+3: route + ground ----
    if (not skip_grounding
            and image_library is not None
            and retrieve_tool is not None):
        t0 = time.time()
        try:
            counts = ground_kg(
                kg, frames, image_library, retrieve_tool, vlm,
            )
            out["ground_counts"] = counts
        except Exception as e:
            logger.exception("Stages 2+3 failed: %s", e)
            out["ground_counts"] = {"error": str(e)[:200]}
        out["stage_timings"]["stage_2_3"] = round(time.time() - t0, 2)
    else:
        out["stage_timings"]["stage_2_3"] = 0.0
        out["ground_counts"] = {"skipped": True}

    out["kg_counts"]["after_stage3"] = {
        "entities":            len(kg.entities),
        "operations":          len(kg.operations),
        "comprehension_level": kg.comprehension_level,
    }

    # ---- Stage 4: KG → answer ----
    t0 = time.time()
    try:
        ans = answer_from_kg(
            kg, item, frames, vlm,
            mc_max_tokens=mc_max_tokens,
            gen_max_tokens=gen_max_tokens,
        )
    except Exception as e:
        logger.exception("Stage 4 failed: %s", e)
        return {**out, "error": f"stage4: {e}", "score": 0.0,
                  "pred": None, "raw": ""}
    out["stage_timings"]["stage4"] = round(time.time() - t0, 2)

    out["pred"]        = ans.get("pred")
    out["raw"]         = ans.get("raw", "")
    out["score"]       = ans.get("score", 0.0)
    out["gold"]        = ans.get("gold")
    out["abstained"]   = ans.get("abstained", False)
    out["kg_summary"]  = ans.get("kg_summary", {})
    if ans.get("error"):
        out["error"] = ans["error"]
    return out
