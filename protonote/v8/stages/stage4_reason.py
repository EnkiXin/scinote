"""Stage 4: KG-based reasoning → final answer.

Takes a grounded KnowledgeGraph plus the original video frames + the
benchmark item (question / options / task_type) and produces the MC
letter / sequence / etc. via the same V6 BUILDERS + parse_for_task +
SCORERS scaffolding the rest of the project uses.

The KG is rendered to markdown by `kg.render()` and passed in the
`notes_md` slot that V6/V7 already supported. So Stage 4 is a thin
wrapper:

    notes_md = kg.render()
    messages = BUILDERS[task_type](item, frames, notes_md, benchmark?)
    raw = vlm._impl.generate(messages, max_new_tokens=…)
    pred = parse_for_task(raw, task_type, item)
    score = SCORERS[task_type](pred, item)

If the VLM client doesn't expose `._impl.generate(messages, …)` (e.g.
a pure mock), we fall back to ``vlm.generate_text(prompt, …)``.

Public API:
    answer_from_kg(kg, item, frames, vlm,
                       *, mc_max_tokens=8, gen_max_tokens=64,
                       force_no_notes=False) -> dict

Returns a dict containing ``pred``, ``raw``, ``score``, plus a
``kg_summary`` block (counts + comprehension level) and an
``abstained`` flag for upstream logging.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from PIL import Image

logger = logging.getLogger(__name__)


# ============================================================
# Public API
# ============================================================


def answer_from_kg(
    kg,
    item: dict,
    frames: list[Image.Image],
    vlm,
    *,
    mc_max_tokens: int = 8,
    gen_max_tokens: int = 64,
    force_no_notes: bool = False,
    abstain_on_empty: bool = True,
) -> dict:
    """Run Stage 4: render KG → ask VLM → score.

    Args:
        kg: KnowledgeGraph (typically from `ground_kg`).
        item: benchmark item with question, options, task_type, benchmark, gold.
        frames: list[PIL.Image] of the same video the KG was extracted from.
        vlm: client. Production uses `QwenVL72BClient` (despite the name
             works for the 7B model too — see scripts/v8_stage1_smoke.py).
        mc_max_tokens / gen_max_tokens: per-task answer cap.
        force_no_notes: skip notes_md entirely (Stage-4 "no KG" ablation).
        abstain_on_empty: if KG has zero entities/ops AND comprehension==0,
             do not pass notes_md (= C0 behaviour). Mirrors V7 abstain.

    Returns:
        dict with keys:
            pred         — task-specific answer (letter for MC, list for seq, …)
            raw          — first 200 chars of the VLM output
            score        — float per the task's SCORER
            gold         — gold answer (from item)
            kg_summary   — {n_entities, n_operations, n_stages, comprehension_level}
            abstained    — bool, True if notes_md was suppressed
            error        — optional error str if anything failed
    """
    # Lazy imports to keep module loadable without the heavy V6 stack.
    try:
        from evaluate_c0_test_split import BUILDERS, gold_for, parse_for_task
        from evaluate_unified import SCORERS
    except ImportError as e:
        logger.warning("Stage 4 evaluator imports unavailable: %s", e)
        return _err_result(item, kg, f"evaluator imports failed: {e}")

    out = {
        "sample_id":  item.get("sample_id"),
        "benchmark":  item.get("benchmark"),
        "task":       item.get("task"),
        "task_type":  item.get("task_type", "mc"),
        "gold":       gold_for(item),
        "kg_summary": _kg_summary(kg),
        "abstained":  False,
    }

    task_type = out["task_type"]
    builder = BUILDERS.get(task_type)
    if builder is None:
        return {**out, "error": f"no builder for task_type={task_type}"}

    notes_md = _render_notes(
        kg,
        force_no_notes=force_no_notes,
        abstain_on_empty=abstain_on_empty,
    )
    out["abstained"] = notes_md is None

    if task_type == "mc":
        messages = builder(item, frames, notes_md, item.get("benchmark"))
    else:
        messages = builder(item, frames, notes_md)
    max_new = mc_max_tokens if task_type == "mc" else gen_max_tokens

    raw = _call_vlm_with_messages(vlm, messages, max_new)
    if raw is None:
        return {**out, "error": "VLM call failed", "pred": None,
                  "score": 0.0, "raw": ""}

    try:
        pred = parse_for_task(raw, task_type, item)
        score = float(SCORERS[task_type](pred, out["gold"]))
    except Exception as e:
        logger.warning("parse/score failed: %s", e)
        return {**out, "error": str(e)[:200], "pred": None,
                  "score": 0.0, "raw": (raw or "")[:200]}

    return {
        **out,
        "pred":  pred,
        "raw":   (raw or "")[:200],
        "score": score,
    }


# ============================================================
# Internals
# ============================================================


def _kg_summary(kg) -> dict:
    return {
        "n_entities":         len(kg.entities),
        "n_operations":       len(kg.operations),
        "n_stages":           len(kg.stages),
        "comprehension_level": kg.comprehension_level,
    }


def _render_notes(
    kg,
    *,
    force_no_notes: bool,
    abstain_on_empty: bool,
) -> Optional[str]:
    """Render KG to markdown, or return None to suppress notes."""
    if force_no_notes:
        return None
    if abstain_on_empty:
        if (len(kg.entities) == 0
                and len(kg.operations) == 0
                and kg.comprehension_level == 0.0):
            return None
    try:
        return kg.render()
    except Exception as e:
        logger.debug("KG render failed: %s", e)
        return None


def _call_vlm_with_messages(vlm, messages, max_new_tokens: int) -> Optional[str]:
    """Send V6-style messages to the VLM.

    Production V6 client exposes `._impl.generate(messages, max_new_tokens)`.
    For unit-test mocks we accept a duck-typed object that also
    implements `.generate_messages(messages, max_new_tokens)`.
    """
    impl = getattr(vlm, "_impl", None)
    if impl is not None and hasattr(impl, "generate"):
        try:
            return impl.generate(messages, max_new_tokens=max_new_tokens)
        except Exception as e:
            logger.warning("vlm._impl.generate failed: %s", e)
            return None
    if hasattr(vlm, "generate_messages"):
        try:
            return vlm.generate_messages(messages,
                                              max_new_tokens=max_new_tokens)
        except Exception as e:
            logger.warning("vlm.generate_messages failed: %s", e)
            return None
    logger.warning("vlm has no compatible message generator")
    return None


def _err_result(item: dict, kg, msg: str) -> dict:
    return {
        "sample_id":  item.get("sample_id"),
        "benchmark":  item.get("benchmark"),
        "task":       item.get("task"),
        "task_type":  item.get("task_type", "mc"),
        "gold":       None,
        "kg_summary": _kg_summary(kg),
        "abstained":  True,
        "error":      msg,
        "pred":       None,
        "raw":        "",
        "score":      0.0,
    }
