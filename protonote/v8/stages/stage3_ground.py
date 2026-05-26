"""Stage 3: Selective grounding (4 paths).

Each routing action has its own grounding function:

  IMAGE_MATCH           crop → SigLIP2 → top-k → VLM verify
  RETRIEVE_PLUS_IMAGE   KB retrieve → candidate names → image verify
  RETRIEVE_ONLY         KB retrieve → candidate names (Materials)
  OCR                   read numbers/labels from bbox

Failed grounding leaves `entity.grounded` either as a tightly-scoped
"ungrounded" with candidates for downstream reasoning, or in a state
the caller can escalate (e.g. low-similarity IMAGE_MATCH → returns
False, leaves grounded=None so the orchestrator can drop into the
retrieve fallback).
"""

from __future__ import annotations

import logging
from typing import Optional

from PIL import Image

from protonote.v8.grounding.crop_utils import crop_entity
from protonote.v8.grounding.verifier import vlm_verify_match
from protonote.v8.kg.entity import Entity, GroundingInfo

logger = logging.getLogger(__name__)


# Thresholds (V8_RESEARCH_PLAN_V3.md, 实测 motivated)
IMAGE_MATCH_MIN          = 0.65
IMAGE_MATCH_VERIFY_MIN   = 0.70
RETRIEVE_PLUS_IMAGE_MIN  = 0.55
HIGH_CONFIDENCE_LABEL    = 0.80


# ============================================================
# IMAGE_MATCH path (W4D2)
# ============================================================


def ground_via_image_match(
    entity: Entity,
    frames: list[Image.Image],
    image_library,
    llm_client,
    top_k: int = 5,
) -> bool:
    """IMAGE_MATCH path: crop → SigLIP2 top-k → VLM verify.

    Updates ``entity.grounded`` in place.

    Returns:
      True  — entity was successfully grounded via image_match.
      False — could not ground:
              * no crop  → grounded set to ``ungrounded`` (terminal)
              * no candidates → grounded set to ``ungrounded`` (terminal)
              * low SigLIP2 similarity → leaves `grounded=None` so the
                caller can escalate to a retrieve fallback.
              * VLM verify rejected → leaves `grounded=None`.
    """
    crop = crop_entity(frames, entity)
    if crop is None:
        entity.grounded = GroundingInfo(
            identity=None, confidence=0.0, method="ungrounded",
            evidence="no frames / no crop available",
        )
        return False

    candidates = image_library.top_k(
        crop,
        k=top_k,
        filter_entity_type=entity.type,
    )

    if not candidates:
        entity.grounded = GroundingInfo(
            identity=None, confidence=0.0, method="ungrounded",
            evidence=f"no {entity.type} candidates in library",
        )
        return False

    top = candidates[0]
    if top.score < IMAGE_MATCH_MIN:
        logger.debug(
            "%s: top library similarity %.2f < %.2f — caller may escalate",
            entity.id, top.score, IMAGE_MATCH_MIN,
        )
        return False  # leave grounded=None for orchestrator escalation

    try:
        ref_image = Image.open(top.image_path).convert("RGB")
    except Exception as e:
        logger.warning("could not load candidate image %s: %s",
                          top.image_path, e)
        return False

    verification = vlm_verify_match(
        query_crop=crop,
        candidate_image=ref_image,
        candidate_label=top.label,
        llm_client=llm_client,
    )
    if (not verification["match"]
            or verification["confidence"] < IMAGE_MATCH_VERIFY_MIN):
        logger.debug(
            "%s: VLM verify rejected (match=%s conf=%.2f)",
            entity.id, verification["match"], verification["confidence"],
        )
        return False

    entity.grounded = GroundingInfo(
        identity=top.label,
        confidence=top.score,
        method="image_match",
        source_dataset=top.dataset,
        evidence=(
            f"crop similar to {top.label} "
            f"(sim {top.score:.2f}); VLM verified "
            f"({verification['confidence']:.2f}): "
            f"{verification['reasoning'][:120]}"
        ),
    )
    return True
