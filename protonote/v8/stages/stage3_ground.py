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

import numpy as np
from PIL import Image

from protonote.v8.grounding.candidate_extractor import (
    extract_candidates_from_passages,
)
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


# ============================================================
# RETRIEVE_PLUS_IMAGE path (W4D3)
# ============================================================


def ground_via_retrieve_plus_image(
    entity: Entity,
    frames: list[Image.Image],
    image_library,
    retrieve_tool,
    llm_client,
) -> bool:
    """KB retrieve → candidate names → image-library lookup + visual verify.

    Used as a fallback for low-similarity IMAGE_MATCH cases and as the
    primary path for low-confidence Container/Instrument entities.

    Updates ``entity.grounded`` in place. Returns True only if a candidate
    was found in the library AND a cosine-similarity above
    ``RETRIEVE_PLUS_IMAGE_MIN`` (0.55) was achieved.
    """
    passages = retrieve_tool.retrieve_for_entity(entity, top_k=5)
    if not passages:
        entity.grounded = GroundingInfo(
            identity=None, confidence=0.0, method="ungrounded",
            evidence="no KB passages retrieved",
        )
        return False

    candidates = extract_candidates_from_passages(
        entity, passages, llm_client,
    )
    if not candidates:
        entity.grounded = GroundingInfo(
            identity=None, confidence=0.0, method="ungrounded",
            evidence=(
                f"KB returned {len(passages)} passages, no candidates "
                f"extracted"
            ),
        )
        return False

    crop = crop_entity(frames, entity)
    if crop is None:
        entity.grounded = GroundingInfo(
            identity=None, confidence=0.0, method="ungrounded",
            candidates=candidates,
            evidence=(
                f"candidates from KB but no crop available: "
                f"{', '.join(candidates)}"
            ),
        )
        return False

    # Compute crop embedding once.
    try:
        crop_emb = image_library.embedder.embed_images([crop])[0]
    except Exception as e:
        logger.warning("crop embedding failed: %s", e)
        entity.grounded = GroundingInfo(
            identity=None, confidence=0.0, method="ungrounded",
            candidates=candidates,
            evidence=f"crop-embedding failed: {e}",
        )
        return False

    best_label: Optional[str] = None
    best_dataset: Optional[str] = None
    best_score: float = 0.0

    for cand in candidates:
        refs = image_library.get_by_label(cand)
        if not refs:
            continue
        ref = refs[0]
        try:
            ref_img = Image.open(ref.image_path).convert("RGB")
            ref_emb = image_library.embedder.embed_images([ref_img])[0]
        except Exception as e:
            logger.debug("reference embed failed for %s: %s", cand, e)
            continue
        score = float(np.dot(crop_emb, ref_emb))   # both L2-normalized
        if score > best_score:
            best_score = score
            best_label = cand
            best_dataset = ref.dataset

    if best_label is not None and best_score >= RETRIEVE_PLUS_IMAGE_MIN:
        entity.grounded = GroundingInfo(
            identity=best_label,
            confidence=best_score,
            method="retrieve_plus_image",
            source_dataset=best_dataset,
            candidates=candidates,
            evidence=(
                f"KB candidates: {', '.join(candidates)}. "
                f"Visual match: {best_label} (sim {best_score:.2f})"
            ),
        )
        return True

    entity.grounded = GroundingInfo(
        identity=None, confidence=0.0, method="ungrounded",
        candidates=candidates,
        evidence=(
            f"KB candidates ({', '.join(candidates)}) not visually "
            f"verifiable (best sim {best_score:.2f} < "
            f"{RETRIEVE_PLUS_IMAGE_MIN})"
        ),
    )
    return False


# ============================================================
# RETRIEVE_ONLY path (W4D3) — Materials only
# ============================================================


def ground_via_retrieve_only(
    entity: Entity,
    retrieve_tool,
    llm_client,
) -> bool:
    """KB retrieve → candidate names; NO image library check.

    For Material entities where library hit rate is 0 % (实测). We
    still don't commit to an identity (no visual verification), but
    we attach the candidate list so Stage 4 reasoning can use it as
    a hypothesis (e.g. "if Entity3 is MOF, then …").

    Always returns False (entity stays officially ungrounded); the
    ``candidates`` field is the useful side-effect.
    """
    passages = retrieve_tool.retrieve_for_entity(entity, top_k=5)
    if not passages:
        entity.grounded = GroundingInfo(
            identity=None, confidence=0.0, method="ungrounded",
            evidence="no KB passages retrieved",
        )
        return False

    candidates = extract_candidates_from_passages(
        entity, passages, llm_client,
    )
    if not candidates:
        entity.grounded = GroundingInfo(
            identity=None, confidence=0.0, method="ungrounded",
            evidence=f"KB returned {len(passages)} passages, no candidates",
        )
        return False

    entity.grounded = GroundingInfo(
        identity=None, confidence=0.0, method="ungrounded",
        candidates=candidates,
        evidence=(
            f"KB candidates: {', '.join(candidates)}. "
            f"Material entity — no visual library verification."
        ),
    )
    return False


# ============================================================
# OCR path (W4D4) — Display + Measurement entities
# ============================================================


def ground_via_ocr(
    entity: Entity,
    frames: list[Image.Image],
    vlm,
) -> bool:
    """Crop entity region and OCR for text/numbers.

    Used for Display and Measurement entities (always-OCR per Stage 2
    policy). Returns True if OCR returned any text, False if blank or
    on error. ``entity.grounded.ocr_text`` is set either way.
    """
    from protonote.v8.tools.ocr_tool import ocr_for_entity

    result = ocr_for_entity(entity, frames, vlm)
    text = result.get("text", "")
    err = result.get("error")

    if text:
        entity.grounded = GroundingInfo(
            identity=None,
            confidence=0.9,           # OCR-content trust, not identity
            method="ocr",
            ocr_text=text,
            evidence=(
                f"OCR read text from {entity.type} crop "
                f"(frame {result.get('frame_idx', '?')})"
            ),
        )
        return True

    entity.grounded = GroundingInfo(
        identity=None, confidence=0.0, method="ungrounded",
        ocr_text=None,
        evidence=err if err else "OCR returned no text (NO_TEXT_VISIBLE)",
    )
    return False


# ============================================================
# Orchestrator (W4D5)
# ============================================================


def ground_kg(
    kg,
    frames: list[Image.Image],
    image_library,
    retrieve_tool,
    vlm,
) -> dict:
    """Run Stages 2 + 3 end-to-end.

    Sequence:
      1. Stage 2 ``route_kg`` partitions entities + pre-populates
         USE_AS_IS grounding.
      2. For each IMAGE_MATCH entity → ``ground_via_image_match``.
         If it returns False AND left ``entity.grounded=None`` (i.e.
         low SigLIP2 similarity or VLM-verify rejection), escalate
         to ``ground_via_retrieve_plus_image``.
      3. For each entity directly routed to RETRIEVE_PLUS_IMAGE
         (low-confidence Container/Instrument) → that path.
      4. For each RETRIEVE_ONLY entity (Materials) → that path.
      5. For each OCR entity (Display/Measurement) → that path.

    Returns a counts dict with grounding outcomes per path. The KG
    itself is mutated in place — every entity ends with a non-None
    ``entity.grounded`` (either a real ground or a deliberate
    ``ungrounded`` record).
    """
    # Local import to keep module-load light.
    from protonote.v8.stages.stage2_route import route_kg

    routing = route_kg(kg)
    counts = {
        "use_as_is":            len(routing.use_as_is),
        "image_match_success":  0,
        "image_match_escalated":0,
        "retrieve_plus_image_success": 0,
        "retrieve_only":        0,
        "ocr_success":          0,
        "ocr_blank":            0,
        "ungrounded_total":     0,
    }

    # --- IMAGE_MATCH (with escalation) ---
    for ent in routing.image_match:
        ok = ground_via_image_match(ent, frames, image_library, vlm)
        if ok:
            counts["image_match_success"] += 1
            continue
        # If still None, escalate.
        if ent.grounded is None:
            counts["image_match_escalated"] += 1
            ok2 = ground_via_retrieve_plus_image(
                ent, frames, image_library, retrieve_tool, vlm,
            )
            if ok2:
                counts["retrieve_plus_image_success"] += 1

    # --- direct RETRIEVE_PLUS_IMAGE ---
    for ent in routing.retrieve_plus_image:
        if ground_via_retrieve_plus_image(
            ent, frames, image_library, retrieve_tool, vlm,
        ):
            counts["retrieve_plus_image_success"] += 1

    # --- RETRIEVE_ONLY (Materials) ---
    for ent in routing.retrieve_only:
        ground_via_retrieve_only(ent, retrieve_tool, vlm)
        counts["retrieve_only"] += 1

    # --- OCR ---
    for ent in routing.ocr:
        if ground_via_ocr(ent, frames, vlm):
            counts["ocr_success"] += 1
        else:
            counts["ocr_blank"] += 1

    # Final tally + sanity: every entity should have entity.grounded set.
    for ent in kg.entities.values():
        if ent.grounded is None:
            # Should not happen, but be defensive — mark as ungrounded.
            ent.grounded = GroundingInfo(
                identity=None, confidence=0.0, method="ungrounded",
                evidence="orchestrator: no path produced a grounding",
            )
        if (ent.grounded.method == "ungrounded"
                or ent.grounded.identity is None):
            counts["ungrounded_total"] += 1

    return counts
