"""V9 Stage 1.1 — Core entity identification + fork detection.

Replaces V8's single-shot KG extraction. This stage:
  1. Looks at the (downsampled) video frames.
  2. Emits a SHORT list of important, persistent entities — not every
     object that appears in any frame.
  3. Applies the "same-track aggregation, different-track separation"
     principle so control vs experimental subjects are not fused.

It does NOT track state changes — that is Stage 1.2's job.

The prompt enforces (via few-shot examples):
  • Cross-frame identity (one physical object = one entity_id).
  • Fork detection (`is_individually_operated`).
  • 6-class type taxonomy reused from V8.
  • Strict JSON output with no markdown fences.

See V9_RESEARCH_PLAN.md §3.2.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Optional

from PIL import Image

# Reuse V8's battle-tested JSON envelope + repair helpers.
from protonote.v8.stages.stage1_extract import (
    _extract_json_envelope,
    _try_parse_json,
)
from protonote.v9.kg.state_entity import StateMachineEntity

logger = logging.getLogger(__name__)


_ALLOWED_TYPES = {
    "Operator", "Instrument", "Container", "Material", "Display", "Measurement",
}
_ALLOWED_ROLES = {
    "starting_material", "tool", "intermediate_product", "final_product",
    "control", "experimental", "observer", "byproduct",
}


_STAGE_1_1_PROMPT = """You are an expert scientific-experiment video analyst.

Look at these frames and identify the CORE entities in this video.

OUTPUT REQUIREMENTS
  - Return only entities that are important and persistent across the
    video. DO NOT list every object that appears in any single frame.
  - 5 – 15 entities total is typical. Outliers (single-frame objects,
    backgrounds, decorations) MUST be skipped.

CRITICAL RULE 1 — cross-frame identity
A single physical object that moves, changes color, or otherwise evolves
is STILL ONE entity. Never create separate entity_ids for the same
object at different timestamps.

CRITICAL RULE 2 — same-track aggregation, different-track separation
Aggregate multiple peers of the SAME class into ONE entity ONLY IF, for
the entire video:
  - they undergo the SAME operations,
  - they share the SAME state at all times,
  - they occupy the SAME spatial bucket (e.g. one tube rack).
If any individual is singled out at ANY time — picked up alone, reagent
added singly, marked with a label, heated alone — that individual MUST
be its own entity. Default to splitting when unsure.

  ✗ WRONG:  one entity for 10 tubes, one of which later turns red
  ✓ RIGHT:  Entity_1 = 9 untouched empty tubes (control)
            Entity_2 = 1 tube with reagent (experimental)

CRITICAL RULE 3 — canonical name
Use the MOST SPECIFIC identification you can give (e.g. "saturated
potassium sulfate solution" rather than "clear liquid"). Embed quantity
and role hints in the name when relevant ("9 unused empty test tubes
(control)").

CRITICAL RULE 4 — type taxonomy (use exactly ONE of these)
  Operator      = a human / hand performing the experiment
  Instrument    = active lab equipment (microscope, centrifuge, ...)
  Container     = passive vessel (flask, test tube, petri dish, ...)
  Material      = chemical / biological substance (solution, powder, ...)
  Display       = a screen / readout showing text or numbers
  Measurement   = a gauge / scale / sensor that emits values

FEW-SHOT EXAMPLES

Example A — aggregation (no fork)
  Video: 5 already-labelled tubes (S1–S5), each gets the same reagent
  added in the same step.
  Output:
    [{"entity_id":"Entity_1",
      "canonical_name":"5 labelled tubes treated identically",
      "type":"Container", "estimated_quantity":5,
      "is_individually_operated":false, "core_role":"experimental",
      "first_appearance":0}]

Example B — fork (control vs experimental)
  Video: 5 same-shape tubes. Only the middle one is reagent-loaded;
  the other four stay empty as controls.
  Output:
    [{"entity_id":"Entity_1",
      "canonical_name":"4 control tubes (untouched)",
      "type":"Container", "estimated_quantity":4,
      "is_individually_operated":false, "core_role":"control",
      "first_appearance":0},
     {"entity_id":"Entity_2",
      "canonical_name":"1 experimental tube (reagent A added)",
      "type":"Container", "estimated_quantity":1,
      "is_individually_operated":true, "core_role":"experimental",
      "first_appearance":3}]

Example C — fork with different treatments
  Video: 10 tubes; 5 get reagent A (turn red), 5 get reagent B (turn blue).
  Output:
    [{"entity_id":"Entity_1",
      "canonical_name":"5 tubes treated with reagent A",
      "type":"Container", "estimated_quantity":5,
      "is_individually_operated":true, "core_role":"experimental",
      "first_appearance":2},
     {"entity_id":"Entity_2",
      "canonical_name":"5 tubes treated with reagent B",
      "type":"Container", "estimated_quantity":5,
      "is_individually_operated":true, "core_role":"experimental",
      "first_appearance":5}]

OUTPUT FORMAT
Strict JSON, no markdown fences, no commentary, exactly this schema:

{{
  "entities": [
    {{
      "entity_id": "Entity_1",
      "canonical_name": "...",
      "type": "Operator|Instrument|Container|Material|Display|Measurement",
      "estimated_quantity": 1,
      "first_appearance": 0,
      "core_role": "starting_material|tool|intermediate_product|final_product|control|experimental",
      "is_individually_operated": false
    }}
  ]
}}
"""


@dataclass
class Stage1_1Result:
    """Outcome of Stage 1.1 extraction (compact, easy to feed into 1.2)."""
    entities: list[StateMachineEntity]
    raw_response: str
    parse_ok: bool
    error: Optional[str] = None

    def to_prompt_block(self) -> str:
        """Render entities as a markdown block for Stage 1.2 to consume."""
        lines = ["### CORE ENTITIES (from Stage 1.1)"]
        if not self.entities:
            lines.append("(none identified)")
            return "\n".join(lines)
        for e in self.entities:
            individuality = (
                "individually-operated" if e.is_individually_operated
                else f"group of {e.estimated_quantity}"
            )
            role_part = f", role={e.core_role}" if e.core_role else ""
            lines.append(
                f"- {e.entity_id} [{e.type}] "
                f"\"{e.canonical_name}\" ({individuality}{role_part}, "
                f"first_seen ~{e.first_appearance:.0f}s)"
            )
        return "\n".join(lines)


# ── parser helpers ────────────────────────────────────────────────

def _coerce_entity_id(raw: object, idx: int) -> str:
    if isinstance(raw, str) and raw.strip().startswith("Entity_"):
        return raw.strip()
    if isinstance(raw, str) and raw.strip():
        return f"Entity_{raw.strip().lstrip('Entity').lstrip('_').strip()}" \
            if raw.strip() else f"Entity_{idx}"
    return f"Entity_{idx}"


def _coerce_type(raw: object) -> Optional[str]:
    if not isinstance(raw, str):
        return None
    t = raw.strip()
    if t in _ALLOWED_TYPES:
        return t
    # Case-insensitive recovery (some models lowercase).
    for at in _ALLOWED_TYPES:
        if t.lower() == at.lower():
            return at
    return None


def _coerce_role(raw: object) -> Optional[str]:
    if not isinstance(raw, str):
        return None
    r = raw.strip()
    if r in _ALLOWED_ROLES:
        return r
    if r.lower() in _ALLOWED_ROLES:
        return r.lower()
    return None  # tolerate unknown role; downstream ignores None


def _coerce_int(raw: object, default: int = 1, lo: int = 1) -> int:
    if isinstance(raw, bool):    # bool is subclass of int, skip it
        return default
    if isinstance(raw, int) and raw >= lo:
        return raw
    if isinstance(raw, float):
        return max(lo, int(raw))
    if isinstance(raw, str):
        try:
            return max(lo, int(float(raw.strip())))
        except (TypeError, ValueError):
            return default
    return default


def _coerce_float(raw: object, default: float = 0.0) -> float:
    if isinstance(raw, bool):
        return default
    if isinstance(raw, (int, float)):
        return float(raw)
    if isinstance(raw, str):
        try:
            return float(raw.strip())
        except (TypeError, ValueError):
            return default
    return default


def _coerce_bool(raw: object, default: bool = False) -> bool:
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        return raw.strip().lower() in {"true", "yes", "1"}
    if isinstance(raw, (int, float)):
        return bool(raw)
    return default


# ── parse a complete response ─────────────────────────────────────

def parse_stage1_1_response(raw: str) -> Stage1_1Result:
    """Convert a raw VLM response into a Stage1_1Result."""
    if not raw or not raw.strip():
        return Stage1_1Result(entities=[], raw_response=raw or "",
                                parse_ok=False, error="empty response")

    envelope = _extract_json_envelope(raw)
    if envelope is None:
        return Stage1_1Result(entities=[], raw_response=raw,
                                parse_ok=False,
                                error="no JSON envelope found")

    data = _try_parse_json(envelope)
    if not isinstance(data, dict):
        return Stage1_1Result(entities=[], raw_response=raw,
                                parse_ok=False,
                                error="JSON parse failed")

    entities_raw = data.get("entities", [])
    if not isinstance(entities_raw, list):
        return Stage1_1Result(entities=[], raw_response=raw,
                                parse_ok=False,
                                error="entities field is not a list")

    out: list[StateMachineEntity] = []
    seen_ids: set[str] = set()
    for idx, item in enumerate(entities_raw, start=1):
        if not isinstance(item, dict):
            continue
        etype = _coerce_type(item.get("type"))
        if etype is None:
            logger.debug(
                "skip entity %s: invalid type %r", idx, item.get("type"),
            )
            continue
        eid = _coerce_entity_id(item.get("entity_id"), idx)
        if eid in seen_ids:
            # Collision — append a suffix.
            eid = f"{eid}_{idx}"
        seen_ids.add(eid)

        name = item.get("canonical_name") or item.get("name") or ""
        if not isinstance(name, str) or not name.strip():
            continue
        name = name.strip()

        ent = StateMachineEntity(
            entity_id=eid,
            canonical_name=name,
            type=etype,
            estimated_quantity=_coerce_int(item.get("estimated_quantity"), 1, 1),
            first_appearance=_coerce_float(item.get("first_appearance"), 0.0),
            core_role=_coerce_role(item.get("core_role")),
            is_individually_operated=_coerce_bool(
                item.get("is_individually_operated"), False,
            ),
        )
        out.append(ent)

    return Stage1_1Result(entities=out, raw_response=raw, parse_ok=True)


# ── runner ────────────────────────────────────────────────────────

def run_stage1_1(
    frames: list[Image.Image],
    vlm,
    *,
    max_tokens: int = 1024,
    temperature: float = 0.0,
) -> Stage1_1Result:
    """Run Stage 1.1 on the given frames.

    `vlm` must expose ``generate_video(prompt, frames, max_tokens=...,
    temperature=...)`` — same shape as the V8 client.
    """
    if not frames:
        return Stage1_1Result(entities=[], raw_response="",
                                parse_ok=False, error="no frames")

    try:
        raw = vlm.generate_video(
            _STAGE_1_1_PROMPT, frames,
            max_tokens=max_tokens, temperature=temperature,
        )
    except Exception as e:
        logger.warning("Stage 1.1 VLM call failed: %s", e)
        return Stage1_1Result(entities=[], raw_response="",
                                parse_ok=False, error=f"vlm error: {e}")

    return parse_stage1_1_response(raw)
