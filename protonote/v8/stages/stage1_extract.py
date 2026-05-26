"""Stage 1: extract a KnowledgeGraph from a video.

A single VLM call (or a small retry budget) given:
  - the uniformly-sampled frames of a video
  - optionally the downstream question

The VLM emits a JSON envelope (per ``kg/stoa.py`` prompt template):

    {
      "entities":   [ {id, type, features, identity_guess,
                         initial_confidence, bbox?, appearance_intervals?,
                         ocr_candidate?}, ... ],
      "operations": [ {id, action, subject, object, timestamp,
                         duration?, confidence?, description?}, ... ]
    }

We then parse this into Entity / Operation objects and build a
KnowledgeGraph. The parser is deliberately forgiving — truncated /
malformed responses degrade to "skip this entity" rather than raising.

Public API:
    extract_kg(frames, vlm, question=None, duration_sec=None,
                  *, max_tokens=2048, temperature=0.0) -> KnowledgeGraph
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional

from PIL import Image

from protonote.v8.kg.entity import Entity
from protonote.v8.kg.knowledge_graph import KnowledgeGraph
from protonote.v8.kg.operation import Operation
from protonote.v8.kg.stoa import (
    ACTION_VOCAB,
    ENTITY_TYPES,
    STAGE1_SYSTEM_PROMPT,
    build_extraction_prompt,
)

logger = logging.getLogger(__name__)

# ============================================================
# Public API
# ============================================================


def extract_kg(
    frames: list[Image.Image],
    vlm,
    question: Optional[str] = None,
    duration_sec: Optional[float] = None,
    *,
    max_tokens: int = 2048,
    temperature: float = 0.0,
) -> KnowledgeGraph:
    """Extract a KnowledgeGraph by VLM call on the given frames.

    Args:
        frames: list of PIL images (typically 32 uniform samples).
        vlm: object exposing ``.generate_video(prompt, frames, system=,
             max_tokens=, temperature=)`` returning the raw text reply.
        question: optional downstream question to bias extraction.
        duration_sec: video duration in seconds (used to map
             appearance_intervals to frame indices). Defaults to
             ``len(frames)`` if not supplied.
        max_tokens / temperature: forwarded to the VLM call.

    Returns:
        KnowledgeGraph (possibly empty if extraction failed cleanly).
    """
    if not frames:
        logger.debug("extract_kg: no frames supplied")
        return KnowledgeGraph()

    duration = float(duration_sec) if duration_sec else float(len(frames))
    prompt = build_extraction_prompt(
        n_frames=len(frames),
        duration_sec=duration,
        question=question,
    )

    try:
        raw = vlm.generate_video(
            prompt, frames,
            system=STAGE1_SYSTEM_PROMPT,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    except Exception as e:
        logger.warning("Stage 1 VLM call failed: %s", e)
        return KnowledgeGraph()

    return parse_kg_from_response(raw)


# ============================================================
# JSON envelope extraction (robust)
# ============================================================


_VALID_ENTITY_TYPES = set(ENTITY_TYPES)
_VALID_ACTIONS = set(ACTION_VOCAB)


def parse_kg_from_response(raw: str | None) -> KnowledgeGraph:
    """Parse the VLM raw text into a KnowledgeGraph.

    Robustness:
      - strips ```json fences``` if present
      - extracts the FIRST balanced JSON object via brace counting
      - tries permissive JSON repairs (single-quote, trailing comma)
      - returns empty KG on total parse failure
      - drops malformed individual entities/ops, keeps the rest
    """
    if not raw:
        return KnowledgeGraph()

    payload = _extract_json_envelope(raw)
    if payload is None:
        logger.debug("parse_kg: could not extract JSON envelope")
        return KnowledgeGraph()

    data = _try_parse_json(payload)
    if data is None:
        return KnowledgeGraph()
    if not isinstance(data, dict):
        return KnowledgeGraph()

    kg = KnowledgeGraph()
    _add_entities(kg, data.get("entities") or [])
    _add_operations(kg, data.get("operations") or [])
    return kg


def _extract_json_envelope(raw: str) -> Optional[str]:
    """Find the first balanced ``{ ... }`` object in raw text.

    Handles common LLM cruft: leading prose, ``` fences, trailing
    explanations. If the text is truncated (depth never returns to 0
    — e.g. the VLM hit max_tokens mid-enumeration), trim to the last
    complete element and pad missing braces / brackets so that the
    parser can recover the partial KG.
    """
    s = raw.strip()
    # Strip code fences if present.
    fence = re.search(r"```(?:json|JSON)?\s*\n?", s)
    if fence:
        s = s[fence.end():]
        end_fence = s.rfind("```")
        if end_fence >= 0:
            s = s[:end_fence]

    start = s.find("{")
    if start < 0:
        return None

    depth = 0
    bracket_depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        ch = s[i]
        if esc:
            esc = False
            continue
        if ch == "\\":
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[start:i + 1]
        elif ch == "[":
            bracket_depth += 1
        elif ch == "]":
            bracket_depth -= 1

    # --- Truncation recovery ---
    # The string ended while still inside an object (depth > 0). Most
    # common cause: VLM hit max_tokens mid-enumeration. Trim back to
    # the last complete element and add the missing closing brackets
    # so the rest of the parser can recover whatever was emitted.
    tail = s[start:]
    return _repair_truncated(tail)


def _repair_truncated(s: str) -> Optional[str]:
    """Pad a truncated JSON object with its missing brackets.

    Trims back to the LAST point where depth + bracket_depth returned
    to (depth=1, bracket_depth=1) — i.e. just after a complete entity
    object closed inside an array of objects — then pads the closing
    brackets to balance.

    If no such safe point exists, we fall back to "trim to last `,`
    + balance" as a coarse second attempt.
    """
    in_str = False
    esc = False
    depth = 0
    bracket_depth = 0
    # Track position immediately AFTER a `}` that brings us back to
    # exactly (depth=1, bracket_depth=1). That's the end of one entity
    # in the "entities" array.
    last_array_item_end: Optional[int] = None
    last_comma: Optional[int] = None

    for i, ch in enumerate(s):
        if esc:
            esc = False
            continue
        if ch == "\\":
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 1 and bracket_depth == 1:
                last_array_item_end = i + 1
        elif ch == "[":
            bracket_depth += 1
        elif ch == "]":
            bracket_depth -= 1
        elif ch == ",":
            last_comma = i

    trim_at = last_array_item_end if last_array_item_end is not None \
                  else last_comma
    if trim_at is None:
        return None

    trimmed = s[:trim_at]
    # Recompute current depths on trimmed text.
    in_str = False
    esc = False
    depth = 0
    bracket_depth = 0
    for ch in trimmed:
        if esc:
            esc = False
            continue
        if ch == "\\":
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{": depth += 1
        elif ch == "}": depth -= 1
        elif ch == "[": bracket_depth += 1
        elif ch == "]": bracket_depth -= 1

    if depth < 0 or bracket_depth < 0:
        return None
    return trimmed + ("]" * bracket_depth) + ("}" * depth)


def _try_parse_json(payload: str) -> Optional[Any]:
    """Try strict parse, then permissive repairs."""
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        pass

    repaired = payload
    repaired = repaired.replace("'", '"')                    # single → double
    repaired = re.sub(r",\s*}", "}", repaired)               # trailing comma
    repaired = re.sub(r",\s*]", "]", repaired)               # trailing comma
    try:
        return json.loads(repaired)
    except json.JSONDecodeError as e:
        logger.debug("JSON parse failed after repair: %s", e)
        return None


# ============================================================
# Entity / Operation construction (defensive)
# ============================================================


_OPERATOR_SINGLETON_ID = "Entity_Operator"


def _coerce_id(raw_id: Any, prefix: str) -> Optional[str]:
    """Ensure id starts with `prefix`. If empty/None, return None."""
    if not isinstance(raw_id, str) or not raw_id.strip():
        return None
    rid = raw_id.strip()
    if not rid.startswith(prefix):
        # The schema mandates the prefix — repair if possible.
        if rid.startswith(prefix.lower()):
            rid = prefix + rid[len(prefix):]
        else:
            rid = f"{prefix}{rid}"
    return rid


def _coerce_float(x: Any, default: float = 0.0,
                       lo: float = 0.0, hi: float = 1.0) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, v))


def _coerce_int(x: Any, default: int = 0, lo: int | None = None) -> int:
    try:
        v = int(x)
    except (TypeError, ValueError):
        try:
            v = int(float(x))
        except (TypeError, ValueError):
            return default
    if lo is not None and v < lo:
        return default
    return v


def _coerce_bbox(raw: Any) -> Optional[tuple[int, int, int, int]]:
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        return None
    try:
        x1, y1, x2, y2 = (int(v) for v in raw)
    except (TypeError, ValueError):
        return None
    if x2 <= x1 or y2 <= y1 or x1 < 0 or y1 < 0:
        return None
    return (x1, y1, x2, y2)


def _coerce_intervals(raw: Any) -> list[tuple[int, int]]:
    """Accept [[s, e], …] or [(s, e), …]; drop malformed pairs."""
    if not isinstance(raw, list):
        return []
    out: list[tuple[int, int]] = []
    for item in raw:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue
        try:
            s = int(item[0])
            e = int(item[1])
        except (TypeError, ValueError):
            continue
        if e < s:
            s, e = e, s
        out.append((s, e))
    return out


def _add_entities(kg: KnowledgeGraph, raw_entities: list[Any]) -> None:
    if not isinstance(raw_entities, list):
        return
    seen_ids: set[str] = set()
    for raw in raw_entities:
        if not isinstance(raw, dict):
            continue
        ent = _build_entity(raw)
        if ent is None:
            continue
        if ent.id in seen_ids:
            # de-dup: keep first occurrence
            continue
        try:
            kg.add_entity(ent)
            seen_ids.add(ent.id)
        except ValueError as e:
            logger.debug("entity rejected by KG: %s", e)


def _build_entity(raw: dict) -> Optional[Entity]:
    eid = raw.get("id")
    if not isinstance(eid, str):
        return None
    eid_s = eid.strip()
    if not eid_s:
        return None

    # Special-case the operator singleton id.
    if eid_s == _OPERATOR_SINGLETON_ID:
        # Re-write to a valid "Entity*" form for the KG schema.
        eid_s = "EntityOperator"

    if not eid_s.startswith("Entity"):
        coerced = _coerce_id(eid_s, "Entity")
        if coerced is None:
            return None
        eid_s = coerced

    etype = raw.get("type")
    if etype not in _VALID_ENTITY_TYPES:
        # If type missing / invalid, try one defaulting heuristic:
        #   if ocr_candidate=True, this is a Display.
        if bool(raw.get("ocr_candidate")):
            etype = "Display"
        else:
            return None

    try:
        ent = Entity(
            id=eid_s,
            type=etype,
            features=str(raw.get("features") or ""),
            identity_guess=str(raw.get("identity_guess") or ""),
            initial_confidence=_coerce_float(
                raw.get("initial_confidence"), default=0.5,
            ),
            appearance_intervals=_coerce_intervals(
                raw.get("appearance_intervals"),
            ),
            bbox=_coerce_bbox(raw.get("bbox")),
            ocr_candidate=bool(raw.get("ocr_candidate", False)),
        )
    except ValueError as e:
        logger.debug("entity construction failed: %s", e)
        return None
    return ent


def _add_operations(kg: KnowledgeGraph, raw_ops: list[Any]) -> None:
    if not isinstance(raw_ops, list):
        return
    seen_ids: set[str] = set()
    for raw in raw_ops:
        if not isinstance(raw, dict):
            continue
        op = _build_operation(raw, kg)
        if op is None:
            continue
        if op.id in seen_ids:
            continue
        try:
            kg.add_operation(op)
            seen_ids.add(op.id)
        except ValueError as e:
            logger.debug("operation rejected by KG: %s", e)


def _build_operation(raw: dict, kg: KnowledgeGraph) -> Optional[Operation]:
    op_id = raw.get("id")
    if not isinstance(op_id, str):
        return None
    op_id_s = _coerce_id(op_id, "Op")
    if op_id_s is None:
        return None

    action = raw.get("action")
    if action not in _VALID_ACTIONS:
        # Coerce to closed vocab if possible
        if isinstance(action, str) and action.lower() in _VALID_ACTIONS:
            action = action.lower()
        else:
            action = "use"   # last-resort fallback

    subject = raw.get("subject")
    obj = raw.get("object")
    # Normalize Entity_Operator → EntityOperator
    if subject == _OPERATOR_SINGLETON_ID:
        subject = "EntityOperator"
    if obj == _OPERATOR_SINGLETON_ID:
        obj = "EntityOperator"
    if not isinstance(subject, str) or not isinstance(obj, str):
        return None
    # subject/object SHOULD reference existing entities, but Stage 1
    # output sometimes references entities it forgot to declare. We
    # don't enforce this here — Stage 2/3/4 simply ignore missing refs.

    ts_raw = raw.get("timestamp", 0)
    ts = _coerce_int(ts_raw, default=0, lo=0)

    try:
        op = Operation(
            id=op_id_s,
            action=str(action),
            subject=subject,
            object=obj,
            timestamp=ts,
            duration=_coerce_int(raw.get("duration")) or None,
            confidence=_coerce_float(
                raw.get("confidence"), default=1.0,
            ),
            description=(
                str(raw["description"])
                if isinstance(raw.get("description"), str) else None
            ),
        )
    except ValueError as e:
        logger.debug("operation construction failed: %s", e)
        return None
    return op
