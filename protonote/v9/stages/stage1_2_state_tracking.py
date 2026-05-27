"""V9 Stage 1.2 — State lifecycle tracking + OCR ledger reverse injection.

Inputs:
  - the (down-sampled) video frames
  - the entity list produced by Stage 1.1
  - the OCR ledger built ahead of time by
    `protonote.v9.preprocessing.ocr_preprocessor.OCRLedgerBuilder`

The VLM emits, for every entity from Stage 1.1, a list of lifecycle
states (with timestamps + visual features + lifecycle_status +
optional transmutation links + quantitative_value pulled from the OCR
ledger), plus a list of state-transition operations with explicit
`input_states` / `output_states`.

Post-processing:
  - Parse JSON (truncation-aware envelope extraction + repair, reused
    from V8's stage1_extract).
  - Build the StateMachineKG container, attaching states to the right
    StateMachineEntity (by entity_id).
  - Validate every state's `quantitative_value` against the OCR ledger
    window via `ocr_preprocessor.validate_ocr_alignment` — sets the
    `ocr_alignment_warning` flag when tokens are missing from the
    ledger entries that overlap the state's time_interval.

See V9_RESEARCH_PLAN.md §3.3.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Optional

from PIL import Image

from protonote.v8.stages.stage1_extract import (
    _extract_json_envelope,
    _try_parse_json,
)
from protonote.v9.kg.state_entity import (
    EntityState,
    LifecycleStatus,
    StateMachineEntity,
)
from protonote.v9.kg.state_operation import StateTransitionOperation
from protonote.v9.kg.state_machine_kg import StateMachineKG
from protonote.v9.preprocessing.ocr_preprocessor import (
    OCRLedgerEntry,
    format_ocr_ledger_for_prompt,
    validate_ocr_alignment,
)
from protonote.v9.stages.stage1_1_core_entities import Stage1_1Result

logger = logging.getLogger(__name__)


_ALLOWED_LIFECYCLE = {
    "active", "consumed", "transformed", "merged", "split",
}

# Sub-set of action categories the Stage 2 enrichment trigger keys on.
# Stage 1.2 may emit values outside this set; we keep them verbatim.
_KNOWN_ACTION_CATEGORIES = {
    "mixing", "heating", "cooling", "centrifuging", "measuring",
    "observing", "transferring", "preparing", "incubating",
    "weighing", "titrating", "filtering", "washing", "drying",
    "sampling", "stirring", "diluting", "extracting",
}


_STAGE_1_2_PROMPT_TEMPLATE = """You are an expert scientific-experiment video analyst.

Your job: for every entity already identified in this video, track the
*lifecycle* of states it goes through and the *operations* that connect
those states.

EXISTING ENTITIES (from Stage 1.1 — do not invent new ones unless a
transmutation creates one, see Rule 4):
{entities_block}

OCR LEDGER (pre-processed text/numbers detected in each frame):
{ocr_block}

CRITICAL RULE 1 — entity_id consistency
A single physical entity keeps the SAME entity_id even when its color,
shape, position, or quantity changes. Do NOT mint a new entity for the
same object at a different timestamp.

CRITICAL RULE 2 — state snapshots
Every entity has at least one state. Create a NEW state whenever any of
these changes substantively:
  - color
  - shape / form / phase
  - position / container
  - quantity
  - label / annotation

A state describes the entity over a time interval, not at a single
instant.

CRITICAL RULE 3 — lifecycle_status per state
Tag every state with exactly ONE of:
  - "active"      : entity still present in recognisable form
  - "consumed"    : entity fully consumed and untraceable (e.g. ethanol
                    diluted into bulk solvent)
  - "transformed" : entity became one or more new entities (chemical
                    reaction product, etc.)
  - "merged"      : entity merged with peers into a single new entity
  - "split"       : entity split into multiple new entities (e.g.
                    centrifugation separating supernatant from pellet)

CRITICAL RULE 4 — transmutation links
When `lifecycle_status` is one of {{transformed, merged, split}}, you
MUST list the downstream new entity ids in `transmuted_to_entity_ids`
on the last state of the source entity, AND create new entity records
whose first state lists the source ids in `transmuted_from_entity_ids`.

Naming for new entities created here: use the NEXT free integer (e.g.
if Stage 1.1 ended with Entity_7, the first new entity is Entity_8).

CRITICAL RULE 5 — quantitative_value MUST come from the OCR ledger
NEVER hallucinate numbers. When you need a numeric value for a state:
  - pick one or more LEDGER entries whose timestamps fall inside the
    state's time_interval (±0.5 s),
  - whose `type` field is numeric, unit, or compound,
  - whose `text` is consistent with the entity (e.g. a balance reading
    near the weighed sample, not a background oven dial).
Set `quantitative_value` to the concatenated token string (e.g.
"0.535 g"), and put each contributing ledger token into
`raw_ocr_tokens`. If no suitable ledger entry exists, set
`quantitative_value` to null and leave `raw_ocr_tokens` empty.

CRITICAL RULE 6 — operations link states
Every operation MUST list at least one `input_states` and at least one
`output_states`. They are state_ids, not entity_ids. An operation
without explicit inputs/outputs is invalid.

CRITICAL RULE 7 — action_category
Pick the closest match from the menu below; if none fits, invent a
short lowercase verb-noun tag (e.g. "centrifuging", "transferring").
Allowed core values:
  mixing, heating, cooling, centrifuging, measuring, observing,
  transferring, preparing, incubating, weighing, titrating, filtering,
  washing, drying, sampling, stirring, diluting, extracting

OUTPUT FORMAT
Strict JSON, no markdown fences, no commentary, exactly this schema:

{{
  "entity_states": [
    {{
      "entity_id": "Entity_1",
      "states": [
        {{
          "state_id": "E1_S1",
          "time_interval": [start_sec, end_sec],
          "visual_features": "...",
          "lifecycle_status": "active",
          "transmuted_to_entity_ids": [],
          "transmuted_from_entity_ids": [],
          "quantitative_value": null,
          "raw_ocr_tokens": []
        }}
      ]
    }}
  ],
  "operations": [
    {{
      "operation_id": "Op_1",
      "action": "...",
      "timestamp": 0.0,
      "duration": null,
      "input_states": ["E1_S1"],
      "output_states": ["E1_S2"],
      "action_category": "mixing"
    }}
  ]
}}
"""


@dataclass
class Stage1_2Result:
    kg: StateMachineKG
    raw_response: str
    parse_ok: bool
    n_states_parsed: int = 0
    n_operations_parsed: int = 0
    error: Optional[str] = None
    ocr_alignment_warnings: int = 0


# ── helpers ────────────────────────────────────────────────────────

def _coerce_lifecycle(raw: object) -> LifecycleStatus:
    if isinstance(raw, str) and raw.strip().lower() in _ALLOWED_LIFECYCLE:
        return raw.strip().lower()  # type: ignore[return-value]
    return "active"


def _coerce_interval(raw: object) -> Optional[tuple[float, float]]:
    if isinstance(raw, (list, tuple)) and len(raw) >= 2:
        try:
            a = float(raw[0])
            b = float(raw[1])
        except (TypeError, ValueError):
            return None
        if b < a:
            a, b = b, a
        return (a, b)
    return None


def _coerce_str_list(raw: object) -> list[str]:
    if not isinstance(raw, list):
        return []
    return [str(x).strip() for x in raw if isinstance(x, (str, int, float)) and str(x).strip()]


def _coerce_float_or_none(raw: object) -> Optional[float]:
    if raw is None:
        return None
    if isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    if isinstance(raw, str):
        try:
            return float(raw.strip())
        except (TypeError, ValueError):
            return None
    return None


def _coerce_action_category(raw: object) -> Optional[str]:
    if not isinstance(raw, str):
        return None
    v = raw.strip().lower()
    if not v:
        return None
    return v


# ── parse a complete response ─────────────────────────────────────

def parse_stage1_2_response(
    raw: str,
    stage1_1_entities: list[StateMachineEntity],
) -> tuple[StateMachineKG, dict]:
    """Return a built KG and the bookkeeping dict (parse stats)."""
    kg = StateMachineKG()
    # Seed KG with Stage 1.1 entities (preserves entity-level metadata).
    for ent in stage1_1_entities:
        # Make a fresh copy so callers can re-use the original objects.
        kg.add_entity(StateMachineEntity(
            entity_id=ent.entity_id,
            canonical_name=ent.canonical_name,
            type=ent.type,
            is_individually_operated=ent.is_individually_operated,
            estimated_quantity=ent.estimated_quantity,
            core_role=ent.core_role,
            first_appearance=ent.first_appearance,
            canonical_id=ent.canonical_id,
        ))

    stats = {"n_states_parsed": 0, "n_operations_parsed": 0,
             "parse_ok": False, "error": None, "raw_data": None}

    envelope = _extract_json_envelope(raw)
    if envelope is None:
        stats["error"] = "no JSON envelope"
        return kg, stats
    data = _try_parse_json(envelope)
    if not isinstance(data, dict):
        stats["error"] = "JSON parse failed"
        return kg, stats
    stats["raw_data"] = data

    # 1) Attach states to existing entities; create transmutation-target
    #    entities lazily when referenced.
    entity_states = data.get("entity_states", [])
    if not isinstance(entity_states, list):
        entity_states = []

    for ent_record in entity_states:
        if not isinstance(ent_record, dict):
            continue
        eid = ent_record.get("entity_id")
        if not isinstance(eid, str) or not eid.startswith("Entity_"):
            continue
        if eid not in kg.entities:
            # Stage 1.2 invented a new entity (likely a transmutation
            # product). Create a minimal Container record; later passes
            # can refine.
            kg.entities[eid] = StateMachineEntity(
                entity_id=eid,
                canonical_name=eid,            # placeholder
                type="Material",                # safe default
            )

        ent = kg.entities[eid]
        for s in ent_record.get("states", []):
            if not isinstance(s, dict):
                continue
            interval = _coerce_interval(s.get("time_interval"))
            if interval is None:
                continue
            sid = s.get("state_id")
            if not isinstance(sid, str) or not sid:
                # synthesize a stable id
                sid = f"{eid.replace('Entity_', 'E')}_S{len(ent.states) + 1}"

            try:
                ent.add_state(EntityState(
                    state_id=sid,
                    time_interval=interval,
                    visual_features=str(s.get("visual_features", "")).strip(),
                    lifecycle_status=_coerce_lifecycle(s.get("lifecycle_status")),
                    transmuted_to_entity_ids=_coerce_str_list(
                        s.get("transmuted_to_entity_ids")),
                    transmuted_from_entity_ids=_coerce_str_list(
                        s.get("transmuted_from_entity_ids")),
                    quantitative_value=(
                        s.get("quantitative_value")
                        if isinstance(s.get("quantitative_value"), str)
                        and s.get("quantitative_value").strip()
                        else None
                    ),
                    raw_ocr_tokens=_coerce_str_list(s.get("raw_ocr_tokens")),
                ))
                stats["n_states_parsed"] += 1
            except ValueError:
                # Duplicate state_id — skip, Stage 1.2 occasionally
                # emits the same sid twice. We keep the first.
                continue

    # 2) Operations.
    ops = data.get("operations", [])
    if not isinstance(ops, list):
        ops = []
    for op in ops:
        if not isinstance(op, dict):
            continue
        opid = op.get("operation_id")
        if not isinstance(opid, str) or not opid:
            opid = f"Op_{len(kg.operations) + 1}"
        ts = _coerce_float_or_none(op.get("timestamp"))
        if ts is None:
            continue
        try:
            kg.add_operation(StateTransitionOperation(
                operation_id=opid,
                action=str(op.get("action", "")).strip(),
                timestamp=ts,
                duration=_coerce_float_or_none(op.get("duration")),
                input_states=_coerce_str_list(op.get("input_states")),
                output_states=_coerce_str_list(op.get("output_states")),
                operator_id=str(op.get("operator_id", "Entity_Operator")),
                confidence=_coerce_float_or_none(op.get("confidence")) or 1.0,
                action_category=_coerce_action_category(op.get("action_category")),
                description=(op.get("description")
                              if isinstance(op.get("description"), str)
                              else None),
            ))
            stats["n_operations_parsed"] += 1
        except ValueError:
            continue

    stats["parse_ok"] = True
    return kg, stats


def _apply_ocr_alignment(
    kg: StateMachineKG, ledger: list[OCRLedgerEntry],
) -> int:
    """Validate every state's quantitative_value against the OCR ledger.

    Returns the number of states marked with `ocr_alignment_warning`.
    """
    if not ledger:
        return 0

    # Build a (mutable) dict representation, run validate_ocr_alignment
    # on it, then copy flags back. This re-uses the dict-oriented
    # implementation in the preprocessor.
    dict_view = {
        "entity_states": [
            {
                "entity_id": ent.entity_id,
                "states": [s.to_dict() for s in ent.states],
            }
            for ent in kg.entities.values()
        ],
    }
    validate_ocr_alignment(dict_view, ledger, window_pad_sec=0.5)

    warnings = 0
    for ent_record in dict_view["entity_states"]:
        eid = ent_record["entity_id"]
        ent = kg.entities.get(eid)
        if ent is None:
            continue
        for sid_state in ent_record["states"]:
            sid = sid_state.get("state_id")
            warn = bool(sid_state.get("ocr_alignment_warning", False))
            if not warn:
                continue
            for s in ent.states:
                if s.state_id == sid:
                    s.ocr_alignment_warning = True
                    warnings += 1
                    break
    return warnings


# ── runner ────────────────────────────────────────────────────────

def run_stage1_2(
    frames: list[Image.Image],
    stage1_1: Stage1_1Result,
    ocr_ledger: list[OCRLedgerEntry],
    vlm,
    *,
    max_tokens: int = 4096,
    temperature: float = 0.0,
) -> Stage1_2Result:
    if not stage1_1.entities:
        return Stage1_2Result(
            kg=StateMachineKG(), raw_response="", parse_ok=False,
            error="no entities from Stage 1.1",
        )
    if not frames:
        return Stage1_2Result(
            kg=StateMachineKG(), raw_response="", parse_ok=False,
            error="no frames",
        )

    prompt = _STAGE_1_2_PROMPT_TEMPLATE.format(
        entities_block=stage1_1.to_prompt_block(),
        ocr_block=format_ocr_ledger_for_prompt(ocr_ledger),
    )

    try:
        raw = vlm.generate_video(
            prompt, frames,
            max_tokens=max_tokens, temperature=temperature,
        )
    except Exception as e:
        logger.warning("Stage 1.2 VLM call failed: %s", e)
        return Stage1_2Result(
            kg=StateMachineKG(), raw_response="", parse_ok=False,
            error=f"vlm error: {e}",
        )

    kg, stats = parse_stage1_2_response(raw, stage1_1.entities)
    warnings = _apply_ocr_alignment(kg, ocr_ledger)

    return Stage1_2Result(
        kg=kg,
        raw_response=raw,
        parse_ok=stats["parse_ok"],
        n_states_parsed=stats["n_states_parsed"],
        n_operations_parsed=stats["n_operations_parsed"],
        ocr_alignment_warnings=warnings,
        error=stats["error"],
    )
