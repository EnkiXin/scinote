"""V9 Stage 4 — Multi-view KG render + reasoning prompt.

Builds the final VLM prompt for answering the question by:
  1. Rendering ONE markdown block per active view (quantitative,
     hypothetical, conceptual, procedural), each highlighting the
     parts of the StateMachineKG most useful for that reasoning type.
  2. Composing those blocks into a single multi-view KG section.
  3. Generating adaptive reasoning instructions matched to the active
     views (so the LLM knows what kind of reasoning to do per view).

The renderers deliberately repeat some information across views —
this is cheaper than letting the LLM hunt across one giant KG dump.

See V9_RESEARCH_PLAN.md §6.2 / §8.
"""

from __future__ import annotations

from typing import Optional

from protonote.v9.kg.state_entity import EntityState, StateMachineEntity
from protonote.v9.kg.state_machine_kg import StateMachineKG


# ── per-view renderers ────────────────────────────────────────────

def _states_with_quant(entity: StateMachineEntity) -> list[EntityState]:
    return [s for s in entity.states if s.quantitative_value]


def render_quantitative_view(kg: StateMachineKG) -> str:
    lines = ["#### Quantitative timeline"]
    any_qv = False
    for ent in kg.entities.values():
        qs = _states_with_quant(ent)
        if not qs:
            continue
        any_qv = True
        for s in sorted(qs, key=lambda x: x.time_interval[0]):
            warn = " ⚠️" if s.ocr_alignment_warning else ""
            lines.append(
                f"- {s.time_interval[0]:.0f}-{s.time_interval[1]:.0f}s | "
                f"{ent.entity_id} ({ent.canonical_name}) → "
                f"**{s.quantitative_value}**{warn}"
            )
    if not any_qv:
        lines.append("(no on-screen quantitative readings captured)")

    # Operation parameters (from RAG enrichment if available).
    lines.append("")
    lines.append("#### Operation parameters")
    any_param = False
    for op in kg.operations:
        params = op.operation_enrichment.get("typical_parameters") \
            if op.operation_enrichment else None
        if not params:
            continue
        any_param = True
        param_str = ", ".join(f"{k}={v}" for k, v in params.items())
        lines.append(
            f"- {op.operation_id} ({op.action}): {param_str}"
        )
    if not any_param:
        lines.append("(no operation parameters from KB)")
    return "\n".join(lines)


def render_hypothetical_view(kg: StateMachineKG) -> str:
    lines = ["#### State transitions"]
    if not kg.operations:
        lines.append("(no operations extracted)")
    for op in kg.operations:
        in_states = ", ".join(op.input_states) or "(none)"
        out_states = ", ".join(op.output_states) or "(none)"
        cat = f" [{op.action_category}]" if op.action_category else ""
        lines.append(
            f"- **{op.operation_id}** at {op.timestamp:.0f}s{cat}: "
            f"{op.action}"
        )
        lines.append(f"    inputs: {in_states}")
        lines.append(f"    outputs: {out_states}")

    # Per-entity lifecycle / transmutation links.
    lines.append("")
    lines.append("#### Entity lifecycles")
    for ent in kg.entities.values():
        lines.append(f"- **{ent.entity_id}** ({ent.canonical_name}) [{ent.type}]")
        for s in sorted(ent.states, key=lambda x: x.time_interval[0]):
            extras = []
            if s.lifecycle_status != "active":
                extras.append(s.lifecycle_status)
            if s.transmuted_to_entity_ids:
                extras.append(
                    "→ " + ", ".join(s.transmuted_to_entity_ids)
                )
            if s.transmuted_from_entity_ids:
                extras.append(
                    "← " + ", ".join(s.transmuted_from_entity_ids)
                )
            extras_str = f"  ({'; '.join(extras)})" if extras else ""
            lines.append(
                f"    - {s.state_id} "
                f"[{s.time_interval[0]:.0f}-{s.time_interval[1]:.0f}s]: "
                f"{s.visual_features or '(no description)'}{extras_str}"
            )

    # Operation failure modes (from RAG enrichment).
    failures = []
    for op in kg.operations:
        modes = (op.operation_enrichment or {}).get("failure_modes")
        if modes:
            failures.append(
                f"- {op.operation_id} ({op.action_category or op.action}): "
                + "; ".join(modes)
            )
    if failures:
        lines.append("")
        lines.append("#### Known failure modes")
        lines.extend(failures)

    return "\n".join(lines)


def render_conceptual_view(kg: StateMachineKG) -> str:
    lines = ["#### Core entities (compact)"]
    if not kg.entities:
        lines.append("(no entities extracted)")
        return "\n".join(lines)
    for ent in kg.entities.values():
        role = f", role={ent.core_role}" if ent.core_role else ""
        qty = f" ×{ent.estimated_quantity}" \
            if ent.estimated_quantity > 1 else ""
        lines.append(
            f"- {ent.entity_id} [{ent.type}]: "
            f"\"{ent.canonical_name}\"{qty}{role}"
        )
    # Entity-level enrichment (implicit properties).
    enrichments = [
        (eid, ent.entity_level_enrichment)
        for eid, ent in kg.entities.items()
        if ent.entity_level_enrichment
    ]
    if enrichments:
        lines.append("")
        lines.append("#### Implicit properties (from KB)")
        for eid, enr in enrichments:
            props = enr.get("implicit_properties") if isinstance(enr, dict) else None
            if not props:
                continue
            for k, v in props.items():
                lines.append(f"- {eid}: {k} = {v}")
    return "\n".join(lines)


def render_procedural_view(kg: StateMachineKG) -> str:
    lines = ["#### Operations in temporal order"]
    if not kg.operations:
        lines.append("(no operations extracted)")
        return "\n".join(lines)
    for op in kg.operations:
        cat = f" [{op.action_category}]" if op.action_category else ""
        dur = f" (~{op.duration:.0f}s)" if op.duration else ""
        lines.append(
            f"- **{op.timestamp:.0f}s**{cat}{dur}: {op.action}"
        )
    return "\n".join(lines)


_RENDERERS = {
    "quantitative": render_quantitative_view,
    "hypothetical": render_hypothetical_view,
    "conceptual":   render_conceptual_view,
    "procedural":   render_procedural_view,
}


# ── multi-view stratagist ─────────────────────────────────────────

def render_multi_view_kg(
    kg: StateMachineKG, active_views: list[str],
) -> str:
    """Build the multi-view markdown for Stage 4 prompt consumption."""
    parts = [
        "# Knowledge Graph (multi-view)",
        f"Active views: {', '.join(active_views) if active_views else '(none)'}",
        "",
    ]
    for v in active_views:
        renderer = _RENDERERS.get(v)
        if renderer is None:
            continue
        parts.append(f"## {v.upper()} view")
        parts.append(renderer(kg))
        parts.append("")
    return "\n".join(parts)


_VIEW_INSTRUCTIONS = {
    "quantitative": (
        "For numerical aspects: extract values from the Quantitative "
        "timeline and any Operation parameters. If a value carries the "
        "⚠️ warning it was not grounded in OCR — treat it as low "
        "confidence. Perform the calculation step by step."
    ),
    "hypothetical": (
        "For causal / what-if aspects: trace State transitions, "
        "consult Entity lifecycles (esp. consumed/transformed/merged/"
        "split states), and use Known failure modes from KB to predict "
        "outcomes. Reason about pre- and post-conditions explicitly."
    ),
    "conceptual": (
        "For conceptual aspects: identify the principle / technique by "
        "matching the Core entities and any Implicit properties from KB. "
        "Avoid over-elaboration."
    ),
    "procedural": (
        "For procedural aspects: follow the Operations in temporal "
        "order. Be careful about control vs experimental entities — "
        "individually-operated entities follow their own track."
    ),
}


def _format_options(options) -> str:
    if isinstance(options, dict):
        return "\n".join(f"{k}. {v}" for k, v in options.items())
    if isinstance(options, list):
        labels = "ABCDEFGHIJ"
        return "\n".join(
            f"{labels[i]}. {opt}" for i, opt in enumerate(options[:10])
        )
    return str(options)


def build_stage4_prompt(
    *,
    question: str,
    options,
    kg: StateMachineKG,
    active_views: list[str],
    task_type: str = "mc",
) -> str:
    """Build the final Stage 4 prompt with multi-view KG + reasoning hints.

    `task_type` is "mc" (answer A/B/...) or "seq_gen" (list output).
    The instruction footer adapts to the task type.
    """
    kg_block = render_multi_view_kg(kg, active_views)
    instr_lines = [
        f"- {_VIEW_INSTRUCTIONS[v]}"
        for v in active_views
        if v in _VIEW_INSTRUCTIONS
    ]
    instructions = "\n".join(instr_lines) if instr_lines else (
        "- Reason carefully step by step, then answer."
    )

    if task_type == "mc":
        format_footer = (
            "Provide brief step-by-step reasoning, then on the FINAL line "
            "output exactly one letter (A, B, C, ...) matching your chosen "
            "option. Do not output anything after that letter."
        )
    else:
        format_footer = (
            "Provide brief step-by-step reasoning, then on the FINAL line "
            "output your answer in the format the question requested."
        )

    return f"""You are answering a question about a scientific experiment video.

Question: {question.strip()}

Options:
{_format_options(options)}

This question activates the following reasoning views: {', '.join(active_views)}.

{kg_block}

Reasoning instructions:
{instructions}

{format_footer}

Reasoning:"""
