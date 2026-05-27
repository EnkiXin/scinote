"""KG → Markdown renderer for LLM consumption.

Renders a KnowledgeGraph as structured markdown that an LLM can read
in the final answer prompt (Stage 4).
"""


def render_kg_markdown(kg) -> str:
    """Render KG as markdown.

    Output sections:
    1. Header + comprehension summary
    2. Entities (with grounding info)
    3. Operations (temporal order)
    4. Stages (high-level grouping, if any)
    """
    lines: list[str] = []

    # --- Header ---
    lines.append("# Video Knowledge Graph")
    lines.append("")

    # --- Comprehension summary ---
    lines.extend(_render_comprehension_summary(kg))
    lines.append("")

    # --- Entities ---
    lines.append("## Entities")
    lines.append("")
    if not kg.entities:
        lines.append("*No entities extracted.*")
        lines.append("")
    else:
        for entity in kg.entities.values():
            lines.extend(_render_entity(entity))
            lines.append("")

    # --- Operations (temporal order) ---
    lines.append("## Operations (in temporal order)")
    lines.append("")
    if not kg.operations:
        lines.append("*No operations extracted.*")
        lines.append("")
    else:
        for op_id in kg.temporal_chain:
            op = kg.get_operation(op_id)
            if op is not None:
                lines.append(_render_operation(op))
        lines.append("")

    # --- Stages (high-level grouping, if any) ---
    if kg.stages:
        lines.append("## Procedural Stages")
        lines.append("")
        for stage in kg.stages:
            lines.extend(_render_stage(stage))
            lines.append("")

    return "\n".join(lines)


def _render_comprehension_summary(kg) -> list[str]:
    """Render comprehension level summary."""
    m = kg.metadata
    lines = [
        f"**Comprehension level**: {m.comprehension_level:.0%}",
        f"- Grounded via image library: {m.grounded_via_image}",
        f"- Grounded via retrieve + image: {m.grounded_via_retrieve}",
        f"- Grounded via OCR: {m.grounded_via_ocr}",
        f"- Ungrounded: {m.ungrounded}",
    ]
    return lines


def _render_entity(entity) -> list[str]:
    """Render single entity."""
    lines = [f"### {entity.id} [{entity.type}]"]

    # Identity (from grounding or guess)
    if entity.grounded is not None:
        g = entity.grounded
        if g.identity:
            lines.append(
                f"- **Identity**: {g.identity} "
                f"(grounded via {g.method}, confidence {g.confidence:.2f})"
            )
            if g.source_dataset:
                lines.append(f"- **Source dataset**: {g.source_dataset}")
        else:
            lines.append("- **Identity**: unknown")
            if g.candidates:
                lines.append(
                    f"- **Candidates from KB**: {', '.join(g.candidates)}"
                )

        if g.ocr_text:
            lines.append(f'- **OCR text**: "{g.ocr_text}"')

        if g.evidence:
            lines.append(f"- **Visual evidence**: {g.evidence}")
    else:
        lines.append(
            f"- **Identity guess** (ungrounded): {entity.identity_guess}"
        )
        lines.append(
            f"- **Initial confidence**: {entity.initial_confidence:.2f}"
        )

    # Visual features
    lines.append(f"- **Features**: {entity.features}")

    # Appearance intervals (Approach 1 temporal)
    if entity.appearance_intervals:
        intervals_str = ", ".join(
            f"[{s}s-{e}s]" for s, e in entity.appearance_intervals
        )
        lines.append(f"- **Visible at**: {intervals_str}")

    # Optional domain-specific fields
    if entity.quantity:
        lines.append(
            f"- **Quantity**: {entity.quantity.get('value')} "
            f"{entity.quantity.get('unit', '')}"
        )
    if entity.state:
        lines.append(f"- **State**: {entity.state}")
    if entity.role_in_procedure:
        lines.append(f"- **Role**: {entity.role_in_procedure}")

    return lines


def _render_operation(op) -> str:
    """Render single operation as a temporal-ordered line."""
    base = f"- **{op.timestamp}s**: {op.action} — {op.subject} → {op.object}"
    if op.duration:
        base += f" (duration {op.duration}s)"
    if op.description:
        base += f" — *{op.description}*"
    return base


def _render_stage(stage) -> list[str]:
    """Render single stage."""
    lines = [
        f"### {stage.name} ({stage.interval[0]}s-{stage.interval[1]}s)",
        f"Operations: {', '.join(stage.operations)}",
    ]
    if stage.description:
        lines.append(f"*{stage.description}*")
    return lines
