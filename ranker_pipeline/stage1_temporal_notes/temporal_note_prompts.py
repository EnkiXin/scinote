"""Prompt templates for Stage 1 temporal note generation."""

SYSTEM_PROMPT = (
    "You are a careful, precise scientific video annotator. "
    "You describe ONLY what is visible in the video, with specific detail. "
    "You output ONLY valid JSON."
)


TEMPORAL_NOTE_PROMPT = """You are observing a segment of a scientific experiment video.

This is segment {segment_id} of {total_segments}, covering time {start_sec:.0f}s to {end_sec:.0f}s of the full video.

Describe EVERYTHING visible in this segment in detail. Be exhaustive and specific.

CRITICAL: Be specific, not generic.
- BAD:  "researcher holds tube"
- GOOD: "researcher holds 1.5mL microtube labeled 'DMEM' with blue cap"

- BAD:  "adds liquid"
- GOOD: "pipettes ~200uL of pink-colored DMEM medium into petri dish"

Output STRICT JSON:
{{
  "phase": "<one of: preparation | execution | observation | measurement | conclusion>",
  "actions_observed": ["<specific action 1>", "<specific action 2>"],
  "objects_visible": ["<object with distinguishing features>"],
  "visible_text_labels": ["<text/label visible on screen>"],
  "quantities": ["<specific quantity, volume, count, or reading>"],
  "key_distinguishing_features": "<2-3 sentences highlighting what is distinctive about this segment>"
}}

Output ONLY valid JSON, no other text."""
