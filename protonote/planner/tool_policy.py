"""tool_policy.py — task-conditional tool routing.

For Phase 3 each task is mapped to a small ordered list of tools the agent
should consult before producing the final answer. The choice is grounded in
what the question is actually asking about:

  - sequence_generation: list actions in order → visual descriptions
  - sequence_ordering:   pick order from options → visual + temporal
  - step_prediction:     predict next step → visual on the late part of video
  - video_verification:  judge whether shown step matches stated step → OCR + visual
  - experimental_conclusion / scientific_discovery: open-ended → visual + OCR

`note_read` is implicit (the controller always renders running notes before
the final answer call) and `note_write` is implicit too (every tool result
that succeeds is appended). So the lists below name only the *evidence-
gathering* tools, not the bookkeeping tools.
"""
from __future__ import annotations

TASK_TO_TOOLS: dict[str, list[str]] = {
    "sequence_generation":     ["visual_inspect"],
    "sequence_ordering":       ["visual_inspect"],
    "step_prediction":         ["visual_inspect"],
    "video_verification":      ["ocr", "visual_inspect"],
    "experimental_conclusion": ["visual_inspect", "ocr"],
    "scientific_discovery":    ["visual_inspect", "ocr"],
    # SciVideoBench: conceptual / hypothetical MC — visual descriptions
    # are the most useful signal.
    "scivideobench":           ["visual_inspect"],
}

DEFAULT_TOOLS = ["visual_inspect"]


def tools_for_task(task: str | None) -> list[str]:
    """Return the ordered tool subset for a task. Unknown task → DEFAULT_TOOLS."""
    if not task:
        return list(DEFAULT_TOOLS)
    return list(TASK_TO_TOOLS.get(task, DEFAULT_TOOLS))
