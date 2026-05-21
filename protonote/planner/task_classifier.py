"""task_classifier.py — map a question to a task name.

ExpVid items already carry a `task` field, so for Phase 3 this is a passthrough.
A real classifier (trained on question text) will go here in a later phase.
"""
from __future__ import annotations

from typing import Optional


# canonical task names used throughout the codebase (also keys in
# protonote.planner.tool_policy.TASK_TO_TOOLS)
KNOWN_TASKS = {
    "sequence_generation",
    "sequence_ordering",
    "step_prediction",
    "video_verification",
    "experimental_conclusion",
    "scientific_discovery",
    # SciVideoBench (single bucket)
    "scivideobench",
}


def classify_task(item: dict) -> Optional[str]:
    """Return the task name for an item, or None if unknown."""
    t = item.get("task")
    if t in KNOWN_TASKS:
        return t
    # SciVB items use benchmark='scivideobench' without a fine-grained task
    if item.get("benchmark") == "scivideobench":
        return "scivideobench"
    return t  # unknown — caller decides fallback
