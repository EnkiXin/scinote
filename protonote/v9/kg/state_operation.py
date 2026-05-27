"""V9 state-transition operation schema.

Each operation is an edge in the state graph: it consumes a set of
`input_states` (referenced by state_id) and produces `output_states`.
Stage 1.2 produces these explicitly so that hypothetical-reasoning
questions ("if X failed, what happens to Y?") can be answered by
walking the graph rather than guessing.

See V9_RESEARCH_PLAN.md §2.2.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional

# Categorical tag used by Stage 2 enrichment trigger to decide whether
# the operation deserves a RAG lookup, and by Stage 4 prompts to weight
# operation parameters appropriately.
ActionCategory = str
# Common values:
#   "mixing", "heating", "cooling", "centrifuging", "measuring",
#   "observing", "transferring", "preparing", "incubating",
#   "weighing", "titrating"
# We keep this as a free-form str so Stage 1.2 can introduce new tags
# without code changes.


@dataclass
class StateTransitionOperation:
    operation_id: str
    action: str
    timestamp: float
    duration: Optional[float] = None

    # Explicit state-graph edges.
    input_states: list[str] = field(default_factory=list)
    output_states: list[str] = field(default_factory=list)

    operator_id: str = "Entity_Operator"
    confidence: float = 1.0
    action_category: Optional[ActionCategory] = None

    # Optional descriptive context kept for Stage 4 hypothetical view.
    description: Optional[str] = None

    # Stage 2 operation-level RAG enrichment: typical parameters +
    # failure modes drawn from the wet-lab KB.
    operation_enrichment: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def operation_from_dict(d: dict) -> StateTransitionOperation:
    return StateTransitionOperation(**d)
