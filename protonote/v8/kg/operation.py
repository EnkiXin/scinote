"""Operation and Stage schemas for V8 KnowledgeGraph.

Operations represent atomic actions in the experiment.
Stages group operations into higher-level procedural phases (e.g.,
"Sample Preparation", "Measurement").
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Operation:
    """Atomic operation in experiment temporal chain.

    Approach 2 temporal representation: operations have timestamps and
    can be linked in temporal order.
    """

    id: str                # "Op1", "Op2", …
    action: str            # Action verb: "add", "mix", "heat", "transfer", …
    subject: str           # Entity ID (who performs action; usually Operator)
    object: str            # Entity ID (what is acted upon)

    # Approach 2 temporal
    timestamp: int                  # Approximate timestamp in seconds
    duration: Optional[int] = None  # Optional: how long action took (sec)

    # Confidence in extraction
    confidence: float = 1.0

    # Stage grouping (Approach 2 expansion)
    stage_id: Optional[str] = None

    # Temporal links (optional sequencing)
    follows_op: Optional[str] = None     # Op ID this follows
    precedes_op: Optional[str] = None    # Op ID this precedes

    # Additional context
    description: Optional[str] = None    # Optional natural language

    def __post_init__(self):
        if not self.id.startswith("Op"):
            raise ValueError(f"Operation id must start with 'Op', got {self.id}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"Confidence must be in [0.0, 1.0], got {self.confidence}"
            )
        if self.timestamp < 0:
            raise ValueError(f"Timestamp must be >= 0, got {self.timestamp}")


@dataclass
class Stage:
    """High-level procedural stage grouping operations.

    e.g., "Sample Preparation" containing Op1, Op2, Op3.
    Approach 2 expansion: supports ExpVid L3 reasoning over extended workflows.
    """

    id: str                        # "Stage1", "Stage2", …
    name: str                      # "Sample Preparation", "Measurement", …
    interval: tuple[int, int]      # (start_sec, end_sec)
    operations: list[str] = field(default_factory=list)   # Operation IDs

    # Optional description
    description: Optional[str] = None

    def __post_init__(self):
        if not self.id.startswith("Stage"):
            raise ValueError(f"Stage id must start with 'Stage', got {self.id}")
        if self.interval[0] > self.interval[1]:
            raise ValueError(
                f"Stage interval must have start <= end, got {self.interval}"
            )

    @property
    def duration(self) -> int:
        """Stage duration in seconds."""
        return self.interval[1] - self.interval[0]
