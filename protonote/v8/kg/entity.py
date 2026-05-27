"""Entity schema for V8 KnowledgeGraph.

Represents an entity (Operator, Instrument, Container, Material, Display, or
Measurement) extracted from a scientific video, along with grounding
information.
"""

from dataclasses import dataclass, field
from typing import Literal, Optional

EntityType = Literal[
    "Operator",      # Person performing experiment
    "Instrument",    # Equipment (centrifuge, microscope, …)
    "Container",     # Vessel (tube, flask, beaker, …)
    "Material",      # Substance (buffer, sample, reagent, …)
    "Display",       # Visual display (screen, label with numbers, …)
    "Measurement",   # Numeric measurement entity
]

GroundingMethod = Literal[
    "image_match",         # SigLIP2 + VLM verify (MED conf)
    "retrieve_plus_image", # KB retrieve + image library (LOW conf)
    "ocr",                 # OCR for Display/Measurement
    "ungrounded",          # 仍然 unknown
]
# NOTE: "vlm_direct" was removed (2026-05-27). USE_AS_IS routing is a
# compute-saving skip, NOT a grounding mechanism — Stage 1's guess is
# the INPUT to grounding, not a verified output.


@dataclass
class GroundingInfo:
    """Result of selective grounding for an entity."""

    identity: Optional[str]   # e.g., "centrifuge"; None if ungrounded
    confidence: float         # 0.0-1.0
    method: GroundingMethod

    # Optional method-specific fields
    source_dataset: Optional[str] = None             # for image_match
    candidates: list[str] = field(default_factory=list)  # for retrieve_plus_image
    ocr_text: Optional[str] = None                   # for ocr
    evidence: Optional[str] = None                   # VLM's visual reasoning

    def __post_init__(self):
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"Confidence must be in [0.0, 1.0], got {self.confidence}"
            )


@dataclass
class Entity:
    """Entity in a video KG.

    Initial extraction (Stage 1) populates id, type, features,
    identity_guess, initial_confidence. Stage 3 populates grounded.
    """

    # Required fields (Stage 1 extraction)
    id: str                       # "Entity1", "Entity2", …
    type: EntityType
    features: str                 # 可见 features (what we SEE)
    identity_guess: str           # VLM's best guess
    initial_confidence: float     # 0.0-1.0

    # Temporal (Approach 1: appearance intervals)
    appearance_intervals: list[tuple[int, int]] = field(default_factory=list)

    # Spatial (VL-KnG style)
    bbox: Optional[tuple[int, int, int, int]] = None  # x1, y1, x2, y2
    color: Optional[str] = None
    material: Optional[str] = None
    size: Optional[str] = None
    spatial_relationships: dict = field(default_factory=dict)

    # Domain-specific (scientific)
    state: Optional[str] = None              # "liquid"/"solid"/"gas"
    quantity: Optional[dict] = None          # {"value": 50, "unit": "mL"}
    role_in_procedure: Optional[str] = None  # "source"/"destination"/"tool"

    # Routing flag (Stage 2)
    ocr_candidate: bool = False

    # Grounding result (Stage 3)
    grounded: Optional[GroundingInfo] = None

    def __post_init__(self):
        if not 0.0 <= self.initial_confidence <= 1.0:
            raise ValueError(
                f"Confidence must be in [0.0, 1.0], got {self.initial_confidence}"
            )
        if not self.id.startswith("Entity"):
            raise ValueError(f"Entity id must start with 'Entity', got {self.id}")

    @property
    def is_grounded(self) -> bool:
        """Whether entity has a specific identity."""
        return (
            self.grounded is not None
            and self.grounded.identity is not None
        )

    @property
    def final_confidence(self) -> float:
        """Final confidence after grounding (initial if not grounded)."""
        if self.grounded is None:
            return self.initial_confidence
        return self.grounded.confidence
