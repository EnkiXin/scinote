"""Stage 2: Route entities to grounding paths.

Per-type, per-confidence policy from V8_RESEARCH_PLAN_V3.md.
Motivated by V8_LIBRARY_COVERAGE 实测 (Material 0% hit rate, Container
92% hit rate) — different entity types route differently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from protonote.v8.kg.entity import Entity, GroundingInfo
from protonote.v8.kg.knowledge_graph import KnowledgeGraph


class RoutingAction(str, Enum):
    """Possible routing decisions for an entity."""

    USE_AS_IS = "use_as_is"
    IMAGE_MATCH = "image_match"
    RETRIEVE_PLUS_IMAGE = "retrieve_plus_image"
    RETRIEVE_ONLY = "retrieve_only"   # Material → 0% image-library hit
    OCR = "ocr"


@dataclass
class TypePolicy:
    """Routing policy for a single entity type."""

    high_conf_threshold: float
    med_conf_threshold: float
    high_route: RoutingAction
    med_route: RoutingAction
    low_route: RoutingAction
    always_route: Optional[RoutingAction] = None  # override


# Per-type policies — locked from V8_RESEARCH_PLAN_V3.md.
ROUTING_POLICIES: dict[str, TypePolicy] = {
    "Container": TypePolicy(
        high_conf_threshold=0.80,
        med_conf_threshold=0.50,
        high_route=RoutingAction.USE_AS_IS,
        med_route=RoutingAction.IMAGE_MATCH,        # 92 % library hit
        low_route=RoutingAction.RETRIEVE_PLUS_IMAGE,
    ),
    "Instrument": TypePolicy(
        high_conf_threshold=0.75,
        med_conf_threshold=0.40,
        high_route=RoutingAction.USE_AS_IS,
        med_route=RoutingAction.IMAGE_MATCH,        # 8 % hit, still worth it
        low_route=RoutingAction.RETRIEVE_PLUS_IMAGE,
    ),
    "Material": TypePolicy(
        high_conf_threshold=0.90,  # strict (chemicals visually ambiguous)
        med_conf_threshold=0.60,
        high_route=RoutingAction.USE_AS_IS,
        med_route=RoutingAction.RETRIEVE_ONLY,      # 实测 0% library hit
        low_route=RoutingAction.RETRIEVE_ONLY,
    ),
    "Operator": TypePolicy(
        high_conf_threshold=0.60,
        med_conf_threshold=0.30,
        high_route=RoutingAction.USE_AS_IS,
        med_route=RoutingAction.USE_AS_IS,          # no library coverage
        low_route=RoutingAction.USE_AS_IS,
    ),
    "Display": TypePolicy(
        high_conf_threshold=0.0, med_conf_threshold=0.0,
        high_route=RoutingAction.OCR, med_route=RoutingAction.OCR,
        low_route=RoutingAction.OCR,
        always_route=RoutingAction.OCR,             # numeric content
    ),
    "Measurement": TypePolicy(
        high_conf_threshold=0.0, med_conf_threshold=0.0,
        high_route=RoutingAction.OCR, med_route=RoutingAction.OCR,
        low_route=RoutingAction.OCR,
        always_route=RoutingAction.OCR,
    ),
}


@dataclass
class RoutingResult:
    """Result of routing: which entities go to which path."""

    use_as_is: list[Entity] = field(default_factory=list)
    image_match: list[Entity] = field(default_factory=list)
    retrieve_plus_image: list[Entity] = field(default_factory=list)
    retrieve_only: list[Entity] = field(default_factory=list)
    ocr: list[Entity] = field(default_factory=list)

    def by_action(self, action: RoutingAction) -> list[Entity]:
        return {
            RoutingAction.USE_AS_IS:           self.use_as_is,
            RoutingAction.IMAGE_MATCH:         self.image_match,
            RoutingAction.RETRIEVE_PLUS_IMAGE: self.retrieve_plus_image,
            RoutingAction.RETRIEVE_ONLY:       self.retrieve_only,
            RoutingAction.OCR:                 self.ocr,
        }[action]

    def total(self) -> int:
        return (
            len(self.use_as_is)
            + len(self.image_match)
            + len(self.retrieve_plus_image)
            + len(self.retrieve_only)
            + len(self.ocr)
        )

    def counts(self) -> dict[str, int]:
        """Per-action count summary."""
        return {
            "use_as_is":            len(self.use_as_is),
            "image_match":          len(self.image_match),
            "retrieve_plus_image":  len(self.retrieve_plus_image),
            "retrieve_only":        len(self.retrieve_only),
            "ocr":                  len(self.ocr),
        }


def route_entity(entity: Entity) -> RoutingAction:
    """Decide routing for a single entity."""
    policy = ROUTING_POLICIES.get(entity.type)
    if policy is None:
        # Unknown type → safest default: keep VLM's guess
        return RoutingAction.USE_AS_IS

    # Always-route override (Display / Measurement)
    if policy.always_route is not None:
        return policy.always_route

    conf = entity.initial_confidence
    if conf >= policy.high_conf_threshold:
        return policy.high_route
    elif conf >= policy.med_conf_threshold:
        return policy.med_route
    else:
        return policy.low_route


def route_kg(kg: KnowledgeGraph) -> RoutingResult:
    """Route all entities in a KG.

    For USE_AS_IS entities, populates `entity.grounded` immediately
    (no Stage 3 work needed). For other paths, leaves `.grounded` as
    None — Stage 3 will fill it in.
    """
    result = RoutingResult()

    for entity in kg.entities.values():
        action = route_entity(entity)

        if action == RoutingAction.USE_AS_IS:
            entity.grounded = GroundingInfo(
                identity=entity.identity_guess,
                confidence=entity.initial_confidence,
                method="vlm_direct",
                evidence=(
                    f"VLM direct (HIGH conf {entity.initial_confidence:.2f})"
                ),
            )
            result.use_as_is.append(entity)
        elif action == RoutingAction.IMAGE_MATCH:
            result.image_match.append(entity)
        elif action == RoutingAction.RETRIEVE_PLUS_IMAGE:
            result.retrieve_plus_image.append(entity)
        elif action == RoutingAction.RETRIEVE_ONLY:
            result.retrieve_only.append(entity)
        elif action == RoutingAction.OCR:
            result.ocr.append(entity)

    return result
