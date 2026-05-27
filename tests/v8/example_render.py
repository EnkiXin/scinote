"""Generate example KG markdown for visual review.

Builds the HURT 6 case (MOF NMR sample) to show what a typical V8 KG
will look like when fed into the Stage-4 answer prompt.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from protonote.v8.kg.entity import Entity, GroundingInfo
from protonote.v8.kg.knowledge_graph import KnowledgeGraph
from protonote.v8.kg.operation import Operation, Stage


def make_example_kg() -> KnowledgeGraph:
    """Build example KG mimicking HURT 6 case (MOF NMR sample)."""
    kg = KnowledgeGraph()

    kg.add_entity(
        Entity(
            id="Entity1",
            type="Container",
            features="long thin glass tube, ~20cm, narrow diameter",
            identity_guess="NMR tube",
            initial_confidence=0.85,
            appearance_intervals=[(340, 400)],
            grounded=GroundingInfo(
                identity="NMR tube", confidence=0.85, method="image_match"
            ),
        )
    )
    kg.add_entity(
        Entity(
            id="Entity2",
            type="Material",
            features="white crystalline powder",
            identity_guess="unknown crystalline solid",
            initial_confidence=0.25,
            appearance_intervals=[(340, 365)],
            grounded=GroundingInfo(
                identity=None,
                confidence=0,
                method="ungrounded",
                candidates=["MOF", "salt", "polymer"],
            ),
        )
    )
    kg.add_entity(
        Entity(
            id="Entity3",
            type="Display",
            features="label on tube showing text",
            identity_guess="concentration label",
            initial_confidence=0.4,
            ocr_candidate=True,
            appearance_intervals=[(380, 390)],
            grounded=GroundingInfo(
                identity=None, confidence=0.9, method="ocr",
                ocr_text="50 mg/mL",
            ),
        )
    )
    kg.add_entity(
        Entity(
            id="Entity4",
            type="Instrument",
            features="benchtop NMR spectrometer",
            identity_guess="NMR spectrometer",
            initial_confidence=0.9,
            appearance_intervals=[(400, 500)],
            grounded=GroundingInfo(
                identity="NMR spectrometer", confidence=0.9,
                method="image_match", source_dataset="Chemistry-25",
            ),
        )
    )

    kg.add_operation(
        Operation(
            id="Op1", action="transfer", subject="Entity_Operator",
            object="Entity2", timestamp=345, confidence=0.8,
        )
    )
    kg.add_operation(
        Operation(
            id="Op2", action="add solvent", subject="Entity_Operator",
            object="Entity1", timestamp=370, confidence=0.7,
        )
    )
    kg.add_operation(
        Operation(
            id="Op3", action="insert into NMR", subject="Entity_Operator",
            object="Entity1", timestamp=410, confidence=0.95,
        )
    )

    kg.add_stage(
        Stage(
            id="Stage1", name="Sample preparation",
            interval=(340, 380), operations=["Op1", "Op2"],
        )
    )
    kg.add_stage(
        Stage(
            id="Stage2", name="NMR measurement",
            interval=(400, 500), operations=["Op3"],
        )
    )

    return kg


if __name__ == "__main__":
    kg = make_example_kg()
    print(kg.render())
