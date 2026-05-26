"""Tests for Operation and Stage schemas."""

import pytest

from protonote.v8.kg.operation import Operation, Stage


class TestOperation:
    def test_basic_creation(self):
        op = Operation(
            id="Op1",
            action="add",
            subject="Entity1",
            object="Entity2",
            timestamp=45,
        )
        assert op.id == "Op1"
        assert op.action == "add"
        assert op.confidence == 1.0  # default

    def test_id_validation(self):
        with pytest.raises(ValueError):
            Operation(
                id="X1", action="add", subject="E1", object="E2",
                timestamp=0,
            )

    def test_timestamp_validation(self):
        with pytest.raises(ValueError):
            Operation(
                id="Op1", action="add", subject="E1", object="E2",
                timestamp=-5,
            )

    def test_confidence_validation(self):
        with pytest.raises(ValueError):
            Operation(
                id="Op1", action="add", subject="E1", object="E2",
                timestamp=0, confidence=1.5,
            )

    def test_full_operation(self):
        op = Operation(
            id="Op1",
            action="centrifuge",
            subject="Entity_Operator",
            object="Entity1",
            timestamp=120,
            duration=300,
            confidence=0.95,
            stage_id="Stage2",
            follows_op="Op0",
            description="Centrifuging sample at 4000g for 5 min",
        )
        assert op.duration == 300
        assert op.stage_id == "Stage2"


class TestStage:
    def test_basic_creation(self):
        s = Stage(
            id="Stage1",
            name="Sample Preparation",
            interval=(40, 80),
            operations=["Op1", "Op2"],
        )
        assert s.id == "Stage1"
        assert s.duration == 40
        assert len(s.operations) == 2

    def test_id_validation(self):
        with pytest.raises(ValueError):
            Stage(id="S1", name="Foo", interval=(0, 10), operations=[])

    def test_interval_validation(self):
        with pytest.raises(ValueError):
            Stage(id="Stage1", name="Foo", interval=(50, 40), operations=[])

    def test_empty_operations(self):
        s = Stage(id="Stage1", name="Foo", interval=(0, 10), operations=[])
        assert s.operations == []

    def test_duration_property(self):
        s = Stage(id="Stage1", name="Foo", interval=(100, 250), operations=[])
        assert s.duration == 150
