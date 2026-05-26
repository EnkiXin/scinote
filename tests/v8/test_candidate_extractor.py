"""Tests for candidate_extractor."""

from unittest.mock import MagicMock

import pytest

from protonote.v8.grounding.candidate_extractor import (
    extract_candidates_from_passages,
)
from protonote.v8.kg.entity import Entity


def _ent() -> Entity:
    return Entity(
        id="Entity1", type="Material",
        features="white powder", identity_guess="solid",
        initial_confidence=0.3,
    )


def _llm(candidates):
    c = MagicMock()
    c.generate_json = MagicMock(return_value={"candidates": candidates})
    return c


def test_basic_extraction():
    out = extract_candidates_from_passages(
        _ent(),
        passages=[{"text": "MOF synthesis with crystalline powder"}],
        llm_client=_llm(["MOF", "salt"]),
    )
    assert out == ["MOF", "salt"]


def test_no_passages():
    out = extract_candidates_from_passages(_ent(), [], _llm(["x"]))
    assert out == []


def test_empty_passage_text():
    out = extract_candidates_from_passages(
        _ent(),
        passages=[{"text": ""}, {"content": ""}],
        llm_client=_llm(["x"]),
    )
    assert out == []


def test_llm_call_raises():
    client = MagicMock()
    client.generate_json = MagicMock(side_effect=RuntimeError("boom"))
    out = extract_candidates_from_passages(
        _ent(),
        passages=[{"text": "some text"}],
        llm_client=client,
    )
    assert out == []


def test_caps_at_max_candidates():
    out = extract_candidates_from_passages(
        _ent(),
        passages=[{"text": "x"}],
        llm_client=_llm(["a", "b", "c", "d", "e", "f", "g"]),
        max_candidates=3,
    )
    assert out == ["a", "b", "c"]


def test_dedups_case_insensitive():
    out = extract_candidates_from_passages(
        _ent(),
        passages=[{"text": "x"}],
        llm_client=_llm(["MOF", "mof", "MOF "]),
    )
    assert out == ["MOF"]


def test_non_string_filtered():
    out = extract_candidates_from_passages(
        _ent(),
        passages=[{"text": "x"}],
        llm_client=_llm([{"x": 1}, None, "good", 42]),
    )
    assert out == ["good"]


def test_uses_content_key_if_text_missing():
    """Passages from some KB engines use 'content' instead of 'text'."""
    out = extract_candidates_from_passages(
        _ent(),
        passages=[{"content": "MOF synthesis details"}],
        llm_client=_llm(["MOF"]),
    )
    assert out == ["MOF"]
