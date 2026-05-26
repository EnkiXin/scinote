"""Tests for kg/stoa.py — STOA vocab + prompt builder."""

import pytest

from protonote.v8.kg.stoa import (
    ACTION_VOCAB,
    CONFIDENCE_GUIDE,
    ENTITY_TYPES,
    build_extraction_prompt,
)
from protonote.v8.kg.entity import EntityType


def test_entity_types_match_schema():
    """ENTITY_TYPES must exactly cover the Literal in kg/entity.py."""
    from typing import get_args
    schema_types = set(get_args(EntityType))
    assert set(ENTITY_TYPES) == schema_types


def test_action_vocab_non_empty_and_unique():
    assert len(ACTION_VOCAB) >= 10
    assert len(set(ACTION_VOCAB)) == len(ACTION_VOCAB)


def test_prompt_has_entity_types_listed():
    p = build_extraction_prompt(n_frames=32, duration_sec=600)
    for et in ENTITY_TYPES:
        assert et in p


def test_prompt_has_action_vocab_listed():
    p = build_extraction_prompt(n_frames=32, duration_sec=600)
    for a in ACTION_VOCAB:
        assert a in p


def test_prompt_includes_confidence_guide():
    p = build_extraction_prompt(n_frames=32, duration_sec=600)
    assert "0.90-1.00" in p   # boundary line from CONFIDENCE_GUIDE


def test_prompt_sec_per_frame_computed():
    p = build_extraction_prompt(n_frames=32, duration_sec=320.0)
    assert "10.0s" in p       # 320/32


def test_prompt_with_question():
    p = build_extraction_prompt(
        n_frames=32, duration_sec=600,
        question="What happens at 4:09?",
    )
    assert "What happens at 4:09?" in p
    assert "downstream question" in p


def test_prompt_without_question():
    p = build_extraction_prompt(n_frames=32, duration_sec=600)
    assert "downstream question" not in p


def test_json_envelope_documented():
    p = build_extraction_prompt(n_frames=32, duration_sec=600)
    assert '"entities"' in p
    assert '"operations"' in p
    assert "Entity_Operator" in p
