"""Tests for V8 timestamp_parser (sub-second fix)."""

import pytest

from protonote.v8.tools.timestamp_parser import parse_timestamp


@pytest.mark.parametrize("inp, expected", [
    (42, 42.0),
    (42.5, 42.5),
    ("42", 42.0),
    ("4:09", 249.0),
    ("4:09.5", 249.5),
    ("0:02.31", 2.31),     # HURT-cases sub-second fraction
    ("1:30:45", 5445.0),
    ("1:30:45.250", 5445.250),
    ("0:00", 0.0),
    (" 4:09 ", 249.0),
])
def test_parses(inp, expected):
    assert abs(parse_timestamp(inp) - expected) < 1e-6


def test_empty_raises():
    with pytest.raises(ValueError):
        parse_timestamp("")


def test_non_numeric_input_raises():
    with pytest.raises(ValueError):
        parse_timestamp([1, 2])  # type: ignore[arg-type]


def test_too_many_colons_raises():
    with pytest.raises(ValueError):
        parse_timestamp("1:2:3:4")
