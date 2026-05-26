"""Tests for RetrieveToolV8 (query rewriter + KB)."""

from unittest.mock import MagicMock

import pytest

from protonote.v8.kg.entity import Entity
from protonote.v8.tools.retrieve_tool import RetrieveToolV8


def _ent(t="Container",
            f="round transparent vessel",
            g="flask") -> Entity:
    return Entity(
        id="Entity1", type=t, features=f, identity_guess=g,
        initial_confidence=0.5,
    )


def _mock_llm(rewritten: str):
    c = MagicMock()
    c.generate = MagicMock(return_value=rewritten)
    return c


def _mock_kb(passages):
    k = MagicMock()
    k.search = MagicMock(return_value=passages)
    return k


def test_rewriter_passes_query_to_kb():
    kb = _mock_kb([{"text": "centrifuge tube ..."}])
    llm = _mock_llm("round-bottom flask reaction vessel")

    tool = RetrieveToolV8(kb_tool=kb, llm_client=llm)
    out = tool.retrieve_for_entity(_ent())
    assert len(out) == 1
    args = kb.search.call_args.args
    assert "flask" in args[0].lower() or "vessel" in args[0].lower()


def test_rewriter_skip_returns_empty():
    kb = _mock_kb([{"text": "anything"}])
    llm = _mock_llm("SKIP")
    tool = RetrieveToolV8(kb_tool=kb, llm_client=llm)
    out = tool.retrieve_for_entity(_ent())
    assert out == []
    kb.search.assert_not_called()


def test_kb_failure_returns_empty():
    kb = MagicMock()
    kb.search = MagicMock(side_effect=RuntimeError("boom"))
    llm = _mock_llm("some query")
    tool = RetrieveToolV8(kb_tool=kb, llm_client=llm)
    out = tool.retrieve_for_entity(_ent())
    assert out == []


def test_rewriter_error_falls_back_to_features():
    kb = _mock_kb([{"text": "ok"}])
    llm = MagicMock()
    llm.generate = MagicMock(side_effect=RuntimeError("model down"))
    tool = RetrieveToolV8(kb_tool=kb, llm_client=llm)
    out = tool.retrieve_for_entity(_ent(f="my features"))
    assert out == [{"text": "ok"}]
    args = kb.search.call_args.args
    assert "my features" in args[0]


def test_strips_quotes_from_rewritten():
    kb = _mock_kb([{"text": "ok"}])
    llm = _mock_llm('"flask reaction vessel".')
    tool = RetrieveToolV8(kb_tool=kb, llm_client=llm)
    tool.retrieve_for_entity(_ent())
    args = kb.search.call_args.args
    assert args[0].startswith("flask")
    assert not args[0].endswith(".")
