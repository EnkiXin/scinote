"""V8 retrieve tool — V6 KB + query rewriter.

Wraps the V6 BioProBench BM25+BGE+rerank pipeline (`kb_tool`) and
adds an LLM-driven query rewriter. The rewriter:

  - converts an entity description into a 5-10-word protocol-style
    query (the corpus is PubMed protocols)
  - returns ``None`` if the entity is clearly out-of-domain (e.g.
    physics equipment) so callers skip retrieve entirely.

This addresses the V6 finding (per KB_COVERAGE.md) that ~38 % of
retrieve calls returned 0 passages because the raw question/feature
text didn't match protocol vocabulary.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


REWRITE_PROMPT = """Rewrite this entity description into a 5-10 word search query matching biology/chemistry protocol vocabulary.

Entity description:
- Type: {entity_type}
- Visible features: {features}
- Identity guess (uncertain): {identity_guess}

Goal: find protocol passages mentioning what this entity might be.

If the entity is NOT likely covered by biology/chemistry protocols
(e.g. physics equipment, computer hardware, generic furniture), output "SKIP".

Examples:

Entity: Material, "white crystalline powder", "unknown solid"
Output: "crystalline powder solid reagent precipitate"

Entity: Container, "round transparent vessel with liquid", "flask"
Output: "round-bottom flask liquid reaction vessel"

Entity: Instrument, "oscilloscope screen", "oscilloscope"
Output: "SKIP"

Entity: Material, "red liquid in tube", "biological sample"
Output: "blood sample tube biological"

Now rewrite:
"""


class RetrieveToolV8:
    """V8 retrieve tool: KB search + query rewriter."""

    def __init__(self, kb_tool, llm_client):
        """
        Args:
            kb_tool: V6 KB instance with a ``.search(query, top_k=…)`` method.
            llm_client: text LLM with ``.generate(prompt, max_tokens, temperature)``.
        """
        self.kb = kb_tool
        self.llm = llm_client

    # ---- public ----

    def retrieve_for_entity(self, entity, top_k: int = 5) -> list[dict]:
        """Rewrite the query → KB search → return passages.

        Returns: list of passage dicts (each with ``text`` + ``score``).
        Empty list on rewriter-SKIP or KB error.

        We accept V5/V6 KB tools that return either:
          - a list of passages directly, or
          - a dict with ``passages``/``scores``/... (V5 schema)
        """
        rewritten = self._rewrite_query(entity)
        if rewritten is None:
            logger.debug("%s: rewriter returned SKIP", entity.id)
            return []
        try:
            result = self.kb.search(rewritten)
        except Exception as e:
            logger.warning("KB search failed: %s", e)
            return []

        if result is None:
            return []
        if isinstance(result, list):
            return result[:top_k]
        # V5 dict shape: {passages: [...], scores: [...], sources: [...], ...}
        passages = result.get("passages") or []
        scores = result.get("scores") or [0.0] * len(passages)
        sources = result.get("sources") or [None] * len(passages)
        out = []
        for i, p in enumerate(passages[:top_k]):
            out.append({
                "text":   p,
                "score":  scores[i] if i < len(scores) else 0.0,
                "source": sources[i] if i < len(sources) else None,
            })
        return out

    # ---- internal ----

    def _rewrite_query(self, entity) -> Optional[str]:
        prompt = REWRITE_PROMPT.format(
            entity_type=entity.type,
            features=entity.features,
            identity_guess=entity.identity_guess,
        )
        # The V6 client exposes `.generate_text` (text-only). Plain `.generate`
        # is the test-only convention. Try text first, fall back to generate.
        try:
            if hasattr(self.llm, "generate_text"):
                raw = self.llm.generate_text(
                    prompt, max_tokens=50, temperature=0.0,
                )
            elif hasattr(self.llm, "generate"):
                raw = self.llm.generate(
                    prompt=prompt, max_tokens=50, temperature=0.0,
                )
            else:
                raise AttributeError("no generate / generate_text on LLM")
        except Exception as e:
            logger.warning("rewrite call failed: %s; using raw features", e)
            return entity.features or None

        if not raw:
            return entity.features or None
        text = str(raw).strip()
        if "SKIP" in text.upper():
            return None
        # strip surrounding quotes / punctuation
        return text.strip('"\'.,;') or None
