"""Extract candidate entity identities from retrieved KB passages.

When KB returns passages, we ask an LLM to pull out specific entity
names that could be the identity of an `entity` (e.g. "lysis buffer",
"MOF crystal", "ethanol"). These candidates are then:

  - in RETRIEVE_PLUS_IMAGE: looked up in the image library to verify
    visually
  - in RETRIEVE_ONLY (Material): stored on the entity as hypotheses
    for Stage 4 reasoning ("if Entity3 is MOF, then …")

We cap at 5 candidates per entity to keep downstream lookups cheap.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


CANDIDATE_EXTRACT_PROMPT = """Extract specific entity names from these protocol passages that could identify the following entity:

Entity:
- Type: {entity_type}
- Features: {features}
- Identity guess (uncertain): {identity_guess}

Protocol passages:
{passages_text}

Output 1-5 specific entity names that this entity might be. Names should be
specific (e.g. "lysis buffer" not just "buffer"; "centrifuge tube" not just
"tube").

Output JSON ONLY:
{{"candidates": ["candidate1", "candidate2", ...]}}

If passages don't mention plausible candidates, output: {{"candidates": []}}
"""


def extract_candidates_from_passages(
    entity,
    passages: list[dict],
    llm_client,
    max_passages: int = 3,
    max_candidates: int = 5,
) -> list[str]:
    """LLM-extract candidate identities from KB passages."""
    if not passages:
        return []

    snippets: list[str] = []
    for p in passages[:max_passages]:
        text = (p.get("text") or p.get("content") or "").strip()
        if text:
            snippets.append(text[:300])
    if not snippets:
        return []

    prompt = CANDIDATE_EXTRACT_PROMPT.format(
        entity_type=entity.type,
        features=entity.features,
        identity_guess=entity.identity_guess,
        passages_text="\n\n---\n\n".join(snippets),
    )

    raw = None
    try:
        if hasattr(llm_client, "generate_json"):
            raw = llm_client.generate_json(prompt=prompt, max_tokens=200)
        elif hasattr(llm_client, "generate_text"):
            raw = llm_client.generate_text(
                prompt, max_tokens=200, temperature=0.0,
            )
        else:
            raise AttributeError("no generate_json/generate_text on LLM")
    except Exception as e:
        logger.debug("candidate extractor LLM call raised: %s", e)

    # Coerce raw string into dict
    data = raw if isinstance(raw, dict) else None
    if data is None and isinstance(raw, str):
        import json as _json
        import re
        m = re.search(r"\{[\s\S]*\}", raw)
        if m:
            try:
                data = _json.loads(m.group(0))
            except Exception:
                data = None

    if not isinstance(data, dict):
        return []

    raw = data.get("candidates", [])
    if not isinstance(raw, list):
        return []

    out: list[str] = []
    for c in raw:
        if not isinstance(c, str):
            continue
        c = c.strip(' "\',.;')
        if c and c.lower() not in (s.lower() for s in out):
            out.append(c)
        if len(out) >= max_candidates:
            break
    return out
