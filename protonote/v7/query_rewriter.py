"""query_rewriter.py — v7 P2.1 fix.

V6 case studies showed planners feeding raw question text or first-
person summary text directly into BM25/BGE — BioProBench is a corpus
of protocol-style snippets, so paraphrased question text rarely
matches. The rewriter converts the question + current notes into a
protocol-style noun-phrase query.

A NOT_APPLICABLE flag tells the caller to skip retrieve entirely —
used when the question is purely visual / temporal / counting.
"""
from __future__ import annotations

_REWRITE_SYSTEM = (
    "You convert scientific video questions into protocol-style search "
    "queries for a biology/biochemistry protocol corpus (BioProBench, "
    "PubMed protocols)."
)

_REWRITE_PROMPT = """Question: {question}

Task: produce a single short protocol-style query (≤ 12 words) that
would match passages describing the technique, reagent, or step
mentioned in the question. Use noun phrases, not full sentences.

Examples:
  Q: "What is the purpose of adding EDTA to the buffer?"
  → "EDTA buffer chelator function in DNA extraction"

  Q: "Which option correctly describes the centrifugation speed?"
  → "centrifugation speed RPM cell pellet protocol"

  Q: "How many beakers are visible on the bench?"
  → NOT_APPLICABLE

  Q: "What happens to the color of the solution after the reagent is added?"
  → "colorimetric assay reagent color change indicator"

Rules:
- If the question is purely visual / counting / temporal (no scientific
  technique to look up), output exactly NOT_APPLICABLE.
- Otherwise output the query string only, no quotes, no prefix.

Q: {question}
→"""


def rewrite_query_for_kb(question: str, vlm,
                                max_tokens: int = 40) -> str | None:
    """Return rewritten query string or None if NOT_APPLICABLE.

    Falls back to the original question on any error (so we never lose
    a retrieve call due to rewriter failure).
    """
    if not question: return None
    prompt = _REWRITE_PROMPT.format(question=question[:300])
    try:
        raw = vlm.generate_text(prompt, system=_REWRITE_SYSTEM,
                                       max_tokens=max_tokens,
                                       temperature=0.0)
    except Exception:
        return question  # fail-safe: use original

    out = (raw or "").strip().splitlines()[0].strip(" '\"`")
    if not out: return question
    if "NOT_APPLICABLE" in out.upper(): return None
    # Strip any leading "→" or label
    for prefix in ("→", "->", "Query:", "query:"):
        if out.startswith(prefix):
            out = out[len(prefix):].strip()
    # Cap length
    return out[:200] or question
