"""query_rewriter.py — LLM-based question → protocol-style query.

v4 KB had 88% zero-passage rate on SciVB because the agent fed raw
questions ("What is the role of antibody staining?") to BioProBench.
v5 mandates rewriting into protocol nouns ("antibody staining protocol
technique") before retrieval.

Usage:
    rewriter = QueryRewriter(vlm=VLMClient(...))
    q = rewriter.rewrite("What buffer composition is used to lyse the cells?")
    # → "lysis buffer composition reagent recipe"

The rewriter is also used as a sanity-check predicate
(`is_protocol_style`) for SFT and RL reward shaping.
"""
from __future__ import annotations

import re

_REWRITE_SYSTEM = (
    "You convert a scientific video question into a SHORT protocol-style "
    "search query for retrieving lab procedures from a protocol corpus. "
    "Output the query ONLY, no explanation, no quotes."
)

_REWRITE_PROMPT_TEMPLATE = """Convert each question into a 3-8 word search query.

Rules:
- Noun phrases only — no "what/how/why/which/did"
- Mention techniques, reagents, instruments, conditions
- No question marks
- 3 to 8 words

Examples:
Q: What buffer composition is used to lyse the cells?
Query: lysis buffer composition reagent recipe

Q: Why is phenol red added to DMEM?
Query: phenol red pH indicator cell culture medium function

Q: What is the role of antibody staining in this protocol?
Query: antibody staining protocol procedure technique

Q: What centrifuge model is shown?
Query: centrifuge instrument specifications laboratory

Q: How long is the incubation step?
Query: incubation time protocol step duration

Q: {question}
Query:"""


_QUESTION_WORDS = ("what", "how", "why", "which", "did", "is", "are",
                    "do", "does", "can", "could", "should", "would", "where")


def is_protocol_style(query: str) -> bool:
    """Heuristic: a "good" protocol query is a noun phrase without
    interrogative leading words and without a question mark."""
    if not query: return False
    q = query.lower().strip().rstrip('.')
    if "?" in q: return False
    first = q.split()[0] if q.split() else ""
    if first in _QUESTION_WORDS: return False
    words = q.split()
    if len(words) < 2 or len(words) > 12: return False
    return True


class QueryRewriter:
    """Wraps a frozen VLM for one-shot text-only query rewriting.

    Falls back to a heuristic (drop question word + question mark) when
    no VLM is provided — useful for unit tests and as a safety net.
    """

    def __init__(self, vlm=None, max_new_tokens: int = 30):
        self.vlm = vlm
        self.max_new_tokens = max_new_tokens

    def rewrite(self, question: str) -> str:
        if not question: return ""
        q = question.strip()
        if self.vlm is None:
            return self._heuristic(q)
        # Text-only LLM call. VLMClient.generate expects messages list;
        # the user content is text-only here.
        messages = [
            {"role": "system", "content": _REWRITE_SYSTEM},
            {"role": "user",   "content": [
                {"type": "text",
                 "text": _REWRITE_PROMPT_TEMPLATE.format(question=q)},
            ]},
        ]
        try:
            raw = self.vlm.generate(messages, max_new_tokens=self.max_new_tokens)
        except Exception:
            return self._heuristic(q)
        out = self._post_clean(raw)
        if is_protocol_style(out):
            return out
        return self._heuristic(q)

    @staticmethod
    def _heuristic(q: str) -> str:
        """Drop interrogative word + punctuation; keep nouns."""
        cleaned = re.sub(r"[?,;:!]", " ", q).strip()
        tokens = cleaned.split()
        if tokens and tokens[0].lower() in _QUESTION_WORDS:
            tokens = tokens[1:]
        # Drop common verbs that aren't search-relevant
        skip = {"is", "are", "was", "were", "be", "been", "do", "does",
                  "did", "the", "a", "an", "of", "in", "on", "at"}
        tokens = [t for t in tokens if t.lower() not in skip]
        return " ".join(tokens[:12]).strip()

    @staticmethod
    def _post_clean(raw: str) -> str:
        """Strip quoting and explanation prefix from raw LLM output."""
        s = raw.strip()
        # Take first line / first sentence
        for sep in ("\n", ". "):
            if sep in s:
                s = s.split(sep, 1)[0]
                break
        # Strip wrapping quotes
        s = s.strip().strip('"').strip("'").strip()
        # Drop "Query:" prefix the model sometimes adds
        for prefix in ("query:", "search query:", "rewritten:"):
            if s.lower().startswith(prefix):
                s = s[len(prefix):].strip()
        return s
