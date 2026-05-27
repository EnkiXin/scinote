"""V9 Stage 3 — Multi-label question router.

Replaces the implicit "every question gets the same KG render" policy
from V8 with explicit per-question selection of one or more
reasoning views. Wet-lab QA mixes types ("if X failed, what would the
yield be?" = hypothetical + quantitative); a single-label router would
drop one half of the question and crash the answer.

Output schema (strict JSON from the LLM):
  {
    "quantitative":   bool,
    "hypothetical":   bool,
    "conceptual":     bool,
    "procedural":     bool,
    "confidence":     float (0–1)
  }

The returned `active_views` list ALWAYS contains ≥ 1 view via two
safety nets:
  (1) If the LLM returns zero true flags → activate a safe default set
      {quantitative, hypothetical, procedural}.
  (2) If `confidence < 0.6` and `conceptual` is missing → append
      conceptual as a low-cost fallback.

See V9_RESEARCH_PLAN.md §6.1.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Optional

from protonote.v8.stages.stage1_extract import (
    _extract_json_envelope,
    _try_parse_json,
)

logger = logging.getLogger(__name__)


VIEW_NAMES = ("quantitative", "hypothetical", "conceptual", "procedural")

DEFAULT_FALLBACK = ["quantitative", "hypothetical", "procedural"]


_ROUTER_PROMPT = """You decide which reasoning view(s) a question needs.
A question can need MORE THAN ONE view — wet-lab questions often mix
quantitative + hypothetical, or procedural + quantitative.

Question:
{question}

Options:
{options}

Decide INDEPENDENTLY for each of the four views — emit `true` whenever
the view is at all relevant, `false` only when clearly irrelevant.

1) quantitative — needs a numeric value, calculation, comparison, rate,
   total, max/min, percentage, concentration, time interval, …
   keywords: "how much", "total", "rate", "calculate", "what value",
   "max", "min", "concentration", "percentage", "time interval"

2) hypothetical — counterfactual / cause-effect / failure-mode / "what
   if" / "why does this happen".
   keywords: "what if", "what would result", "if X fails", "why does",
   "consequence of", "expected outcome if"

3) conceptual — identification of a principle / technique / category.
   keywords: "what is the principle", "which type of", "what method",
   "what technique", "what's the name of"

4) procedural — order / sequence / step verification / timing of steps.
   keywords: "next step", "in what order", "which steps are shown",
   "verify if procedure", "is the procedure correct"

Examples that activate multiple views:
  "If X fails, what is the maximum yield possible?"        → quantitative + hypothetical
  "In what order are these 3 steps done, and how long total?"
                                                            → procedural + quantitative
  "Which technique is shown, and why does adding heat help?"
                                                            → conceptual + hypothetical

OUTPUT FORMAT — strict JSON, no markdown fences, no commentary:

{{
  "quantitative": true/false,
  "hypothetical": true/false,
  "conceptual":   true/false,
  "procedural":   true/false,
  "confidence":   0.0_to_1.0
}}
"""


def _format_options(options) -> str:
    """Render options as A. ... / B. ... lines (matches SciVB MC schema)."""
    if isinstance(options, dict):
        return "\n".join(f"{k}. {v}" for k, v in options.items())
    if isinstance(options, list):
        labels = "ABCDEFGHIJ"
        return "\n".join(
            f"{labels[i]}. {opt}" for i, opt in enumerate(options[:10])
        )
    return str(options)


def determine_active_views(
    question: str,
    options,
    llm_client,
    *,
    max_tokens: int = 80,
    temperature: float = 0.0,
) -> tuple[list[str], dict]:
    """Return (active_views, raw_routing_dict).

    `raw_routing_dict` is the parsed router output (or a default when
    parsing failed). It's returned so callers can log / debug.
    """
    prompt = _ROUTER_PROMPT.format(
        question=question.strip(),
        options=_format_options(options),
    )

    raw: Optional[str] = None
    # The router is a pure-text call. Prefer ``generate_text`` (V8 client
    # API); fall back to a generic ``generate`` for mock clients in tests.
    for method, kw_name in (
        ("generate_text", "max_tokens"),
        ("generate",      "max_new_tokens"),
        ("generate",      "max_tokens"),
    ):
        fn = getattr(llm_client, method, None)
        if not callable(fn):
            continue
        try:
            raw = fn(prompt, **{kw_name: max_tokens,
                                  "temperature": temperature})
            break
        except TypeError:
            continue
        except Exception as e:
            logger.warning("Router %s call failed: %s", method, e)
            break

    routing = _parse_router_response(raw or "")
    active = _resolve_active_views(routing)
    return active, routing


# ── internals ──────────────────────────────────────────────────────

def _parse_router_response(raw: str) -> dict:
    """Parse the JSON router output into a dict with the 4 bools + conf."""
    default = {v: False for v in VIEW_NAMES}
    default["confidence"] = 0.0

    if not raw or not raw.strip():
        return default

    envelope = _extract_json_envelope(raw)
    if envelope is None:
        return default
    data = _try_parse_json(envelope)
    if not isinstance(data, dict):
        return default

    out = dict(default)
    for v in VIEW_NAMES:
        val = data.get(v, False)
        if isinstance(val, bool):
            out[v] = val
        elif isinstance(val, str):
            out[v] = val.strip().lower() in {"true", "yes", "1"}
        elif isinstance(val, (int, float)):
            out[v] = bool(val)
    conf = data.get("confidence", 0.0)
    try:
        out["confidence"] = max(0.0, min(1.0, float(conf)))
    except (TypeError, ValueError):
        out["confidence"] = 0.0
    return out


def _resolve_active_views(routing: dict) -> list[str]:
    """Apply the two safety nets, return ordered list of active views."""
    active = [v for v in VIEW_NAMES if routing.get(v, False)]

    # Safety net 1: nothing activated → default broad set.
    if not active:
        logger.debug("router activated 0 views, defaulting to %s",
                       DEFAULT_FALLBACK)
        return list(DEFAULT_FALLBACK)

    # Safety net 2: low confidence → backstop with conceptual.
    if routing.get("confidence", 0.0) < 0.6 and "conceptual" not in active:
        active.append("conceptual")

    # Preserve canonical ordering for reproducibility.
    return [v for v in VIEW_NAMES if v in set(active)]
