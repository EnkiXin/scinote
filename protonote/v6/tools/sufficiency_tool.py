"""sufficiency_tool.py — v6 explicit sufficiency-judgment tool.

The CRITICAL v6 tool (plan §2.1 Tool 5). The planner can call this to
get an explicit yes/no judgment on whether the current notes are enough
to answer the question, plus a confidence score and a list of missing
info that could direct the next tool call.

Output schema (strict JSON):
  {
    "sufficient": bool,
    "confidence": float 0..1,
    "missing_info": [str, ...],
    "reasoning": str (1-2 sentences),
  }
"""
from __future__ import annotations

import json
import re


_SYSTEM = (
    "You evaluate whether current evidence is sufficient to answer a "
    "scientific video question. Be decisive but cautious — if specific "
    "values are missing, say sufficient=false."
)

_PROMPT_TEMPLATE = """You are evaluating whether the current evidence is
sufficient to answer a scientific video question.

Question: {question}

Current notes:
{notes}

Analyze step by step:

Step 1 - Question analysis:
- What does the question specifically ask for?
- What kind of evidence would directly answer it?

Step 2 - Notes analysis:
- What information is currently in notes?
- Does it provide direct evidence for the answer?

Step 3 - Gap identification:
- Is there missing critical information?
- What specifically would help (if anything)?

Step 4 - Judgment:
- Can you confidently answer based on current notes?

Output JSON ONLY (no prose), this exact schema:
{{
  "sufficient": true/false,
  "confidence": 0.0-1.0,
  "missing_info": ["item 1 if any", "item 2 if any"],
  "reasoning": "1-2 sentences explaining your judgment"
}}"""


def _extract_first_json_object(s: str) -> str | None:
    """Find the first balanced {...} object in `s` (handles nested braces)."""
    if not s: return None
    start = s.find("{")
    if start < 0: return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        ch = s[i]
        if esc: esc = False; continue
        if ch == "\\": esc = True; continue
        if ch == '"': in_str = not in_str
        if in_str: continue
        if ch == "{": depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[start:i+1]
    return None


def _parse_response(raw: str) -> dict:
    """Robust JSON parser with fallback."""
    if not raw: return _default(False, 0.0, "empty_response")
    json_str = _extract_first_json_object(raw)
    if not json_str: return _default(False, 0.0, "no_json_found")
    try:
        obj = json.loads(json_str)
    except Exception:
        # Try to fix trailing commas, single quotes
        s = json_str.replace("'", '"')
        s = re.sub(r",\s*}", "}", s)
        s = re.sub(r",\s*]", "]", s)
        try:
            obj = json.loads(s)
        except Exception:
            return _default(False, 0.0, "json_parse_err")
    # Coerce types
    suf = obj.get("sufficient")
    if isinstance(suf, str):
        suf = suf.lower() in ("true", "yes", "1")
    suf = bool(suf)
    try:
        conf = float(obj.get("confidence", 0.0))
    except Exception:
        conf = 0.0
    conf = max(0.0, min(1.0, conf))
    missing = obj.get("missing_info") or []
    if not isinstance(missing, list):
        missing = [str(missing)]
    missing = [str(x)[:200] for x in missing[:6]]
    reasoning = str(obj.get("reasoning", ""))[:400]
    return {
        "sufficient": suf,
        "confidence": conf,
        "missing_info": missing,
        "reasoning": reasoning,
    }


def _default(suf: bool, conf: float, why: str) -> dict:
    return {
        "sufficient": suf,
        "confidence": conf,
        "missing_info": [],
        "reasoning": f"parse_fallback: {why}",
    }


def is_sufficient(question: str, notes: str, vlm,
                     *, max_tokens: int = 350) -> dict:
    """Explicit sufficiency judgment LLM call.

    Text-only call to the 72B (no video frames needed for this meta-
    judgment).
    """
    prompt = _PROMPT_TEMPLATE.format(
        question=(question or "")[:1000],
        notes=(notes or "(no notes yet)")[:3500],
    )
    raw = vlm.generate_text(prompt, system=_SYSTEM, max_tokens=max_tokens)
    return _parse_response(raw)
