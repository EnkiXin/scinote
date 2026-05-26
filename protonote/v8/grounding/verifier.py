"""VLM-based verification of image-library matches.

After SigLIP2 retrieves top-k candidates, the VLM looks at the query
crop and a candidate reference image side-by-side and decides whether
they're the same TYPE of scientific entity. This adds robustness against
SigLIP2 false positives.

`vlm_verify_match` returns
    {"match": bool, "confidence": float in [0,1], "reasoning": str}
On any error or non-JSON output it returns a `match=False, confidence=0`
dict so callers can treat verification failure as "no match".
"""

from __future__ import annotations

import logging

from PIL import Image

logger = logging.getLogger(__name__)


VLM_VERIFY_PROMPT = """Compare these two images and decide if they show the same TYPE of scientific entity.

Image 1: a region cropped from a scientific video
Image 2: a reference image labeled "{candidate_label}"

Consider:
1. Shape and overall form
2. Color and texture
3. Size and proportions (if visible)
4. Distinctive features

Do NOT require an exact match (lighting / angle / background may differ).
Match the entity type / category.

Output JSON ONLY:
{{
  "match": true/false,
  "confidence": 0.0-1.0,
  "reasoning": "brief comparison of visual features"
}}"""


def vlm_verify_match(
    query_crop: Image.Image,
    candidate_image: Image.Image,
    candidate_label: str,
    llm_client,
    max_tokens: int = 200,
) -> dict:
    """Run the VLM verifier; return a normalized dict.

    The `llm_client` is duck-typed: it must expose
    ``generate_json(prompt, images, max_tokens=...)`` returning a
    Python dict (or None on parse failure).
    """
    prompt = VLM_VERIFY_PROMPT.format(candidate_label=candidate_label)
    try:
        data = llm_client.generate_json(
            prompt=prompt,
            images=[query_crop, candidate_image],
            max_tokens=max_tokens,
        )
    except Exception as e:
        logger.debug(f"VLM verify call raised: {e}")
        data = None

    if not isinstance(data, dict):
        return {
            "match": False,
            "confidence": 0.0,
            "reasoning": "verification call failed",
        }

    try:
        conf = float(data.get("confidence", 0.0))
    except (TypeError, ValueError):
        conf = 0.0
    conf = max(0.0, min(1.0, conf))

    return {
        "match": bool(data.get("match", False)),
        "confidence": conf,
        "reasoning": str(data.get("reasoning", ""))[:500],
    }
