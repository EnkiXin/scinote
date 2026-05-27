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


def _call_vlm_two_images(llm_client, prompt: str, img1, img2,
                                  max_tokens: int):
    """Send 2 images + prompt to VLM, return raw string.

    Routes through ``generate_json`` if the client has it (test mocks),
    else uses V6's ``generate_video`` (which handles N images), else
    raises.
    """
    if hasattr(llm_client, "generate_json"):
        return llm_client.generate_json(
            prompt=prompt, images=[img1, img2], max_tokens=max_tokens,
        )
    if hasattr(llm_client, "generate_video"):
        return llm_client.generate_video(
            prompt, [img1, img2], max_tokens=max_tokens, temperature=0.0,
        )
    raise AttributeError("no generate_json / generate_video on llm_client")


def _try_parse_json(text):
    if isinstance(text, dict):
        return text
    if not isinstance(text, str):
        return None
    import json as _json
    import re
    # Strip ```json fences
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    try:
        return _json.loads(m.group(0))
    except Exception:
        return None


def vlm_verify_match(
    query_crop: Image.Image,
    candidate_image: Image.Image,
    candidate_label: str,
    llm_client,
    max_tokens: int = 200,
) -> dict:
    """Run the VLM verifier; return a normalized dict."""
    prompt = VLM_VERIFY_PROMPT.format(candidate_label=candidate_label)
    try:
        raw = _call_vlm_two_images(
            llm_client, prompt, query_crop, candidate_image, max_tokens,
        )
    except Exception as e:
        logger.debug(f"VLM verify call raised: {e}")
        raw = None
    data = _try_parse_json(raw)

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
