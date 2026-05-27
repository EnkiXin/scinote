"""V9 OCR ledger preprocessor.

Builds a frame-level OCR ledger BEFORE Stage 1.2 runs. The Stage 1.2
prompt then injects this ledger so the VLM picks numerical values from
a verified menu rather than hallucinating them under visual overload.

The ledger entry schema:

    {"timestamp": 3.15, "frame_idx": 7,
     "text": "0.535", "type": "numeric",
     "bbox": [x, y, w, h] | None,
     "confidence": 0.92}

Type tagging is regex-based:
  - numeric  : pure number, possibly signed/decimal
  - unit     : standalone unit token (g, mL, °C, ...)
  - compound : number+unit jammed together ("220°C", "50mL")
  - label    : anything else readable (textual labels, axis names...)

VLM call uses ``generate_image`` (same as V8 OCR tool) but on the whole
frame, not a per-entity crop.

See V9_RESEARCH_PLAN.md §3.3.1.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field, asdict
from typing import Optional

from PIL import Image

logger = logging.getLogger(__name__)


_FRAME_OCR_PROMPT = (
    "Read all visible text, numbers, labels, instrument readings, and "
    "unit symbols in this image. Output each item on its own line in "
    "the strict format:\n"
    "  <text>\n"
    "Do not include positions or descriptions — just the raw text tokens, "
    "one per line, in the order you see them. If there is no readable "
    "text in the image, output exactly: NO_TEXT_VISIBLE."
)

NO_TEXT_MARKER = "NO_TEXT_VISIBLE"
DEFAULT_RESOLUTION = (768, 768)
DEFAULT_MAX_TOKENS = 220
DEFAULT_CONFIDENCE = 0.85   # VLM-OCR has no native conf score; use a
                              # fixed prior. Stage 1.2 prompt treats
                              # everything ≥0.5 as usable.

# Tokens we choose to recognize as units. Keep tight to avoid coloring
# normal English words as units.
_UNIT_TOKENS = {
    "g", "mg", "kg", "ng", "μg", "ug",
    "L", "mL", "uL", "μL", "nL",
    "M", "mM", "μM", "uM", "nM",
    "°C", "K", "Pa", "kPa", "MPa", "atm", "bar",
    "rpm", "rcf", "g-force",
    "mol", "mmol", "μmol", "umol",
    "Hz", "kHz", "MHz",
    "V", "mV", "A", "mA", "W", "Ω",
    "min", "s", "sec", "h", "hr",
    "%", "ppm", "ppb",
    "RH", "OD", "pH", "Da", "kDa",
}


# ── Ledger entry ───────────────────────────────────────────────────

@dataclass
class OCRLedgerEntry:
    timestamp: float
    frame_idx: int
    text: str
    type: str = "label"
    bbox: Optional[list[int]] = None
    confidence: float = DEFAULT_CONFIDENCE

    def to_dict(self) -> dict:
        return asdict(self)


# ── Token classification ───────────────────────────────────────────

_NUMERIC_RE = re.compile(r"^-?\d+(?:[.,]\d+)?$")
_COMPOUND_RE = re.compile(
    r"^-?\d+(?:[.,]\d+)?\s*"
    r"(?:°C|°F|μL|uL|μg|ug|μmol|umol|μM|uM|"
    r"[a-zA-Z%Ω°]+)$",
)


def classify_text(token: str) -> str:
    """Return one of {numeric, unit, compound, label}."""
    t = token.strip()
    if not t:
        return "label"
    if _NUMERIC_RE.match(t):
        return "numeric"
    if t in _UNIT_TOKENS:
        return "unit"
    if _COMPOUND_RE.match(t):
        return "compound"
    return "label"


# ── VLM frame OCR ──────────────────────────────────────────────────

def _ocr_one_frame(
    frame: Image.Image,
    vlm,
    resolution: tuple[int, int] = DEFAULT_RESOLUTION,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> list[str]:
    """Run VLM-OCR on a single frame; return list of raw text tokens.

    Empty list means either no text or the VLM call failed (logged).
    """
    img = frame
    try:
        if img.size != resolution:
            img = img.resize(resolution, Image.BILINEAR)
    except Exception as e:
        logger.debug("resize failed: %s (using original)", e)

    try:
        raw = vlm.generate_image(_FRAME_OCR_PROMPT, img, max_tokens=max_tokens)
    except Exception as e:
        logger.warning("VLM OCR call failed: %s", e)
        return []

    text = (raw or "").strip()
    if not text or NO_TEXT_MARKER in text.upper():
        return []

    tokens: list[str] = []
    for line in text.splitlines():
        line = line.strip().strip("-•").strip()
        if not line:
            continue
        if NO_TEXT_MARKER in line.upper():
            continue
        # Some VLMs prefix lines with bullets / quotes — strip them.
        line = line.strip("\"' \t")
        # Split a line on commas/semicolons so each unit becomes its own
        # ledger entry, but keep number+unit pairs together.
        for piece in re.split(r"[,;]\s*", line):
            piece = piece.strip()
            if piece:
                tokens.append(piece)
    return tokens


# ── Public API ─────────────────────────────────────────────────────

class OCRLedgerBuilder:
    """Builds the full-video OCR ledger.

    Caller is responsible for supplying frames (typically the 32-frame
    uniform sample) and their seconds-from-start timestamps. The VLM
    instance must expose a ``generate_image(prompt, image, max_tokens=...)``
    method (V8's `QwenVL72BClient` and V8's mock VLMs both qualify).
    """

    def __init__(
        self,
        vlm,
        resolution: tuple[int, int] = DEFAULT_RESOLUTION,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        confidence_floor: float = 0.5,
    ) -> None:
        self.vlm = vlm
        self.resolution = resolution
        self.max_tokens = max_tokens
        self.confidence_floor = confidence_floor

    def build_ledger(
        self,
        frames: list[Image.Image],
        timestamps: list[float],
    ) -> list[OCRLedgerEntry]:
        if len(frames) != len(timestamps):
            raise ValueError(
                f"frames ({len(frames)}) / timestamps ({len(timestamps)}) "
                "length mismatch"
            )

        ledger: list[OCRLedgerEntry] = []
        for idx, (frame, ts) in enumerate(zip(frames, timestamps)):
            tokens = _ocr_one_frame(
                frame, self.vlm,
                resolution=self.resolution,
                max_tokens=self.max_tokens,
            )
            for tok in tokens:
                ledger.append(OCRLedgerEntry(
                    timestamp=float(ts),
                    frame_idx=idx,
                    text=tok,
                    type=classify_text(tok),
                ))
        return [e for e in ledger if e.confidence >= self.confidence_floor]


def format_ocr_ledger_for_prompt(ledger: list[OCRLedgerEntry]) -> str:
    """Compact OCR ledger as markdown text for LLM consumption.

    Drops low-confidence entries already filtered by `build_ledger`.
    """
    if not ledger:
        return "(empty — no readable text found in the video frames)"

    lines = [
        "## OCR LEDGER (pre-processed, per-frame text capture)",
        "Format: timestamp_sec | text | type",
        "---",
    ]
    # Sort by timestamp so the LLM reads in time order; tie-break by
    # original ledger position (stable on Python list sort).
    for entry in sorted(ledger, key=lambda e: e.timestamp):
        lines.append(
            f"- {entry.timestamp:>6.2f}s | \"{entry.text}\" | {entry.type}"
        )
    return "\n".join(lines)


# ── Post-validation ────────────────────────────────────────────────

def validate_ocr_alignment(
    states_and_ops: dict,
    ledger: list[OCRLedgerEntry],
    window_pad_sec: float = 0.5,
) -> dict:
    """Verify every state's `quantitative_value` is grounded in the ledger.

    For each state with a non-null `quantitative_value`:
      - Find ledger entries whose timestamp falls in
        (state.time_interval[0] - pad, state.time_interval[1] + pad).
      - Each whitespace-separated token in `quantitative_value` must
        appear (case-insensitive substring match) in at least one ledger
        entry in that window.
      - If any token is unmatched, set `ocr_alignment_warning=True` on
        the state.

    Mutates `states_and_ops` in place and returns it for chaining.
    """
    if not isinstance(states_and_ops, dict):
        return states_and_ops

    entity_states = states_and_ops.get("entity_states", [])
    for ent_record in entity_states:
        for st in ent_record.get("states", []):
            qv = st.get("quantitative_value")
            if not qv:
                continue

            ti = st.get("time_interval")
            if not (isinstance(ti, (list, tuple)) and len(ti) == 2):
                continue
            t0, t1 = float(ti[0]) - window_pad_sec, float(ti[1]) + window_pad_sec
            window = [e for e in ledger if t0 <= e.timestamp <= t1]
            window_text = " ".join(e.text.lower() for e in window)

            missing = []
            for tok in qv.split():
                if tok.lower() not in window_text:
                    missing.append(tok)

            if missing:
                st["ocr_alignment_warning"] = True
                logger.debug(
                    "state %s qv=%r contains unmatched tokens %s (window %.1fs-%.1fs)",
                    st.get("state_id"), qv, missing, t0, t1,
                )
            else:
                st["ocr_alignment_warning"] = False

    return states_and_ops
