"""build_corpus.py — BioProBench corpus loader + JoVE leak filter + chunking.

Sources (downloaded under data/bioprobench/):
  Bio-protocol.json        4,157 protocols (Bio-protocol journal,
                            doi prefix 10.21769/BioProtoc)
  Protocol-exchange.json     933 protocols (Nature Protocol Exchange,
                            doi prefix 10.21203/rs.pex)
  Protocol-io.json         9,585 protocols (protocols.io entries)
  Total                   14,675

Per the v4 plan §7 mandatory JoVE 4-layer filter:
  - journal field must not be "visualized experiments" / "jove"
  - doi must not start with "10.3791/" (JoVE DOI prefix)
  - title must not contain "jove"
  - url must not contain "jove.com"

Run leak audit: expected leak rate = 0 % (BioProBench has zero JoVE
overlap by source — DOI prefixes are BioProtoc / pex / protocols.io,
none of which is 10.3791/).

Chunking: split each protocol into <= 300-token passages. Within a
protocol, prefer breaking at:
  1. Step boundaries (^Step \d+:, ^\d+\.)
  2. Section boundaries (^## ..., ^# ...)
  3. Sentence boundaries (fallback)

Output (data/bioprobench/filtered_corpus.jsonl): one JSON per line
with schema:
  {"chunk_id":  str,    # "{source_id}_chunk_{i}"
   "source_id": str,    # original protocol id
   "source":    str,    # "bio-protocol" / "protocol-exchange" / "protocols-io"
   "doi":       str,    # extracted from url where possible
   "title":     str,
   "text":      str,    # the chunk text (<= 300 tokens)
   "domain":    str,    # primary_domain from classification (if available)
  }
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

# ── JoVE 4-layer leak filter ────────────────────────────────────────────────


def is_jove_leak(entry: dict, source: str = "") -> bool:
    """Return True if the entry comes from JoVE (must be filtered out)."""
    # Layer 1: journal field
    journal = str(entry.get("journal", "")).lower()
    if "visualized experiments" in journal or "jove" in journal:
        return True

    # Layer 2: DOI prefix
    doi = str(entry.get("doi", "")).strip()
    if not doi:
        # Try to extract DOI from url
        url = str(entry.get("url", ""))
        m = re.search(r"10\.\d{4,9}/[^\s]+", url)
        if m:
            doi = m.group(0)
    if doi.startswith("10.3791/"):
        return True

    # Layer 3: title field
    title = str(entry.get("title", "")).lower()
    if "jove" in title:
        return True

    # Layer 4: URL host
    url = str(entry.get("url", "")).lower()
    if "jove.com" in url:
        return True

    return False


# ── chunking ────────────────────────────────────────────────────────────────


def _approx_token_count(text: str) -> int:
    """Rough token count: ~4 chars per token (whitespace-aware)."""
    return max(len(text) // 4, len(text.split()))


_STEP_RE = re.compile(r"^(?:Step\s+\d+:|\d+\.\s|##\s|#\s)",
                       flags=re.MULTILINE)


def chunk_text(text: str, *, max_tokens: int = 300) -> list[str]:
    """Split `text` into chunks of <= max_tokens.

    Prefer breaking at step/section boundaries, then sentence boundaries.
    """
    if not text or not text.strip():
        return []
    if _approx_token_count(text) <= max_tokens:
        return [text.strip()]

    # Try splitting by step / section markers first
    parts: list[str] = []
    starts = [m.start() for m in _STEP_RE.finditer(text)]
    if starts:
        if starts[0] > 0:
            starts = [0] + starts
        starts.append(len(text))
        for i in range(len(starts) - 1):
            seg = text[starts[i]:starts[i + 1]].strip()
            if seg:
                parts.append(seg)
    else:
        parts = [text]

    # Greedy combine adjacent parts up to max_tokens; further split
    # over-sized parts by sentence.
    out: list[str] = []
    buf = ""
    for p in parts:
        if not buf:
            buf = p
            continue
        candidate = buf + "\n\n" + p
        if _approx_token_count(candidate) <= max_tokens:
            buf = candidate
        else:
            out.append(buf.strip())
            buf = p
    if buf:
        out.append(buf.strip())

    # Sentence-level fallback for chunks still too big
    final: list[str] = []
    for chunk in out:
        if _approx_token_count(chunk) <= max_tokens:
            final.append(chunk)
            continue
        # Split by sentence
        sents = re.split(r"(?<=[.!?])\s+", chunk)
        sub_buf = ""
        for s in sents:
            cand = (sub_buf + " " + s).strip() if sub_buf else s
            if _approx_token_count(cand) <= max_tokens:
                sub_buf = cand
            else:
                if sub_buf:
                    final.append(sub_buf)
                sub_buf = s
        if sub_buf:
            final.append(sub_buf)
    return [c for c in final if c]


# ── domain extraction ──────────────────────────────────────────────────────


def extract_domain(entry: dict) -> str:
    cls = entry.get("classification")
    if isinstance(cls, dict):
        primary = cls.get("primary_domain", "")
        if primary:
            return str(primary)
    return ""


def extract_doi(entry: dict) -> str:
    doi = str(entry.get("doi", "")).strip()
    if doi:
        return doi
    url = str(entry.get("url", ""))
    m = re.search(r"10\.\d{4,9}/[^\s]+", url)
    return m.group(0) if m else ""


# ── corpus build ────────────────────────────────────────────────────────────


SOURCES = [
    ("Bio-protocol.json", "bio-protocol"),
    ("Protocol-exchange.json", "protocol-exchange"),
    ("Protocol-io.json", "protocols-io"),
]


def load_raw_corpus(data_dir: Path) -> list[tuple[dict, str]]:
    """Concatenate all raw BioProBench JSON files, tagging each entry
    with its source name."""
    out: list[tuple[dict, str]] = []
    for fname, source_name in SOURCES:
        path = data_dir / fname
        if not path.exists():
            print(f"  [warn] missing {path}; skipping")
            continue
        items = json.load(open(path))
        print(f"  loaded {len(items):>6} from {fname}")
        for it in items:
            out.append((it, source_name))
    return out


def build(data_dir: Path, out_path: Path,
            *, max_tokens: int = 300, max_protocols: int | None = None) -> None:
    raw = load_raw_corpus(data_dir)
    print(f"  total raw protocols: {len(raw)}")

    # JoVE filter
    kept: list[tuple[dict, str]] = []
    n_leak = 0
    for entry, source in raw:
        if is_jove_leak(entry, source):
            n_leak += 1
        else:
            kept.append((entry, source))
    print(f"  JoVE leaks filtered out: {n_leak} "
          f"({100 * n_leak / max(len(raw), 1):.4f} %)")
    print(f"  kept after JoVE filter: {len(kept)}")
    if n_leak > 0:
        print(f"  ⚠️  WARNING: JoVE leaks found! Investigate.")
    else:
        print(f"  ✓ JoVE leak rate = 0 % (filter verified clean)")

    if max_protocols:
        kept = kept[:max_protocols]
        print(f"  capped to first {max_protocols} for debug")

    # Chunk each kept protocol
    n_chunks = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fout:
        for entry, source in kept:
            source_id = str(entry.get("id", "?"))
            title = str(entry.get("title", "")).strip()
            doi = extract_doi(entry)
            domain = extract_domain(entry)

            # Concatenate the most useful fields, then chunk.
            body_parts = []
            for field in ("abstract", "input", "protocol",
                           "method", "description"):
                v = entry.get(field, "")
                if isinstance(v, str) and v.strip():
                    body_parts.append(v.strip())
            body_text = "\n\n".join(body_parts)

            chunks = chunk_text(body_text, max_tokens=max_tokens)
            for i, ch in enumerate(chunks):
                row = {
                    "chunk_id":  f"{source_id}_chunk_{i}",
                    "source_id": source_id,
                    "source":    source,
                    "doi":       doi,
                    "title":     title,
                    "domain":    domain,
                    "text":      ch,
                }
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_chunks += 1
    print(f"  wrote {n_chunks} chunks → {out_path}")

    # Sanity: domain distribution
    domains: dict[str, int] = {}
    for entry, _ in kept:
        d = extract_domain(entry) or "(none)"
        domains[d] = domains.get(d, 0) + 1
    print("\n  Top-10 domain distribution (across kept protocols):")
    for d, c in sorted(domains.items(), key=lambda x: -x[1])[:10]:
        print(f"    {d:<40} {c:>5}")


# ── self-test on a tiny subset ─────────────────────────────────────────────


def _self_test() -> None:
    """Verify chunking + filter on a tiny synthetic input."""
    # 1. JoVE filter
    assert is_jove_leak({"journal": "Journal of Visualized Experiments"})
    assert is_jove_leak({"doi": "10.3791/12345"})
    assert is_jove_leak({"title": "A JoVE protocol for ..."})
    assert is_jove_leak({"url": "https://www.jove.com/v/12345"})
    assert not is_jove_leak({"doi": "10.21769/BioProtoc.5130"})
    print("  ✓ JoVE filter passes all 4 layers")

    # 2. Chunking
    text = "Step 1: Wash cells.\nStep 2: Add reagent.\nStep 3: Incubate."
    chunks = chunk_text(text, max_tokens=300)
    assert len(chunks) == 1, f"short text should be 1 chunk; got {len(chunks)}"
    long_text = "Step 1: " + ("A " * 1000) + "\nStep 2: " + ("B " * 1000)
    chunks = chunk_text(long_text, max_tokens=300)
    assert len(chunks) >= 2, "long text should split into >=2 chunks"
    print(f"  ✓ chunking: short → 1 chunk, long ({len(chunks)} chunks)")

    print("\n  ✅ build_corpus self-test passed")


# ── CLI ─────────────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/bioprobench",
                     help="dir containing Bio-protocol/Protocol-exchange/Protocol-io JSON")
    ap.add_argument("--output", default="data/bioprobench/filtered_corpus.jsonl")
    ap.add_argument("--max_tokens", type=int, default=300)
    ap.add_argument("--max_protocols", type=int, default=0,
                     help="0 = all; useful values: 100 for smoke")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _self_test()
        return
    build(Path(args.data_dir), Path(args.output),
           max_tokens=args.max_tokens,
           max_protocols=(args.max_protocols or None))


if __name__ == "__main__":
    main()
