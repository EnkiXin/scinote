"""Wikimedia Commons targeted crawler for V8 image library rebuild.

For each target Wikimedia category, fetch member files via the
MediaWiki API, download images that satisfy:
  - JPG/PNG/JPEG content type
  - min 256x256 resolution
  - reasonable aspect (0.4 < w/h < 2.5) to exclude long banners
  - file size 30 KB – 5 MB (excludes thumbnails and giant scans)

Per category, cap at N images (default 200) to keep total bounded.
All downloads saved under
   cache/image_library/raw/wikimedia/<slug>/
together with a manifest.jsonl recording attribution + license per file.

This crawler does NOT distinguish photos vs schematic illustrations.
A follow-up filter step does that (separate script).

Usage:
    python scripts/v8_crawl_wikimedia.py --max-per-cat 200 [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from urllib.parse import quote

import requests
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "cache/image_library/raw/wikimedia"
USER_AGENT = (
    "UNT-ProtoNote-V8-Research/0.1 "
    "(xin.yang@unt.edu; research; benchmark grounding library)"
)

# High-priority cross-discipline categories. Curate by matching V8 entity
# types and SciVB/ExpVid disciplines: physics, chemistry, biology,
# medicine, engineering, materials, imaging, bioengineering.
#
# NOTE: 2026-05-27 — first pass found 12/26 categories returned 0 files
# because the plural / "_devices" / "_equipment" suffixes don't exist as
# Wikimedia Commons categories. Replaced with the actual canonical names
# (singular, or the term Commons curates them under). All 26 should now
# yield >0 after recursion. Slugs preserved to match WIKIMEDIA_MAP in
# dataset_mappers.py (some slugs changed; mapper updated in tandem).
CATEGORIES = [
    # Imaging / microscopy (Instrument)
    ("Atomic_force_microscopes",          "atomic force microscope", "Instrument"),
    ("Scanning_electron_microscope",      "scanning electron microscope", "Instrument"),
    ("Transmission_electron_microscopes", "transmission electron microscope", "Instrument"),
    ("Stereo_microscopes",                "optical microscope", "Instrument"),
    ("Confocal_microscopes",              "confocal microscope", "Instrument"),
    # Physics measurement (Instrument / Display)
    ("Oscilloscopes",                     "oscilloscope", "Instrument"),
    ("Mass_spectrometers",                "mass spectrometer", "Instrument"),
    ("Nuclear_magnetic_resonance_spectroscopy", "NMR spectrometer", "Instrument"),
    ("Infrared_spectrometers",            "infrared spectrometer", "Instrument"),
    ("Spectrophotometers",                "UV-Vis spectrometer", "Instrument"),
    # Optics / lasers
    ("Optical_tables",                    "optical table", "Instrument"),
    ("Lasers",                            "laser", "Instrument"),
    # Bioengineering / nanofab
    ("Microfluidic_chips",                "microfluidic device", "Instrument"),
    ("Vacuum_chambers",                   "vacuum chamber", "Instrument"),
    ("Sputter_coating",                   "sputter deposition system", "Instrument"),
    ("Chemical_vapour_deposition",        "chemical vapor deposition system", "Instrument"),
    ("Photolithography_(microfabrication)", "photolithography aligner", "Instrument"),
    # Biology / medicine
    ("Gel_electrophoresis",               "gel electrophoresis apparatus", "Instrument"),
    ("Centrifuges",                       "centrifuge", "Instrument"),
    ("Thermocyclers",                     "PCR thermocycler", "Instrument"),
    ("Incubators_(microbiology)",         "laboratory incubator", "Instrument"),
    ("Surgical_instruments",              "surgical instrument", "Instrument"),
    # General laboratory containers (boost coverage)
    ("Round-bottom_flasks",               "round-bottom flask", "Container"),
    ("Erlenmeyer_flasks",                 "Erlenmeyer flask", "Container"),
    ("Glass_beakers_(laboratory_equipment)", "beaker", "Container"),
    ("Petri_dishes",                      "Petri dish", "Container"),
]


def slugify(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", s).strip("_").lower()


def api_get(session: requests.Session, params: dict) -> dict:
    r = session.get(
        "https://commons.wikimedia.org/w/api.php",
        params={**params, "format": "json"},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def list_subcategories(
    session: requests.Session,
    category: str,
    limit: int = 30,
) -> list[str]:
    """Return immediate subcategory names (without 'Category:' prefix)."""
    subs: list[str] = []
    cmcontinue = None
    while len(subs) < limit:
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": f"Category:{category}",
            "cmtype": "subcat",
            "cmlimit": min(50, limit - len(subs)),
        }
        if cmcontinue:
            params["cmcontinue"] = cmcontinue
        data = api_get(session, params)
        for m in data.get("query", {}).get("categorymembers", []):
            t = m.get("title", "")
            if t.startswith("Category:"):
                subs.append(t[len("Category:"):])
        cmcontinue = data.get("continue", {}).get("cmcontinue")
        if not cmcontinue:
            break
        time.sleep(0.3)
    return subs


def list_files_direct(
    session: requests.Session,
    category: str,
    cap: int,
) -> list[str]:
    """List File: titles directly in `category` (no recursion)."""
    titles: list[str] = []
    cmcontinue = None
    while len(titles) < cap:
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": f"Category:{category}",
            "cmtype": "file",
            "cmlimit": min(50, cap - len(titles)),
        }
        if cmcontinue:
            params["cmcontinue"] = cmcontinue
        data = api_get(session, params)
        for m in data.get("query", {}).get("categorymembers", []):
            titles.append(m["title"])
            if len(titles) >= cap:
                break
        cmcontinue = data.get("continue", {}).get("cmcontinue")
        if not cmcontinue:
            break
        time.sleep(0.3)
    return titles


def list_files_in_category(
    session: requests.Session,
    category: str,
    cap: int,
    recurse: int = 1,
) -> list[str]:
    """Return File: titles in `category` and (optionally) its subcategories.

    Wikimedia commons category trees usually keep files in deeper subcats;
    a single level of recursion is enough to multiply yield 5-50x.
    """
    titles: list[str] = []
    seen: set[str] = set()

    def absorb(new_titles):
        for t in new_titles:
            if t not in seen:
                seen.add(t)
                titles.append(t)
                if len(titles) >= cap:
                    return True
        return False

    # 1) Files directly under the target category
    if absorb(list_files_direct(session, category, cap=cap)):
        return titles

    # 2) Recurse one level into subcategories
    if recurse > 0:
        subs = list_subcategories(session, category, limit=30)
        for sub in subs:
            try:
                sub_titles = list_files_direct(
                    session, sub, cap=max(20, (cap - len(titles)) // 2),
                )
            except Exception:
                continue
            if absorb(sub_titles):
                return titles
            time.sleep(0.2)
    return titles


def get_imageinfo(
    session: requests.Session,
    titles: list[str],
) -> dict[str, dict]:
    """Return per-title imageinfo metadata (url, width, height, mime, etc.)."""
    info: dict[str, dict] = {}
    for i in range(0, len(titles), 25):
        chunk = titles[i:i + 25]
        params = {
            "action": "query",
            "prop": "imageinfo",
            "iiprop": "url|size|mime|extmetadata",
            "titles": "|".join(chunk),
        }
        data = api_get(session, params)
        pages = data.get("query", {}).get("pages", {})
        for _, p in pages.items():
            t = p.get("title")
            ii = p.get("imageinfo")
            if t and ii:
                info[t] = ii[0]
        time.sleep(0.4)
    return info


def passes_quality_filter(meta: dict, dest: Path) -> tuple[bool, str]:
    mime = (meta.get("mime") or "").lower()
    if mime not in {"image/jpeg", "image/png", "image/jpg"}:
        return False, f"mime={mime}"
    w = meta.get("width", 0)
    h = meta.get("height", 0)
    if w < 256 or h < 256:
        return False, f"too_small {w}x{h}"
    ar = w / max(1, h)
    if ar < 0.4 or ar > 2.5:
        return False, f"bad_aspect {ar:.2f}"
    size = meta.get("size", 0)
    if size < 30 * 1024:
        return False, f"too_few_bytes {size}"
    if size > 5 * 1024 * 1024:
        return False, f"too_many_bytes {size}"
    return True, "ok"


def attribution_from(meta: dict, title: str) -> dict:
    em = meta.get("extmetadata", {}) or {}
    def gv(k):
        v = em.get(k)
        return (v or {}).get("value") if isinstance(v, dict) else v
    return {
        "title": title,
        "license": gv("LicenseShortName") or gv("License"),
        "author": gv("Artist") or gv("Credit"),
        "source_url": meta.get("descriptionurl") or meta.get("url"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-per-cat", type=int, default=200)
    ap.add_argument("--dry-run", action="store_true",
                     help="list candidates, do not download")
    ap.add_argument("--only-cat", default=None,
                     help="restrict to one category for testing")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    cats = CATEGORIES
    if args.only_cat:
        cats = [c for c in cats if c[0] == args.only_cat]
        if not cats:
            print(f"category {args.only_cat} not in list", file=sys.stderr)
            sys.exit(2)

    total_dl = 0
    total_skipped = 0
    grand_log = []

    for cat_name, unified_label, entity_type in cats:
        slug = slugify(cat_name)
        cat_dir = OUT_DIR / slug
        cat_dir.mkdir(parents=True, exist_ok=True)
        manifest = []
        print(f"\n=== {cat_name} → '{unified_label}' [{entity_type}] ===",
              flush=True)
        try:
            titles = list_files_in_category(
                session, cat_name, cap=args.max_per_cat * 3,
            )
        except Exception as e:
            print(f"  list error: {e}", flush=True)
            continue
        print(f"  candidates: {len(titles)}", flush=True)
        if not titles:
            continue
        info = get_imageinfo(session, titles)
        dl_this = 0
        for title in titles:
            if dl_this >= args.max_per_cat:
                break
            meta = info.get(title)
            if not meta:
                continue
            ok, reason = passes_quality_filter(meta, cat_dir)
            if not ok:
                total_skipped += 1
                continue
            url = meta.get("url")
            if not url:
                continue
            fname = re.sub(
                r"[^A-Za-z0-9_.-]+", "_", title.replace("File:", "")
            )[:160]
            dest = cat_dir / fname
            if dest.exists():
                manifest.append({
                    "fname": dest.name,
                    "label": unified_label, "entity_type": entity_type,
                    "raw_label": cat_name,
                    "source_dataset": "wikimedia",
                    **attribution_from(meta, title),
                })
                dl_this += 1
                continue
            if args.dry_run:
                print(f"  [dry-run] would dl {url}", flush=True)
                dl_this += 1
                continue
            ok_dl = False
            backoff = 2.0
            for attempt in range(4):
                try:
                    r = session.get(url, timeout=60, stream=True)
                    if r.status_code == 429:
                        wait = float(r.headers.get("Retry-After", backoff))
                        print(f"  429 throttled; sleeping {wait:.1f}s",
                              flush=True)
                        time.sleep(wait)
                        backoff *= 2
                        continue
                    r.raise_for_status()
                    with dest.open("wb") as f:
                        for chunk in r.iter_content(64 * 1024):
                            f.write(chunk)
                    try:
                        with Image.open(dest) as im:
                            im.verify()
                    except Exception:
                        dest.unlink(missing_ok=True)
                        break
                    manifest.append({
                        "fname": dest.name,
                        "label": unified_label, "entity_type": entity_type,
                        "raw_label": cat_name,
                        "source_dataset": "wikimedia",
                        **attribution_from(meta, title),
                    })
                    dl_this += 1
                    total_dl += 1
                    ok_dl = True
                    if dl_this % 25 == 0:
                        print(f"  …{dl_this}/{args.max_per_cat}", flush=True)
                    break
                except requests.exceptions.HTTPError as e:
                    if e.response is not None and e.response.status_code == 429:
                        time.sleep(backoff)
                        backoff *= 2
                        continue
                    print(f"  dl fail {title}: {e}", flush=True)
                    break
                except Exception as e:
                    print(f"  dl fail {title}: {e}", flush=True)
                    break
            if not ok_dl:
                total_skipped += 1
            # gentle pacing — Wikimedia asks for slow bots
            time.sleep(0.6)

        # write per-category manifest
        with (cat_dir / "manifest.jsonl").open("w") as f:
            for m in manifest:
                f.write(json.dumps(m) + "\n")
        grand_log.append({
            "category": cat_name, "unified_label": unified_label,
            "entity_type": entity_type, "downloaded": dl_this,
        })
        print(f"  ✓ saved {dl_this} to {cat_dir}", flush=True)

    # write grand summary
    sum_path = OUT_DIR / "crawl_summary.json"
    with sum_path.open("w") as f:
        json.dump({
            "total_downloaded": total_dl,
            "total_skipped": total_skipped,
            "per_category": grand_log,
        }, f, indent=2)

    print(f"\nTotal downloaded: {total_dl}")
    print(f"Total skipped:    {total_skipped}")
    print(f"Summary written:  {sum_path}")


if __name__ == "__main__":
    main()
