#!/usr/bin/env python3
"""Filter STGR-SFT/RL json to media-complete samples (covered subset).

Reuses the official path resolution (tools/check_open_o3_data.official_media_paths)
so the covered set is consistent with the data-sanity report. Rows whose every
resolved media path exists are kept; this avoids training crashes on missing
media (TVG/TreeVGR/GQA/VideoEspresso gaps).
"""
import json
import sys
from pathlib import Path

REPO = Path("/home/yz0392@unt.ad.unt.edu/xin_ai/Open-o3-Video")
DATA_ROOT = Path("/home/yz0392@unt.ad.unt.edu/xin_ai/open_o3/data/Open-o3-Video-data")
sys.path.insert(0, str(REPO / "tools"))
from check_open_o3_data import official_media_paths, iter_records  # noqa: E402


def filter_file(name: str) -> None:
    src = DATA_ROOT / "json_data" / name
    records = iter_records(json.loads(src.read_text()))
    kept = []
    for r in records:
        paths = official_media_paths(DATA_ROOT, r)
        if paths and all(p.exists() for p in paths):
            kept.append(r)
    out = DATA_ROOT / "json_data" / name.replace(".json", "-covered.json")
    out.write_text(json.dumps(kept))
    print(f"{name}: {len(kept)}/{len(records)} covered -> {out.name}", flush=True)


for n in ("STGR-SFT.json", "STGR-RL.json"):
    filter_file(n)
