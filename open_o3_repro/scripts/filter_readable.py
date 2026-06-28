#!/usr/bin/env python3
"""Filter covered subset to DECORD-READABLE media (not just existing).

The covered subset removed missing media, but some videos exist yet are
corrupt (h264 errors) and crash training because torchvision 0.26 lost the
read_video fallback. Parallel decord-open test keeps only readable samples.
"""
import json, sys, os
from concurrent.futures import ProcessPoolExecutor

REPO = "/home/yz0392@unt.ad.unt.edu/xin_ai/Open-o3-Video"
DATA_ROOT = "/home/yz0392@unt.ad.unt.edu/xin_ai/open_o3/data/Open-o3-Video-data"
sys.path.insert(0, REPO + "/tools")
from pathlib import Path
from check_open_o3_data import official_media_paths  # noqa: E402


def readable(row):
    paths = official_media_paths(Path(DATA_ROOT), row)
    if not paths:
        return False
    for p in paths:
        sp = str(p)
        if sp.lower().endswith((".jpg", ".jpeg", ".png")):
            if not p.exists():
                return False
            continue
        try:
            import decord
            vr = decord.VideoReader(sp)
            if len(vr) < 1:
                return False
            vr[0]  # actually decode first frame (catches h264 corruption)
        except Exception:
            return False
    return True


def filt(name):
    recs = json.loads(open(f"{DATA_ROOT}/json_data/{name}").read())
    with ProcessPoolExecutor(max_workers=24) as ex:
        flags = list(ex.map(readable, recs, chunksize=8))
    kept = [r for r, ok in zip(recs, flags) if ok]
    out = f"{DATA_ROOT}/json_data/{name.replace('-covered.json','-readable.json')}"
    json.dump(kept, open(out, "w"))
    print(f"{name}: {len(kept)}/{len(recs)} readable -> {os.path.basename(out)}", flush=True)


for n in ("STGR-SFT-covered.json", "STGR-RL-covered.json"):
    filt(n)
