#!/usr/bin/env python3
"""Robust per-file mirror downloader for HF datasets.

snapshot_download + HF_ENDPOINT=hf-mirror fails (metadata API incompat), and
HF direct is throttled to ~79 KB/s. The mirror's direct resolve URLs are fast
(~19 MB/s). So: list files via the HF API (metadata works), then wget each from
hf-mirror.com with resume (-c). Single wget per file (no offset corruption).

Usage: python dl_mirror.py <repo_id> <local_dir> [allow_substr]
"""
import os
import subprocess
import sys

from huggingface_hub import HfApi

repo_id = sys.argv[1]
local_dir = sys.argv[2]
allow = sys.argv[3] if len(sys.argv) > 3 else ""
os.makedirs(local_dir, exist_ok=True)

allows = [a for a in allow.split(",") if a]
files = HfApi().list_repo_files(repo_id, repo_type="dataset")
files = [
    f for f in files
    if (not allows or any(a in f for a in allows)) and not f.endswith(".gitattributes")
]
print(f"{repo_id}: {len(files)} files to fetch (filter={allows!r})", flush=True)

base = f"https://hf-mirror.com/datasets/{repo_id}/resolve/main"
for i, f in enumerate(files):
    out = os.path.join(local_dir, f)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    url = f"{base}/{f}"
    print(f"[{i+1}/{len(files)}] {f}", flush=True)
    subprocess.run(
        ["wget", "-c", "-q", "--tries=10", "--timeout=60", "--waitretry=5",
         "--retry-connrefused", "-O", out, url],
        check=False,
    )
print(f"{repo_id} DONE", flush=True)
