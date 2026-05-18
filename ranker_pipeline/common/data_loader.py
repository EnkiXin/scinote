"""Unified sample/manifest loader for ExpVid L2+L3 and SciVideoBench.

Returns one canonical record shape so Stage 1 (note gen), Stage 2 (labelling),
Stage 3 (training), and Stage 5 (eval) all consume the same structure.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evaluate_unified import TASKS, LEVEL_TASKS, REPO_ID  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402

SCIVB_ROOT = Path("/home/yz0392@unt.ad.unt.edu/xin_ai/scivideobench")
SCIVB_ANN_PATH = SCIVB_ROOT / "scivideobench_1k.jsonl"
SCIVB_VIDEO_DIR = SCIVB_ROOT / "videos"

L2_L3_TASKS = LEVEL_TASKS["all_level2_3"]  # 6 tasks


@dataclass
class Sample:
    """Canonical training/eval sample."""

    id: str
    benchmark: str          # "expvid" | "scivideobench"
    task: str               # e.g. "experimental_conclusion" | "mc"
    task_type: str          # "mc" | "seqgen" | "steppred" | "fitb"
    video_id: str           # benchmark-internal video identifier
    video_path: str         # absolute (or HF-hub) path the loader produces
    question: str
    options: dict[str, str]
    gold: Optional[str]     # the gold answer (only used at Stage-2 label time)
    raw: dict               # original record, for fields like discipline

    @property
    def cache_key(self) -> str:
        return hashlib.md5(self.id.encode()).hexdigest()[:16]


def _expvid_video_path(video_path_field: str) -> str:
    """Resolve an ExpVid record's relative `video_path` to a local file."""
    return hf_hub_download(repo_id=REPO_ID, filename=video_path_field, repo_type="dataset")


def _scivb_video_path(video_id: str) -> str:
    for pat in (f"jove_{video_id}.mp4", f"{video_id}.mp4"):
        p = SCIVB_VIDEO_DIR / pat
        if p.exists():
            return str(p)
    return ""


def load_expvid_samples(tasks: Iterable[str] = L2_L3_TASKS, limit: Optional[int] = None) -> list[Sample]:
    """Load ExpVid items for the given tasks. Resolves video paths lazily."""
    out: list[Sample] = []
    for t in tasks:
        ann_path, task_type = TASKS[t]
        local = hf_hub_download(repo_id=REPO_ID, filename=ann_path, repo_type="dataset")
        with open(local) as f:
            items = [json.loads(l) for l in f if l.strip()]
        if limit:
            items = items[:limit]
        for it in items:
            sid = f"expvid_{t}_{it['video_path'].replace('/', '_')}_{it.get('id', 'noid')}"
            out.append(Sample(
                id=sid,
                benchmark="expvid",
                task=t,
                task_type=task_type,
                video_id=it["video_path"],
                video_path=it["video_path"],  # resolved lazily on demand
                question=it["question"],
                options=it.get("options", {}),
                gold=it.get("answer"),
                raw=it,
            ))
    return out


def load_scivideobench_samples(limit: Optional[int] = None) -> list[Sample]:
    out: list[Sample] = []
    with open(SCIVB_ANN_PATH) as f:
        items = [json.loads(l) for l in f if l.strip()]
    if limit:
        items = items[:limit]
    for it in items:
        vid = str(it["video_id"]); qid = str(it.get("question_id", ""))
        sid = f"scivideobench_mc_{vid}_{qid}"
        out.append(Sample(
            id=sid,
            benchmark="scivideobench",
            task="mc",
            task_type="mc",
            video_id=vid,
            video_path=_scivb_video_path(vid),
            question=it["question"],
            options=it.get("options", {}),
            gold=it.get("answer"),
            raw=it,
        ))
    return out


def load_all_training_samples(limit: Optional[int] = None) -> list[Sample]:
    """Stage 2/3 training pool: ExpVid L2+L3 + SciVideoBench."""
    return load_expvid_samples(L2_L3_TASKS, limit=limit) + load_scivideobench_samples(limit=limit)


def resolve_video_path(sample: Sample) -> str:
    """Materialise the actual on-disk path for a sample (handles ExpVid HF cache)."""
    if sample.benchmark == "expvid":
        return _expvid_video_path(sample.video_path)
    return sample.video_path  # SciVideoBench: already absolute
