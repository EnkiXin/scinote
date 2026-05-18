"""Unified sample/manifest loader for ExpVid L2+L3 and SciVideoBench.

Returns one canonical record shape so Stage 1 (note gen), Stage 2 (labelling),
Stage 3 (training), and Stage 5 (eval) all consume the same structure.

**Train/test split**: paper 2 trains and evaluates *within* each benchmark
(no cross-benchmark transfer claims). For each task we keep a deterministic
80/20 split keyed by sample-id-md5, so Stage 2 only labels training items
and Stage 5 only evaluates held-out test items.
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


# ─── Deterministic per-task train/test split ─────────────────────────────
TRAIN_FRAC = 0.80
SPLIT_SEED = "ranker_pipeline_v1"


def _split_bucket(sample: Sample) -> str:
    """Deterministically map a sample to 'train' or 'test' using id md5.

    Per-task uniformity: hashing includes both `task` and `id`, so the same
    fraction is held out within every task — there's no risk of a task being
    skipped from training or evaluation purely by chance.
    """
    h = hashlib.md5(f"{SPLIT_SEED}|{sample.task}|{sample.id}".encode()).hexdigest()
    # First 8 hex chars -> int in [0, 2^32)
    bucket = int(h[:8], 16) / 0xFFFFFFFF
    return "train" if bucket < TRAIN_FRAC else "test"


def filter_by_split(samples: list[Sample], split: str) -> list[Sample]:
    """Keep only samples whose deterministic bucket equals `split`.

    `split` is one of {"train", "test", "all"}; "all" disables filtering.
    """
    if split == "all":
        return samples
    if split not in ("train", "test"):
        raise ValueError(f"split must be train/test/all, got {split!r}")
    return [s for s in samples if _split_bucket(s) == split]


def load_all_training_samples_split(split: str = "train",
                                       limit: Optional[int] = None) -> list[Sample]:
    """Stage 2/3 training pool restricted to the requested split."""
    samples = load_expvid_samples(L2_L3_TASKS, limit=limit) + load_scivideobench_samples(limit=limit)
    return filter_by_split(samples, split)


def load_eval_samples_split(benchmark: str, split: str = "test",
                              tasks: Optional[Iterable[str]] = None,
                              limit: Optional[int] = None) -> list[Sample]:
    """Stage 5 eval pool restricted to `split` items of one benchmark."""
    if benchmark == "scivideobench":
        samples = load_scivideobench_samples(limit=limit)
    elif benchmark in ("expvid", "expvid_l2", "expvid_l3"):
        if tasks is None:
            if benchmark == "expvid_l3":
                tasks = ["experimental_conclusion", "scientific_discovery"]
            elif benchmark == "expvid_l2":
                tasks = [t for t in L2_L3_TASKS
                          if t not in ("experimental_conclusion", "scientific_discovery")]
            else:
                tasks = L2_L3_TASKS
        samples = load_expvid_samples(tasks, limit=limit)
    else:
        raise ValueError(f"unknown benchmark: {benchmark}")
    return filter_by_split(samples, split)
