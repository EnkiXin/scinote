"""data/loaders.py — thin wrappers around evaluate_unified + the existing
20% test split JSONL. ProtoNote evaluates on the same 745 ExpVid + 218 SciVB
items paper-1 used (L2 + L3 difficulty), AND on the ExpVid L1 split loaded
directly from HuggingFace.

The records returned here are dicts with the schema used throughout the
existing scinote evaluators (sample_id / benchmark / task / task_type /
video_path / id / question / options / gold).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

TEST_SPLIT_PATH = ROOT / "train_data" / "v2_split_test.jsonl"
SCIVB_VIDEO_DIR = Path("/home/yz0392@unt.ad.unt.edu/xin_ai/scivideobench/videos")


def load_test_split(benchmark: str | None = None, limit: int | None = None) -> list[dict]:
    """Return all test samples (or filtered to a single benchmark)."""
    items = [json.loads(l) for l in open(TEST_SPLIT_PATH)]
    if benchmark:
        items = [it for it in items if it.get("benchmark") == benchmark]
    if limit:
        items = items[:limit]
    return items


# ── ExpVid L1 (Level-1, 4 single-clip MC sub-tasks) ─────────────────────────
#
# The HuggingFace ExpVid dataset exposes L1 as four separate configs:
# level1_tools / level1_materials / level1_operation / level1_quantity.
# Items are 4-choice MC, one question per video clip. We normalize them
# into the same schema the rest of ProtoNote expects.

_L1_CONFIGS = ["level1_tools", "level1_materials", "level1_operation",
                "level1_quantity"]


def _normalize_l1_item(raw: dict, cfg: str) -> dict:
    """Map an HF L1 record to our internal sample schema."""
    task_name = cfg.replace("level1_", "l1_")  # e.g. l1_tools
    return {
        "sample_id":   f"expvid_{task_name}_{raw['id']}",
        "benchmark":   "expvid",
        "task":        task_name,
        "task_type":   "mc",
        "video_path":  raw["video_path"],
        "id":          raw["id"],
        "question":    raw["question"],
        "options":     raw["options"],
        "gold":        raw["answer"],
        "category":    raw.get("category", ""),
        "asr_caption": raw.get("asr_caption", ""),
    }


def load_expvid_l1(subtask: str | None = None, limit: int | None = None) -> list[dict]:
    """Load ExpVid L1 test items, optionally filtered to one sub-task.

    Args:
        subtask: one of 'tools', 'materials', 'operation', 'quantity', or
            None for all four concatenated.
        limit: cap total returned items (None = all).
    """
    from datasets import load_dataset
    if subtask:
        cfgs = [f"level1_{subtask}"]
    else:
        cfgs = _L1_CONFIGS

    out: list[dict] = []
    for cfg in cfgs:
        ds = load_dataset("OpenGVLab/ExpVid", cfg, split="test")
        out.extend(_normalize_l1_item(r, cfg) for r in ds)
    if limit:
        out = out[:limit]
    return out


def resolve_video_path(item: dict) -> str:
    """Mirror train_notetaker_vl_v2.resolve_video_path — dispatch on benchmark."""
    benchmark = item.get("benchmark", "expvid")
    if benchmark == "scivideobench":
        vp = str(item.get("video_path", ""))
        vid = vp.split(":")[-1] if ":" in vp else vp
        for pat in (f"jove_{vid}.mp4", f"{vid}.mp4"):
            p = SCIVB_VIDEO_DIR / pat
            if p.exists():
                return str(p)
        return ""
    # ExpVid (L1, L2, L3 all use the same HF repo)
    from huggingface_hub import hf_hub_download
    from evaluate_unified import REPO_ID
    return hf_hub_download(repo_id=REPO_ID, filename=item["video_path"], repo_type="dataset")
