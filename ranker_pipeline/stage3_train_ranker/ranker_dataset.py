"""Dataset + collator for Stage 3 ranker training.

The ranker is *text-only* — it sees the question, options, and the four
segment notes (no video), and is trained to emit a JSON of per-segment
relevance scores. This is the same Qwen2.5-VL-3B base used in paper 1,
but only its language attention is LoRA-adapted; the vision tower is
frozen and unused at training time.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ranker_pipeline"))

from ranker_pipeline.common.formatting import (  # noqa: E402
    format_options,
    format_segments_for_ranker,
)

STAGE1_CACHE = Path(__file__).resolve().parents[1] / "stage1_temporal_notes" / "cache"


RANKER_SYSTEM = (
    "You are a video segment ranker. Given a question about a scientific "
    "experiment video and notes describing video segments, score each segment's "
    "relevance for answering the question. "
    'Output JSON: {"segment_0": <score>, "segment_1": <score>, ...} where scores '
    "are in [0.0, 1.0]."
)

RANKER_USER_TEMPLATE = """Question: {question}

Options:
{options}

Segments to rank:
{segments}

Output JSON only:"""


def load_temporal_notes(video_id: str) -> dict | None:
    safe = video_id.replace("/", "_").replace(".mp4", "")
    p = STAGE1_CACHE / f"{safe}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def build_target_json(relevance_scores: dict, num_segments: int) -> str:
    """Render the relevance dict as the canonical training target string."""
    return json.dumps({
        f"segment_{i}": float(relevance_scores.get(str(i), relevance_scores.get(i, 0.0)))
        for i in range(num_segments)
    })


class RankerDataset(Dataset):
    """Text-only causal-LM SFT dataset for the ranker."""

    def __init__(self, labels_paths: list[Path] | list[str], tokenizer,
                 split: str = "train", val_frac: float = 0.05, seed: int = 42,
                 max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Load every record from one or more JSONL files
        rows: list[dict] = []
        for p in labels_paths:
            p = Path(p)
            if not p.exists():
                continue
            with p.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        # Drop rows with no relevance scores (Stage 2 failures)
        rows = [r for r in rows if "relevance_scores" in r and r["relevance_scores"]]
        random.Random(seed).shuffle(rows)

        # val_frac=0.0 disables splitting (whole dataset is train; val empty)
        if val_frac <= 0.0:
            self.rows = rows if split == "train" else []
        else:
            n_val = max(1, int(len(rows) * val_frac))
            self.rows = rows[n_val:] if split == "train" else rows[:n_val]
        print(f"RankerDataset({split}): {len(self.rows)} rows", flush=True)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        rec = self.rows[idx]
        tn = load_temporal_notes(rec["video_id"])
        if tn is None:
            raise RuntimeError(f"missing stage-1 notes for video {rec['video_id']}")
        segments = tn["segments"]

        user_text = RANKER_USER_TEMPLATE.format(
            question=rec["question"],
            options=format_options(rec["options"]),
            segments=format_segments_for_ranker(segments),
        )
        target_text = build_target_json(rec["relevance_scores"], len(segments))

        # Build prompt + target via the tokenizer's chat template
        prompt_msgs = [
            {"role": "system", "content": RANKER_SYSTEM},
            {"role": "user", "content": user_text},
        ]
        full_msgs = prompt_msgs + [{"role": "assistant", "content": target_text}]

        prompt_str = self.tokenizer.apply_chat_template(
            prompt_msgs, tokenize=False, add_generation_prompt=True
        )
        full_str = self.tokenizer.apply_chat_template(
            full_msgs, tokenize=False, add_generation_prompt=False
        )

        full = self.tokenizer(full_str, return_tensors="pt", truncation=True,
                                max_length=self.max_length)
        prompt = self.tokenizer(prompt_str, return_tensors="pt", truncation=True,
                                  max_length=self.max_length)

        input_ids = full["input_ids"][0]
        attn = full["attention_mask"][0]
        plen = min(prompt["input_ids"].shape[1], len(input_ids))
        labels = input_ids.clone()
        labels[:plen] = -100

        return {"input_ids": input_ids, "attention_mask": attn, "labels": labels}


def ranker_collator(batch: list[dict]) -> dict:
    """Right-pad to max length in batch. Pads input_ids/attention_mask with 0
    and labels with -100 so padded tokens contribute nothing to loss."""
    batch = [b for b in batch if b is not None]
    input_ids = pad_sequence([b["input_ids"] for b in batch], batch_first=True, padding_value=0)
    attn = pad_sequence([b["attention_mask"] for b in batch], batch_first=True, padding_value=0)
    labels = pad_sequence([b["labels"] for b in batch], batch_first=True, padding_value=-100)
    return {"input_ids": input_ids, "attention_mask": attn, "labels": labels}
