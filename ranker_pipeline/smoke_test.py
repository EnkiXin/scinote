"""smoke_test.py — verify imports + small synthetic forward through the dataset
pipeline. Does NOT touch GPU and does NOT call vLLM.

Run from repo root:
  python ranker_pipeline/smoke_test.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_imports():
    """Just import every module and confirm they parse."""
    print("test_imports() ...", flush=True)
    import ranker_pipeline.common.video_utils  # noqa: F401
    import ranker_pipeline.common.data_loader  # noqa: F401
    import ranker_pipeline.common.formatting  # noqa: F401
    import ranker_pipeline.stage1_temporal_notes.temporal_note_prompts  # noqa: F401
    import ranker_pipeline.stage2_counterfactual_labels.subset_eval  # noqa: F401
    import ranker_pipeline.stage3_train_ranker.ranker_dataset  # noqa: F401
    import ranker_pipeline.stage4_inference.pipeline_inference  # noqa: F401
    import ranker_pipeline.stage5_evaluation.bootstrap_ci  # noqa: F401
    import ranker_pipeline.stage5_evaluation.evaluate_all_conditions  # noqa: F401
    import ranker_pipeline.stage5_evaluation.ablations  # noqa: F401
    print("  ✅ all modules import cleanly")


def test_formatting():
    print("test_formatting() ...", flush=True)
    from ranker_pipeline.common.formatting import (
        format_options, format_note, format_segments_for_ranker, parse_letter,
    )
    opts = {"A": "first", "B": "second", "C": "third"}
    assert "A. first" in format_options(opts)
    assert "C. third" in format_options(opts)
    note = {
        "phase": "execution",
        "actions_observed": ["pipettes liquid"],
        "objects_visible": ["microtube"],
        "quantities": ["200uL"],
        "key_distinguishing_features": "researcher pipettes 200uL into petri dish",
    }
    s = format_note(note)
    assert "execution" in s and "pipettes liquid" in s and "200uL" in s
    seg = {"segment_id": 0, "time_range": [0.0, 120.0], "note": note}
    ranker_in = format_segments_for_ranker([seg])
    assert "Segment 0" in ranker_in and "0-120s" in ranker_in
    assert parse_letter("Answer: B") == "B"
    assert parse_letter("the answer is C.") == "C"
    assert parse_letter("") == ""
    print("  ✅ formatting helpers work")


def test_bootstrap():
    print("test_bootstrap() ...", flush=True)
    from ranker_pipeline.stage5_evaluation.bootstrap_ci import accuracy_ci, paired_delta_ci
    correct = [1.0] * 80 + [0.0] * 20
    m, lo, hi = accuracy_ci(correct, n_resamples=500, seed=1)
    assert abs(m - 0.80) < 1e-6
    assert lo < hi
    d, lo, hi = paired_delta_ci(correct, [1.0] * 50 + [0.0] * 50, n_resamples=500, seed=1)
    assert abs(d - 0.30) < 1e-6
    print("  ✅ bootstrap CI numerically reasonable")


def test_dataset_with_synthetic_data():
    """Write a tiny Stage-1 cache file + a Stage-2 labels file, then build
    a RankerDataset and inspect one sample. No tokenizer load — use a fake
    text-only tokenizer."""
    print("test_dataset_with_synthetic_data() ...", flush=True)

    from ranker_pipeline.stage3_train_ranker import ranker_dataset as RDS

    fake_video_id = "smoke_test_video_0"
    seg_note = {
        "phase": "execution",
        "actions_observed": ["pipettes"], "objects_visible": ["tube"],
        "visible_text_labels": [], "quantities": ["200uL"],
        "key_distinguishing_features": "researcher pipettes into petri dish",
    }
    cache_dir = Path(ranker_dataset_cache_dir())
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{fake_video_id}.json"
    cache_path.write_text(json.dumps({
        "video_id": fake_video_id,
        "duration_seconds": 480.0,
        "num_segments": 4,
        "segments": [
            {"segment_id": i,
             "time_range": [i * 120.0, (i + 1) * 120.0],
             "frame_indices": list(range(i * 8, (i + 1) * 8)),
             "note": seg_note}
            for i in range(4)
        ],
    }))

    with tempfile.TemporaryDirectory() as tmp:
        labels_path = Path(tmp) / "labels.jsonl"
        labels_path.write_text(json.dumps({
            "sample_id": "smoke_0",
            "benchmark": "scivideobench", "task": "mc",
            "video_id": fake_video_id,
            "question": "Which segment shows pipetting?",
            "options": {"A": "first", "B": "second", "C": "third", "D": "fourth"},
            "gold": "B",
            "all_segments": [0, 1, 2, 3],
            "subset_accuracy": {"[1]": True, "[0,1,2,3]": True},
            "minimal_sufficient_set": [1],
            "relevance_scores": {"0": 0.0, "1": 1.0, "2": 0.0, "3": 0.0},
        }) + "\n")

        # Minimal tokenizer stub: enough to satisfy apply_chat_template + tokenizer()
        class FakeTok:
            pad_token = "<pad>"; eos_token = "<eos>"
            def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=False):
                return "\n".join(f"{m['role']}: {m['content']}" for m in msgs)
            def __call__(self, text, return_tensors="pt", truncation=True, max_length=2048):
                import torch
                ids = torch.tensor([[ord(c) % 256 for c in text[:max_length]]])
                return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

        ds = RDS.RankerDataset([labels_path], FakeTok(), split="train", val_frac=0.0, max_length=512)
        item = ds[0]
        assert "input_ids" in item and "labels" in item
        # Some prompt tokens should be masked to -100
        assert (item["labels"] == -100).any()
        print(f"  ✅ dataset sample built; input_ids len={len(item['input_ids'])}, "
              f"masked prefix={(item['labels']==-100).sum().item()}")

    # Cleanup synthetic cache file
    cache_path.unlink(missing_ok=True)


def ranker_dataset_cache_dir() -> Path:
    """Where ranker_dataset.load_temporal_notes looks for cached notes."""
    from ranker_pipeline.stage3_train_ranker.ranker_dataset import STAGE1_CACHE
    return STAGE1_CACHE


def main():
    test_imports()
    test_formatting()
    test_bootstrap()
    test_dataset_with_synthetic_data()
    print("\nAll smoke tests passed.")


if __name__ == "__main__":
    main()
