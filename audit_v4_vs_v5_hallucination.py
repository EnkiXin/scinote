"""audit_v4_vs_v5_hallucination.py — qualitative comparison of v4a vs v5a
student-noter outputs on 20 sampled test items.

For each sample, prints:
  - question + gold answer + task_type
  - v4a generated note (from results_v4_split/v4a_noter_notes)
  - v5a generated note (from results_v4_split/v5a_noter_notes)

Saves report to V5_HALLUCINATION_AUDIT.md. Use this to manually assess:
  - Does v4a emit structured fields (verbatim_specifics etc.) with
    plausible-but-hallucinated content?
  - Does v5a use cleaner prose with frame-anchored cues that are visible?
  - On which task_types does each fail / succeed?
"""
import argparse, hashlib, json, random
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def load_note(notes_dir: Path, sample_id: str, benchmark: str):
    safe = hashlib.md5(sample_id.encode()).hexdigest()[:16] + ".json"
    p = notes_dir / benchmark / safe
    if not p.exists():
        return None
    try:
        return json.load(open(p)).get("note")
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=20260520)
    ap.add_argument("--out", default="V5_HALLUCINATION_AUDIT.md")
    args = ap.parse_args()
    random.seed(args.seed)

    test_items = [json.loads(l) for l in open(ROOT / "train_data" / "v5_split_test.jsonl")]
    # Spread sampling across task_types
    by_tt = {}
    for it in test_items:
        by_tt.setdefault(it.get("task_type", "?"), []).append(it)
    picks = []
    per_tt = max(1, args.n // len(by_tt))
    for tt, lst in by_tt.items():
        random.shuffle(lst)
        picks.extend(lst[:per_tt])
    picks = picks[:args.n]

    v4_dir = ROOT / "results_v4_split" / "v4a_noter_notes"
    v5_dir = ROOT / "results_v4_split" / "v5a_noter_notes"

    lines = ["# v4a vs v5a hallucination audit — qualitative side-by-side\n"]
    lines.append(f"Sampled {len(picks)} items from v5_split_test (seed={args.seed}, "
                 f"per-task-type quota={per_tt}).\n")
    lines.append("Compare:\n")
    lines.append("- **v4a** = MiMo-VL-7B-RL + LoRA trained on v4 task-aware oracle "
                 "(per_option_evidence / observed_step_indices / verbatim_specifics / "
                 "fill_in_index structured fields)\n")
    lines.append("- **v5a** = MiMo-VL-7B-RL + LoRA trained on v5 statement-grounded oracle "
                 "(unified `supporting_cues` with embedded frame ranges; no per-task schema)\n\n")
    lines.append("---\n")

    for i, it in enumerate(picks, 1):
        sid = it["sample_id"]; bench = it["benchmark"]; tt = it.get("task_type", "?")
        task = it.get("task", "?")
        v4 = load_note(v4_dir, sid, bench)
        v5 = load_note(v5_dir, sid, bench)
        lines.append(f"## Example {i} — `{task}` ({tt})\n")
        lines.append(f"- benchmark: `{bench}`")
        lines.append(f"- sample_id: `{sid[:80]}{'...' if len(sid) > 80 else ''}`")
        q = (it.get("question", "") or "")[:400]
        lines.append(f"- question: {q}{'…' if len(q) >= 400 else ''}")
        if it.get("options"):
            opts_text = "\n".join(f"  {k}. {str(v)[:120]}" for k, v in sorted(it["options"].items()))
            lines.append(f"- options:\n```\n{opts_text}\n```")
        lines.append(f"- gold: `{it.get('gold') or it.get('answer')}`\n")
        lines.append("**v4a note:**\n")
        lines.append("```json\n" + (v4[:1500] if v4 else "(no v4a note)") + "\n```\n")
        lines.append("**v5a note:**\n")
        lines.append("```json\n" + (v5[:1500] if v5 else "(no v5a note)") + "\n```\n")
        lines.append("---\n")

    out = ROOT / args.out
    out.write_text("\n".join(lines))
    print(f"wrote {out} ({len(picks)} examples)")


if __name__ == "__main__":
    main()
