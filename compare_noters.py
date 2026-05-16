"""3-way comparison: Video-only / Video+7B-note / Video+72B-note per task."""
import json, os

ORDER = [
    ("L1", "materials"), ("L1", "tools"), ("L1", "operation"), ("L1", "quantity"),
    ("L2", "sequence_generation"), ("L2", "sequence_ordering"),
    ("L2", "step_prediction"), ("L2", "video_verification"),
    ("L3", "experimental_conclusion"), ("L3", "scientific_discovery"),
]

# (label, path-template)  — None path means look it up as "{dir}/eval_{task}.json"
SOURCES = [
    ("video",     "results_h200/qwen7b/eval_{task}.json"),
    ("7B-note",   "results_h200_unified/c2/eval_{task}.json"),
    ("72B-note",  "results_h200_unified_q72/c2/eval_{task}.json"),
]


def load(template, task):
    f = template.format(task=task)
    if not os.path.exists(f):
        return None
    return json.load(open(f))


def main():
    rows = []
    print(f"{'Level':<4}{'Task':<26}{'video':>9}{'+7B note':>10}{'+72B note':>11}"
          f"{'Δ72-vid':>10}{'Δ72-7B':>10}")
    print("-" * 80)
    for level, task in ORDER:
        cells = [load(t, task) for _, t in SOURCES]
        if any(c is None for c in cells):
            avail = ", ".join(lbl for (lbl, _), c in zip(SOURCES, cells) if c)
            print(f"{level:<4}{task:<26}  (incomplete: have {avail})")
            continue
        v, n7, n72 = (c["accuracy"] for c in cells)
        rows.append((level, task, v, n7, n72, cells[0]["n_valid"]))
        print(f"{level:<4}{task:<26}{v:>8.2f}%{n7:>9.2f}%{n72:>10.2f}%"
              f"{n72-v:>+9.2f}{n72-n7:>+9.2f}")
    print("-" * 80)
    if rows:
        for lvl in ("L1", "L2", "L3", None):
            rs = rows if lvl is None else [r for r in rows if r[0] == lvl]
            if not rs:
                continue
            v_ = sum(r[2] for r in rs) / len(rs)
            n7_ = sum(r[3] for r in rs) / len(rs)
            n72_ = sum(r[4] for r in rs) / len(rs)
            tag = "macro" if lvl is None else lvl
            label = "avg (all)" if lvl is None else f"avg ({lvl})"
            print(f"{tag:<4}{label:<26}{v_:>8.2f}%{n7_:>9.2f}%{n72_:>10.2f}%"
                  f"{n72_-v_:>+9.2f}{n72_-n7_:>+9.2f}")


if __name__ == "__main__":
    main()
