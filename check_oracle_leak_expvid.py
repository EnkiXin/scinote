"""
check_oracle_leak_expvid.py — Static lexical leak check on ExpVid 72B oracle notes.
Mirrors the SciVideoBench version but iterates 10 tasks (L2+L3 actually
generated).
"""
import hashlib
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate_unified import TASKS, REPO_ID
from huggingface_hub import hf_hub_download

CACHE = Path("/home/yz0392@unt.ad.unt.edu/xin_ai/scinote/results_h200_unified/oracle_notes")
OUT = Path("/home/yz0392@unt.ad.unt.edu/xin_ai/scinote/analysis_oracle_leak/expvid_leak_static.txt")
OUT.parent.mkdir(exist_ok=True)

STOPWORDS = set("a an the of and to in on at by for with as is are was were be been being do does did this that these those it its their they them his her he she we i you your our my or but nor so yet if then than when where which what who why how to from out into".split())


def tokens(s):
    return [w.lower() for w in re.findall(r"[a-zA-Z][a-zA-Z]+", s or "")]


def content(s):
    return [w for w in tokens(s) if w not in STOPWORDS and len(w) > 2]


def main():
    out_lines = []
    def p(line=""):
        out_lines.append(line); print(line)
    overall = {"n": 0, "letter": 0, "verbatim": 0, "high_paraphrase": 0}
    for task in ["sequence_generation", "sequence_ordering", "step_prediction",
                 "video_verification", "experimental_conclusion", "scientific_discovery"]:
        ann, task_type = TASKS[task]
        try:
            local = hf_hub_download(repo_id=REPO_ID, filename=ann, repo_type="dataset")
            items = [json.loads(l) for l in open(local) if l.strip()]
        except Exception as e:
            p(f"{task}: cannot load annotations: {e}"); continue
        n = letter = verbatim = high_para = 0
        para_scores = []
        for it in items:
            key = f"{it['video_path']}|{it.get('id','')}"
            cache_file = CACHE / task / (hashlib.md5(key.encode()).hexdigest()[:16] + ".json")
            if not cache_file.exists():
                continue
            try:
                note = json.load(open(cache_file)).get("note", "")
            except Exception:
                continue
            n += 1
            gold_raw = it.get("answer")
            if isinstance(gold_raw, str) and len(gold_raw) == 1 and gold_raw.upper() in "ABCD":
                # MC task with letter gold
                if gold_raw != "A":  # skip A=article false positives
                    if re.search(rf"\b{gold_raw}\b", note):
                        letter += 1
                opts = it.get("options", {})
                gold_text = opts.get(gold_raw, "")
                if gold_text and gold_text.lower() in note.lower():
                    verbatim += 1
                gold_words = set(content(gold_text))
                other_words = set()
                for k, v in opts.items():
                    if k == gold_raw: continue
                    other_words.update(content(v))
                dist = gold_words - other_words
                if dist:
                    nw = set(content(note))
                    score = len(dist & nw) / len(dist)
                    para_scores.append(score)
                    if score > 0.5:
                        high_para += 1
            else:
                # fitb / seqgen / steppred: gold is a list or string of answer tokens
                gold_str = " ".join(map(str, gold_raw)) if isinstance(gold_raw, list) else str(gold_raw)
                if gold_str and gold_str.lower() in note.lower():
                    verbatim += 1
                gold_words = set(content(gold_str))
                if gold_words:
                    nw = set(content(note))
                    score = len(gold_words & nw) / len(gold_words)
                    para_scores.append(score)
                    if score > 0.5:
                        high_para += 1
        mean_para = (sum(para_scores)/len(para_scores)*100) if para_scores else 0
        p(f"{task:<25s} n={n:>4}  letter={letter:>3} ({letter/max(n,1)*100:.2f}%)  "
          f"verbatim={verbatim:>3} ({verbatim/max(n,1)*100:.2f}%)  "
          f"high_para={high_para:>3} ({high_para/max(n,1)*100:.2f}%)  "
          f"mean_para={mean_para:.1f}%")
        overall["n"] += n; overall["letter"] += letter
        overall["verbatim"] += verbatim; overall["high_paraphrase"] += high_para

    n = overall["n"]
    p()
    p(f"OVERALL  n={n}  letter={overall['letter']/n*100:.2f}%  "
      f"verbatim={overall['verbatim']/n*100:.2f}%  "
      f"high_para={overall['high_paraphrase']/n*100:.2f}%")

    OUT.write_text("\n".join(out_lines))
    p(f"\n💾 saved → {OUT}")


if __name__ == "__main__":
    main()
