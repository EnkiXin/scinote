"""
check_oracle_leak.py — Are 72B oracle notes leaking the answer?

Three checks on the SciVideoBench oracle cache:

  1. Static lexical leak:
      - standalone gold letter (\\b[A-J]\\b) appearing in note
      - verbatim substring match of the gold option text in note
      - high n-gram overlap of gold option text vs note (could be paraphrased leak)
  2. Distinctive-keyword leak: words that appear in the gold option but NOT in any
     other option, found in the note. Score = #distinctive gold words appearing
     / #total distinctive gold words.
  3. Mention-rate of competing-option distinctive words vs gold: if note mentions
     gold's distinctive words much more than competitors', that's a leak.

Writes summary + flagged examples to analysis_oracle_leak/.
"""

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from collections import Counter

ANN_PATH = "/home/yz0392@unt.ad.unt.edu/xin_ai/scivideobench/scivideobench_1k.jsonl"
CACHE_DIR = "/home/yz0392@unt.ad.unt.edu/xin_ai/scivideobench/results_scivideobench/oracle_notes"
OUT_DIR = Path("/home/yz0392@unt.ad.unt.edu/xin_ai/scinote/analysis_oracle_leak")
OUT_DIR.mkdir(exist_ok=True, parents=True)

STOPWORDS = set("""
a an the of and to in on at by for with as is are was were be been being do does did
this that these those it its their they them his her he she we i you your our my
or but nor so yet if then than when where which what who why how to from out into of
""".split())


def tokens(s):
    return [w.lower() for w in re.findall(r"[a-zA-Z][a-zA-Z]+", s or "")]


def content_words(s):
    return [w for w in tokens(s) if w not in STOPWORDS and len(w) > 2]


def load_oracle(vid, qid):
    key = f"{vid}|{qid}"
    p = Path(CACHE_DIR) / (hashlib.md5(key.encode()).hexdigest()[:16] + ".json")
    if not p.exists(): return None
    return json.load(open(p)).get("note", "")


def main():
    with open(ANN_PATH) as f:
        items = [json.loads(l) for l in f if l.strip()]

    n_total = 0
    n_letter_leak = 0
    n_verbatim_leak = 0
    n_high_paraphrase = 0
    distinctive_scores = []
    leaks_flagged = []

    for it in items:
        note = load_oracle(it["video_id"], it["question_id"])
        if not note:
            continue
        n_total += 1
        gold = it["answer"]
        opts = it["options"]
        gold_text = opts.get(gold, "")

        # 1a. standalone gold letter
        letter_pat = rf"\b{re.escape(gold)}\b"
        letter_hits = re.findall(letter_pat, note)
        # Excludes mentions inside JSON keys; check after stripping quotes
        # but more importantly skip A-J inside json keys / unlikely.
        # Just count crude mentions.
        if letter_hits:
            n_letter_leak += 1
            if len(leaks_flagged) < 30:
                leaks_flagged.append({"reason": "letter", "item": it,
                                       "note": note, "hits": letter_hits})

        # 1b. verbatim option text substring
        if gold_text and gold_text.lower() in note.lower():
            n_verbatim_leak += 1
            if len(leaks_flagged) < 30:
                leaks_flagged.append({"reason": "verbatim", "item": it, "note": note})

        # 2. distinctive keyword overlap
        gold_words = set(content_words(gold_text))
        other_words = set()
        for k, v in opts.items():
            if k == gold: continue
            other_words.update(content_words(v))
        distinctive = gold_words - other_words
        if distinctive:
            note_words = set(content_words(note))
            hits = distinctive & note_words
            score = len(hits) / max(len(distinctive), 1)
            distinctive_scores.append(score)
            if score > 0.5:
                n_high_paraphrase += 1
                if len(leaks_flagged) < 30:
                    leaks_flagged.append({
                        "reason": f"distinctive {len(hits)}/{len(distinctive)}",
                        "item": it, "note": note,
                        "distinctive_gold": sorted(distinctive),
                        "hits": sorted(hits),
                    })

    # competing-option mention-rate
    competing_dist_scores = []
    for it in items:
        note = load_oracle(it["video_id"], it["question_id"])
        if not note: continue
        gold = it["answer"]; opts = it["options"]
        note_words = set(content_words(note))
        per_opt_score = {}
        for k, v in opts.items():
            gold_words = set(content_words(v))
            other_words = set()
            for k2, v2 in opts.items():
                if k2 == k: continue
                other_words.update(content_words(v2))
            dist = gold_words - other_words
            if dist:
                per_opt_score[k] = len(dist & note_words) / len(dist)
            else:
                per_opt_score[k] = 0.0
        sorted_keys = sorted(per_opt_score, key=lambda k: -per_opt_score[k])
        # is the gold ranked #1 by distinctive-word coverage?
        rank_of_gold = sorted_keys.index(gold) + 1
        competing_dist_scores.append((rank_of_gold, per_opt_score[gold],
                                       max(per_opt_score[k] for k in opts if k != gold)
                                       if len(opts) > 1 else 0))

    print(f"=== Oracle leak audit (n={n_total} notes) ===\n")
    print(f"1a. Standalone gold-letter leak: {n_letter_leak} / {n_total} "
          f"({n_letter_leak/n_total*100:.2f}%)")
    print(f"1b. Verbatim option-text substring leak: {n_verbatim_leak} / {n_total} "
          f"({n_verbatim_leak/n_total*100:.2f}%)")
    print(f"2.  High-distinctive-word coverage (>50%): {n_high_paraphrase} / {n_total} "
          f"({n_high_paraphrase/n_total*100:.2f}%)")
    if distinctive_scores:
        import statistics
        print(f"    distinctive-word coverage mean = {statistics.mean(distinctive_scores)*100:.2f}%, "
              f"median = {statistics.median(distinctive_scores)*100:.2f}%, "
              f"95th pct = {statistics.quantiles(distinctive_scores, n=20)[-1]*100:.2f}%")
    if competing_dist_scores:
        # rank of gold among options by distinctive-coverage
        rank_counts = Counter(t[0] for t in competing_dist_scores)
        n = len(competing_dist_scores)
        print(f"\n3.  Where does the gold option rank vs competitors by distinctive-coverage?")
        for r in sorted(rank_counts):
            print(f"      rank {r}: {rank_counts[r]:>4} / {n} "
                  f"({rank_counts[r]/n*100:.1f}%)")
        gold_avg = sum(t[1] for t in competing_dist_scores) / n
        comp_avg = sum(t[2] for t in competing_dist_scores) / n
        print(f"      mean coverage: gold = {gold_avg*100:.2f}%, "
              f"best competitor = {comp_avg*100:.2f}%, "
              f"gap = {(gold_avg-comp_avg)*100:+.2f} pp")

    # write flagged examples
    with open(OUT_DIR / "flagged_examples.json", "w") as f:
        json.dump(leaks_flagged[:30], f, indent=2, default=str)
    print(f"\n💾 flagged examples → {OUT_DIR}/flagged_examples.json")


if __name__ == "__main__":
    main()
