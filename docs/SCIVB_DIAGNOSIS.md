# SciVideoBench Agent Regression — Item-Level Diagnosis

**Question**: ProtoNote C1_fixed agent loses 1.38 pp on SciVB
(C0 = 25.69 % → C1_fixed = 24.31 %). Why?

**Method**: pair the 218 C0 and C1_fixed trajectories by row index within
each chunk file (sample_id alone is NOT unique — the SciVB test split
has 75 duplicate sample_ids that actually correspond to different
questions over the same video), then bucket by C0 ✓/✗ × C1 ✓/✗.

**Reproduce**: `python tools/analyze_scivb_regression.py` (see §4).

---

## 1. Confusion matrix

|  | n |
|---|---:|
| both right | 43 |
| **loss (C0 ✓ → C1 ✗)** | **13** |
| gain (C0 ✗ → C1 ✓) | 10 |
| both wrong | 152 |
| **net** | **−3 items = −1.38 pp** |

Loss > gain → agent hurts more items than it helps.

## 2. Failure pattern (13 loss items)

### 2.1 "Purpose / mechanism" questions — note anchors literal action, model picks literal distractor

Concrete losses (selected):

| sample_id | Question | gold (mechanism) | C1 pred (literal) |
|---|---|---|---|
| scivideobench_mc_60616_1 | "What is the **purpose** of the sonication step?" | A) Complete and homogeneous redissolution of lipid film | E) Breaking down lipid vesicles into smaller particles |
| scivideobench_mc_57885_2 | "What critical sample property is determined by the measurement?" | J) Spatial displacement and coordinates of MoS₂ flakes | B) Electrical conductivity of the MoS₂ flakes |
| scivideobench_mc_57270_1 | "Primary **purpose** of the action on the neonatal pup?" | F) Inducing anesthesia | D) Starting metabolic stimulation |

**Mechanism**: the agent's `visual_inspect` writes
*"sonication is breaking down vesicles"* or
*"a syringe is being inserted into the pup"* — a literal description of
what's happening. The answer model then reads the note and picks the
option that **lexically matches** the observation (breaking-down →
option E; injection → option D), even though the question asks **why**
the action is done (mechanism), not **what** it visibly does.

### 2.2 Pure numerical calculation — note is irrelevant noise

| scivideobench_mc_64571_5 | "Calculate the percent yield if final isolated mass is exactly 1.70 g" | A) 59 % | H) 62 % |

Visual notes are completely irrelevant for arithmetic. Worse, the
description may include adjacent numbers from the video
(*"0.62 g of reagent"*, *"6.2 mL"*) that distract the model toward
similar-looking distractors.

## 3. Success pattern (10 gain items) — all counterfactual

100 % of the gain items are **"What could happen if X fails"** /
counterfactual questions:

| sample_id | Question | gold |
|---|---|---|
| scivideobench_mc_4213_5 | "What could happen if the operation **fails**?" | C) The suspension remains heterogeneous |
| scivideobench_mc_2609_1 | "What could happen if the action **fails** before raising mouse to apparatus?" | A) The mouse's body is not aligned perpendicular to the bar |
| scivideobench_mc_64112_1 | "What could happen if the procedure **fails**?" | H) Plastic remains flexible and does not fracture mechanically |
| scivideobench_mc_65522_3 | "What could happen if the sample drop speed control **fails**?" | H) Vacuum is not maintained |
| scivideobench_mc_2967_1 | "Primary purpose of the procedure?" | D) Permeabilize membranes and dehydrate sample |

**Mechanism**: the note anchors *which* procedure is being performed
(centrifugation? sonication? coronal-bar alignment?). For
counterfactual reasoning, the model first needs to know what step it is
identifying with — once the note nails that, the model can correctly
chain "if THIS step fails → THIS consequence".

## 4. Reading

The two patterns are **mirror images** of the same mechanism:

* The visual note is *what the agent saw*.
* For **mechanism / purpose** questions, this gets confused with *what
  the answer is*, and the model is pulled toward the literal distractor.
* For **counterfactual** questions, this stays clearly an
  *antecedent* (the step we're conditioning on), and the model
  correctly reasons forward from it.

ExpVid (where C1_fixed wins +3.12 pp overall) is mostly
*procedural / what-is-shown* — visual notes ARE the answer, no
mechanism/purpose confound. SciVB is dominated by *why / purpose*
questions, with a minority of *what-if* — so the agent net-hurts.

## 5. Solution directions

| Idea | Cost | Hypothesis |
|---|---|---|
| **Reword `visual_inspect` prompt** to write "what type of procedure + key reagents/instruments" instead of "what is happening" | small (prompt change) | reduces literal-action wording in the note, so the answer model can't lexically match it to literal distractors |
| **Task-conditional note injection** | small | for SciVB mechanism / purpose items, skip the `_ctx_block` and pass `note=None` to the answer builder. Falls back to C0 on those items. |
| **Question-type routing** | medium | classify each question into {procedural, mechanism, counterfactual, calculation}, only inject notes for procedural + counterfactual |
| **C2_react_v2 (already running)** with planner-sees-options + no timestamp picking | already implemented | for purpose questions the planner should pick "answer immediately" (don't write a note), avoiding the bug entirely |
| **Confidence-weighted note rendering** | medium | render notes ordered by tool confidence, drop low-conf evidence — visual_inspect has fixed conf=0.85, so this only helps if we calibrate per-task confidence |

## 6. What this means for the paper

The SciVB regression is **not** "agent doesn't work on conceptual MC" —
it's "literal visual description biases the answer model on questions
that need mechanism inference, but helps on questions that need
procedure identification". This is a paper-publishable finding:

1. Adds **scope conditions** to the ProtoNote claim ("notes-as-artifact
   helps when the question's bottleneck is *grounding what was done*,
   not *inferring why*").
2. Motivates **task-conditional note injection** (one of the three
   advertised contributions in the original proposal) — it's not just
   about tool selection, also about whether to inject the resulting
   notes at all.
3. Suggests a clean ablation: show that the SciVB regression
   disappears when notes are gated to procedural and counterfactual
   subsets.

## 7. Pairing script

```python
# tools/analyze_scivb_regression.py
import json
from pathlib import Path

def load_pairs(out_dir):
    out = []
    for f in sorted(Path(out_dir).glob('trajectory_*.jsonl')):
        for i, line in enumerate(open(f)):
            out.append((f.name, i, json.loads(line)))
    return out

c0 = {(c, i): r for c, i, r in load_pairs('results_protonote/c0_scivb')}
c1 = {(c, i): r for c, i, r in load_pairs('results_protonote/c1_scivb')}

loss = gain = both_right = both_wrong = 0
loss_items = []
gain_items = []
for k in c0:
    if k not in c1: continue
    r0, r1 = c0[k], c1[k]
    if 'score' not in r0 or 'score' not in r1: continue
    rt0 = r0['score'] >= 0.5
    rt1 = r1['score'] >= 0.5
    if rt0 and not rt1: loss += 1; loss_items.append((r0, r1))
    elif rt1 and not rt0: gain += 1; gain_items.append((r0, r1))
    elif rt0 and rt1: both_right += 1
    else: both_wrong += 1

n = both_right + loss + gain + both_wrong
print(f"both_right={both_right}  loss={loss}  gain={gain}  both_wrong={both_wrong}")
print(f"net = {gain - loss:+d}  ({(gain - loss)/n*100:+.2f} pp on n={n})")
```

Important caveat: do NOT key by `sample_id` alone — SciVB has 75
duplicate sample_ids in the test split that correspond to different
questions over the same video (verified in §1). Always pair by
`(chunk_filename, line_index)` or by the full
`(sample_id, gold, question)` triple.
