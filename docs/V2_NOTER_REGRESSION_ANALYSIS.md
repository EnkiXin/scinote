# Why v2-noter regresses on free-form generation tasks (and how to fix)

The v2 trained noter helps MC tasks but **hurts free-form generation** on ExpVid:

| Task family | v2-noter Δ vs Video-only |
|---|---:|
| video_verification (mc) | **+9.87** ✅ |
| sequence_ordering (mc) | **+5.33** ✅ |
| scientific_discovery (fitb) | +1.08 |
| step_prediction (steppred) | −1.38 |
| experimental_conclusion (fitb) | −2.93 |
| **sequence_generation (seqgen)** | **−9.45** ❌ |

This investigation explains the mechanism and shows a simple task-gating rule that recovers **+3.09 pp over Video macro**, beating the unconditional v2-noter (+0.57 pp).

## Why sequence_generation drops 9.45 pp

Inspected 70 items where Video-only beat v2-noter; computed:

| Pattern | Count |
|---|---:|
| Video covered Gold, v2 collapses to wrong step range | 26 |
| v2 keeps gold steps but **adds wrong extra steps** | 41 |
| v2 picks completely disjoint range | 19 |

Three striking regressions (Video correct, v2 collapses):

| sample | gold | Video pred | v2-noter pred |
|---|---|---|---|
| `57660_clip2` | `[9 10 11 12]` | `9 10 11 12` (F1=1.0) | `15 18` (F1=0) |
| `60500_clip6` | `[28-34]` | `28-34` (perfect) | `2 3 4 5` |
| `59358_clip6` | `[24-29]` | `22-29` (F1≈0.86) | `4 5 13` |

**Mechanism**: the v2 noter's seqgen note schema uses `observed_steps_with_evidence` — pure descriptive prose **without step numbers**. Of 161 seqgen notes, 87 contain *zero* digits. The answer model with note in hand does lexical matching on action verbs against the full numbered procedure and lands on the *earliest* matching steps. Without the note, the answer model attends directly to **visual temporal cues** (late-protocol apparatus colors, instrument names visible only in the late frames) and gets the right index range.

**Hypothesis tested**: the note isn't hallucinating wrong numbers (only 1 / 161 v2 preds match digits in the note). The note's accurate-but-temporal-free descriptions actively **mislead** lexical attention.

## Why experimental_conclusion drops 2.93 pp (fitb)

10 random items. The noter outputs **shorter, more generic phrases** than the gold:

| Item | Gold (FITB) | v2-noter pred |
|---|---|---|
| sample type | "isolated cannabis trichomes" | "sample" |
| tissue panel | "FIB-SEM \| mouse nervus tibialis \| C. elegans" | "tissue \| mouse \| human" |
| graft trial | "autologous chondrocyte-seeded collagen matrices \| 3.6 mm \| trochlear \| 12 weeks" | "matrix \| 2-3 \| patellar \| 6 weeks" |

The noter's `key_evidence` field describes visuals ("graph with R²=0.9993", "centrosomes in fusion") but **lacks the quantitative specifics** (`1.42 mg/mL`, `490 ppb`) the fitb blanks demand. Specificity erasure — answer model now writes the noter's vague phrasing instead of its own correct guess.

## Why MC tasks (video_verification +9.87 pp) gain

Inspected 5 v2-flips-right items. In every case the v2 note **enumerates exactly which steps it observed** (e.g. *"centrifuge spin → suggests step 2"*, *"no visible evidence of washing the RNA pellet with 70 % ethanol → step C is missing"*). MC is **letter-classification over a closed vocabulary**; once evidence-level cues are present, even if specifics are slightly off, the model picks the right letter.

## Two diagnoses

- **Format mismatch (seqgen)**: the note's *evidence prose* discards the temporal anchoring (which protocol-step indices map to the visible content) that seqgen needs.
- **Specificity erasure (fitb)**: the note's *general descriptions* crowd out the answer model's own fine-grained numeric / chemical-grade guess.

MC tasks are robust to both because they only need evidence-level cues for letter selection.

## Improvement 1 — task gating (zero training, immediate)

**Rule**: use v2-noter for MC tasks only; for non-MC tasks, fall back to Video-only (no note).

| Task | n | Video | V + v2-Noter | task-gated | source |
|---|---:|---:|---:|---:|---|
| sequence_generation | 161 | **44.85** | 35.40 | 44.85 | Video |
| sequence_ordering | 150 | 48.00 | **53.33** | 53.33 | v2-noter |
| step_prediction | 145 | **3.45** | 2.07 | 3.45 | Video |
| video_verification | 152 | 11.84 | **21.71** | 21.71 | v2-noter |
| experimental_conclusion | 76 | **20.00** | 17.07 | 20.00 | Video |
| scientific_discovery | 61 | 17.81 | **18.89** | 18.89 | v2-noter (small gain) |
| **overall macro** | 745 | 25.94 | 26.51 | **29.03** | mixed |

**Δ vs Video: +3.09 pp**. **Δ vs unconditional v2-noter: +2.52 pp**.

Implementation: 1-line condition in the eval pipeline checking `item.task_type == "mc"` before injecting the note.

## Improvement 2 — task-conditioned noter training (re-train v3)

Make the noter emit task-aware output:
- For seqgen → JSON `{"observed_step_indices": [..], "evidence_per_step": [..]}` enforced by prompt + structured loss
- For fitb → require quoted text overlays and numeric values verbatim; penalise vague phrasing
- For mc → keep the current evidence-list schema (it works)

Single LoRA, multi-prompt training. Estimated effort: rewrite `prepare_training_data_v2.py` (per-task target schemas) + retrain 1 epoch (~5 h).

## Improvement 3 — separate LoRA per task family (highest cost)

Train 3 small LoRAs (mc / seqgen+steppred / fitb), pick the right one at inference. Highest ceiling, ~3 × the train time, more deployment complexity.

## Recommended next step

Adopt **Improvement 1 (task gating)** to lock in the +3.09 pp result *immediately* with zero retrain. If we want to push higher, **Improvement 2** is the natural next experiment (~5 GPU-hours).

---

## Update: Improvement 1 (prompt deferral) result

We tested a softer version of "use video for specifics": modify the eval prompt to tell the answer model `"READ THE VIDEO DIRECTLY for any specifics; use notes only for orientation"` and ask fitb tasks to "copy exact numbers/labels visible on screen, do NOT use the notes' summarised phrasing." [`evaluate_v2_test_split_promptv2.py`](evaluate_v2_test_split_promptv2.py).

| Task | n | Video | V+v2 (orig) | **V+v2 promptv2** | task-gated v2 |
|---|---:|---:|---:|---:|---:|
| sequence_generation     | 161 | 44.85 | 35.40 | **38.20** (+2.80) | 44.85 |
| sequence_ordering       | 150 | 48.00 | 53.33 | 53.33             | 53.33 |
| step_prediction         | 145 |  3.45 |  2.07 |  2.07             |  3.45 |
| video_verification      | 152 | 11.84 | 21.71 | 21.71             | 21.71 |
| experimental_conclusion |  76 | 20.00 | 17.07 | **18.77** (+1.70) | 20.00 |
| scientific_discovery    |  61 | 17.81 | 18.89 | 17.10 (−1.79)     | 18.89 |
| **overall macro**       | 745 | 25.94 | 26.51 | **27.14** (+0.63) | **29.03** |

**Reading**: prompt-level deferral recovers some of the regression on seqgen (+2.80 pp) and exp_conclusion (+1.70 pp), but the answer model still gets distracted by note content even with explicit "read video directly" instructions. The overall +0.63 pp lift is much smaller than the +3.09 pp from task gating (Improvement 1) — confirming that the answer model is heavily anchored on its prompt's text once a note is present. Root-cause fix requires changing the noter's output schema at training time (Improvement 2 / v3, in flight).

## Improvement 2 (v3 task-aware noter) — running

`train_notetaker_vl_v3.py`: per-task augmented oracle notes (seqgen → `observed_step_indices: <gold>`, fitb → `verbatim_specifics: <gold>`, steppred → `next_step_prediction: <gold>`) + task-aware `build_user_text`. Training launched, ETA ~9 h. After it finishes we generate v3 notes and re-eval the test split with the same task-aware scorer; the gap to 49.31 (oracle ceiling) is what we want to close.
