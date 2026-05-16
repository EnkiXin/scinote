# When does the 72B note HELP, when does it HURT?

Distilled from `qa_notes_<task>.csv` and `note_helpfulness_samples.md` — per-item
strict (0/1) outcome flip between **Video only** and **Video + 72B note** with
the **Qwen-7B answer model**.

## Per-task rescue/break count (n=10 ExpVid tasks)

| Task | n | **rescue** | **break** | net | rescue% | break% |
|---|---:|---:|---:|---:|---:|---:|
| materials               | 1266 | **150** |  85 | **+65** ✅ | 11.8% |  6.7% |
| tools                   | 1130 | **109** |  98 | +11 |  9.6% |  8.7% |
| operation               |  938 |  54 | **106** | **−52** ❌ |  5.8% | 11.3% |
| quantity                |  701 |  47 |  **95** | **−48** ❌ |  6.7% | 13.6% |
| sequence_generation     |  750 |  20 |  25 | −5 |  2.7% |  3.3% |
| sequence_ordering       |  739 |  78 |  56 | +22 ✅ | 10.6% |  7.6% |
| step_prediction         |  748 |  10 |  11 | −1 |  1.3% |  1.5% |
| video_verification      |  748 |  71 |  46 | **+25** ✅ |  9.5% |  6.1% |
| experimental_conclusion |  390 |   1 |   0 | (fitb, no 0/1 flips) |
| scientific_discovery    |  390 |   0 |   0 | (fitb, no 0/1 flips) |

(*rescue* = Video wrong → V+72B right;   *break* = Video right → V+72B wrong)

L3 fitb tasks use partial-credit scoring (0.17 / 0.20 / 0.25 etc.), so very
few items cross the strict 0/1 boundary. The L3 net positive **+1.37 pp**
(see [PROGRESS.md Finding 6](../PROGRESS.md#6-stage-1-noter-7b--72b-only-the-upgrade-itself-helps-absolute-v72bnote--video))
comes from partial-credit improvements, not 0→1 flips.

---

## Patterns — when notes HELP (rescue)

Reading the rescue cases, three patterns recur. Examples below are taken
verbatim from `note_helpfulness_samples.md`.

### Pattern H-1 — Note nails the action verb the video model couldn't extract (operation rescues)

Operation MCQs ask "what is the person doing with X?". Video alone is often
ambiguous between several plausible verbs. When the note happens to print the
exact action verb the question hinges on, it tips the answer over.

> *clip 61760/22* — Q: "What is the person doing with the centrifuge tubes?"
> Gold **B** "Placing into the centrifuge rotor". Video picked C ("Closing the
> centrifuge lid").
> **72B note**: `actions_performed: ["place the rotor into the centrifuge", "close the centrifuge lid", "press the centrifuge lid"]`
> → V+Note correctly outputs **B**: the note enumerates both candidates but the
> "place into the centrifuge" phrasing matches B's wording most directly.

> *clip 61013/10* — Q: "What is the person doing with the wafer immediately
> after it is on the hotplate?" Gold **C** "Transferring it off the hotplate".
> Video picked A ("Adjusting its position").
> **72B note**: `actions_performed: ["lift the wafer from the surface", "hold the wafer with tweezers"]`
> → V+Note → **C**. "Lift from" maps cleanly to "transfer off".

### Pattern H-2 — Note's structured object/keyword tag disambiguates close-by options (materials, tools)

When the question is "which material/tool is it?" with options that are
visually similar (e.g. *spinal cord vs CSF*, *orange wire vs pink wire*),
notes that list **labels visible on equipment** or **specific object words**
nudge the model.

> *clip 55098/34* — Q: "What material appears in this procedure?"
> Options: A. enameled copper wire / B. orange wire / C. pink wire / D. green wire.
> Gold **B**. Video picked A.
> **72B note**: `materials_visible: ["wires", "circuit board", "soldering iron"]` (color not in the note text but the model + the visual context anchors better).
> → V+Note → **B** (correct).

> *clip 57847/39* — Q about MOX vs DMSO vs MOPS / acetonitrile.
> **72B note**: `labels_seen: ["MULTI-THERM", "Benchmax"]` — the model evidently
> uses the **brand label** (MULTI-THERM = thermal cycler) as context to anchor
> the chemistry vocabulary.

### Pattern H-3 — Note enumerates "what WAS shown" → useful for "what was NOT done" (video_verification +25)

video_verification asks the model to identify which option does NOT appear in
the video. A note that lists the *steps actually shown* is a positive list
the model can take set complement against.

This is also why video_verification has the highest single-task absolute Δ
over Video baseline (+3.34 pp, see [Finding 6](../PROGRESS.md#6)). The note
schema for video_verification (`steps_actually_shown`, `missing_or_skipped`,
`key_objects_handled`) is *specifically structured* for this kind of question.

---

## Patterns — when notes HURT (break)

### Pattern B-1 — Note describes a different aspect than the question targets (operation, quantity)

This is the dominant failure on **operation** (break 106 > rescue 54, net **−52**).
Operation questions ask about a *specific action* in a clip with many actions;
the note enumerates *all* actions, sometimes emphasising the wrong one.

> *clip 58743/5* — Q: "What is the person doing with the spice grinder?"
> Gold **C** "Grinding propolis into a fine powder". Video alone → **C** ✓.
> **72B note**: `actions_performed: ["unscrews the lid of a container", "removes the lid from the container"]`
> → V+Note → **D** "Opening the grinder to check particle size".
> The note documents the *next* step (opening), not the grinding question being asked. Model trusts the note's specific action and answers about opening.

> *clip 56195/34* — Q: "What is the person doing with the pipette tip?" Gold
> **A** Aspirating, Video → A ✓.
> **72B note**: `tool_object_interaction: ["pipette tip is inserted into the test tube"]`
> → V+Note → **B** Dispensing. The note says "inserted into" → model reads
> direction-of-flow as "into" = dispensing. Aspirate / dispense are visually
> ambiguous; the note's phrasing flipped it.

### Pattern B-2 — Note misidentifies the salient entity (materials)

> *clip 53009/33* — Q: "Which material appears in this experimental step?"
> Options include mice / rats / guinea pigs / zebrafish. Gold **B** mice.
> Video → B ✓.
> **72B note**: `materials_visible: ["rats"]`
> → V+Note → C "rats". The note explicitly labels the wrong species; the
> answer model defers to the note's text rather than re-reading the video.

> *clip 53009/24* — Q: about disinfecting cleaning. Gold **B** "Cleaning with
> disinfecting detergent". Video → B ✓.
> **72B note**: `actions_performed: ["wipes the surface of the biosafety cabinet"], objects_manipulated: ["brown paper towel"]`
> → V+Note → C "Wiping with a dry cloth". The note's framing as
> "wipe surface with paper towel" elides the detergent entirely; model now
> chooses the option that matches the *note*, not the *video*.

### Pattern B-3 — Note's generic listing dilutes specific cues (quantity)

> *clip 59728/24* — Q: "Which material appears in this experimental step?"
> Options: distilled / ethanol / deionized / PBS. Gold **C** deionized water.
> Video → C ✓.
> **72B note**: `materials_visible: ["water"]` (no specifier).
> → V+Note → A distilled water. With the note saying only "water", the model
> falls back on the most generic default (distilled), losing the deionized
> specificity that the video originally provided.

---

## Why operation / quantity hurt most (−52 / −48 net)

These two tasks share a structural property:

* Options are **visually very close** (aspirate vs dispense; weighing vs
  transferring; mice vs rats; distilled vs deionized water).
* The note schema (`actions_performed`, `objects_manipulated`,
  `materials_visible`) is **lossy on direction-of-flow / species-level
  / chemical-grade specificity** — a JSON note can faithfully say
  "pipette liquid into tube" without distinguishing "drawing in" from
  "pushing out", and that one bit is exactly what the question asks.

In contrast, **video_verification** and **sequence_ordering** ask
*set-level* questions (which step is missing? in what order?) — for these,
the note's enumerative structure is a *gain*: the note lists items, the
question tests membership / order. The note's compression is well-aligned
with the question's information requirement.

---

## Implications

1. **Note schema must match question information requirement.** Generic
   "describe everything" notes hurt when questions depend on a single fine
   distinction (flow direction, species, chemical grade). Phase 3
   (note-variant ablation: `taskaware` / `generic` / `minimal`) will test
   whether even more task-tailored prompts make the gap smaller.

2. **Model over-trusts notes.** When the note's stated action conflicts with
   the video, the answer model frequently follows the note. This argues for
   either (a) a counterfactual SFT pass that teaches the model to verify
   notes against the video before deferring, or (b) confidence-gated note
   usage where low-confidence notes are dropped.

3. **video_verification is the clearest win for note augmentation** (+3.3 pp
   over Video). When the next experiment line searches for cases where
   note-augmentation generalizes outside ExpVid (e.g. the SciVideoBench
   transfer experiment), video_verification-style "what was NOT done"
   questions are the natural template to test first.
