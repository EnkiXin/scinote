"""oracle_prompts_v5.py — Statement-grounded evidence-retrieval oracle prompts
for the paper-1 extension's v5 oracle redesign.

Key design change from v4:
  v4 used four task-specific structured schemas (mc / seqgen / steppred / fitb)
  with fields like `per_option_evidence`, `observed_steps[].step_index`,
  `verbatim_specifics`, `fills[].fill_in_index`. v3 / v4 student noters learned
  to emit those structured fields at 100% coverage but hallucinated plausible
  content, producing worse fitb scores than v2's prose. v4 also exposes option
  letter structure to the oracle (answer-leak risk).

  v5 collapses (question, gold_answer) into a single declarative `<STATEMENT>`
  and asks the 72B oracle to "find visible cues in the video that support the
  statement". A single system + user prompt is used across all 4 task types;
  the ONLY per-task code is the statement builder. Frame ranges are embedded
  inline inside cue strings (no separate structured frame field for the student
  to be forced to hallucinate).
"""
from __future__ import annotations

import re

# ── System prompt: identical for every task type ───────────────────────────────
ORACLE_SYSTEM_V5 = (
    "You are a careful, precise observer of scientific experiment videos.\n"
    "You will be shown a video and a STATEMENT about its content. The statement\n"
    "is known to be true.\n\n"
    "Your task: identify the specific visual cues in the video that support\n"
    "the statement. Describe ONLY what is actually visible in the video frames.\n\n"
    "STRICT CONSTRAINTS:\n"
    "  - Only describe content that is actually visible in the video.\n"
    "  - Do NOT mention answer letters (A, B, C, ...) or option positions.\n"
    "  - Do NOT copy answer text verbatim if it appears in the statement; paraphrase.\n"
    "  - Do NOT include speculation that is not grounded in visible evidence.\n"
    "  - Embed approximate frame ranges (e.g. 'frame 0-5') inside each cue,\n"
    "    using frame indices from the 32 sampled frames.\n"
    "  - Output ONLY valid JSON, no extra text or markdown fences."
)

# ── User template: identical for every task type ───────────────────────────────
ORACLE_USER_TEMPLATE_V5 = (
    "Statement: {statement}\n\n"
    "Find the visual cues in the video that support this statement.\n\n"
    "Output ONLY this JSON:\n"
    "{{\n"
    '  "supporting_cues": [\n'
    "    \"specific visible cue with embedded frame range, e.g. 'frame 5-9: pipette transferring liquid into well containing clear medium'\"\n"
    "  ],\n"
    '  "salient_objects_or_text": [\n'
    "    \"distinctive labels, numbers, or objects visible on screen\"\n"
    "  ]\n"
    "}}"
)


# ── Statement builders ─────────────────────────────────────────────────────────

def build_statement_sequence_ordering(item: dict) -> str:
    """For sequence_ordering: gold option's full text becomes the declarative."""
    gold = str(item.get("answer") or item.get("gold") or "").strip()
    options = item.get("options", {})
    if gold not in options:
        raise ValueError(f"sequence_ordering: gold {gold!r} not in options {list(options)}")
    return (
        "The correct sequence of steps for the procedure is:\n"
        f"{options[gold]}"
    )


def build_statement_video_verification(item: dict) -> str:
    """For video_verification: the gold option is the step that is NOT shown.
    The annotation has `removed_step: {step_number, action, ...}` giving the
    full step text directly. `options[gold]` is just the step index (e.g. "1"),
    not the text — so we use removed_step for the statement.

    We frame the statement so that the oracle's cue-finding task becomes
    "find positive evidence that the OTHER steps ARE present" — easier to
    ground than searching for the absence of a specific step.
    """
    removed = item.get("removed_step")
    if isinstance(removed, dict) and "action" in removed:
        step_num = removed.get("step_number", "?")
        step_text = removed["action"]
        missing_line = f"  {step_num}. {step_text}"
    else:
        # Fallback: options[gold] is the step index; look up in segment_step_list
        gold = str(item.get("answer") or item.get("gold") or "").strip()
        options = item.get("options", {})
        if gold not in options:
            raise ValueError(f"video_verification: gold {gold!r} not in options {list(options)}")
        step_idx = options[gold]
        seg = item.get("segment_step_list", []) or []
        step_text = next(
            (s.get("action", "") for s in seg if str(s.get("step_number", "")) == str(step_idx)),
            ""
        )
        missing_line = f"  {step_idx}. {step_text}" if step_text else f"  step {step_idx}"
    return (
        "The step that is NOT shown in the video is:\n"
        f"{missing_line}\n\n"
        "All OTHER steps in the procedure (not listed above) ARE shown in the video."
    )


def _step_list_text(item: dict) -> list[tuple[int, str]]:
    """Return list of (step_index, step_text) for either seqgen or steppred.

    - seqgen rows include `step_list: [{step_number, action}, ...]`.
    - steppred rows do NOT — full procedure is embedded in the question after
      'Complete step list:'. Parse via regex `^(\\d+)\\.\\s+(.+)$`.
    """
    sl = item.get("step_list")
    if isinstance(sl, list) and sl and isinstance(sl[0], dict):
        return [(int(s["step_number"]), str(s["action"])) for s in sl]
    # Fallback: parse from question text
    q = item.get("question", "")
    parts = re.findall(r"^\s*(\d+)\.\s+(.+)$", q, re.MULTILINE)
    if parts:
        return [(int(i), t.strip()) for i, t in parts]
    return []


def build_statement_sequence_generation(item: dict) -> str:
    """For sequence_generation: gold answer is a list of step indices visible
    in the video. Interleave with step text from `item['step_list']`."""
    gold = item.get("answer") or item.get("groundtruth") or []
    if isinstance(gold, str):
        gold = [s.strip() for s in re.findall(r"\d+", gold)]
    steps = dict(_step_list_text(item))
    if not steps:
        # Fallback: just list the indices without text
        gold_lines = [f"  step {i}" for i in gold]
    else:
        gold_lines = []
        for idx in gold:
            try:
                i = int(idx)
            except (ValueError, TypeError):
                gold_lines.append(f"  step {idx}")
                continue
            txt = steps.get(i)
            if txt:
                gold_lines.append(f"  {i}. {txt}")
            else:
                gold_lines.append(f"  {i}.")
    return "The steps shown in this video are:\n" + "\n".join(gold_lines)


def build_statement_step_prediction(item: dict) -> str:
    """For step_prediction: gold answer is the next step index. Use the full
    procedure (parsed from question for steppred — it doesn't have step_list)
    to look up the step text."""
    gold = str(item.get("answer") or item.get("gold") or "").strip()
    try:
        gold_int = int(gold)
    except ValueError:
        raise ValueError(f"step_prediction: gold {gold!r} not an integer")
    steps = dict(_step_list_text(item))
    txt = steps.get(gold_int)
    if txt:
        return (
            "After the steps shown in the video, the next step in the procedure is:\n"
            f"  {gold_int}. {txt}"
        )
    # Fallback: no step text available
    return (
        "After the steps shown in the video, the next step in the procedure is:\n"
        f"  step {gold_int}"
    )


def build_statement_fitb(item: dict) -> str:
    """For fitb (experimental_conclusion, scientific_discovery): substitute
    `____` placeholders one-at-a-time with the gold fill-ins."""
    q = item.get("question", "")
    fills = item.get("answer") or item.get("gold") or []
    if isinstance(fills, str):
        fills = [fills]
    n_blanks = q.count("____")
    if n_blanks != len(fills):
        # Pad or trim defensively (verified on samples that they always match)
        if n_blanks < len(fills):
            fills = fills[:n_blanks]
    statement = q
    for f in fills:
        statement = statement.replace("____", str(f), 1)
    if "____" in statement:
        # Bug guard: if any blanks remain after substitution, replace with placeholder
        statement = statement.replace("____", "<UNFILLED>")
    return statement


_TASK_TYPE_TO_BUILDER = {
    "mc": None,  # dispatched on item['task'] when task_type == 'mc'
    "seqgen": build_statement_sequence_generation,
    "steppred": build_statement_step_prediction,
    "fitb": build_statement_fitb,
}

_TASK_NAME_TO_MC_BUILDER = {
    "sequence_ordering": build_statement_sequence_ordering,
    "video_verification": build_statement_video_verification,
}


def build_statement_scivb(item: dict) -> str:
    """SciVideoBench is a single-task MC benchmark (A-J options) with
    conceptual / hypothetical / quantitative scientific reasoning questions.
    The statement is a simple "question + correct option text" declarative.
    """
    gold = str(item.get("answer") or item.get("gold") or "").strip()
    options = item.get("options", {})
    if gold not in options:
        raise ValueError(f"scivb: gold {gold!r} not in options {list(options)}")
    option_text = options[gold]
    q = item.get("question", "").strip()
    return (
        f"Question: {q}\n"
        f"The correct answer is: {option_text}"
    )


def build_statement_v5(item: dict, task_type: str, task: str | None = None) -> str:
    """Dispatch to the right statement builder.

    task_type values:
      'mc' (sequence_ordering / video_verification),
      'seqgen', 'steppred', 'fitb' — ExpVid task types
      'scivb_mc' — SciVideoBench single-task MC
    For 'mc', further dispatches on `task` (sequence_ordering vs
    video_verification) since the negation framing differs. `task` falls back
    to item['task'] if not explicitly passed.
    """
    if task_type == "scivb_mc":
        return build_statement_scivb(item)
    if task_type == "mc":
        t = task or item.get("task", "")
        builder = _TASK_NAME_TO_MC_BUILDER.get(t)
        if builder is None:
            raise ValueError(f"Unknown MC task {t!r} (expected sequence_ordering or video_verification)")
        return builder(item)
    builder = _TASK_TYPE_TO_BUILDER.get(task_type)
    if builder is None:
        raise ValueError(f"Unknown task_type {task_type!r}")
    return builder(item)


def build_oracle_prompt_v5(item: dict, task_type: str, task: str | None = None) -> str:
    """Build the full user-side oracle prompt for v5 = unified template wrapping
    the per-task statement."""
    statement = build_statement_v5(item, task_type, task=task)
    return ORACLE_USER_TEMPLATE_V5.format(statement=statement)


# ── Smoke test when run as __main__ ───────────────────────────────────────────
if __name__ == "__main__":
    import sys, json
    sys.path.insert(0, "/home/yz0392@unt.ad.unt.edu/xin_ai/scinote")
    from huggingface_hub import hf_hub_download
    from evaluate_unified import TASKS, REPO_ID

    SAMPLE_TASKS = [
        ("sequence_ordering", "mc"),
        ("video_verification", "mc"),
        ("sequence_generation", "seqgen"),
        ("step_prediction", "steppred"),
        ("experimental_conclusion", "fitb"),
        ("scientific_discovery", "fitb"),
    ]
    print("=" * 78)
    print("oracle_prompts_v5 smoke test")
    print("=" * 78)
    for task, tt in SAMPLE_TASKS:
        ann_path = TASKS[task][0]
        local = hf_hub_download(repo_id=REPO_ID, filename=ann_path, repo_type="dataset")
        items = [json.loads(l) for l in open(local) if l.strip()]
        if not items:
            print(f"\n=== {task} ({tt}) — no items"); continue
        item = items[0]
        statement = build_statement_v5(item, tt, task=task)
        prompt = build_oracle_prompt_v5(item, tt, task=task)
        print(f"\n=== task={task}  task_type={tt}  item_id={item.get('id','?')} ===")
        print(f"--- statement ({len(statement)} chars) ---")
        print(statement[:600] + ("..." if len(statement) > 600 else ""))
        print(f"--- full user prompt ({len(prompt)} chars) ---")
        print(prompt[:600] + ("..." if len(prompt) > 600 else ""))
        # Specific sanity checks
        if tt == "fitb":
            assert "____" not in statement, f"unfilled blanks in fitb statement: {statement[:200]}"
            print(f"  ✓ fitb statement has no remaining ____")
        elif tt == "mc":
            # Statement must not start with the gold letter / option label
            first_line = statement.splitlines()[0] if statement else ""
            assert item.get("answer", "") not in first_line.strip().split(":")[0], \
                f"answer letter leaked into first line of statement: {first_line!r}"
            print(f"  ✓ mc statement first line does not contain gold letter")
        elif tt in ("seqgen", "steppred"):
            assert any(c.isdigit() for c in statement), "no digits in seqgen/steppred statement"
            print(f"  ✓ {tt} statement includes step indices")
    print("\nAll smoke tests passed.")
