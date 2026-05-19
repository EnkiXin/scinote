"""Task-aware oracle prompts for v4 oracle regeneration (Paper-1 Extension Week 1).

Replaces the v2 oracle prompts in generate_oracle_notes_expvid.py with the
schemas specified in PAPER1_EXTENSION_PLAN.md item (3). Key changes:

  * mc      : per-option supporting / refuting evidence + frame_locations
              (was: single key_evidence list paraphrasing the answer-supportive cues)
  * seqgen  : per-step {step_index, visual_evidence, frame_range, verbatim_on_screen_text}
              (was: prose-only observed_steps_with_evidence)
  * steppred: per-step + current_state_at_end (with frame_range) + why_next_step
  * fitb    : per-blank {fill_in_index, verbatim_on_screen, frame_location, context}
              CRITICAL: verbatim_on_screen must literally match a visible string,
              null if not visible (forces honest gap measurement, addresses
              v2/v3 specificity erasure).

These prompts force the 78B oracle to:
  - emit step indices / verbatim text directly (no prose-only fields)
  - anchor evidence to approximate frame ranges
  - balance coverage across MC options (reduces selective bias)
"""

ORACLE_SYSTEM_V4 = (
    "You are a careful, precise observer of scientific experiment videos. "
    "You will be shown a video, a question about it, and the CORRECT answer. "
    "Your task: write structured visual notes that describe ONLY what is VISIBLE "
    "in the video, in enough detail that someone who reads only your notes "
    "(without watching the video) could derive the correct answer through reasoning "
    "over the visible evidence.\n\n"
    "STRICT CONSTRAINTS:\n"
    "  - Only describe content that is actually visible in the video.\n"
    "  - Do NOT mention the answer letter (A, B, C, ...) anywhere.\n"
    "  - Do NOT copy any of the option texts verbatim.\n"
    "  - Do NOT include any speculation that is not grounded in visible evidence.\n"
    "  - For any field requesting 'verbatim_on_screen' text, the value must literally "
    "    match a string visible somewhere in the video frames; use null if not present.\n"
    "  - For frame ranges, use approximate frame indices from the 32 sampled frames.\n"
    "  - Output ONLY valid JSON, no extra text or markdown fences."
)


def fmt_options(opts):
    return "\n".join(f"  {k}. {v}" for k, v in sorted(opts.items()))


def build_oracle_prompt_v4(item, task_type):
    if task_type == "mc":
        return (
            f"Question: {item['question']}\n\n"
            f"Options:\n{fmt_options(item['options'])}\n\n"
            f"Correct answer: {item['answer']} "
            f"(use this only to know what to highlight; do NOT reveal the letter)\n\n"
            f"Output ONLY this JSON:\n"
            f"{{\n"
            f'  "per_option_evidence": {{\n'
            f'    "A": {{\n'
            f'      "supporting_evidence": ["visible cues that support option A"],\n'
            f'      "refuting_evidence": ["visible cues that rule out option A"],\n'
            f'      "frame_locations": ["frame ranges where evidence appears, e.g. 5-9"]\n'
            f'    }},\n'
            f'    "B": {{ "supporting_evidence": [], "refuting_evidence": [], "frame_locations": [] }},\n'
            f'    "...": "and so on for each option"\n'
            f'  }},\n'
            f'  "salient_objects_or_text": ["distinctive objects, labels, readings on screen"]\n'
            f"}}\n\n"
            f"Output JSON with balanced coverage across ALL options."
        )

    elif task_type == "seqgen":
        gold = item.get("answer", item.get("groundtruth", []))
        return (
            f"Question: {item['question']}\n\n"
            f"The correct steps visible in this video are: {gold}\n\n"
            f"Output ONLY this JSON:\n"
            f"{{\n"
            f'  "observed_steps": [\n'
            f'    {{\n'
            f'      "step_index": <integer step number>,\n'
            f'      "visual_evidence": "specific visible cue for this step",\n'
            f'      "frame_range": "approximate frame indices where visible, e.g. 0-3",\n'
            f'      "verbatim_on_screen_text": "any visible label/number for this step (null if none)"\n'
            f'    }}\n'
            f'  ],\n'
            f'  "salient_objects_or_text": ["distinctive labels/objects/readings on screen"]\n'
            f"}}\n\n"
            f"Include EACH visible step with its integer step_index."
        )

    elif task_type == "steppred":
        gold = item.get("answer")
        return (
            f"Question: {item['question']}\n\n"
            f"The correct next step is: {gold}\n\n"
            f"Output ONLY this JSON:\n"
            f"{{\n"
            f'  "observed_steps_so_far": [\n'
            f'    {{\n'
            f'      "step_index": <integer>,\n'
            f'      "visual_evidence": "specific visible cue",\n'
            f'      "frame_range": "e.g. 0-7"\n'
            f'    }}\n'
            f'  ],\n'
            f'  "current_state_at_end": {{\n'
            f'    "description": "what is visible at the end of the video",\n'
            f'    "frame_range": "frame range covering the end state"\n'
            f'  }},\n'
            f'  "why_next_step": "specific visible evidence that the next observable step would be {gold}",\n'
            f'  "salient_objects_or_text": ["on-screen labels/numbers"]\n'
            f"}}"
        )

    elif task_type == "fitb":
        gold = item.get("answer", [])
        return (
            f"Question: {item['question']}\n\n"
            f"The correct fill-in answers (in order): {gold}\n\n"
            f"Output ONLY this JSON:\n"
            f"{{\n"
            f'  "fills": [\n'
            f'    {{\n'
            f'      "fill_in_index": <integer, 0-based blank position>,\n'
            f'      "verbatim_on_screen": "exact text/number as visible on screen (null if not literally visible)",\n'
            f'      "frame_location": "frame where visible, e.g. 12",\n'
            f'      "context": "surrounding visual context that justifies the fill-in"\n'
            f'    }}\n'
            f'  ],\n'
            f'  "salient_objects_or_text": ["readable labels, signals, equipment on screen"]\n'
            f"}}\n\n"
            f"CRITICAL: verbatim_on_screen must be exactly what is visible. "
            f"If the gold answer is not literally visible in the frames (e.g. inferred), "
            f"set verbatim_on_screen to null. This forces honest gap measurement."
        )

    return ""
