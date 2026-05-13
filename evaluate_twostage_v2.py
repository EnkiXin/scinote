"""
Two-stage v2: TASK-AWARE prompt.

Stage 1: VLM watches video AND knows task type (but NOT options) → focused note.
Stage 2: Same VLM (text-only) answers using focused notes.

Key change from v1:
- v1: Generic note ("describe everything")  → too vague to disambiguate distractors
- v2: Task-specific note ("focus on the SPECIFIC materials with exact names")

This is the highest-effort prompt design before falling back to fine-tuning.

Usage:
    python evaluate_twostage_v2.py --task materials --limit 50 --output results_twostage_v2
"""

import argparse, json, os, re, time
from pathlib import Path
from typing import Optional

import av
import torch
from PIL import Image
from huggingface_hub import hf_hub_download
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

TASKS = {
    "materials":              ("annotations/level1/materials.jsonl",    "mc"),
    "tools":                  ("annotations/level1/tools.jsonl",         "mc"),
    "operation":              ("annotations/level1/operation.jsonl",     "mc"),
    "quantity":               ("annotations/level1/quantity.jsonl",      "mc"),
    "sequence_generation":    ("annotations/level2/sequence_generation.jsonl",  "seqgen"),
    "sequence_ordering":      ("annotations/level2/sequence_ordering.jsonl",    "mc"),
    "step_prediction":        ("annotations/level2/step_prediction.jsonl",      "steppred"),
    "video_verification":     ("annotations/level2/video_verification.jsonl",   "mc"),
    "experimental_conclusion": ("annotations/level3/experimental_conclusion.jsonl", "fitb"),
    "scientific_discovery":    ("annotations/level3/scientific_discovery.jsonl",    "fitb"),
}

LEVEL_TASKS = {
    "all_level1": ["materials", "tools", "operation", "quantity"],
    "all_level2": ["sequence_generation", "sequence_ordering", "step_prediction", "video_verification"],
    "all_level3": ["experimental_conclusion", "scientific_discovery"],
}

REPO_ID = "OpenGVLab/ExpVid"


# ── Task-aware Stage 1 prompts ──────────────────────────────────────────────

NOTE_PROMPTS = {
    "materials": (
        "Watch this scientific lab clip and FOCUS ON MATERIALS / SUBSTANCES.\n\n"
        "Identify SPECIFIC materials visible — use exact scientific names when readable on labels "
        "(e.g., 'PBS buffer', 'tracheal cannula', 'hippocampal slices', NOT generic words like "
        "'liquid' or 'sample').\n\n"
        "Output ONLY this JSON (no extra text):\n"
        "{\n"
        '  "primary_material": "the most prominent specific material/substance being handled",\n'
        '  "container_or_form": "container, state, or preparation form (e.g., \'in 1.5mL tube\', \'fixed tissue slice\')",\n'
        '  "labels_or_text_visible": ["any labels, package text, container markings you can read"],\n'
        '  "color_texture": "color and texture details",\n'
        '  "all_materials_seen": ["comprehensive list of substances/samples in view"]\n'
        "}"
    ),
    "tools": (
        "Watch this scientific lab clip and FOCUS ON TOOLS / EQUIPMENT.\n\n"
        "Identify the SPECIFIC tool being used — use exact lab equipment names "
        "(e.g., 'Hamilton syringe' not 'syringe', 'orbital shaker' not 'machine', "
        "'microcentrifuge tube' not 'tube'). Look at shape, size, labels.\n\n"
        "Output ONLY this JSON:\n"
        "{\n"
        '  "primary_tool": "the SPECIFIC tool actively being used",\n'
        '  "distinguishing_features": "shape/size/markings that identify the exact type",\n'
        '  "labels_or_brand": ["any brand or model text visible"],\n'
        '  "how_tool_is_used": "what the user does with the tool",\n'
        '  "other_tools_in_scene": ["other tools visible in background"]\n'
        "}"
    ),
    "quantity": (
        "Watch this scientific lab clip and FOCUS ON QUANTITIES / NUMBERS / COUNTS.\n\n"
        "Look very carefully at:\n"
        "- Numbers on displays (e.g., shaker RPM, temperature, timer)\n"
        "- Volume markings on tubes/syringes (e.g., '50 mL', '1.5 mL')\n"
        "- Counts of items (wells, stitches, samples)\n"
        "- Concentrations from labels\n\n"
        "Output ONLY this JSON:\n"
        "{\n"
        '  "numbers_on_displays": ["values shown on any digital/analog displays"],\n'
        '  "volume_markings": ["values from graduated containers"],\n'
        '  "counts": ["object: count, e.g., \'wells: 12\', \'stitches: 3\'"],\n'
        '  "any_other_quantitative_info": ["temperatures, percentages, times"]\n'
        "}"
    ),
    "operation": (
        "Watch this scientific lab clip and FOCUS ON THE ACTION being performed.\n\n"
        "Identify the EXACT action — use precise verbs and full action phrases "
        "(e.g., 'pipetting 200 uL into well A1' not 'using pipette', "
        "'inserting needle into vial' not 'using needle').\n\n"
        "Pay attention to direction, technique, and what object the action affects.\n\n"
        "Output ONLY this JSON:\n"
        "{\n"
        '  "main_action": "exact concise action being performed (verb + object)",\n'
        '  "action_target": "what the action is being done TO (the object being affected)",\n'
        '  "hands_motion": "describe hand/tool motion direction and manner",\n'
        '  "before_state": "state of objects before the action begins",\n'
        '  "after_state": "state of objects after the action completes (if visible)"\n'
        "}"
    ),
    "sequence_generation": (
        "Watch this scientific lab clip and identify the SEQUENCE of distinct procedural steps shown, in order.\n\n"
        "Output ONLY this JSON:\n"
        "{\n"
        '  "observed_steps_in_order": ["step 1 description", "step 2 description", ...],\n'
        '  "key_actions_per_step": ["specific action and object for each"],\n'
        '  "materials_or_tools_used": ["across all steps"]\n'
        "}"
    ),
    "sequence_ordering": (
        "Watch this scientific lab clip and identify the SEQUENCE of distinct procedural steps shown, in order.\n\n"
        "Output ONLY this JSON:\n"
        "{\n"
        '  "observed_steps_in_order": ["step 1 description", "step 2 description", ...],\n'
        '  "first_step": "what happens first",\n'
        '  "last_step": "what happens last",\n'
        '  "key_transitions": ["how each step transitions to next"]\n'
        "}"
    ),
    "step_prediction": (
        "Watch this scientific lab clip and identify the steps shown, focusing on the LAST/MOST RECENT action.\n"
        "The model needs to predict the NEXT step that would logically follow.\n\n"
        "Output ONLY this JSON:\n"
        "{\n"
        '  "observed_steps": ["concise description of each step seen"],\n'
        '  "current_state_at_end": "exactly what state things are in at the END of the video",\n'
        '  "next_logical_action_hint": "what immediately needs to happen based on the current state"\n'
        "}"
    ),
    "video_verification": (
        "Watch this scientific lab clip and list the procedural steps shown.\n"
        "The task will ask which step from a list was NOT performed in the video.\n\n"
        "Output ONLY this JSON:\n"
        "{\n"
        '  "steps_actually_shown": ["complete list of actions performed in the clip"],\n'
        '  "missing_or_skipped": "any obvious gap or skipped step",\n'
        '  "key_objects_handled": ["materials/tools touched in this clip"]\n'
        "}"
    ),
    "experimental_conclusion": (
        "Watch this scientific experiment video and produce DETAILED structured observations "
        "for downstream reasoning about experimental results.\n\n"
        "Output ONLY this JSON:\n"
        "{\n"
        '  "experiment_overview": "what the experiment is about and its goal",\n'
        '  "procedures_observed": ["all major procedures performed, in order"],\n'
        '  "materials_and_subjects": ["all samples, animals, materials, tissues used"],\n'
        '  "tools_and_setup": ["equipment, instruments, apparatus seen"],\n'
        '  "quantitative_observations": ["numbers, volumes, times, temperatures"],\n'
        '  "outcomes_visible": ["any results/data/observations visible in the video"]\n'
        "}"
    ),
    "scientific_discovery": (
        "Watch this scientific experiment video and produce DETAILED structured observations "
        "to support reasoning about scientific significance.\n\n"
        "Output ONLY this JSON:\n"
        "{\n"
        '  "experiment_overview": "what the experiment is about",\n'
        '  "novel_techniques_shown": ["any unique or novel methods/setups visible"],\n'
        '  "key_steps": ["critical procedural steps"],\n'
        '  "materials_subjects": ["all biological/chemical samples"],\n'
        '  "tools_and_setup": ["specific equipment that enables the experiment"],\n'
        '  "anything_unusual_or_notable": ["distinctive features that suggest method significance"]\n'
        "}"
    ),
}

NOTE_SYSTEM = (
    "You are a careful, precise observer of scientific experiment videos. "
    "You produce structured visual notes grounded in visible evidence. "
    "Use exact scientific terminology when you can read it on labels or recognize the equipment. "
    "Do NOT speculate beyond what you actually see. "
    "Output ONLY valid JSON with no extra text or markdown fences."
)


def build_note_messages(frames, task: str):
    prompt = NOTE_PROMPTS.get(task, NOTE_PROMPTS["materials"])
    return [
        {"role": "system", "content": NOTE_SYSTEM},
        {
            "role": "user",
            "content": [
                {"type": "video", "video": frames, "max_pixels": 360 * 420},
                {"type": "text", "text": prompt},
            ],
        },
    ]


# ── Stage 2 (same as v1) ────────────────────────────────────────────────────

MC_SYSTEM_S2 = (
    "You are an expert evaluator. You will see structured visual notes from a scientific experiment "
    "video, plus a multiple-choice question. Match the notes to the best option. "
    "Respond with ONLY one letter: A, B, C, or D."
)

FITB_SYSTEM_S2 = (
    "You are an expert evaluator. You will see structured visual notes from a scientific experiment "
    "video, plus a fill-in-the-blank question. Provide concise answers, separated by '|'."
)


def build_mc_messages_s2(item, notes):
    opts = item["options"]
    options_text = "\n".join(f"{k}. {v}" for k, v in opts.items())
    user_text = (
        f"Visual notes:\n{notes}\n\n"
        f"Question: {item['question']}\n\n"
        f"Options:\n{options_text}\n\n"
        f"Pick the option that BEST matches the notes. Output ONLY one letter."
    )
    return [
        {"role": "system", "content": MC_SYSTEM_S2},
        {"role": "user", "content": user_text},
    ]


def build_seqgen_messages_s2(item, notes):
    user_text = (
        f"Visual notes:\n{notes}\n\n"
        f"{item['question']}\n\n"
        "Output ONLY the step numbers visible in this video, separated by spaces. Nothing else."
    )
    return [{"role": "system", "content": "Expert procedural reasoner."},
            {"role": "user", "content": user_text}]


def build_steppred_messages_s2(item, notes):
    user_text = (
        f"Visual notes:\n{notes}\n\n"
        f"{item['question']}\n\n"
        "Output ONLY the step number (single integer)."
    )
    return [{"role": "system", "content": "Expert procedural reasoner."},
            {"role": "user", "content": user_text}]


def build_fitb_messages_s2(item, notes):
    n_blanks = item["question"].count("____")
    user_text = (
        f"Visual notes:\n{notes}\n\n"
        f"Question: {item['question']}\n\n"
        f"Fill {n_blanks} blank(s). Separate answers with ' | '."
    )
    return [{"role": "system", "content": FITB_SYSTEM_S2},
            {"role": "user", "content": user_text}]


# ── Scoring (same as v1) ────────────────────────────────────────────────────

def parse_mc_answer(r):
    r = r.strip()
    m = re.search(r'\b([A-D])\b', r)
    if m: return m.group(1)
    return r[0].upper() if r and r[0].upper() in "ABCD" else ""

def score_mc(p, g): return 1.0 if p.upper() == g.upper() else 0.0

def score_seqgen(p, g):
    pn, gn = set(re.findall(r'\d+', p)), set(g)
    if not gn: return 1.0
    c = pn & gn
    if not c: return 0.0
    pr = len(c)/len(pn) if pn else 0
    rc = len(c)/len(gn)
    return 2*pr*rc/(pr+rc) if pr+rc>0 else 0.0

def score_steppred(p, g):
    nums = re.findall(r'\d+', p.strip())
    return 1.0 if nums and nums[0] == str(g) else 0.0

def score_fitb(p, g):
    pp = [x.strip().lower() for x in p.split("|")]
    out = []
    for i, ref in enumerate(g):
        if i < len(pp):
            pt, gt = set(pp[i].split()), set(ref.lower().split())
            if not gt: out.append(1.0); continue
            c = pt & gt
            if not c: out.append(0.0); continue
            pr = len(c)/len(pt) if pt else 0
            rc = len(c)/len(gt)
            out.append(2*pr*rc/(pr+rc) if pr+rc>0 else 0)
        else:
            out.append(0.0)
    return sum(out)/len(out) if out else 0.0


# ── Video & Model ───────────────────────────────────────────────────────────

def get_video_path(p): return hf_hub_download(repo_id=REPO_ID, filename=p, repo_type="dataset")

def load_annotations(task, limit=None):
    ann_path, _ = TASKS[task]
    local = hf_hub_download(repo_id=REPO_ID, filename=ann_path, repo_type="dataset")
    with open(local) as f:
        items = [json.loads(l) for l in f if l.strip()]
    return items[:limit] if limit else items

def extract_frames(video_path, fps=1.0, max_frames=32):
    c = av.open(video_path)
    s = c.streams.video[0]
    vfps = float(s.average_rate); tf = s.frames
    n = max(1, min(max_frames, int((tf/vfps)*fps))) if (tf>0 and vfps>0) else max_frames
    frames = []
    c.seek(0)
    for f in c.decode(video=0):
        frames.append(f.to_image())
    c.close()
    if not frames: return []
    if len(frames) <= n: return frames
    idx = [int(i*len(frames)/n) for i in range(n)]
    return [frames[i] for i in idx]


class QwenVLModel:
    def __init__(self, name):
        print(f"Loading {name}", flush=True)
        self.processor = AutoProcessor.from_pretrained(name)
        d = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        self.device = d
        dtype = torch.float16 if d in ("cuda","mps") else torch.float32
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            name, torch_dtype=dtype, device_map=d if d!="mps" else "cpu")
        if d == "mps": self.model = self.model.to("mps")
        self.model.eval()
        print(f"Loaded on {d}", flush=True)

    def gen_video(self, msgs, max_new=400):
        text = self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        img, vid, vkw = process_vision_info(msgs, return_video_kwargs=True)
        inp = self.processor(text=[text], images=img, videos=vid, return_tensors="pt", **vkw)
        inp = {k: v.to(self.model.device) if hasattr(v,'to') else v for k,v in inp.items()}
        with torch.no_grad():
            out = self.model.generate(**inp, max_new_tokens=max_new)
        result = self.processor.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        if self.device == "cuda": torch.cuda.empty_cache()
        return result

    def gen_text(self, msgs, max_new=64):
        text = self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = self.processor(text=[text], return_tensors="pt")
        inp = {k: v.to(self.model.device) if hasattr(v,'to') else v for k,v in inp.items()}
        with torch.no_grad():
            out = self.model.generate(**inp, max_new_tokens=max_new)
        result = self.processor.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        if self.device == "cuda": torch.cuda.empty_cache()
        return result


def evaluate_task(model, task, limit=None):
    _, ttype = TASKS[task]
    items = load_annotations(task, limit)
    print(f"\n=== Task: {task} | {ttype} | {len(items)} samples ===", flush=True)
    results, total = [], 0.0
    for i, item in enumerate(items):
        try:
            v = get_video_path(item["video_path"])
        except Exception as e:
            results.append({"id": item["id"], "error": str(e)}); continue
        # Lower frame count for L3 to fit in 24GB GPU memory
        if ttype == "fitb":
            fps, max_frames = 0.2, 8
        else:
            fps, max_frames = 1.0, 32
        frames = extract_frames(v, fps=fps, max_frames=max_frames)
        if not frames:
            results.append({"id": item["id"], "error": "no frames"}); continue

        t0 = time.time()
        try:
            notes = model.gen_video(build_note_messages(frames, task), max_new=400)
        except Exception as e:
            print(f"[{i+1}/{len(items)}] S1 ERR: {e}", flush=True)
            results.append({"id": item["id"], "error": f"s1:{e}"}); continue
        tn = time.time()-t0

        t0 = time.time()
        try:
            if ttype == "mc":
                m = build_mc_messages_s2(item, notes); mn = 8
            elif ttype == "seqgen":
                m = build_seqgen_messages_s2(item, notes); mn = 64
            elif ttype == "steppred":
                m = build_steppred_messages_s2(item, notes); mn = 8
            else:
                m = build_fitb_messages_s2(item, notes); mn = 128
            r = model.gen_text(m, max_new=mn)
        except Exception as e:
            print(f"[{i+1}/{len(items)}] S2 ERR: {e}", flush=True)
            results.append({"id": item["id"], "notes": notes, "error": f"s2:{e}"}); continue
        ta = time.time()-t0

        if ttype == "mc":
            pred = parse_mc_answer(r); sc = score_mc(pred, item["answer"])
        elif ttype == "seqgen":
            pred = r; sc = score_seqgen(pred, item["answer"])
        elif ttype == "steppred":
            pred = r; sc = score_steppred(pred, item["answer"])
        else:
            pred = r; sc = score_fitb(pred, item["answer"])

        total += sc
        results.append({"id": item["id"], "pred": pred, "gold": item["answer"],
                        "score": sc, "notes": notes, "response": r,
                        "t_note": round(tn,2), "t_ans": round(ta,2)})
        status = "✅" if sc==1.0 else ("🔶" if sc>0 else "❌")
        print(f"[{i+1}/{len(items)}] {status} pred={pred!r} gold={item['answer']!r} (n{tn:.1f}s a{ta:.1f}s)", flush=True)

    valid = [r for r in results if "error" not in r]
    acc = total/len(valid)*100 if valid else 0
    print(f"\nAcc: {acc:.1f}% ({len(valid)}/{len(results)})", flush=True)
    return {"task": task, "accuracy": round(acc,2),
            "n_valid": len(valid), "n_error": len(results)-len(valid),
            "results": results}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="all_level1")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--output", default="results_twostage_v2")
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()

    od = Path(a.output); od.mkdir(exist_ok=True)
    tl = LEVEL_TASKS.get(a.task, [a.task] if a.task in TASKS else list(TASKS.keys()))

    model = QwenVLModel(a.model)
    all_r = []
    for t in tl:
        out = od / f"eval_{t}.json"
        if a.resume and out.exists():
            print(f"Skip {t}", flush=True)
            with open(out) as f: all_r.append(json.load(f))
            continue
        r = evaluate_task(model, t, a.limit)
        all_r.append(r)
        with open(out, "w") as f: json.dump(r, f, indent=2, ensure_ascii=False)
        print(f"Saved {out}", flush=True)

    print("\n" + "="*50)
    print(f"{'Task':<30} {'Acc':>10} {'N':>6}")
    for r in all_r:
        if "accuracy" in r:
            print(f"{r['task']:<30} {r['accuracy']:>9.1f}% {r['n_valid']:>6}")


if __name__ == "__main__":
    main()
