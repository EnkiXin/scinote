"""protonote/cli.py — entry-point for the ProtoNote agent.

Phase 0: agent is a single Qwen2.5-VL-7B call (no tools, no notes). Reproduces
the fresh-pipeline C0 baseline (ExpVid 26.73% overall).

Phase 3+ will swap this for a ReAct loop that uses tools (Phase 2) and the
NoteBuffer (Phase 1).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from evaluate_unified import (  # noqa: E402
    MC_SYSTEM, FITB_SYSTEM, SEQGEN_SYSTEM, STEPPRED_SYSTEM,
    SCORERS, MAX_PIXELS,
)
# Reuse the C0 evaluator's prompt builders + parsers (same fresh-pipeline behavior)
from evaluate_c0_test_split import (  # noqa: E402
    BUILDERS, parse_for_task, gold_for, extract_frames,
)

from protonote.data.loaders import (  # noqa: E402
    load_test_split, load_expvid_l1, resolve_video_path,
)


# ── LLM client (HF transformers in-process; multi-arch dispatcher) ─────────

# Qwen2.5-VL family + MiMo-VL-7B-RL (Qwen-derived arch). InternVL3 has its
# own loader/chat-API and is dispatched as a separate client below.
_QWEN_VL_FAMILY = (
    "Qwen/Qwen2.5-VL-",            # all three sizes (3B/7B/72B)
    "Qwen2.5-VL-",                 # short form
    "XiaomiMiMo/MiMo-VL",         # MiMo-VL inherits Qwen2.5-VL architecture
    "MiMo-VL",                     # short form
)


def _is_qwen_family(model_name: str) -> bool:
    return any(p in model_name for p in _QWEN_VL_FAMILY)


def _is_internvl(model_name: str) -> bool:
    return "InternVL" in model_name


class VLMClient:
    """Polymorphic VLM client. Dispatches to a Qwen-family or InternVL3
    loader based on `model_name`.

    The `.generate(messages, max_new_tokens)` interface is the same for
    every backbone — callers (the agent, tools) do not need to know which
    model they are talking to.

    `device` semantics:
      * "cuda:0" / "cuda:N" — single GPU placement (Qwen-3B/7B, MiMo, InternVL)
      * "auto"              — HF `device_map="auto"` for tensor-parallel
                              (Qwen-72B across all visible GPUs)
    """

    def __init__(self, model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
                 device: str = "cuda:0", dtype=torch.bfloat16):
        self.model_name = model_name
        self.device = device
        print(f"[VLMClient] loading {model_name} on {device}", flush=True)
        dm = "auto" if device == "auto" else device

        if _is_internvl(model_name):
            self._impl = _InternVLImpl(model_name, dm, dtype)
        elif _is_qwen_family(model_name):
            self._impl = _QwenVLImpl(model_name, dm, dtype)
        else:
            # Fall back to Qwen loader (covers most VL models with same arch)
            print(f"[VLMClient] unknown family for {model_name}; "
                  f"trying Qwen2.5-VL loader", flush=True)
            self._impl = _QwenVLImpl(model_name, dm, dtype)
        print(f"[VLMClient] loaded ({self._impl.__class__.__name__})",
              flush=True)
        self.model = self._impl.model          # exposed for LoRA wrapping
        self.processor = self._impl.processor   # exposed for legacy callers

    @torch.no_grad()
    def generate(self, messages: list, max_new_tokens: int = 64) -> str:
        return self._impl.generate(messages, max_new_tokens=max_new_tokens)


# ── Qwen-2.5-VL family + MiMo-VL implementation ─────────────────────────────

class _QwenVLImpl:
    def __init__(self, model_name: str, device_map, dtype):
        self.processor = AutoProcessor.from_pretrained(
            model_name, trust_remote_code=True)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name, dtype=dtype, device_map=device_map)
        self.model.eval()

    @torch.no_grad()
    def generate(self, messages: list, max_new_tokens: int = 64) -> str:
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages, return_video_kwargs=True)
        if "fps" in video_kwargs and isinstance(video_kwargs["fps"], list):
            video_kwargs["fps"] = video_kwargs["fps"][0] if video_kwargs["fps"] else 1.0
        inputs = self.processor(
            text=[text], images=image_inputs, videos=video_inputs,
            return_tensors="pt", **video_kwargs)
        inputs = {k: v.to(self.model.device) if hasattr(v, "to") else v
                  for k, v in inputs.items()}
        outputs = self.model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False)
        raw = self.processor.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True).strip()
        return raw


# ── InternVL3 implementation ────────────────────────────────────────────────

class _InternVLImpl:
    """InternVL3 uses a different chat API (`model.chat(tokenizer, ...)`)
    and its own per-frame patch tokenizer. We render frames to a
    pixel-value tensor and call `.chat()` once per generate."""

    def __init__(self, model_name: str, device_map, dtype):
        from transformers import AutoModel, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, use_fast=False)
        # transformers 5.8 has a `all_tied_weights_keys` check in its
        # caching allocator warmup that fails for InternVL's custom model
        # class (which only exposes `_tied_weights_keys`). Patch the class
        # before loading.
        from transformers.dynamic_module_utils import get_class_from_dynamic_module
        import transformers.models.auto.auto_factory as _af
        # Soft monkey-patch: add the missing alias to any class that has _tied_weights_keys.
        _orig_from = AutoModel.from_pretrained
        def _patched_from(*args, **kwargs):
            kwargs.setdefault("torch_dtype", dtype)
            kwargs["trust_remote_code"] = True
            if "device_map" in kwargs:
                kwargs.pop("device_map")
            # First load to CPU to apply the patch, then move.
            return _orig_from(*args, **kwargs)
        try:
            self.model = AutoModel.from_pretrained(
                model_name, dtype=dtype,
                trust_remote_code=True, device_map=device_map)
        except AttributeError as e:
            if "all_tied_weights_keys" not in str(e):
                raise
            # Add the property to the class then retry without auto warmup.
            # Use the dynamic module loader to grab the class, then alias.
            from transformers import AutoConfig
            cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
            cls = get_class_from_dynamic_module(
                cfg.auto_map["AutoModel"], model_name)
            if not hasattr(cls, "all_tied_weights_keys"):
                # _tied_weights_keys is a list of strings; we need a dict-like
                # mapping for the new API. Provide a minimal dict view.
                cls.all_tied_weights_keys = property(
                    lambda self: {k: k for k in (self._tied_weights_keys or [])})
            self.model = AutoModel.from_pretrained(
                model_name, dtype=dtype,
                trust_remote_code=True, device_map=device_map)
        self.model.eval()
        # InternVL processor differs from the Qwen one; keep `processor` as
        # the tokenizer so the public attribute is set but unused.
        self.processor = self.tokenizer

    @torch.no_grad()
    def generate(self, messages: list, max_new_tokens: int = 64) -> str:
        # Convert OpenAI-style messages -> (text_question, frame_pil_list).
        text_parts: list[str] = []
        frames: list = []
        for msg in messages:
            content = msg["content"]
            if isinstance(content, str):
                text_parts.append(content)
                continue
            for block in content:
                if block.get("type") == "text":
                    text_parts.append(block["text"])
                elif block.get("type") == "video":
                    frames.extend(block["video"])
                elif block.get("type") == "image":
                    frames.append(block["image"])
        question = "\n\n".join(text_parts).strip()

        from PIL import Image as _PILImage
        import torchvision.transforms as T

        # InternVL transform: 448x448 normalized.
        tf = T.Compose([
            T.Resize((448, 448), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
        ])
        if not frames:
            pixel_values = None
        else:
            tensors = [tf(f if isinstance(f, _PILImage.Image)
                          else _PILImage.fromarray(f)) for f in frames]
            pixel_values = torch.stack(tensors).to(self.model.device).to(
                next(self.model.parameters()).dtype)

        num_patches_list = ([pixel_values.shape[0]]
                             if pixel_values is not None else None)
        gen_cfg = dict(max_new_tokens=max_new_tokens, do_sample=False)
        if pixel_values is not None:
            video_prefix = "".join(
                f"Frame{i + 1}: <image>\n"
                for i in range(pixel_values.shape[0]))
            question_with_video = video_prefix + question
            response = self.model.chat(
                self.tokenizer, pixel_values, question_with_video,
                gen_cfg, num_patches_list=num_patches_list,
                history=None, return_history=False)
        else:
            response = self.model.chat(
                self.tokenizer, None, question, gen_cfg,
                history=None, return_history=False)
        return response.strip()


# ── Agent ───────────────────────────────────────────────────────────────────

class ProtoNoteAgent:
    """Phase 0 implementation: single-call C0 baseline agent.

    The trajectory format is generic (list of steps) so Phase 3+ can extend
    without changing the return type.
    """

    def __init__(self, vlm: VLMClient, condition: str = "C0"):
        self.vlm = vlm
        self.condition = condition  # Phase 0 only supports "C0"

    def answer(self, item: dict, max_frames: int = 32) -> dict:
        """Returns: dict with sample_id / pred / gold / score / trajectory."""
        out = {
            "sample_id": item["sample_id"],
            "benchmark": item["benchmark"], "task": item.get("task"),
            "task_type": item.get("task_type", "mc"),
            "gold": gold_for(item),
            "condition": self.condition,
            "trajectory": [],
        }
        try:
            vp = resolve_video_path(item)
            if not vp:
                return {**out, "error": "no_video"}
            frames = extract_frames(vp, max_frames=max_frames)
            if not frames:
                return {**out, "error": "no_frames"}
        except Exception as e:
            return {**out, "error": f"video err: {str(e)[:120]}"}

        task_type = item.get("task_type", "mc")
        builder = BUILDERS[task_type]
        # C0: no note context (None)
        if task_type == "mc":
            messages = builder(item, frames, None, item["benchmark"])
        else:
            messages = builder(item, frames, None)

        max_new = 8 if task_type == "mc" else 64
        try:
            t0 = time.time()
            raw = self.vlm.generate(messages, max_new_tokens=max_new)
            elapsed = time.time() - t0
        except Exception as e:
            return {**out, "error": f"gen err: {str(e)[:120]}"}

        pred = parse_for_task(raw, task_type, item)
        scorer = SCORERS[task_type]
        sc = float(scorer(pred, out["gold"]))

        out["trajectory"].append({
            "step": 0,
            "action": "answer",  # Phase 0: single answer step
            "raw": raw[:200],
            "elapsed_s": round(elapsed, 3),
        })
        out["pred"] = pred
        out["raw"] = raw[:120]
        out["score"] = sc
        return out


# ── CLI ─────────────────────────────────────────────────────────────────────

def _build_agent(condition: str, vlm, notes_cache_dir: str,
                  max_react_steps: int = 2, planner_adapter: str = ""):
    """Factory: build the agent that matches `condition`.

    Conditions:
      C0          single VLM call (no tools, no notes)
      C1_fixed    deterministic task-routed tools → NoteBuffer → answer
      C2_react    ReAct (planner picks tool + timestamp_range, no options shown)
                  — the original C2; underperforms C1_fixed on video_verification
      C2_react_v2 ReAct with (B) no timestamp picking + (C) MC options shown
                  to planner. Designed to fix the C2_react regression.
      C3_learned  ReAct with a trained planner LoRA. Loads adapter from
                  `planner_adapter`; uses adapter for planner JSON output and
                  base model for tool/answer calls (peft disable_adapter).
    """
    if condition == "C0":
        return ProtoNoteAgent(vlm=vlm, condition="C0")
    if condition in ("C1_fixed", "C2_react", "C2_react_v2", "C3_learned"):
        from protonote.notes.note_buffer import NoteBuffer
        from protonote.tools import build_default_tools
        buf = NoteBuffer(cache_dir=notes_cache_dir)
        tools = build_default_tools(vlm=vlm, note_buffer=buf)
        if condition == "C1_fixed":
            from protonote.planner.controller import FixedScheduleAgent
            return FixedScheduleAgent(vlm=vlm, tools=tools, note_buffer=buf)
        if condition == "C3_learned":
            if not planner_adapter:
                raise ValueError("C3_learned requires --planner_adapter <path>")
            from protonote.planner.learned_controller import LearnedReActAgent
            return LearnedReActAgent(vlm=vlm, tools=tools, note_buffer=buf,
                                       adapter_path=planner_adapter,
                                       max_react_steps=max_react_steps)
        from protonote.planner.react_controller import ReActAgent
        if condition == "C2_react_v2":
            return ReActAgent(vlm=vlm, tools=tools, note_buffer=buf,
                                max_react_steps=max_react_steps,
                                allow_timestamp_picking=False,
                                show_options_to_planner=True)
        return ReActAgent(vlm=vlm, tools=tools, note_buffer=buf,
                            max_react_steps=max_react_steps,
                            allow_timestamp_picking=True,
                            show_options_to_planner=False)
    raise ValueError(f"unknown condition: {condition!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--benchmark", default="expvid",
                     choices=["expvid", "scivideobench", "expvid_l1", "all"])
    ap.add_argument("--l1_subtask", default="",
                     help="ExpVid L1 only: filter to one sub-task "
                          "(tools / materials / operation / quantity). "
                          "Empty = all four.")
    ap.add_argument("--limit", type=int, default=50, help="Pilot limit; 0 = full split")
    ap.add_argument("--split", default="test", choices=["test", "train"],
                     help="Which split of v2_split_*.jsonl to evaluate (ignored for expvid_l1).")
    ap.add_argument("--max_frames", type=int, default=32)
    ap.add_argument("--output_dir", default="results_protonote/pilot")
    ap.add_argument("--chunk_id", type=int, default=0)
    ap.add_argument("--num_chunks", type=int, default=1)
    ap.add_argument("--condition", default="C0",
                     choices=["C0", "C1_fixed", "C2_react", "C2_react_v2",
                                "C3_learned"],
                     help="C0/C1_fixed/C2_react/C2_react_v2 — see _build_agent. "
                          "C3_learned = ReAct with a trained planner LoRA "
                          "(--planner_adapter <path>).")
    ap.add_argument("--notes_cache", default="",
                     help="NoteBuffer cache dir (C1+/C2/C3). Defaults to "
                          "<output_dir>/notes_cache.")
    ap.add_argument("--max_react_steps", type=int, default=2,
                     help="C2/C3 only: max LLM-planned tool calls after seed.")
    ap.add_argument("--planner_adapter", default="",
                     help="C3_learned: path to the trained LoRA adapter dir "
                          "(e.g. checkpoints/planner_lora_A/final).")
    args = ap.parse_args()

    if args.benchmark == "expvid_l1":
        benchmark = "expvid_l1"
        items = load_expvid_l1(subtask=args.l1_subtask or None,
                                 limit=args.limit if args.limit > 0 else None)
    else:
        benchmark = None if args.benchmark == "all" else args.benchmark
        items = load_test_split(benchmark=benchmark,
                                  limit=args.limit if args.limit > 0 else None,
                                  split=args.split)
    if args.num_chunks > 1:
        items = [it for i, it in enumerate(items) if i % args.num_chunks == args.chunk_id]
    print(f"[cli] {len(items)} items (benchmark={benchmark}, "
          f"chunk={args.chunk_id}/{args.num_chunks}, "
          f"condition={args.condition})", flush=True)

    vlm = VLMClient(model_name=args.model, device=args.device)

    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    notes_cache_dir = args.notes_cache or str(out_dir / "notes_cache")
    agent = _build_agent(args.condition, vlm=vlm,
                          notes_cache_dir=notes_cache_dir,
                          max_react_steps=args.max_react_steps,
                          planner_adapter=args.planner_adapter)

    out_path = out_dir / (
        f"trajectory_{args.benchmark}"
        + (f"_chunk{args.chunk_id}of{args.num_chunks}" if args.num_chunks > 1 else "")
        + ".jsonl"
    )

    results = []
    with open(out_path, "w") as fout:
        for i, item in enumerate(items):
            r = agent.answer(item, max_frames=args.max_frames)
            fout.write(json.dumps(r, default=str) + "\n")
            fout.flush()
            results.append(r)
            if i % 25 == 0:
                valid = [x for x in results if "score" in x]
                acc = sum(x["score"] for x in valid) / max(len(valid), 1) * 100
                print(f"  [{i+1}/{len(items)}] running acc={acc:.2f}% n={len(valid)}", flush=True)

    # Per-task summary
    by_task = {}
    n_err = 0
    for r in results:
        if "score" not in r:
            n_err += 1; continue
        t = r.get("task", "?")
        by_task.setdefault(t, []).append(r["score"])
    print()
    print(f"=== ProtoNote {args.benchmark} {args.condition} (limit={args.limit}) ===")
    for t, s in sorted(by_task.items()):
        print(f"  {t:<30} acc={100*sum(s)/len(s):.2f}%  n={len(s)}")
    all_scores = [r["score"] for r in results if "score" in r]
    if all_scores:
        print(f"  overall acc={100*sum(all_scores)/len(all_scores):.2f}%  "
              f"n_valid={len(all_scores)}  n_err={n_err}")
    print(f"\nTrajectory: {out_path}")


if __name__ == "__main__":
    main()
