#!/usr/bin/env python3
"""Preflight checks for Open-o3-Video reproduction runs."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path


EVAL_IMPORTS = (
    "cv2",
    "numpy",
    "PIL",
    "qwen_vl_utils",
    "torch",
    "tqdm",
    "transformers",
    "yaml",
)
TRAIN_IMPORTS = (
    "cv2",
    "datasets",
    "deepspeed",
    "flash_attn",
    "numpy",
    "PIL",
    "qwen_vl_utils",
    "torch",
    "tqdm",
    "transformers",
    "trl",
    "vllm",
    "yaml",
)


def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def check_path(path: Path, label: str, errors: list[str]) -> None:
    if not path.exists():
        errors.append(f"missing {label}: {path}")


def resolve_nvcc() -> str | None:
    for key in ("CUDA_HOME", "CUDA_PATH"):
        value = os.environ.get(key)
        if value:
            nvcc = Path(value) / "bin" / "nvcc"
            if nvcc.exists():
                return str(nvcc)
    return shutil.which("nvcc")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--open-o3-root", type=Path, default=Path("/home/yz0392@unt.ad.unt.edu/xin_ai/open_o3"))
    parser.add_argument("--repo-dir", type=Path, default=Path("/home/yz0392@unt.ad.unt.edu/xin_ai/Open-o3-Video"))
    parser.add_argument("--min-gpus", type=int, default=8)
    parser.add_argument("--mode", choices=("eval", "train", "all"), default="all")
    args = parser.parse_args()

    root = args.open_o3_root
    data_root = root / "data" / "Open-o3-Video-data"
    errors: list[str] = []
    report: dict[str, object] = {
        "python": sys.version,
        "imports": {},
        "paths": {},
        "cuda": {},
        "tools": {},
    }

    required_imports = set(EVAL_IMPORTS if args.mode == "eval" else TRAIN_IMPORTS)
    if args.mode == "all":
        required_imports = set(EVAL_IMPORTS) | set(TRAIN_IMPORTS)

    for module in sorted(required_imports):
        available = has_module(module)
        report["imports"][module] = available  # type: ignore[index]
        if not available:
            errors.append(f"missing Python module: {module}")

    required_paths = {
        "repo": args.repo_dir,
        "official_model_config": root / "models" / "Open-o3-Video-official" / "config.json",
        "base_model_config": root / "models" / "Qwen2.5-VL-7B-Instruct" / "config.json",
        "judge_model_config": root / "models" / "Qwen2.5-72B-Instruct" / "config.json",
        "vstar_annotation": root / "data" / "V-STaR" / "V_STaR_test.json",
        "vstar_videos": root / "data" / "V-STaR" / "videos",
        "sft_json": data_root / "json_data" / "STGR-SFT.json",
        "rl_json": data_root / "json_data" / "STGR-RL.json",
        "stgr_videos": data_root / "videos" / "stgr",
        "timerft_videos": data_root / "videos" / "timerft",
    }
    for label, path in required_paths.items():
        exists = path.exists()
        report["paths"][label] = {"path": str(path), "exists": exists}  # type: ignore[index]
        check_path(path, label, errors)

    if has_module("torch"):
        import torch

        gpu_count = torch.cuda.device_count()
        report["cuda"] = {  # type: ignore[assignment]
            "available": torch.cuda.is_available(),
            "device_count": gpu_count,
            "devices": [torch.cuda.get_device_name(i) for i in range(gpu_count)],
        }
        if not torch.cuda.is_available():
            errors.append("torch.cuda is not available")
        if gpu_count < args.min_gpus:
            errors.append(f"expected at least {args.min_gpus} GPUs, found {gpu_count}")

    if args.mode in ("train", "all"):
        nvcc = resolve_nvcc()
        report["tools"]["nvcc"] = nvcc  # type: ignore[index]
        if not nvcc:
            errors.append("missing nvcc; CUDA_HOME/CUDA_PATH or PATH must point to a full CUDA toolkit")

    print(json.dumps(report, indent=2, sort_keys=True))
    if errors:
        print("\nRuntime readiness check failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
