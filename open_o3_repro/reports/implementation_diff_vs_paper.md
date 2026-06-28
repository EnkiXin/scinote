# Implementation Difference Check vs Open-o3-Video Paper

Date: 2026-06-18

## Verdict

The current runnable training path is not paper-faithful yet. It is useful as a smoke-test/debug run, but its checkpoint should not be compared directly with the Open-o3-Video paper results.

The official paper/repo recipe is:

- Stage 1: full-model cold-start SFT from `Qwen2.5-VL-7B-Instruct` on `STGR-CoT-30k` / `STGR-SFT.json`.
- Stage 2: RL with GSPO on `STGR-RL-36k` / `STGR-RL.json`.
- Official scripts use 8 GPUs, DeepSpeed ZeRO, and `flash_attention_2`.

The current runnable path produced only an intermediate LoRA SFT checkpoint:

- checkpoint: `open_o3/ckpts/sft_repro/checkpoint-500`
- checkpoint type: PEFT/LoRA adapter only, `adapter_model.safetensors`
- global step: 500 / 3291
- world size: 6
- no completed SFT checkpoint and no RL checkpoint found

## Major Differences

| Area | Paper / official implementation | Current runnable path | Impact |
| --- | --- | --- | --- |
| Training stage coverage | SFT then RL with GSPO | SFT only; RL has not run | Cannot reproduce final paper model or V-STAR gains. |
| Model update | Full model fine-tuning | LoRA/PEFT adapter, rank 16, alpha 32, target `q_proj/k_proj/v_proj/o_proj` | Very different capacity and optimization dynamics. |
| SFT data | `STGR-SFT.json`, 31,166 examples | `STGR-SFT-covered.json`, 19,746 examples | Removes 11,420 examples and changes source/task distribution. |
| RL data | `STGR-RL.json`, 37,231 examples | `STGR-RL-covered.json`, 22,026 examples if used | Removes 15,205 examples; RL distribution would not match paper. |
| Missing source coverage | Includes temporal, spatial, spatio-temporal, and general QA data | SFT covered set drops TreeVGR visual QA and TVG temporal QA entirely | Weakens/removes two supervision categories from SFT. |
| GPUs / world size | Official scripts use 8 GPUs | Actual checkpoint training args show world size 6 | Changes effective global batch and step count. |
| Distributed optimizer | Official SFT uses DeepSpeed `zero2.json`; official RL uses `zero3.json` | Current SFT checkpoint has `deepspeed=None` | Memory/optimizer behavior differs from official. |
| Attention kernel | `flash_attention_2` | `sdpa` | Mostly a runtime/kernel substitution, but not identical numerically/performance-wise. |
| Video preprocessing | Official defaults/script configuration | `VIDEO_MAX_PIXELS=1605632` in local wrapper | Potentially lower visual budget than official processing. |
| Corrupt media handling | Official collator raises on media-processing failure | Local code reuses the previous successfully loaded example | Silently duplicates examples and hides bad samples. |
| Checkpoint format | Full model checkpoint expected for downstream RL/eval | PEFT adapter checkpoint only | Eval/RL must explicitly load or merge adapter; not directly equivalent. |

## Data Distribution Change

SFT original:

- 31,166 examples
- tasks: 13,000 general video QA MCQ, 7,047 temporal-spatial free-form QA, 5,000 visual QA, 4,119 temporal QA, 2,000 general free-form QA

SFT covered subset:

- 19,746 examples
- tasks: 10,699 general video QA MCQ, 7,047 temporal-spatial free-form QA, 2,000 general free-form QA
- removed: TreeVGR visual QA 5,000, TVG temporal QA 4,119, plus 2,301 VideoR1 MCQ media-missing examples

RL original:

- 37,231 examples
- tasks: 13,000 general video QA MCQ, 12,047 temporal-spatial free-form QA, 5,000 visual QA, 2,904 temporal QA MCQ, 2,280 temporal QA, 2,000 general free-form QA

RL covered subset:

- 22,026 examples
- tasks: 10,699 general video QA MCQ, 7,047 temporal-spatial free-form QA, 2,280 temporal QA, 2,000 general free-form QA
- removed: GQA visual QA 5,000, VideoEspresso temporal-spatial 5,000, TVG temporal QA MCQ 2,904, plus 2,301 VideoR1 MCQ media-missing examples

## Local Changes That Are Mostly Infrastructure

These are probably acceptable for reproduction if they stay limited to paths/environment:

- `Open-o3-Video/eval/scripts/eval_all.sh`: local model/data paths and `PYTHON_BIN`.
- `Open-o3-Video/src/r1-v/configs/data_root.py`: env-based local data root.
- `Open-o3-Video/src/scripts/run_sft_video.sh`: local path defaults and `PYTHONPATH`.
- `Open-o3-Video/src/scripts/run_grpo_video.sh`: local path defaults and `PYTHONPATH`.

## Local Changes That Affect Method

These must be treated as non-paper-faithful:

- `open_o3/scripts/run_sft_gpu.sh`: LoRA, subset data, SDPA, 6/7 GPU wrapper, no DeepSpeed.
- `Open-o3-Video/src/r1-v/src/open_r1/sft_multi_task.py`: fallback to previous good example on corrupt-video failure.
- `Open-o3-Video/src/r1-v/local_scripts/zero2_torch.json` and `zero3_torch.json`: torch Adam workaround, if used.

## Recommended Path To Paper-Faithful Reproduction

1. Complete/fix the missing media rather than filtering to `*-covered.json`.
2. Revert the collator fallback or replace it with an explicit pre-validation step that repairs/removes bad media before training.
3. Run official SFT with full model, full `STGR-SFT.json`, 8 GPUs, DeepSpeed ZeRO-2, and `flash_attention_2`.
4. Run official GSPO RL from the completed SFT checkpoint with full `STGR-RL.json`, 8 GPUs, DeepSpeed ZeRO-3, and `flash_attention_2`.
5. Evaluate the final RL checkpoint with the official V-STAR script and compare to paper metrics.

