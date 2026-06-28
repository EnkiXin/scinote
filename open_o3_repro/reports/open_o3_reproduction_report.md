# Open-o3-Video Reproduction Report

## Status

Reproduction workspace is prepared much further than the initial setup:

- Official repo is cloned and locally patched only for paths/wrappers.
- Official released model, Qwen2.5-VL-7B base model, and Qwen2.5-72B judge are downloaded.
- V-STaR test annotation and videos are downloaded and unpacked.
- STGR JSON, STGR media, TimeRFT media, VideoR1 LLaVA/STAR media, and VideoR1 free-form coverage are available.
- Official V-STaR run is still blocked in this shell by runtime environment: no conda env, no torch/deps, no visible CUDA driver.
- Remaining train-media gaps require more network access or manual source-data setup.

## Official Sources

- GitHub: https://github.com/marinero4972/Open-o3-Video
- Dataset: https://huggingface.co/datasets/marinero4972/Open-o3-Video
- Official model: https://huggingface.co/marinero4972/Open-o3-Video
- V-STaR: https://huggingface.co/datasets/V-STaR-Bench/V-STaR

## Commit

- Local official repo: `/home/yz0392@unt.ad.unt.edu/xin_ai/Open-o3-Video`
- Commit: `4797fd1adfd261eea821e0fcb57c876afdbc0362`
- Commit record: `/home/yz0392@unt.ad.unt.edu/xin_ai/open_o3/open_o3_commit.txt`

## Directory Layout

The requested `/work/open_o3` path could not be created on this machine:

```text
mkdir: cannot create directory '/work': Permission denied
```

Fallback root used:

```text
/home/yz0392@unt.ad.unt.edu/xin_ai/open_o3/
  data/
  models/
  ckpts/
  logs/
  reports/
  repos/Open-o3-Video -> ../../Open-o3-Video
  scripts/
```

## Downloaded Assets

Approximate local sizes:

| asset | local path | status |
| --- | --- | --- |
| Open-o3 STGR data/media | `open_o3/data/Open-o3-Video-data` | 315G |
| V-STaR benchmark | `open_o3/data/V-STaR` | 56G, 743 videos |
| source zip archives | `open_o3/data/source_datasets` | 223G |
| Qwen2.5-VL-7B-Instruct | `open_o3/models/Qwen2.5-VL-7B-Instruct` | 13G |
| Open-o3-Video official model | `open_o3/models/Open-o3-Video-official` | 13G |
| Qwen2.5-72B-Instruct judge | `open_o3/models/Qwen2.5-72B-Instruct` | 107G, 37 safetensor shards |

Media counts:

- STGR extracted files: 40,404
- TimeRFT extracted videos: 2,475
- VideoR1 extracted files: 45,116
- V-STaR extracted videos: 743

## Data Sanity

Latest report:

- `open_o3/reports/data_sanity_report.md`
- `open_o3/reports/data_sanity_stats.json`
- `open_o3/reports/data_sanity_missing_media_examples.csv`

Current official-path coverage:

| split | checked | existing | missing |
| --- | ---: | ---: | ---: |
| STGR-SFT | 48,154 | 36,734 | 11,420 |
| STGR-RL | 49,278 | 34,073 | 15,205 |

Covered sources:

- STGR SFT/RL: 100%
- TimeRFT: 100%
- VideoR1 free-form: 100%
- VideoR1 MCQ: 82.30%
- VideoEspresso keyframes: present

Remaining media gaps:

| source | remaining gap |
| --- | ---: |
| TVG-R1 media | SFT 4,119 paths, RL 2,904 paths |
| TreeVGR images | SFT 5,000 paths |
| GQA images | RL 5,000 paths |
| VideoEspresso full videos | RL 5,000 videos |
| VideoR1 MCQ residual | 2,301 paths, from `NeXT-QA`, `CLEVRER`, `PerceptionTest` |

Attempted next download for `NeXT-QA`, `CLEVRER`, and `PerceptionTest`, but the required network escalation was rejected by the session usage limit. No workaround was attempted.

## Runtime Check

Current shell:

- system Python: 3.13.5
- conda exists at `/home/yz0392@unt.ad.unt.edu/miniconda3/bin/conda`
- best existing env found: `vc_train`
- `vc_train` has `torch`, `transformers`, `datasets`, `deepspeed`, `trl`, `vllm`, `qwen_vl_utils`
- `vc_train` is missing `flash_attn`
- `nvcc`: missing; `/usr/local/cuda/bin/nvcc` does not exist
- `nvidia-smi -L`: cannot communicate with NVIDIA driver
- `/dev/nvidia*`: missing

`open_o3/scripts/check_runtime_ready.py --mode train` confirms all key local paths exist, but CUDA training runtime is not ready: no visible GPUs, no `nvcc`, and no `flash_attn`.

## Local Code Changes

Only path/wrapper/checking helpers were added or changed; no model architecture, reward logic, or method behavior was changed.

- `Open-o3-Video/src/r1-v/configs/data_root.py`: local `DATA_ROOT` default, overridable with `OPEN_O3_DATA_ROOT`.
- `Open-o3-Video/eval/scripts/eval_all.sh`: local model, V-STaR, Python-bin, and local 72B judge defaults.
- `Open-o3-Video/src/scripts/run_sft_video.sh`: local base model/output defaults.
- `Open-o3-Video/src/scripts/run_grpo_video.sh`: local SFT checkpoint/output defaults.
- `Open-o3-Video/tools/check_open_o3_data.py`: official media-path sanity checker.
- `open_o3/scripts/*`: download, sanity, setup, eval, SFT/RL wrapper, runtime preflight, and unpack helpers.
- `open_o3/scripts/run_trained_pipeline.sh`: trained-result pipeline for SFT -> SFT V-STaR eval -> RL -> RL V-STaR eval.

## Official V-STaR Evaluation

Path preflight is now satisfied:

- official model config exists
- V-STaR annotation exists
- V-STaR video folder exists
- local 72B judge config exists

Latest run attempt:

```bash
cd /home/yz0392@unt.ad.unt.edu/xin_ai
PYTHON_BIN=python3 open_o3/scripts/run_official_eval_vstar.sh
```

Result on 2026-06-18 19:19 CDT: failed before model inference because required packages are absent in this shell.

- generation log: `ModuleNotFoundError: No module named 'tqdm'`
- judge log: `ModuleNotFoundError: No module named 'numpy'`
- logs:
  - `Open-o3-Video/eval/logs/vstar_logs/test_open_o3_official_eval_vstar.log`
  - `Open-o3-Video/eval/logs/vstar_logs/eval_open_o3_official_eval_vstar.log`

Expected command on the target H200 environment:

```bash
cd /home/yz0392@unt.ad.unt.edu/xin_ai
open_o3/scripts/setup_target_env.sh
PYTHON_BIN=python open_o3/scripts/run_official_eval_vstar.sh
```

## SFT And RL

The user-requested target is trained results, not only official-checkpoint evaluation. A real SFT launch was attempted:

```bash
cd /home/yz0392@unt.ad.unt.edu/xin_ai
/home/yz0392@unt.ad.unt.edu/miniconda3/bin/conda run -n vc_train bash open_o3/scripts/run_sft_repro.sh
```

Initial issue fixed:

- `torchrun` child processes could not import `configs`; fixed by exporting `PYTHONPATH="$PWD:${PYTHONPATH:-}"` inside official SFT/RL launcher scripts.

Latest SFT result on 2026-06-18 19:24 CDT:

- `torchrun` starts.
- It fails before model/data loading during DeepSpeed initialization.
- Root cause: `FileNotFoundError: /usr/local/cuda/bin/nvcc`.
- Additional training blockers on this node: no `/dev/nvidia*`, `torch.cuda.device_count() == 0`, missing `flash_attn`.
- Log: `open_o3/logs/sft_repro.log`.

So SFT/RL trained checkpoints do not exist yet:

- `open_o3/ckpts/sft_repro/`: missing
- `open_o3/ckpts/rl_repro/`: missing

The trained-result wrapper command is ready for a proper GPU node:

```bash
cd /home/yz0392@unt.ad.unt.edu/xin_ai
CONDA_BIN=/home/yz0392@unt.ad.unt.edu/miniconda3/bin/conda \
CONDA_ENV=vc_train \
open_o3/scripts/run_trained_pipeline.sh
```

Equivalent manual sequence:

```bash
cd /home/yz0392@unt.ad.unt.edu/xin_ai
open_o3/scripts/run_sft_repro.sh
open_o3/scripts/run_rl_repro.sh
```

Before full-paper training, finish the remaining source-data gaps or decide whether to train only on currently covered subsets.

## Next Steps

1. Move to a node with 8 visible GPUs and a full CUDA toolkit (`nvcc` available).
2. Install/activate an env with `flash_attn`; `vc_train` is close but currently missing it on this node.
3. Run `python open_o3/scripts/check_runtime_ready.py --open-o3-root open_o3 --repo-dir Open-o3-Video --mode train --min-gpus 8`.
4. Run trained pipeline with `CONDA_ENV=... open_o3/scripts/run_trained_pipeline.sh`.
5. Run official V-STaR eval with `PYTHON_BIN=python open_o3/scripts/run_official_eval_vstar.sh` for any checkpoint that should be compared.
6. Resume downloading residual media:
   - `Video-R1/Video-R1-data`: include `NeXT-QA/*`, `CLEVRER/*`, `PerceptionTest/*`.
   - `hshjerry0315/VideoEspresso_train_video`: full video split archive.
   - `Boshenxx/TimeR1-Dataset`: done for TimeRFT.
   - TVG-R1, TreeVGR, GQA via their official source instructions.
7. Re-run `open_o3/scripts/run_data_sanity.sh`.
8. Compare SFT/RL V-STaR summaries to paper numbers.
