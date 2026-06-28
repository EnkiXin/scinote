# Open-o3-Video Reproduction Workspace

This folder is intentionally placed next to `scinote` as the local fallback for the requested `/work/open_o3` layout.

## Layout

```text
open_o3/
  data/Open-o3-Video-data/          # STGR json/media plus externally sourced media
  data/V-STaR/                       # V-STaR annotation and videos
  data/source_datasets/              # source zip archives used to unpack media
  models/Qwen2.5-VL-7B-Instruct/    # base model
  models/Open-o3-Video-official/    # official released model
  models/Qwen2.5-72B-Instruct/      # local V-STaR judge model
  ckpts/                            # SFT/RL checkpoint outputs
  logs/                             # local run logs
  reports/                          # reproduction and sanity reports
  repos/Open-o3-Video               # symlink to the local official repo clone
  scripts/                          # helper scripts for download/checks
```

## Helpers

```bash
open_o3/scripts/download_official_assets.sh
open_o3/scripts/run_data_sanity.sh
open_o3/scripts/check_runtime_ready.py
open_o3/scripts/setup_target_env.sh
open_o3/scripts/run_official_eval_vstar.sh
open_o3/scripts/run_sft_repro.sh
open_o3/scripts/run_rl_repro.sh
open_o3/scripts/run_trained_pipeline.sh
```

Official method code is kept in the official repo. This workspace is only for paths, downloads, checkpoints, logs, and reports.

Latest state is summarized in:

```text
open_o3/reports/open_o3_reproduction_report.md
open_o3/reports/data_sanity_report.md
```
