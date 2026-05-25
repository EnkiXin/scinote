#!/usr/bin/env bash
#
# run_v7_full.sh — sequential SciVB 218 then ExpVid 745 on GPUs 4-7.
#
# Runs in foreground; intended for nohup. Pushes progress after each
# benchmark finishes.

set -e
cd /home/yz0392@unt.ad.unt.edu/xin_ai/scinote
source activate crag

export CUDA_VISIBLE_DEVICES=4,5,6,7

echo "=== [$(date)] Starting V7 SciVB 218 ==="
python -m protonote.v7.run_react \
    --benchmark scivideobench \
    --limit 0 \
    --output_dir results_protonote_v7/v7_react_scivb \
    --condition_label v7_react

echo "=== [$(date)] SciVB done. Starting V7 ExpVid 745 ==="
python -m protonote.v7.run_react \
    --benchmark expvid \
    --limit 0 \
    --output_dir results_protonote_v7/v7_react_expvid \
    --condition_label v7_react

echo "=== [$(date)] All V7 runs done ==="
