#!/usr/bin/env bash
# Run V-STaR eval (Stage A inference + Stage B judge) for baseline / official / sft,
# sequentially, each using all 6 free GPUs (2-7) for inference and 2-5 for the judge.
# rl_faithful is evaluated separately (already running). Priority order: baseline first
# (needed for the +delta vs paper), then official (pipeline validation), then sft.
set -u
D=/home/yz0392@unt.ad.unt.edu/xin_ai
# name -> model path
run_one () {
  local name="$1" mpath="$2"
  echo "######## EVAL $name ($mpath) ########"
  MODEL_PATH="$mpath" EXP_NAME="$name" bash $D/open_o3/scripts/run_eval.sh
  echo "######## DONE $name ########"
}
run_one "baseline_qwen25vl" "$D/open_o3/models/Qwen2.5-VL-7B-Instruct"
run_one "official"          "$D/open_o3/models/Open-o3-Video-official"
run_one "sft_faithful"      "$D/open_o3/ckpts/sft_faithful/checkpoint-5083"
echo "ALL_REMAINING_EVALS_DONE"
