#!/usr/bin/env bash
# Re-run ONLY the Stage B judge (deterministic 72B) on the already-complete inference
# jsons (2094 items each), sequentially with NO concurrency, to get trustworthy metrics
# after the earlier concurrent-eval mixup. Inference jsons are atomic/consistent; only the
# judge logs needed a clean re-run. Writes to eval_<name>_clean.log.
set -u
D=/home/yz0392@unt.ad.unt.edu/xin_ai
EENV=/home/yz0392@unt.ad.unt.edu/miniconda3/envs/open-o3-eval
cd $D/Open-o3-Video/eval
export PATH=$EENV/bin:$PATH
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
vlog=./logs/vstar_logs
for m in rl_faithful baseline_qwen25vl official sft_faithful; do
  echo "######## CLEAN JUDGE $m ########"
  CUDA_VISIBLE_DEVICES=2,3,4,5 $EENV/bin/python ./test/eval_vstar.py \
      --result_file "$vlog/${m}_vstar.json" \
      --model_path $D/open_o3/models/Qwen2.5-72B-Instruct > "$vlog/eval_${m}_clean.log" 2>&1
  r=$(grep -oE "mAM:[0-9.]+" "$vlog/eval_${m}_clean.log" | head -1)
  l=$(grep -oE "mLGM:[0-9.]+" "$vlog/eval_${m}_clean.log" | head -1)
  echo "######## $m -> $r $l ########"
done
echo "ALL_CLEAN_JUDGES_DONE"
