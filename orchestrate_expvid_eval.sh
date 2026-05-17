#!/bin/bash
# Two-phase orchestrator for ExpVid C-oracle eval:
#   Phase A: as soon as ExpVid 72B oracle gen finishes, fire chunks 0/7 + 1/7 on GPU 4,6
#   Phase B: as soon as SciVideoBench C-oracle eval finishes, fire chunks 2-6/7 on GPU 0,1,2,3,7
# Each chunk writes to its own eval_<task>_chunk{i}of7.json with --resume.
set -e
cd /home/yz0392@unt.ad.unt.edu/xin_ai/scinote
mkdir -p logs

PY=/home/yz0392@unt.ad.unt.edu/miniconda3/envs/expvid/bin/python
MODEL="Qwen/Qwen2.5-VL-7B-Instruct"
TASK=all_level2_3
NCHUNKS=7

EXPVID_ORACLE_LOG=/home/yz0392@unt.ad.unt.edu/xin_ai/scinote/logs/expvid_oracle_72b_v2.log
SCIVB_EVAL_LOG=/home/yz0392@unt.ad.unt.edu/xin_ai/scivideobench/logs/c_oracle_72b_orchestrator.log

launch_chunk() {
  local gpu=$1
  local i=$2
  local logf="logs/expvid_coracle_eval_gpu${gpu}_chunk${i}of${NCHUNKS}.log"
  CUDA_VISIBLE_DEVICES=$gpu nohup $PY evaluate_oracle_expvid.py \
    --task $TASK --model "$MODEL" --output results_h200_unified \
    --chunk_id $i --num_chunks $NCHUNKS --resume \
    >> "$logf" 2>&1 &
  echo "  GPU $gpu (chunk $i/$NCHUNKS) -> PID $!, log: $logf"
}

echo "[Phase A] waiting for ExpVid 72B oracle gen ..."
until grep -qE "✅ Done\." "$EXPVID_ORACLE_LOG" 2>/dev/null; do sleep 30; done
echo "[Phase A] ExpVid oracle gen DONE. Launching chunks 0,1 on GPU 4,6 ..."
launch_chunk 4 0
launch_chunk 6 1

echo "[Phase B] waiting for SciVideoBench C-oracle eval ..."
until grep -q "ALL 5 CHUNKS DONE" "$SCIVB_EVAL_LOG" 2>/dev/null; do sleep 30; done
echo "[Phase B] SciVideoBench eval DONE. Launching chunks 2-6 on GPU 0,1,2,3,7 ..."
launch_chunk 0 2
launch_chunk 1 3
launch_chunk 2 4
launch_chunk 3 5
launch_chunk 7 6

echo "All 7 chunks running. Waiting for completion ..."
wait
echo "ALL EXPVID CHUNKS DONE"
