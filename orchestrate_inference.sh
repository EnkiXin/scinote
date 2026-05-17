#!/bin/bash
# After training + Step 2 both finish, run 7-chunk parallel inference + eval.
set -e
cd /home/yz0392@unt.ad.unt.edu/xin_ai/scinote
mkdir -p logs

PY=/home/yz0392@unt.ad.unt.edu/miniconda3/envs/crag/bin/python
EXPVID_PY=/home/yz0392@unt.ad.unt.edu/miniconda3/envs/expvid/bin/python

TRAIN_LOG=/home/yz0392@unt.ad.unt.edu/xin_ai/scinote/logs/train_text_single_v3.log
STEP2_LOG=/home/yz0392@unt.ad.unt.edu/xin_ai/scivideobench/logs/2step_step2.log

echo "[Phase 1] waiting for training to finish ..."
until grep -q "saved →" "$TRAIN_LOG" 2>/dev/null; do sleep 60; done
echo "[Phase 1] Training DONE"

echo "[Phase 2] waiting for Step 2 to finish ..."
until grep -q "✅ Step 2 done" "$STEP2_LOG" 2>/dev/null; do sleep 60; done
echo "[Phase 2] Step 2 DONE"

sleep 30  # let GPU memory clear

NCHUNKS=7
GPU_LIST=(0 1 2 3 4 6 7)

echo "[Phase 3] 7-chunk parallel trained-noter inference on SciVideoBench"
PIDS=()
for i in 0 1 2 3 4 5 6; do
  gpu=${GPU_LIST[$i]}
  logf="logs/trained_noter_infer_gpu${gpu}_chunk${i}of${NCHUNKS}.log"
  CUDA_VISIBLE_DEVICES=$gpu TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 nohup $PY generate_notes_with_trained_textonly.py \
    --lora_path checkpoints/notetaker_text_lora_single_v3/final \
    --chunk_id $i --num_chunks $NCHUNKS \
    > "$logf" 2>&1 &
  PIDS+=($!)
  echo "  GPU $gpu chunk $i/$NCHUNKS -> PID $!"
done
for p in "${PIDS[@]}"; do wait $p; done
echo "[Phase 3] inference DONE"

echo "[Phase 4] 7-chunk parallel Qwen-3B answer with trained-noter notes"
cd /home/yz0392@unt.ad.unt.edu/xin_ai/scivideobench
PIDS=()
for i in 0 1 2 3 4 5 6; do
  gpu=${GPU_LIST[$i]}
  logf="logs/c_trained_noter_gpu${gpu}_chunk${i}of${NCHUNKS}.log"
  CUDA_VISIBLE_DEVICES=$gpu nohup $EXPVID_PY evaluate_oracle_flex.py \
    --model Qwen/Qwen2.5-VL-3B-Instruct --output results_scivideobench \
    --notes_subdir trained_noter_notes --key_mode vid_qid \
    --tag c_trained_noter \
    --chunk_id $i --num_chunks $NCHUNKS --resume \
    > "$logf" 2>&1 &
  PIDS+=($!)
done
for p in "${PIDS[@]}"; do wait $p; done
echo "ALL PHASES DONE"
