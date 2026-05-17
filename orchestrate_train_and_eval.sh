#!/bin/bash
# After ExpVid C-oracle eval finishes, launch DDP training of the noter on
# 4 GPUs, then run inference on SciVideoBench, then evaluate.
set -e
cd /home/yz0392@unt.ad.unt.edu/xin_ai/scinote
mkdir -p logs

PY=/home/yz0392@unt.ad.unt.edu/miniconda3/envs/crag/bin/python
TORCHRUN=/home/yz0392@unt.ad.unt.edu/miniconda3/envs/crag/bin/torchrun

EXPVID_ORCH=logs/expvid_eval_orchestrator.log

echo "[Phase 1] waiting for ExpVid C-oracle eval to finish ..."
until grep -q "ALL EXPVID CHUNKS DONE" "$EXPVID_ORCH" 2>/dev/null; do sleep 60; done
echo "[Phase 1] ExpVid eval DONE. Verifying GPUs free ..."
sleep 30

# Give any lingering vLLM/transformers child processes time to die
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

echo "[Phase 2] Launching DDP training on GPU 0,1,2,3,4,6,7 (7-way, skip ml1178's GPU 5)"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,6,7 \
  $TORCHRUN --nproc_per_node=7 \
    --master_port=29501 \
    train_notetaker.py \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --train_jsonl train_data/expvid_oracle_sft_train.jsonl \
    --val_jsonl   train_data/expvid_oracle_sft_val.jsonl \
    --output_dir  checkpoints/notetaker_lora_v1 \
    --lora_r 32 --lora_alpha 64 --learning_rate 1e-4 \
    --per_device_batch_size 1 --grad_accum 8 \
    --epochs 2 --max_frames 16 \
    > logs/train_notetaker_v1.log 2>&1

echo "[Phase 2] Training DONE"

NCHUNKS=7
GPU_LIST=(0 1 2 3 4 6 7)

echo "[Phase 3] Inference: trained noter on SciVideoBench (7 GPUs parallel)"
PIDS=()
for i in 0 1 2 3 4 5 6; do
  gpu=${GPU_LIST[$i]}
  logf="logs/sciv_lora_infer_gpu${gpu}_chunk${i}of${NCHUNKS}.log"
  CUDA_VISIBLE_DEVICES=$gpu nohup $PY generate_notes_with_lora.py \
    --base_model Qwen/Qwen2.5-VL-7B-Instruct \
    --lora_path checkpoints/notetaker_lora_v1/final \
    --chunk_id $i --num_chunks $NCHUNKS \
    > "$logf" 2>&1 &
  PIDS+=($!)
  echo "  GPU $gpu chunk $i/$NCHUNKS -> PID $!"
done
for p in "${PIDS[@]}"; do wait $p; done
echo "[Phase 3] Inference DONE"

echo "[Phase 4] Eval: Qwen-3B + trained-noter notes on SciVideoBench (7-way)"
EXPVID_ENV=/home/yz0392@unt.ad.unt.edu/miniconda3/envs/expvid/bin/python
cd /home/yz0392@unt.ad.unt.edu/xin_ai/scivideobench
PIDS=()
for i in 0 1 2 3 4 5 6; do
  gpu=${GPU_LIST[$i]}
  logf="logs/c_trained_noter_gpu${gpu}_chunk${i}of${NCHUNKS}.log"
  CUDA_VISIBLE_DEVICES=$gpu nohup $EXPVID_ENV evaluate_oracle_flex.py \
    --model Qwen/Qwen2.5-VL-3B-Instruct --output results_scivideobench \
    --notes_subdir trained_noter_notes --key_mode vid_qid \
    --tag c_trained_noter \
    --chunk_id $i --num_chunks $NCHUNKS --resume \
    > "$logf" 2>&1 &
  PIDS+=($!)
done
for p in "${PIDS[@]}"; do wait $p; done
echo "[Phase 4] Eval DONE"
echo "ALL PHASES DONE"
