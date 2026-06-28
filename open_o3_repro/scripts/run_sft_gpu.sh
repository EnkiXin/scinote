#!/usr/bin/env bash
# Open-o3-Video SFT cold-start on THIS GPU node (2026-06-18).
# Faithful to official run_sft_video.sh; only infra overrides:
#  - GPUs 1-7 (GPU0 busy), nproc 7
#  - sdpa (no flash_attn on cu130)
#  - CUDA_HOME=fake_cuda (deepspeed import-time nvcc check; zero2 uses torch AdamW)
#  - media-covered subset (avoid missing-media crash)
#  - vc_train env (transformers 4.49 + trl 0.16.1, official-code-native)
set -u
D=/home/yz0392@unt.ad.unt.edu/xin_ai
REPO=$D/Open-o3-Video
VT=/home/yz0392@unt.ad.unt.edu/miniconda3/envs/vc_train
export OPEN_O3_DATA_ROOT=$D/open_o3/data/Open-o3-Video-data
export CUDA_HOME=$D/fake_cuda
export PATH=$CUDA_HOME/bin:$PATH
export PYTHONPATH="$REPO/src/r1-v:$REPO/src/r1-v/src:${PYTHONPATH:-}"
export WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VIDEO_PIXELS_FACTOR=128
export VIDEO_MAX_PIXELS=1605632   # total video pixel budget (16 frames x 100352); default 90M OOMs

MODEL_PATH="${MODEL_PATH:-$D/open_o3/models/Qwen2.5-VL-7B-Instruct}"
EXP_NAME="${EXP_NAME:-sft_repro}"
OUT_DIR="$D/open_o3/ckpts/${EXP_NAME}"
DATA="${OPEN_O3_DATA_ROOT}/json_data/STGR-SFT-covered.json"
mkdir -p "$OUT_DIR" "$D/open_o3/logs"
cd "$REPO/src/r1-v"

CUDA_VISIBLE_DEVICES=${GPUS:-1,2,3,4,5,6,7} $VT/bin/torchrun --nproc_per_node="${NP:-7}" \
    --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --master_port=12377 \
    src/open_r1/sft_multi_task.py \
    --output_dir "$OUT_DIR" \
    --model_name_or_path "$MODEL_PATH" \
    --dataset_name "$DATA" \
    --use_peft true \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_target_modules q_proj k_proj v_proj o_proj \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-6 \
    --logging_steps 5 \
    --bf16 \
    --report_to none \
    --gradient_checkpointing true \
    --attn_implementation sdpa \
    --num_train_epochs 1 \
    --run_name "$EXP_NAME" \
    --save_steps 500 \
    --max_grad_norm 5 \
    --save_only_model true
