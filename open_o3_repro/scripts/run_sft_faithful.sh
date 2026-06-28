#!/usr/bin/env bash
# Open-o3-Video SFT — FAITHFUL reproduction (replaces the LoRA throwaway).
# Mirrors official src/scripts/run_sft_video.sh EXACTLY except unavoidable infra:
#   - 7 GPUs (1-7); GPU0 is the user's long-running robot-render job -> can't touch.
#     Official used 8 -> global batch 8; here global batch 7 (batch1 x ga1 x 7gpu).
#   - real CUDA via cuda13 (nvcc) so deepspeed zero2 initializes (torch_adam, no FusedAdam build).
# Everything else faithful: full fine-tuning (NO LoRA), deepspeed zero2, flash_attention_2,
# full STGR-SFT.json, lr 1e-6, 1 epoch, save_steps 500, max_grad_norm 5, gradient_checkpointing.
set -u
D=/home/yz0392@unt.ad.unt.edu/xin_ai
REPO=$D/Open-o3-Video
VT=/home/yz0392@unt.ad.unt.edu/miniconda3/envs/vc_train
export OPEN_O3_DATA_ROOT=$D/open_o3/data/Open-o3-Video-data
# real CUDA toolkit (nvcc) for deepspeed; VT/bin on PATH so deepspeed finds the ninja
# binary (needed to JIT-compile FusedAdam); CPATH for cuda headers during that compile.
export CUDA_HOME=$D/cuda13
export PATH=$CUDA_HOME/bin:$VT/bin:$PATH
export CPATH=$CUDA_HOME/include:$CUDA_HOME/targets/x86_64-linux/include:${CPATH:-}
export LIBRARY_PATH=$CUDA_HOME/targets/x86_64-linux/lib:${LIBRARY_PATH:-}
export LD_LIBRARY_PATH=$CUDA_HOME/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH="$REPO/src/r1-v:$REPO/src/r1-v/src:${PYTHONPATH:-}"
export DEBUG_MODE="true"
export WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# VIDEO_MAX_PIXELS: PAPER-FAITHFUL. The paper specifies "uniformly sample 16 frames,
# per-frame resolution not exceeding 128*28*28". Total video budget = 16*128*28*28 =
# 1,605,632. (The official shell omits this env, defaulting vision_process to ~90M total,
# which both contradicts the paper's 128 tokens/frame AND OOMs full-FT on 7B.) So set the
# paper's value explicitly — faithful to the paper and OOM-safe.
export VIDEO_MAX_PIXELS=1605632

MODEL_PATH="${MODEL_PATH:-$D/open_o3/models/Qwen2.5-VL-7B-Instruct}"
EXP_NAME="${EXP_NAME:-sft_faithful}"
OUT_DIR="$D/open_o3/ckpts/${EXP_NAME}"
DATA="${DATA:-${OPEN_O3_DATA_ROOT}/json_data/STGR-SFT.json}"
mkdir -p "$OUT_DIR" "$D/open_o3/logs"
cd "$REPO/src/r1-v"

CUDA_VISIBLE_DEVICES=${GPUS:-1,2,3,4,5,6,7} $VT/bin/torchrun --nproc_per_node="${NP:-7}" \
    --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --master_port=12388 \
    src/open_r1/sft_multi_task.py \
    --output_dir "$OUT_DIR" \
    --model_name_or_path "$MODEL_PATH" \
    --dataset_name "$DATA" \
    --deepspeed "local_scripts/zero2.json" \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-6 \
    --logging_steps 1 \
    --bf16 \
    --report_to none \
    --gradient_checkpointing true \
    --attn_implementation "${ATTN:-flash_attention_2}" \
    --num_train_epochs 1 \
    --run_name "$EXP_NAME" \
    --save_steps 500 \
    --max_grad_norm 5 \
    --save_only_model true
