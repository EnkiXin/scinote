#!/usr/bin/env bash
# Open-o3-Video GRPO RL — FAITHFUL reproduction.
# Mirrors official src/scripts/run_grpo_video.sh EXACTLY except unavoidable infra:
#   - 7 GPUs (1-7); GPU0 = user's robot-render job. Official used 8.
#   - real CUDA via cuda13 so deepspeed zero3 initializes.
# Faithful: grpo.py, zero3.json, max_prompt_length 16384, max_completion_length 768,
# lr 1e-6 cosine, weight_decay 0.01, flash_attention_2, max_pixels 401408, beta 0.04,
# num_generations 4, 1 epoch, save_steps 500, max_grad_norm 5, full STGR-RL.json.
# Starts from the FAITHFUL SFT checkpoint (sft_faithful), per the official two-stage recipe.
set -u
D=/home/yz0392@unt.ad.unt.edu/xin_ai
REPO=$D/Open-o3-Video
VT=/home/yz0392@unt.ad.unt.edu/miniconda3/envs/vc_train
export OPEN_O3_DATA_ROOT=$D/open_o3/data/Open-o3-Video-data
export CUDA_HOME=$D/cuda13
export PATH=$CUDA_HOME/bin:$VT/bin:$PATH
export CPATH=$CUDA_HOME/include:$CUDA_HOME/targets/x86_64-linux/include:${CPATH:-}
export LIBRARY_PATH=$CUDA_HOME/targets/x86_64-linux/lib:${LIBRARY_PATH:-}
export LD_LIBRARY_PATH=$CUDA_HOME/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH="$REPO/src/r1-v:$REPO/src/r1-v/src:${PYTHONPATH:-}"
export DEBUG_MODE="true"
export WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL_PATH="${MODEL_PATH:-$D/open_o3/ckpts/sft_faithful}"
EXP_NAME="${EXP_NAME:-rl_faithful}"
OUT_DIR="$D/open_o3/ckpts/${EXP_NAME}"
DATA="${DATA:-${OPEN_O3_DATA_ROOT}/json_data/STGR-RL.json}"
mkdir -p "$OUT_DIR" "$D/open_o3/logs"
cd "$REPO/src/r1-v"

CUDA_VISIBLE_DEVICES=${GPUS:-1,2,3,4,5,6,7} $VT/bin/torchrun --nproc_per_node="${NP:-7}" \
    --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --master_port=12399 \
    src/open_r1/grpo.py \
    --output_dir "$OUT_DIR" \
    --model_name_or_path "$MODEL_PATH" \
    --dataset_name "$DATA" \
    --deepspeed "local_scripts/zero3.json" \
    --max_prompt_length 16384 \
    --max_completion_length 768 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-6 \
    --lr_scheduler_type "cosine" \
    --weight_decay 0.01 \
    --bf16 \
    --logging_steps 1 \
    --gradient_checkpointing true \
    --attn_implementation "${ATTN:-flash_attention_2}" \
    --max_pixels 401408 \
    --num_train_epochs 1 \
    --run_name "$EXP_NAME" \
    --save_steps 500 \
    --beta 0.04 \
    --max_grad_norm 5 \
    --save_only_model true \
    --num_generations 4
