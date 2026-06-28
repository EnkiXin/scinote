#!/usr/bin/env bash
# V-STaR eval runner — two stages (vLLM inference -> 72B HF judge), adapted to this box.
# Avoids GPU0 (robot job) and GPU1 (benchmark job): uses GPUs 2-7 for inference and 2-5
# for the judge. Uses the dedicated open-o3-eval env (vllm 0.7.2). Must run from eval/.
#
# Usage: MODEL_PATH=<ckpt> EXP_NAME=<name> bash run_eval.sh
#   official: MODEL_PATH=.../open_o3/models/Open-o3-Video-official EXP_NAME=official
#   sft:      MODEL_PATH=.../open_o3/ckpts/sft_faithful/checkpoint-XXXX EXP_NAME=sft_faithful
#   rl:       MODEL_PATH=.../open_o3/ckpts/rl_faithful/checkpoint-XXXX  EXP_NAME=rl_faithful
set -u
D=/home/yz0392@unt.ad.unt.edu/xin_ai
EVAL=$D/Open-o3-Video/eval
EENV=/home/yz0392@unt.ad.unt.edu/miniconda3/envs/open-o3-eval
PY=$EENV/bin/python

MODEL_PATH="${MODEL_PATH:-$D/open_o3/models/Open-o3-Video-official/}"
LLM_PATH="${LLM_PATH:-$D/open_o3/models/Qwen2.5-72B-Instruct}"
EXP_NAME="${EXP_NAME:-official}"
VID="${VSTAR_VIDEO_FOLDER:-$D/open_o3/data/V-STaR/videos/}"
ANNO="${VSTAR_ANNO_FILE:-$D/open_o3/data/V-STaR/V_STaR_test.json}"
INFER_GPUS="${INFER_GPUS:-2,3,4,5,6,7}"
INFER_N="${INFER_N:-6}"
JUDGE_GPUS="${JUDGE_GPUS:-2,3,4,5}"

cd "$EVAL"
mkdir -p ./logs/vstar_logs
export PATH=$EENV/bin:$PATH
# spawned multiprocessing workers need eval/ on sys.path for `from models.model_vllm ...`
export PYTHONPATH="$EVAL:${PYTHONPATH:-}"

echo "[eval:$EXP_NAME] Stage A inference ($INFER_N GPUs: $INFER_GPUS) model=$MODEL_PATH"
NUM_GPUS=$INFER_N CUDA_VISIBLE_DEVICES=$INFER_GPUS "$PY" ./test/test_vstar_multi_images.py \
    --video_folder "$VID" \
    --anno_file "$ANNO" \
    --result_file "./logs/vstar_logs/${EXP_NAME}_vstar.json" \
    --model_path "$MODEL_PATH" \
    --model_kwargs ./config/vstar.yaml \
    --think_mode > "./logs/vstar_logs/test_${EXP_NAME}_vstar.log" 2>&1

echo "[eval:$EXP_NAME] Stage B judge (72B on GPUs: $JUDGE_GPUS)"
CUDA_VISIBLE_DEVICES=$JUDGE_GPUS "$PY" ./test/eval_vstar.py \
    --result_file "./logs/vstar_logs/${EXP_NAME}_vstar.json" \
    --model_path "$LLM_PATH" > "./logs/vstar_logs/eval_${EXP_NAME}_vstar.log" 2>&1

echo "[eval:$EXP_NAME] done. Overall Statistics (mAM/mLGM):"
grep -A 20 "Overall Statistics" "./logs/vstar_logs/eval_${EXP_NAME}_vstar.log" 2>/dev/null | grep -iE "mAM|mLGM|acc|Overall" | head
