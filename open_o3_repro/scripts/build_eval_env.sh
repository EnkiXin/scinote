#!/usr/bin/env bash
# Build the dedicated V-STaR eval env per Open-o3-Video setup.sh.
# Eval needs vllm==0.7.2 (Stage A inference) + transformers (Stage B 72B judge) +
# qwen_vl_utils/numpy/tqdm/cv2/yaml/pillow/decord. vllm 0.7.2 pins torch 2.5.1, so
# this is a SEPARATE env from vc_train (torch 2.11). Isolated per project rule.
set -uo pipefail
CONDA=/home/yz0392@unt.ad.unt.edu/miniconda3
ENV=open-o3-eval
PY=$CONDA/envs/$ENV/bin/python
PIP=$CONDA/envs/$ENV/bin/pip

echo "=== [1/5] create env ==="
$CONDA/bin/conda create -n $ENV python=3.11 -y 2>&1 | tail -2

echo "=== [2/5] vllm 0.7.2 (pulls torch 2.5.1) ==="
$PIP install vllm==0.7.2 2>&1 | tail -3

echo "=== [3/5] transformers @ pinned commit (per setup.sh) ==="
$PIP install "git+https://github.com/huggingface/transformers.git@336dc69d63d56f232a183a3e7f52790429b871ef" 2>&1 | tail -3

echo "=== [4/5] eval deps ==="
$PIP install qwen_vl_utils numpy tqdm opencv-python-headless pyyaml pillow decord pandas nltk rouge_score pysubs2 2>&1 | tail -3

echo "=== [5/5] flash_attn prebuilt (torch 2.5.x) ==="
$PIP install flash_attn==2.7.4.post1 --no-build-isolation 2>&1 | tail -3 || echo "flash_attn optional for eval; sdpa fallback ok"

echo "=== verify ==="
$PY -c "import vllm, transformers, torch; print('vllm',vllm.__version__,'transformers',transformers.__version__,'torch',torch.__version__)" 2>&1 | tail -1
$PY -c "import numpy,tqdm,cv2,yaml,PIL,decord; print('eval deps OK')" 2>&1 | tail -1
echo "EVAL_ENV_BUILD_DONE"
