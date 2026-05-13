#!/bin/bash
# Setup script for ExpVid evaluation on a fresh GPU server (RTX 4090)
# Usage: bash setup_server.sh

set -e
echo "=== ExpVid Server Setup ==="

# HuggingFace cache to work dir (avoid filling /root)
export HF_HOME=~/work/hf_cache
mkdir -p ~/work/hf_cache

# Install dependencies
pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -q transformers accelerate
pip install -q qwen-vl-utils
pip install -q av pillow numpy huggingface_hub

echo "=== Setup complete. Run: bash run_eval.sh ==="
