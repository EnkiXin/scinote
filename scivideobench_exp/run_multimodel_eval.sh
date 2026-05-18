#!/usr/bin/env bash
# Orchestrator: run evaluate_multimodel.py on every open-source MLLM in
# sequence, both C0 and C-vl-noter conditions, 4 chunks each across 8 GPUs.
#
# Per model:
#   GPU 0-3 → C0          chunks 0/1/2/3
#   GPU 4-7 → C-vl-noter  chunks 0/1/2/3
#
# Models that fail to load through vLLM (e.g. unsupported architecture) are
# logged and skipped; the orchestrator keeps going.

set -u
cd "$(dirname "$0")"

PY=/home/yz0392@unt.ad.unt.edu/miniconda3/envs/crag/bin/python
SCRIPT=evaluate_multimodel.py
LOG_DIR=logs/multimodel
RESULTS_DIR=results_scivideobench
mkdir -p "$LOG_DIR"

# Order: easier (Qwen-backbone) → distinct families → larger
MODELS=(
    "XiaomiMiMo/MiMo-VL-7B-RL"
    "OpenGVLab/InternVL3-8B"
    "Kwai-Keye/Keye-VL-8B-Preview"
    "THUDM/GLM-4.1V-9B-Thinking"
    "moonshotai/Kimi-VL-A3B-Thinking"
    "DAMO-NLP-SG/VideoLLaMA3-7B"
)

CONDITIONS=("C0" "C-vl-noter")
N_CHUNKS=4

short_tag() {
    # Strip org/, lowercase, replace -/. with _, keep alnum/_
    echo "$1" | sed 's|.*/||; s/[-\.]/_/g' | tr 'A-Z' 'a-z' | head -c 32
}

for MODEL in "${MODELS[@]}"; do
    TAG_BASE=$(short_tag "$MODEL")
    echo "================================================================"
    echo "[$(date +%H:%M:%S)] Starting $MODEL  (tag base = $TAG_BASE)"
    echo "================================================================"

    # Reap any leftover VLLM::EngineCore zombies from a prior failed run
    pkill -9 -f "VLLM::EngineCore" 2>/dev/null
    sleep 3

    PIDS=()
    for CHUNK in 0 1 2 3; do
        GPU_C0=$CHUNK
        GPU_CN=$((CHUNK + 4))
        # C0
        nohup env CUDA_VISIBLE_DEVICES=$GPU_C0 VLLM_USE_DEEP_GEMM=0 VLLM_USE_DEEP_GEMM_E8M0=0 \
            $PY $SCRIPT --model "$MODEL" --condition C0 \
            --tag "c0_${TAG_BASE}" --chunk_id $CHUNK --num_chunks $N_CHUNKS \
            --resume \
            > "$LOG_DIR/c0_${TAG_BASE}_gpu${GPU_C0}.log" 2>&1 &
        PIDS+=($!)
        # C-vl-noter
        nohup env CUDA_VISIBLE_DEVICES=$GPU_CN VLLM_USE_DEEP_GEMM=0 VLLM_USE_DEEP_GEMM_E8M0=0 \
            $PY $SCRIPT --model "$MODEL" --condition C-vl-noter \
            --notes_subdir trained_vl_noter_notes --key_mode vid_qid \
            --tag "c_vl_noter_${TAG_BASE}" --chunk_id $CHUNK --num_chunks $N_CHUNKS \
            --resume \
            > "$LOG_DIR/c_vl_noter_${TAG_BASE}_gpu${GPU_CN}.log" 2>&1 &
        PIDS+=($!)
    done

    echo "  Launched 8 jobs (PIDs: ${PIDS[*]}); waiting for all to finish..."
    for pid in "${PIDS[@]}"; do
        wait $pid 2>/dev/null
    done
    echo "[$(date +%H:%M:%S)]   all 8 jobs for $MODEL finished."

    # Quick aggregate
    $PY -c "
import json, glob
from collections import defaultdict
for tag in ['c0_${TAG_BASE}', 'c_vl_noter_${TAG_BASE}']:
    rows = []
    for p in sorted(glob.glob(f'$RESULTS_DIR/' + tag + '/eval_scivideobench_chunk*.json')):
        try:
            j = json.load(open(p)); rows.extend(j.get('results', []))
        except Exception as e:
            print(f'    skip {p}: {e}'); continue
    valid = [r for r in rows if 'error' not in r and 'pred' in r]
    n_err = sum(1 for r in rows if 'error' in r)
    if not valid:
        print(f'  {tag}: NO VALID RESULTS ({n_err} errors)'); continue
    acc = sum(r['score'] for r in valid) / max(len(valid), 1) * 100
    print(f'  {tag}: {acc:.2f}% (n={len(valid)}, err={n_err})')
"
    echo ""
done

echo "================================================================"
echo "[$(date +%H:%M:%S)] All models done. See $LOG_DIR/ and $RESULTS_DIR/c{0,_vl_noter}_<model>/"
echo "================================================================"
