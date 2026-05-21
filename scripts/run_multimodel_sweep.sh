#!/usr/bin/env bash
# Multi-model × multi-condition × multi-benchmark sweep for ProtoNote.
#
# Usage:
#   bash scripts/run_multimodel_sweep.sh [<model_alias> ...]
#
# Each (model, condition, benchmark) cell shards across 8 GPUs (or TP=4 for 72B).
# Aggregator runs immediately after each cell completes.
#
# Existing Qwen-7B cells are reused — do not pass `qwen7b` to re-run.

set -u
cd /home/yz0392@unt.ad.unt.edu/xin_ai/scinote
PY=/home/yz0392@unt.ad.unt.edu/miniconda3/envs/crag/bin/python

declare -A MODEL_HF=(
    [qwen3b]="Qwen/Qwen2.5-VL-3B-Instruct"
    [qwen7b]="Qwen/Qwen2.5-VL-7B-Instruct"
    [qwen72b]="Qwen/Qwen2.5-VL-72B-Instruct"
    [mimo]="XiaomiMiMo/MiMo-VL-7B-RL"
    [internvl3]="OpenGVLab/InternVL3-8B"
)

MODELS=${@:-"qwen3b mimo qwen72b internvl3"}
CONDS="C0 C1_fixed"
BENCHES="expvid_l1 expvid scivideobench"

mkdir -p logs results_protonote

for model in $MODELS; do
    HF="${MODEL_HF[$model]}"
    [ -z "$HF" ] && { echo "[err] unknown model alias: $model"; continue; }
    is_72b=0
    if [ "$model" = "qwen72b" ]; then is_72b=1; fi

    for cond in $CONDS; do
        for bench in $BENCHES; do
            out_dir="results_protonote/sweep_${model}_${cond}_${bench}"
            # Skip if summary.json already exists (resumable)
            if [ -f "$out_dir/summary.json" ]; then
                echo "[skip] $out_dir already has summary.json"
                continue
            fi
            mkdir -p "$out_dir"
            ts=$(date +%H:%M:%S)
            echo "[$ts] launching $model × $cond × $bench → $out_dir"
            if [ $is_72b -eq 1 ]; then
                # 72B with TP=4: shard 2-way, each chunk uses 4 GPUs
                for chunk in 0 1; do
                    if [ "$chunk" -eq 0 ]; then gpus="0,1,2,3"; else gpus="4,5,6,7"; fi
                    notes_arg=""
                    if [ "$cond" != "C0" ]; then notes_arg="--notes_cache $out_dir/notes_cache_chunk$chunk"; fi
                    CUDA_VISIBLE_DEVICES=$gpus $PY -m protonote.cli \
                        --model "$HF" --benchmark "$bench" --limit 0 \
                        --condition "$cond" --max_frames 32 \
                        --output_dir "$out_dir" --num_chunks 2 --chunk_id $chunk \
                        $notes_arg --device auto \
                        > logs/sweep_${model}_${cond}_${bench}_chunk${chunk}.log 2>&1 &
                done
            else
                # 7B and smaller: 8-way shard, one GPU per chunk
                for chunk in 0 1 2 3 4 5 6 7; do
                    notes_arg=""
                    if [ "$cond" != "C0" ]; then notes_arg="--notes_cache $out_dir/notes_cache_chunk$chunk"; fi
                    CUDA_VISIBLE_DEVICES=$chunk $PY -m protonote.cli \
                        --model "$HF" --benchmark "$bench" --limit 0 \
                        --condition "$cond" --max_frames 32 \
                        --output_dir "$out_dir" --num_chunks 8 --chunk_id $chunk \
                        $notes_arg --device cuda:0 \
                        > logs/sweep_${model}_${cond}_${bench}_chunk${chunk}.log 2>&1 &
                done
            fi
            # Wait for this cell before launching the next
            wait
            # Aggregate
            $PY -m protonote.eval.eval_expvid --output_dir "$out_dir" \
                2>&1 | tee -a logs/sweep_${model}_${cond}_${bench}.log
            ts2=$(date +%H:%M:%S)
            echo "[$ts2] DONE $model × $cond × $bench"
        done
    done
done

echo "Sweep complete. Summaries:"
for d in results_protonote/sweep_*; do
    [ -f "$d/summary.json" ] || continue
    acc=$(/home/yz0392@unt.ad.unt.edu/miniconda3/envs/crag/bin/python -c \
        "import json; print(json.load(open('$d/summary.json')).get('overall_acc','?'))")
    n=$(/home/yz0392@unt.ad.unt.edu/miniconda3/envs/crag/bin/python -c \
        "import json; print(json.load(open('$d/summary.json')).get('n_valid','?'))")
    echo "  $(basename $d): acc=$acc%  n=$n"
done
