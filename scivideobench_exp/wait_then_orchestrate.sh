#!/usr/bin/env bash
# Wait for Qwen-7B HF eval (evaluate_scivideobench.py + evaluate_oracle_flex.py
# Qwen-7B chunks) to finish, then launch run_multimodel_eval.sh.

cd "$(dirname "$0")"

echo "[$(date +%H:%M:%S)] watcher: starting Qwen-7B wait loop"
while true; do
    a=$(pgrep -fc 'evaluate_scivideobench.py.*Qwen2.5-VL-7B' 2>/dev/null || echo 0)
    b=$(pgrep -fc 'evaluate_oracle_flex.py.*Qwen2.5-VL-7B' 2>/dev/null || echo 0)
    total=$((a + b))
    if [ "$total" -eq 0 ]; then
        echo "[$(date +%H:%M:%S)] watcher: no Qwen-7B HF eval processes; launching orchestrator"
        break
    fi
    sleep 60
done

bash run_multimodel_eval.sh
echo "[$(date +%H:%M:%S)] watcher: ALL 6 models orchestrator finished."
