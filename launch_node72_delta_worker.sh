#!/usr/bin/env bash
set -euo pipefail

gpu=${1:?GPU index required}
case "$gpu" in 2|3|4|6|7) ;; *) echo "GPU must be 2, 3, 4, 6, or 7" >&2; exit 2;; esac
root=/data1/public/yptang/splart-endpoint-node72-data
python_bin=/data/yptang/workspace/20260720_SplArt/.conda/splart/bin/python
handoff="$root/redistribution-v1"
delta="$root/scripts/node72_delta_gpu${gpu}.json"
receipt="$handoff/receipts/worker-gpu${gpu}.json"
session="node72-scratch25k-v1-gpu${gpu}-delta"

[[ "$(stat -c %a "$root")" == "700" ]]
[[ -f "$delta" && ! -e "$receipt" ]]
free_mib=$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
[[ "$free_mib" -ge 22000 ]]
if [[ "$gpu" == 4 ]]; then [[ "$free_mib" -ge 40000 ]]; fi
mkdir -p "$handoff/logs" "$handoff/receipts" "$handoff/claims"
tmux new-session -d -s "$session" \
  "'$python_bin' '$root/scripts/run_order_scratch_baselines.py' \
    --public-manifest '$root/public/profile_manifest.json' --episode-list '$delta' \
    --claim-root '$handoff/claims' --dataset-root '$root/public/materialized-new24-v1' \
    --output-root '$root/scratch-25k-v1/model_ckpts' --log-root '$handoff/logs' \
    --receipt '$receipt' --gpu '$gpu' --shard-index '$gpu' --num-shards 4 \
    --minimum-free-mib 22000 --minimum-output-free-mib 131072 --max-iterations 25000 \
    --timestamp-tag node72-scratch25k-v1 > '$handoff/logs/worker-gpu${gpu}.log' 2>&1"
tmux list-panes -t "$session" -F "gpu${gpu} pane_pid=#{pane_pid} command=#{pane_current_command}"
