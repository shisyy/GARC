#!/usr/bin/env bash
set -euo pipefail

root=/data1/public/yptang/splart-endpoint-node72-data
python_bin=/data/yptang/workspace/20260720_SplArt/.conda/splart/bin/python
session=node72-scratch25k-v1-gpu5-delta
handoff="$root/redistribution-v1"

[[ "$(stat -c %a "$root")" == "700" ]]
[[ ! -e "$handoff/receipts/worker-gpu5.json" ]]
free_mib=$(nvidia-smi -i 5 --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
[[ "$free_mib" -ge 22000 ]]
mkdir -p "$handoff/logs" "$handoff/receipts" "$handoff/claims"
tmux new-session -d -s "$session" \
  "'$python_bin' '$root/scripts/run_order_scratch_baselines.py' \
    --public-manifest '$root/public/profile_manifest.json' \
    --episode-list '$root/scripts/node72_delta_gpu5.json' \
    --claim-root '$handoff/claims' \
    --dataset-root '$root/public/materialized-new24-v1' \
    --output-root '$root/scratch-25k-v1/model_ckpts' \
    --log-root '$handoff/logs' --receipt '$handoff/receipts/worker-gpu5.json' \
    --gpu 5 --shard-index 5 --num-shards 3 --minimum-free-mib 22000 \
    --minimum-output-free-mib 131072 --max-iterations 25000 \
    --timestamp-tag node72-scratch25k-v1 > '$handoff/logs/worker-gpu5.log' 2>&1"
tmux list-panes -t "$session" -F 'pane_pid=#{pane_pid} command=#{pane_current_command}'
