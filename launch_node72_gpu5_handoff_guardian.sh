#!/usr/bin/env bash
set -euo pipefail
root=/data1/public/yptang/splart-endpoint-node72-data
session=node72-gpu5-handoff-guardian-v3
log="$root/redistribution-v1/gpu5_handoff_guardian-v3.log"
[[ ! -e "$log" && ! -e "$root/redistribution-v1/gpu5_handoff_guardian_v3_receipt.json" ]]
tmux new-session -d -s "$session" \
  "python3 '$root/scripts/node72_gpu5_handoff_guardian.py' > '$log' 2>&1"
tmux list-panes -t "$session" -F 'pane_pid=#{pane_pid} command=#{pane_current_command}'
