#!/usr/bin/env bash
set -euo pipefail
root=/data1/public/yptang/splart-gauge-energy-v4
output="$root/node72-profiles-incremental-2-v6"
test "$(stat -c %a "$root")" = 700
test "$(df --output=avail -B1 /data1 | tail -1)" -ge 34359738368
test "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 7)" -lt 1024
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i 7)"
test ! -e "$output"
test -z "$(git -C "$root/d2-frozen-dd78dcc" status --porcelain)"
export CUDA_VISIBLE_DEVICES=7
/data/yptang/workspace/20260720_SplArt/.conda/splart/bin/python \
  "$root/source/export_gauge_profiles.py" \
  --public-manifest /data1/public/yptang/splart-endpoint-node72-data/public/profile_manifest.json \
  --materialization-receipt /data1/public/yptang/splart-endpoint-node72-data/public/materialized-new24-v1/materialization_receipt.json \
  --training-outputs "$root/node72-incremental-completed-v7.json" \
  --d2-source "$root/d2-frozen-dd78dcc" \
  --d2-commit dd78dcc5355fd0f41fb00541d6a18dc36d87ef91 \
  --output-dir "$output" --device cuda
