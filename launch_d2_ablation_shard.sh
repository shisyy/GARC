#!/usr/bin/env bash
set -uo pipefail
gpu="$1"
shard="$2"
root=/data1/public/yptang/splart-gauge-energy-v4/d2-ablation-evidence-v2
python=/data/yptang/workspace/20260720_SplArt/.conda/splart/bin/python
runner=/data1/public/yptang/splart-gauge-energy-v4/public-tools-v2/export_d2_ablation_evidence.py
source=/data1/public/yptang/splart-gauge-energy-v4/d2-frozen-dd78dcc
log="$root/gpu${gpu}.log"
exec >"$log" 2>&1
trap 'status=$?; printf "%s\n" "$status" >"$root/gpu'"$gpu"'.exit.status"; date --iso-8601=seconds >"$root/gpu'"$gpu"'.finished-at.txt"' EXIT
date --iso-8601=seconds >"$root/gpu${gpu}.started-at.txt"
used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu")"
if [ "$used" -gt 1024 ]; then
  echo "GPU${gpu} is not safely free: ${used} MiB" >&2
  exit 73
fi
CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 "$python" -u "$runner" \
  --input "$root/public-only-input-36.json" \
  --d2-source "$source" \
  --d2-commit dd78dcc5355fd0f41fb00541d6a18dc36d87ef91 \
  --output-dir "$root/batch-gpu${gpu}-v1" \
  --device cuda --shard-index "$shard" --shard-count 5
