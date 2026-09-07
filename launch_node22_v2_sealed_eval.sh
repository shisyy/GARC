#!/usr/bin/env bash
set -euo pipefail
run_root=/home/yptang/arbor-runs/splart-node22-sealed-eval-v2
source_root="$run_root/source"
output="$run_root/output"
test ! -e "$output"
cd "$source_root"
export CUDA_VISIBLE_DEVICES=2
export PYTHONPATH=src
exec /data/yptang/workspace/20260720_SplArt/.conda/splart/bin/python run_node22_sealed_evaluator.py \
  --index /data1/public/yptang/splart-gauge-energy-v4/profiles-merged-12-v1/index.json \
  --index /data1/public/yptang/splart-gauge-energy-v4/node72-profiles-cumulative-24-final/index.json \
  --truth /home/yptang/arbor-sealed/splart-node7.2/sealed-truth-v1.json \
  --authorization "$run_root/authorization.json" \
  --baseline-export /data1/public/yptang/splart-gauge-energy-v4/d2-ablation-evidence-v2/node22-baseline-export-v2.json \
  --output "$output" \
  --device cuda
