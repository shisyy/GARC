# CSTR source-gate deployment

CSTR consumes the existing node 8.18 scalar CLIP caches. No renderer or CLIP
encoder rerun is needed. The fixed transform is registered in
`configs/cstr_source_v1.json`: fit the first 17 samples (distance <= 0.25),
then expose absolute residual, absolute residual slope, and centered 9-sample
residual RMS at every tied CR-FPL loop.

Use the unchanged Articraft 13/6 and NJC 11/4 source gates and CR-FPL
references:

```bash
ROOT=/data1/public/yptang/splart-node819-cstr
SRC=$ROOT/source
CACHE=/data1/public/yptang/splart-node818-c-clip-ld/cache
PY=/data/yptang/workspace/20260720_SplArt/.conda/splart/bin/python
ART_INDEX=/data1/public/yptang/splart-node84-source-transfer/source-smoke-v1/index.json
ART_CR_REPORT=/data1/public/yptang/splart-node816-cr-fpl/results/8.16-cr-fpl-source-v1/report.json
ART_CR_CKPT=/data1/public/yptang/splart-node816-cr-fpl/results/8.16-cr-fpl-source-v1/cr_fpl.pt
NJC_INDEX=/data1/public/yptang/splart-node816-cr-fpl-njc/data/njc-full-trajectory-v1/index.json
NJC_CR_REPORT=/data1/public/yptang/splart-node816-cr-fpl-njc/results/8.16-cr-fpl-njc-v1/report.json
NJC_CR_CKPT=/data1/public/yptang/splart-node816-cr-fpl-njc/results/8.16-cr-fpl-njc-v1/cr_fpl.pt
mkdir -p "$ROOT/results"
cd "$SRC"
```

Launch one unchanged control per genuinely free GPU. Repeat for `cstr`, `zero`,
`object_trajectory_shuffle`, `coordinate_only`, and `sign_flip`:

```bash
CONTROL=cstr
CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH="$SRC/src:$SRC" "$PY" -u run_clip_limit_source_gate.py --index "$ART_INDEX" --semantic-index "$CACHE/articraft/index.json" --output-dir "$ROOT/results/art-$CONTROL" --device cuda --control "$CONTROL" --cr-fpl-report "$ART_CR_REPORT" --cr-fpl-checkpoint "$ART_CR_CKPT"
CUDA_VISIBLE_DEVICES=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH="$SRC/src:$SRC" "$PY" -u run_clip_limit_njc_gate.py --index "$NJC_INDEX" --semantic-index "$CACHE/njc/index.json" --output-dir "$ROOT/results/njc-$CONTROL" --device cuda --control "$CONTROL" --cr-fpl-report "$NJC_CR_REPORT" --cr-fpl-checkpoint "$NJC_CR_CKPT"
```

These entrypoints accept only the fixed source indexes and cache contract in
the commands above. Do not substitute Box, node72, B_test, Full22, or Box e/f
inputs.
