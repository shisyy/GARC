# C-CLIP-LD deployment contract (server 98)

The current server has neither OpenCLIP nor a CLIP checkpoint. The node87 RGB banks are intentionally rejected: they contain only two observed states and do not align with the fixed Articraft 13/6 or NJC 11/4 episode identities.

Stage these immutable prerequisites under a new owned directory before running:

- a local `open_clip_torch` wheel and its dependencies;
- an OpenCLIP ViT-B/32 checkpoint;
- the bundled, hash-checked renderer at `renderer/renderer_clip_stream.py`; do not substitute an external renderer. It implements the audited Articraft/NJC mappings and rejects `true_endpoints`/label fields.

```bash
ROOT=/data1/public/yptang/splart-node818-c-clip-ld
SRC=$ROOT/source
PY=/data/yptang/workspace/20260720_SplArt/.conda/splart/bin/python
ART_INDEX=/data1/public/yptang/splart-node84-source-transfer/source-smoke-v1/index.json
ART_CR_REPORT=/data1/public/yptang/splart-node816-cr-fpl/results/8.16-cr-fpl-source-v1/report.json
ART_CR_CKPT=/data1/public/yptang/splart-node816-cr-fpl/results/8.16-cr-fpl-source-v1/cr_fpl.pt
NJC_INDEX=/data1/public/yptang/splart-node816-cr-fpl-njc/data/njc-full-trajectory-v1/index.json
NJC_CR_REPORT=/data1/public/yptang/splart-node816-cr-fpl-njc/results/8.16-cr-fpl-njc-v1/report.json
NJC_CR_CKPT=/data1/public/yptang/splart-node816-cr-fpl-njc/results/8.16-cr-fpl-njc-v1/cr_fpl.pt
CLIP_CKPT=$ROOT/weights/open_clip_vit_b32.bin
RENDERER=$SRC/renderer/renderer_clip_stream.py
ART_ASSETS=/data/yptang/workspace/splart-endpoint-extrapolation-data-scout/articraft-10k
ART_SELECTION=$ART_ASSETS/selection-pilc-v1.json
NJC_ASSETS=/data/yptang/workspace/splart-endpoint-extrapolation-data-scout/njc-articulated-hinge/assets
NJC_PREREG=$ROOT/public-inputs/njc-preregister.json
OPEN_CLIP_WHEEL=$ROOT/wheels/open_clip_torch.whl
OPEN_CLIP_VERSION=3.3.0
TIMM_VERSION=1.0.29
```

Install from staged wheels only and bind the two executable artifacts:

```bash
$PY -m pip install --no-index "$OPEN_CLIP_WHEEL"
CLIP_SHA=$(sha256sum "$CLIP_CKPT" | cut -d' ' -f1)
RENDERER_SHA=$(sha256sum "$RENDERER" | cut -d' ' -f1)
OPEN_CLIP_WHEEL_SHA=$(sha256sum "$OPEN_CLIP_WHEEL" | cut -d' ' -f1)
test "$(sha256sum "$ART_SELECTION" | cut -d' ' -f1)" = b39aba89d3670b2930df7e77fdc8fac0d493e43aea3ca0e2acf340a46b2b5089
test "$(sha256sum "$NJC_PREREG" | cut -d' ' -f1)" = 1928eed04e666e3162d8306ace62a6b195c7202a0ddf84d939dc98d3b1db8f25
test "$CLIP_SHA" = 40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af
test "$(stat -c%s "$CLIP_CKPT")" = 353976522
$PY -c "import importlib.metadata as m; assert m.version('open_clip_torch') == '$OPEN_CLIP_VERSION'"
$PY -c "import importlib.metadata as m; assert m.version('timm') == '$TIMM_VERSION'"
$PY -c "import pytorch3d; assert pytorch3d.__version__ == '0.7.8'; print('bundled renderer dependency OK')"
```

The formal CUDA renderer fails closed unless the installed PyTorch3D version is
exactly 0.7.8. Its installed-file manifest digest, Torch version, and CUDA
runtime version are included in the render configuration and cache audit.

Create the two label-independent render plans:

```bash
cd "$SRC"
PYTHONPATH="$SRC/src:$SRC" $PY prepare_clip_limit_render_plan.py --source-index "$ART_INDEX" --domain articraft --selection "$ART_SELECTION" --output "$ROOT/plans/articraft.json"
PYTHONPATH="$SRC/src:$SRC" $PY prepare_clip_limit_render_plan.py --source-index "$NJC_INDEX" --domain njc --preregister "$NJC_PREREG" --output "$ROOT/plans/njc.json"
```

Stream RGB mini-batches directly through frozen CLIP. Only compact `[2,129,1]` evidence and hash receipts are persisted:

The poses are the unclamped continuation `angle0 + q * (angle1 - angle0)` of
the two public observed-state fractions. Public `f0/f1`, state order, and axis
augmentation remain renderer-only plan metadata: the predictor receives only
the fixed candidate coordinates and signed CLIP evidence, never that metadata.

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$SRC/src:$SRC" $PY build_clip_limit_cache.py --render-plan "$ROOT/plans/articraft.json" --source-index "$ART_INDEX" --domain articraft --selection "$ART_SELECTION" --renderer-script "$RENDERER" --renderer-sha256 "$RENDERER_SHA" --asset-root "$ART_ASSETS" --output "$ROOT/cache/articraft" --clip-checkpoint "$CLIP_CKPT" --clip-checkpoint-sha256 "$CLIP_SHA" --open-clip-wheel "$OPEN_CLIP_WHEEL" --open-clip-wheel-sha256 "$OPEN_CLIP_WHEEL_SHA" --open-clip-version "$OPEN_CLIP_VERSION" --device cuda --batch-size 8
CUDA_VISIBLE_DEVICES=1 PYTHONPATH="$SRC/src:$SRC" $PY build_clip_limit_cache.py --render-plan "$ROOT/plans/njc.json" --source-index "$NJC_INDEX" --domain njc --preregister "$NJC_PREREG" --renderer-script "$RENDERER" --renderer-sha256 "$RENDERER_SHA" --asset-root "$NJC_ASSETS" --output "$ROOT/cache/njc" --clip-checkpoint "$CLIP_CKPT" --clip-checkpoint-sha256 "$CLIP_SHA" --open-clip-wheel "$OPEN_CLIP_WHEEL" --open-clip-wheel-sha256 "$OPEN_CLIP_WHEEL_SHA" --open-clip-version "$OPEN_CLIP_VERSION" --device cuda --batch-size 8
```

After both cache indexes pass validation, launch one unchanged control per genuinely free GPU (repeat with `semantic`, `zero_semantic`, `object_image_shuffle`, and `prompt_swap`):

```bash
CONTROL=semantic
CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH="$SRC/src:$SRC" $PY -u run_clip_limit_source_gate.py --index "$ART_INDEX" --semantic-index "$ROOT/cache/articraft/index.json" --output-dir "$ROOT/results/art-$CONTROL" --device cuda --control "$CONTROL" --cr-fpl-report "$ART_CR_REPORT" --cr-fpl-checkpoint "$ART_CR_CKPT"
CUDA_VISIBLE_DEVICES=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH="$SRC/src:$SRC" $PY -u run_clip_limit_njc_gate.py --index "$NJC_INDEX" --semantic-index "$ROOT/cache/njc/index.json" --output-dir "$ROOT/results/njc-$CONTROL" --device cuda --control "$CONTROL" --cr-fpl-report "$NJC_CR_REPORT" --cr-fpl-checkpoint "$NJC_CR_CKPT"
```

No Box, sealed node72 truth/membership, B_test, Full22, or Box e/f path is accepted by these entrypoints.
