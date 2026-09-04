# SplArt-middle baseline

This branch measures the new intermediate-observation endpoint task without an
endpoint-physics module.  It keeps the original SplArt motion equation and
training schedule from source commit
`2e5e286b3ffc027d37c9c9da1a5dc87d18301849`.

## Information boundary

The training and prediction process receives only a public scene directory.
Every frame has a relabelled state `0` or `1`; articulation metadata may reveal
only the joint type. `endpoint_queries.json` reveals the two query identifiers
and directions (`-1`, `+1`) but no magnitude, true endpoint, or closed side.

The following stay in an evaluator-only process: original physical fractions,
true local endpoint scalars, the closed-query label, URDF limits, held-out
endpoint images, B_test membership, and Full22 membership.  The public-input
validator fails closed if those fields appear in the training tree. Its exact
allowlist is 660 PNG payloads plus `transforms.json`, `endpoint_queries.json`,
`PROXY_RECEIPT.json`, and `COMPLETE.json`; every file is hashed again inside
the capped worker immediately before training.

## Fixed baselines

| name | predicted local scalars | renderer |
|---|---:|---|
| Observed-span | `0.0`, `1.0` | none (limit-only lower bound) |
| Symmetric-linear | `-0.5`, `1.5` | none (fixed diagnostic) |
| SplArt-middle | `0.0`, `1.0` | original SplArt trained from scratch |

Original SplArt-middle has no closed-side prediction head and therefore emits
an explicit abstention: closed coverage is zero and selective accuracy is N/A.
The two scalar-only diagnostics retain a visibly labelled blind fixed prior
(`outside_state_0`), which is never reported as learned closed-side accuracy.

SplArt-middle accepts finite render query scalars outside `[0, 1]` and does not
clamp them.  Its baseline endpoint scalars remain `0` and `1`; later methods may
replace the scalar policy while using the same renderer/evaluator contract.

## Metrics

`endpoint_eval.py` runs outside the model process. The sealed manifest's
`endpoint_metrics` entries are ground-truth camera/asset descriptors, not
numeric scores. A separate post-render measurement record supplies the numeric
PSNR/SSIM/LPIPS/depth/IoU values. The aggregator reports lower/upper and
mean endpoint error normalized by the true local endpoint span, closed-end
accuracy, terminal contact validity, penetration depth, PSNR, SSIM, LPIPS,
depth MAE, static/mobile/background IoU, mIoU, and articulation validity/error
fields.  Macro values are scene means and include coverage counts so missing
metrics cannot silently look like improvements.

## Commands

Validate a public scene:

```bash
python endpoint_baseline.py validate-public --scene-dir PUBLIC_SCENE
```

Create a fresh-training request for independent launch review:

```bash
python endpoint_baseline.py training-request \
  --scene-dir PUBLIC_SCENE \
  --output-dir MODEL_ROOT \
  --experiment-name 100247-Box/RUN_ID \
  --output request.json
```

After training, emit the SplArt-middle endpoint prediction:

```bash
python endpoint_baseline.py predict \
  --scene-dir PUBLIC_SCENE \
  --baseline splart-middle \
  --checkpoint-dir FRESH_CHECKPOINT \
  --output prediction.json
```

Only the evaluator process may open the certified v4 postbuild seal. It first
checks the fixed plan and builder hashes, recomputes the binding ID and all
public/source asset hashes, then loads the candidate checkpoint and renders the
20 held-out endpoint cameras at that candidate's predicted scalars. Partial
view, physics, or articulation coverage is rejected rather than averaged:

```bash
python endpoint_render_eval.py \
  --prediction prediction.json \
  --postbuild-seal /home/yptang/arbor-sealed/splart-endpoint-extrapolation/dev3-v4/postbuild/100247-Box.json \
  --builder-file /home/yptang/arbor-runs/splart-endpoint-extrapolation/builder-v4/benchmark/endpoint_proxy.py \
  --sealed-plan-file /home/yptang/arbor-sealed/splart-endpoint-extrapolation/dev3-v4/manifest.json \
  --public-scene PUBLIC_SCENE \
  --artifact-root FRESH_RENDER_ROOT \
  --measurement-output measurement.json \
  --score-output metrics.json
```
