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
validator fails closed if those fields appear in the training tree.

## Fixed baselines

| name | predicted local scalars | renderer |
|---|---:|---|
| Observed-span | `0.0`, `1.0` | none (limit-only lower bound) |
| Symmetric-linear | `-0.5`, `1.5` | none (fixed diagnostic) |
| SplArt-middle | `0.0`, `1.0` | original SplArt trained from scratch |

All three use the same fixed, label-free closed-side guess
`outside_state_0`.  It is a declared convention rather than a learned signal.
This makes closed-end accuracy measurable without leaking the answer.

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

Only the evaluator process may aggregate against the sealed record:

```bash
python endpoint_eval.py \
  --prediction prediction.json \
  --sealed-evaluator-record SEALED_RECORD.json \
  --measurement-record POST_RENDER_METRICS.json \
  --output metrics.json
```
