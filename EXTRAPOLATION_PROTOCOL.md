# Endpoint extrapolation protocol

## Episode construction

Each episode contains two multi-view observations of one articulated object at
strictly interior configurations. The physical fractions are sampled by the
benchmark builder from disjoint lower and upper interior bands, then erased
from model-facing metadata. The observations are relabelled state `0` and state
`1`, exactly matching SplArt's two-state input interface.

The initial deterministic smoke episode uses physical fractions `0.25` and
`0.75`. This value is public only for plumbing validation and must not support a
performance claim. Formal B_dev uses a sealed per-scene interior-pair manifest;
B_test uses a separately sealed manifest that is never read during iteration.

## Required outputs

The model predicts:

1. the relative joint type, axis, pivot, and observed-state displacement;
2. two extrapolation scalars in the observed coordinate system;
3. which scalar corresponds to the fully closed endpoint;
4. endpoint renderings, depth, and part masks;
5. a physical certificate containing terminal contact and interpenetration
   evidence.

## Baselines

- `Observed-span`: declares the two inputs to be the endpoints.
- `Symmetric-linear`: extends both sides by half the observed displacement;
  this is diagnostic only because the smoke pair makes it oracle-equivalent.
- `SplArt-middle`: original SplArt trained from scratch on the two relabelled
  interior states, with no endpoint-physics module.

## Scoring

Primary endpoint error is the mean absolute error of the two predicted limits,
normalized by the hidden true joint range. Closed-end identification is scored
separately. Endpoint image metrics use held-out cameras at the true limits.
Predicted endpoint renders are evaluated at the model's own predicted scalars,
not at oracle scalars.

A candidate can advance only when it reduces endpoint error and improves closed
endpoint identification without regressing PSNR, SSIM, LPIPS, depth MAE,
static/mobile/background IoU, mIoU, or articulation validity against the new
from-scratch baseline.

## Leakage boundary

URDF limits, physical interior fractions, original endpoint state labels, and
held-out endpoint images may be used by the benchmark builder and evaluator
only. They are forbidden from all model inputs, training manifests, learned
sidecars, adaptive thresholds, and candidate selection.
