# Node 8.1 — Cross-State Radius-Consensus Boundary

## Idea

Preserve every reconstructed state and every D² radius, then estimate each
endpoint from robust agreement among their free-to-penetrating change points.
The intended benefit was to prevent one biased reconstruction or radius from
moving the averaged-energy minimum.

## Changes

- Added a fail-closed, target-free exporter that writes
  `[state=2, side=2, radius=3, sample=257, channel=9]` profiles without
  reducing states or radii.
- Added exact state-swap-equivariant consensus, mean-profile, single-radius,
  and fixed sample-shuffle controls.
- Added a Box a-d-only tuning driver and a held-two-out Box e/f evaluator.
  Neither was used to open e/f because the fail-fast gate pruned the idea on
  Box a.
- All CUDA launches set `CUBLAS_WORKSPACE_CONFIG=:4096:8` before Python.

## Implementation Choices

The default estimator combines each curve's D² posterior expectation with a
signed-gap transition estimate, weights curves by posterior concentration and
transition agreement, and applies a weighted trimmed consensus shared by both
endpoint sides. Every prediction is generated before the development target is
read. Scoring is a separate target-derived operation.

## Baseline vs Result

| Box-a method | Lower NMAE ↓ | Upper NMAE ↓ | Mean NMAE ↓ | Worst-side NMAE ↓ |
|---|---:|---:|---:|---:|
| Frozen D² | 0.283389 | 0.057150 | **0.170270** | **0.283389** |
| Cross-state radius consensus | 0.688222 | 0.544433 | 0.616327 | 0.688222 |
| Mean profile | 0.688222 | 0.633646 | 0.660934 | 0.688222 |
| Single radius | 0.697089 | 0.787363 | 0.742226 | 0.787363 |
| Fixed sample shuffle null | 0.333551 | 0.333353 | 0.333452 | 0.333551 |

The proposed method is 2.43× worse than D² on the preregistered worst-side
criterion, and the shuffle null is also better than the proposal.

## Score

Absolute primary score: **0.6882218205668295 Box-a worst-side NMAE**.

This is a fail-fast development score, not a Box a-f result. Box e/f targets
were not opened.

## Analysis

The mechanistic premise fails on the first public development episode:
all 12 state × side × radius curves have negative signed gap over the entire
outward scan, so there are **0/12 free-to-penetrating zero crossings**. Several
D² posteriors also collapse onto a scan boundary. The consensus therefore
aggregates reconstruction overlap artifacts rather than physical boundary
evidence.

The independent target-free 36-object audit still shows that radius
instability is real: posterior-expected radius standard deviation has mean
0.1196 (median 0.0731, p90 0.3260, max 0.4749), argmin radius standard
deviation averages 0.1964, leave-one-radius prediction range averages 0.1114,
and entropy correlates with instability at r=0.530. Those observations motivate
uncertainty estimation, but they do not validate change-point consensus.

An uncertainty gate that falls back whenever no crossing is observed would
select frozen D² on Box a. That recovers the baseline only; it is not a second
innovation or a performance gain.

## Insights

Per-state/per-radius preservation is a useful reusable measurement artifact,
but a signed-gap change-point method is not identifiable when reconstructed
static and mobile geometry overlaps everywhere. Future second-innovation work
should model the reliability or censoring of counterfactual fields, rather
than treating their nonexistent zero crossings as physical stops.

## Provenance

- Frozen D² commit: `dd78dcc5355fd0f41fb00541d6a18dc36d87ef91`
- Frozen D² tree: `cff0be69750b9d6144c0dffd87f9903c83a7ec11`
- Stable Box-a artifact SHA256:
  `52f40ca8135cbdd91829b02072d9b79a5221415de569bc496aa9805502d571f2`
- Diagnostic SHA256:
  `60f48ba9751880675ac4c45f8325123559e7f16bd2e539e096349cbd7575683f`
- Durable remote evidence:
  `/data1/public/yptang/splart-node81-consensus/aborted-after-box-a`
- Public Box a-d development manifests were inspected as allowed; only Box a
  was scored before fail-fast pruning. Protected inputs read: none. Box e/f
  labels read: no.
