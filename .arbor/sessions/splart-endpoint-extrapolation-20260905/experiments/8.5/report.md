# Node 8.5 — Closure-Anchored Range Completion

## Outcome

**Pruned at the preregistered source gate.** CARC correctly enforces physical
span closure and exact state-swap equivariance, but the geometry-conditioned
range head does not beat the displacement-only null. Its Articraft validation
object-macro range MARE is `0.199629`, versus `0.143619` for range-only. This
39.0% regression directly triggers the fail-fast rule.

The source choice was therefore not frozen for target use. Exact36, public Box
a-d, e/f, old sealed 18/9/9, B_test, and Full22 were not read.

## Changes

- Added the CARC kinematic layer and reliability-weighted affine projection.
  The projected endpoints satisfy the predicted span to numerical precision
  and the complete operation is exactly swap-equivariant.
- Added swap-invariant range, matched-capacity independent-extension, and
  shared-side scalar closure-anchor heads.
- Removed seed choice from initialization: every linear layer uses the same
  analytic `sin(row,col)` rule; all minibatches use one fixed reproducibility
  stream without selection.
- Recovered opaque object groups from the frozen Articraft QC ordering and raw
  finite-joint counts. The runner fails closed unless the recovered grouping
  exactly accounts for all 32-row joint augmentation blocks.
- Scored joint/object macro metrics, every held-out span factor, continuous
  virtual gauges, and NJC end-to-end scalar anchoring.

## Baseline vs result

| Source validation method | Object-macro worst-side NMAE |
|---|---:|
| Range-only + oracle anchor | **0.143619** |
| Full geometry + displacement + oracle anchor | 0.199629 |
| Geometry-only + oracle anchor | 0.201946 |
| Shuffled geometry + oracle anchor | 0.262706 |
| Matched independent extension head | 0.241532 |

The full model beats the newly trained independent head by 17.3%, but this is
insufficient because it loses decisively to the stronger displacement-only
control. Its p90 range error is `0.783253`, versus `0.280069` for range-only.
Continuous virtual-gauge MARE is `0.256934`, so the result is not a hidden win
outside the four synthetic factor templates. Leave-one-factor-out object-macro
MARE ranges from `0.172480` to `0.191560`.

## NJC end-to-end closure test

The shared-side NJC head classifies the closed side with coverage `1.0`, risk
`0.0`, and exact swap error `0.0`; however this binary result is not confused
with a scalar anchor. The actual scalar anchor NMAE is `0.134073` (p90
`0.311086`), and the predicted anchor lies outside the available completed span
on 67.19% of examples.

| NJC calibration method | Mean-side NMAE | Worst-side NMAE |
|---|---:|---:|
| Matched independent | **0.139773** | **0.188940** |
| Evidence-weighted CARC projection | 0.199373 | 0.299933 |
| Hard closure completion | 0.237336 | 0.344772 |

The evidence-weighted projection improves over hard completion while retaining
span identity (`2.38e-7` maximum error), but it remains 58.7% worse than the
matched independent head on the primary worst-side metric.

## Score and insight

The absolute primary score is **0.199629 Articraft object-macro worst-side
NMAE** under oracle closure anchoring. It fails the required 15% margin over
range-only and also fails NJC end-to-end validation.

The useful scientific conclusion is sharper than “geometry does not help.”
The available 47-D two-state descriptors overfit source object range and do not
generalize across continuous observation gauges; meanwhile closed-side
classification is easy but accurate closure *distance* is not. CARC requires a
continuous metric closure anchor and a range representation invariant to the
synthetic four-factor grid. Neither requirement is met by the present public
source artifacts, so Box evaluation would be unjustified.

## Verification

`PYTHONPATH=src python -m pytest tests/test_carc.py -q` → `3 passed`.

Raw full/null result JSON files, summary metrics, and SHA-256 provenance are in
this directory. All three CUDA jobs set `CUBLAS_WORKSPACE_CONFIG=:4096:8`
before Python.
