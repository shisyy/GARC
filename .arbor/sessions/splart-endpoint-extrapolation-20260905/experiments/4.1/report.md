# Arbor Executor Report — Node 4.1 VCGSA

## Idea

Visibility-Censored Gaussian Surface Adaptation (VCGSA) tests whether PILC's
remaining transfer error comes from training on complete mesh vertices while
runtime consumes partial, opacity-bearing 3D Gaussians.

## Changes

- Added a pre-registered deterministic mesh-to-Gaussian adapter with twelve
  fixed Fibonacci candidate views, four hash-selected views, a 24 x 24 union
  z-buffer, fixed 0.82 visible-point censoring, and visibility frequency as
  opacity.
- Added one shared 47D opacity-weighted surface extractor for both synthesized
  training Gaussians and public SplArt runtime Gaussians.
- Preserved the node-4 state-swap-equivariant PILC architecture, optimizer,
  step counts, closed threshold, Articraft object split, NJC object split, and
  staged supervision policy.
- Added data/provenance auditing and a fail-closed internal advance gate.

## Implementation Choices

The source OBJ order is canonicalized before all censoring. View selection and
point retention are pure functions of a domain-separated SHA256 key; no random
seed is used. Static and mobile parts share the z-buffer so inter-part
occlusion is represented. Opacity weights all means, covariances, distance
quantiles, and contact masses without changing the descriptor dimension.

Articraft remains endpoint-only supervision. The endpoint encoder/head is
frozen before NJC trains only the selective closed head. No Box metric or
sealed artifact is an input to data preparation, training, or selection.

## Validation

- Local CPU tests: 22/22 passed.
- Server98 CPU tests: 22/22 passed.
- Articraft: 109 accepted objects, 5,408 rows.
- NJC: 15 objects, 480 rows.
- Both object-disjoint audits passed; Articraft contains no closed labels.
- GPU3 smoke and full 2,000/2,000-step staged training completed.
- Maximum state-swap error: `1.1920928955078125e-07` (pass).
- NJC closed coverage/risk: `1.0 / 0.0`.

## Baseline vs Result

| Domain / metric | Frozen PILC | VCGSA | Statistical prior | Gate |
|---|---:|---:|---:|---|
| Articraft validation endpoint NMAE | 0.114299 | **0.113945** | 0.240316 | PASS |
| NJC calibration endpoint NMAE | 0.106854 | **0.102659** | **0.098711** | FAIL |
| Articraft zero-geometry NMAE | — | 0.355650 | — | model better |
| NJC zero-geometry NMAE | — | 0.132904 | — | model better |

VCGSA slightly improved both learned-domain metrics relative to frozen PILC,
but the pre-registered gate requires both learned metrics to beat their own
statistical priors. NJC missed its prior by `0.00394804` NMAE.

## Score

No Box B_dev score was produced. The internal gate status is **FAIL** and
`box_evaluation_allowed=false`. The absolute gated NJC calibration NMAE is
`0.1026591361`; it is diagnostic evidence, not a replacement Box score.

## Analysis

Matching partial visibility and opacity reduces, but does not eliminate, the
Articraft-to-NJC transfer gap. The remaining error is not explained solely by
full-mesh versus partial-Gaussian statistics. The zero-geometry ablations and
exact swap audit show that the trained geometry signal is active and the
failure is not a constant predictor or ordering artifact.

The first launcher revision did not propagate the gate script's exit 78
because shell `-e` was missing; the authoritative `internal-gate.json` still
recorded FAIL and the launcher contained no Box stage. Final commit fixes this
plumbing defect. No second censor configuration or training run was attempted.

## Insights

Deterministic Gaussian surface adaptation provides a small cross-domain gain,
but NJC remains better served by its statistical range prior. Partial-view
feature alignment alone is therefore insufficient; node 4.1 must stop before
Box evaluation rather than tuning censoring against prior Box outcomes.

## Provenance

- Branch: `arbor/node-4-1-vcgsa`
- Training source commit: `3ff4d56795d0afb30589283bc4769bce46f26e7f`
- Final plumbing commit: `e927e4bfeabbff69ad20cacd5bab958ea0443acc`
- Remote run: `/home/yptang/arbor-runs/splart-endpoint-vcgsa/formal-v1`
- Checkpoint SHA256: `e3fb88baa36c0cf6bb98631aab426bb3d1095bfaac98a00761e98bc9b462b648`
- Metrics SHA256: `e86d94247af67277da104f403952edffd023943f2125b7fd53d4feff4b7de8e7`
- Gate SHA256: `ee06eaf8d16dda11801ff76d6f3e7e336a707d6e11bce9eb1c9c56623ebf6b39`
- Data receipt SHA256: `ebc37a804a1988290fb9c7bb15e2bed5c1230f6edbc2a61486b0fa8731ae6985`
- Box scores read for this node: none.
- Protected B_test / Full22 accesses: none.
