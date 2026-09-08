# Node 8.3 — Virtual-Subinterval Projective Consistency

## Outcome

This node is a **credible negative result** for the learned projective head. It achieved a 37.34% reduction in cross-gauge variance on the 36 target-free training objects, but failed to transfer that stability to the first public Box development episode. On Box-a its variance was 0.78539 versus 0.38330 for the matched no-consistency head, and its absolute worst-side NMAE was 0.41182 versus 0.28339 for exact frozen D2. The preregistered fail-fast rule therefore stops the learned method before Box-b–d aggregation and before any Box-e/f confirmation-label read.

The strict virtual-subinterval machinery itself is valid and reusable. A non-learned five-gauge frozen-energy aggregate reduced Box-a worst-side NMAE from 0.28339 to 0.21505 (24.12%), but it was not the preregistered selected method and did not beat coordinate-only (0.20747), so it is reported only as a future direction rather than promoted post hoc.

## Idea

Given an observed interval `[a,b]` inside the original normalized articulation interval, define the virtual gauge `t=(q-a)/(b-a)`. Recompute D2 directional energy posteriors on the physically corresponding queries `q=a+(b-a)t`, predict outward distances in `t`, map those endpoints back into physical `q`, and penalize disagreement between gauges. This tests a real projective/equivariance property: the physical endpoint should not depend on which interior subinterval is treated as the two observed states.

## Mathematical constructibility

The legacy profiles contain only lower queries `q in [-2,0)` and upper queries `q in (1,3]`. A strict interior gauge `[a,b] subset (0,1)` requires the lower field through `q=a` and the upper field beginning at `q=b`. Since the missing open interval `(0,1)` cannot be recovered by coordinate reindexing, cropping the legacy profiles would be a silent approximation.

We therefore exported fresh target-free full-q fields with shape `[2 reconstructed states, 2 directions, 3 radii, 513 q samples, 9 channels]` for all 36 public objects. The index SHA-256 is `c2fe0e64e1de3aed2689bbdbe70b72e69ff5955720cc48b1a409c3b259c37519`; it records zero target-file and protected-split reads.

## Implementation choices

- Five fixed gauges: `[0,1]`, `[0.1,0.9]`, `[0.2,0.8]`, `[0.15,0.75]`, `[0.25,0.85]`.
- A shared side head consumes no object/category ID. Every learned ablation starts from identical weights (`seed=8300`, reproducibility only).
- The target-free loss combines frozen-energy posterior imitation, an original-gauge anti-collapse anchor, and physical-q cross-gauge variance.
- A target-free sweep over consistency weights `{0.5,1,2,5}` was preregistered before launch. Selection required at least 20% variance reduction versus no-consistency, then the lowest teacher MAE. This selected weight 2.
- Final inference was preregistered as the arithmetic mean of physical-q predictions over all five gauges.
- Every CUDA process was launched with `CUBLAS_WORKSPACE_CONFIG=:4096:8`.

## Target-free comparison (36 objects)

| Method | Teacher absolute MAE | Cross-gauge variance (q²) | Cross-gauge range (q) |
|---|---:|---:|---:|
| Projective consistency (selected w=2) | 0.04810 | **0.06802** | **0.56381** |
| No consistency | **0.00640** | 0.10857 | 0.67863 |
| Frozen D2 posterior | 0.00000 | 0.10759 | 0.67450 |
| Coordinate-only | 0.33883 | 0.07186 | 0.71277 |
| Zero-geometry/range prior | 0.53451 | 0.00000 | 0.00000 |

The zero-geometry result demonstrates why variance cannot be optimized alone: a constant predictor has zero variance while being grossly inaccurate. The preregistered fidelity-plus-consistency rule avoids selecting this collapse.

## Box-a public development screen

All NMAEs below are absolute normalized errors; lower is better. Cross-gauge variance is computed before the fixed mean projection.

| Method | Mean-side NMAE | Worst-side NMAE | Cross-gauge variance (q²) |
|---|---:|---:|---:|
| Projective consistency | 0.27366 | 0.41182 | 0.78539 |
| No consistency | 0.26421 | 0.39766 | 0.38330 |
| Coordinate-only | **0.12736** | **0.20747** | **0.07186** |
| Zero-geometry/range prior | 0.35946 | 0.46716 | n/a |
| Frozen D2 profile, five-gauge analytic mean | 0.21254 | 0.21505 | 0.27584 |
| Exact frozen D2 | 0.17027 | 0.28339 | n/a |

The learned consistency head increases Box-a cross-gauge variance by 104.90% versus its no-consistency twin and increases worst-side NMAE by 45.32% versus exact frozen D2. This is strong evidence of profile-domain shift rather than useful physical-law learning.

## Analysis and insight

The consistency regularizer is effective on its training distribution, but the head learned a domain-specific correction to frozen D2 energy profiles. Under Box geometry, that correction is not equivariant and amplifies gauge changes. The failure is not hidden by averaging: the pre-projection variance exposes it directly.

The promising residue is analytic rather than learned. Recomputing the posterior on valid interior subintervals and aggregating physical-q estimates improved the primary worst-side metric on Box-a by 24.12% over exact D2. A future node should develop a target-free robust M-estimator or uncertainty-weighted analytic consensus with a formal no-worse-than-anchor fallback. It must be preregistered and evaluated on fresh public development episodes; Box-a cannot be reused for selection.

## Leakage and confirmation status

- Old sealed 18/9/9, B_test, and Full22 were not read.
- The 36-object training run read no targets or split membership.
- Box-a is public development only.
- Box-e/f confirmation labels were not read and no confirmation score was produced.
- The checkpoint was frozen at SHA-256 `85a6869ffb3a688fa516a5859dc1c93f39dc91a029d069ccb0f78b54b1f49a44` before Box scoring.

## Artifacts

- Branch: `arbor/node-8-3-projective`
- Remote checkpoint: `/data1/public/yptang/splart-projective-node83/checkpoints/projective-head-v2.pt`
- Freeze receipt: `freeze_receipt.json`
- Box-a detail: `box_a_dev.json`
- Aggregate metrics: `metrics.json`
