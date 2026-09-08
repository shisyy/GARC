# Node 8.2 — Monotone Endpoint Hazard

## Idea

Infer each endpoint as the first stable transition from free motion to terminal interaction. The method converts relative penetration, inside-penetration, contact-mass, support-rise, and energy-turn evidence into a constrained discrete survival process. It does not regress a global range or select an absolute energy minimum.

## Changes

- Added an outward-order canonicalization shared by both endpoint sides.
- Added physically oriented relative-onset channels and an isotonic cumulative hazard.
- Added entropy- and cross-radius-consistency weighting motivated only by 36 target-free public profiles.
- Added energy-only, coordinate-only, channel-shuffle, scalar-coordinate-removal, and frozen-D2 comparisons.
- Added strict support for state-resolved public Box profiles through an exact mean over reconstructed state.

## Implementation choices

The endpoint is the weighted median of the first-transition mass (non-negative increments of cumulative hazard). Three radii are weighted by posterior entropy and agreement of their preliminary transition positions. Channel magnitudes are robustly normalized within scans, so the method uses relative phase change rather than treating the penetrated Gaussian proxy as an absolute collision label.

## Frozen baseline before Box-profile inference

On public Box a-d, the existing frozen-D2 predictions have endpoint NMAE **0.153639** and worst-side NMAE **0.207662**. The coordinator's six-episode endpoint-NMAE baseline is **0.154877**. These are absolute scores, not deltas.

| Episode | D2 lower NMAE | D2 upper NMAE | D2 worst-side NMAE |
|---|---:|---:|---:|
| Box a | 0.283389 | 0.057150 | 0.283389 |
| Box b | 0.056707 | 0.200845 | 0.200845 |
| Box c | 0.161341 | 0.162948 | 0.162948 |
| Box d | 0.183468 | 0.123262 | 0.183468 |

## Target-free mechanism result

On 36 public target-free profiles, monotonicity violations are zero. Channel shuffle changes 95.83% of side predictions (mean absolute shift 0.304475), and energy-only changes them by 0.151211 on average. Thus the implemented predictor is an ordered, semantics-dependent multi-channel mechanism, not an energy-only alias.

## Baseline vs result

The first real state-resolved development artifact, public Box a, falsifies the required free-to-terminal premise. All 12 signed-gap curves (two reconstructed states × two sides × three radii) are entirely non-positive, so there is no free region or zero crossing from which a terminal hazard can begin.

| Method | Worst-side NMAE ↓ | Endpoint NMAE ↓ | Lower ↓ | Upper ↓ |
|---|---:|---:|---:|---:|
| Frozen D2 | 0.283389 | 0.170270 | 0.283389 | 0.057150 |
| **Monotone hazard** | **0.342954** | **0.313357** | **0.283760** | **0.342954** |
| Hazard, no scalar coordinate | 0.341019 | 0.311399 | 0.281778 | 0.341019 |
| Energy-only change point | 0.549764 | 0.434081 | 0.549764 | 0.318399 |
| Coordinate-only | 0.314307 | 0.290849 | 0.267391 | 0.314307 |
| Channel shuffle | 0.265199 | 0.182401 | 0.099604 | 0.265199 |

The full hazard is worse than frozen D2, and channel shuffle improves rather than degrades the score. Consequently the semantic direction of the proposed evidence is not supported by the actual Gaussian trajectory field.

## Score

Absolute representative public-development score: **0.342954 worst-side NMAE** on Box a. This is not presented as a complete a-d or a-f score: the node was pruned after a correctly implemented representative check falsified its mechanism, before consuming e-f confirmation labels.

## Analysis

The target-free exact36 diagnostics showed that the code is genuinely monotone and semantics-sensitive, but Box a demonstrates that those properties do not make the inferred transition physical. Penetration is already present adjacent to both observations and remains present throughout the outward scan. The hazard therefore finds changes near one observed-span unit, while the true Box-a extensions are substantially shorter. This is the same representation failure previously seen in posthoc contact certificates, now expressed as a failed ordered survival model.

## Insight

Do not invest further compute in a monotone free-to-terminal hazard on the current outside-only D2 fields. A viable second innovation must introduce evidence that can distinguish counterfactual states despite globally penetrated proxy geometry—for example reconstruction-state agreement or projective subinterval consistency—rather than imposing an ordered phase interpretation on the same channels.

## Provenance

- Code commits: `c543ebd`, `5cb01ee`, `668db33`
- Frozen D2 commit: `dd78dcc5355fd0f41fb00541d6a18dc36d87ef91`
- Exact36 diagnostic SHA256: `3699dbee023eac4b21a21f56c0c7393bd1483542c51a1b4a024cbf81b1145f65`
- Box-a profile SHA256: `52f40ca8135cbdd91829b02072d9b79a5221415de569bc496aa9805502d571f2`
- Box-a metrics SHA256: `2099cffdbc9f08b0eacc1966db4bb504d3b255a5954f242b413e2cd2fac5c881`
- Confirmation labels read: no
- Protected reads: none
