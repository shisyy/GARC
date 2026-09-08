# Node 8.6 — Dual-Expert Kinematic Projection

Decision: **PRUNE**. The frozen DEKP improves the primary macro worst-side NMAE over both experts, but fails the preregistered mechanistic controls: shuffled uncertainty is better and DEKP wins against both experts on only 2/4 episodes.

## Changes

- Added a six-parameter, shared-side monotone reliability model. D2 reliability can only fall with multi-gauge dispersion and posterior entropy; PILC reliability can only rise with its swap-invariant confidence.
- Fused the frozen D2 and PILC endpoints and applied reliability-weighted exact-span projection.
- Exported target-free full-q D2 trajectories for public Box c/d and reused the frozen a/b trajectories.
- Added immutable input hashes, official per-side NMAE reproduction, signed 3-train/1-test LOEO, state-swap/span audits, and fixed-average, projected-average, coordinate-only, uncertainty-shuffled, D2/PILC, and collapsed-expert controls.

## Protocol and limitations

The evaluation uses only public development episodes Box a–d. Box c/d aggregate comparator scores were received before the config was written, so this is explicitly not a blind test; the six-parameter form and optimizer were frozen before DEKP fitting and were not selected from signed c/d errors. All four episodes are perturbations of the same physical Box object, so LOEO measures episode transfer, not object/category generalization. Box e/f, old sealed18/9/9, B_test, and Full22 were not accessed.

NMAE is the absolute local-scalar error multiplied by `(observed_state_1-observed_state_0)/(upper_limit-lower_limit)`. The builder reproduced every available official D2/PILC lower, upper, and mean NMAE within `1e-6` before fitting.

## Results

Primary metric is absolute macro worst-side NMAE (lower is better).

| Method | Macro mean NMAE | Macro worst-side NMAE |
|---|---:|---:|
| Frozen D2 | 0.153639 | 0.207662 |
| Frozen PILC | **0.074591** | 0.123151 |
| Fixed average | 0.073030 | 0.112265 |
| Projected average | 0.073030 | 0.112265 |
| Coordinate-only | 0.076596 | 0.113018 |
| Shuffled uncertainty | 0.076571 | **0.105756** |
| **DEKP** | 0.078202 | 0.109606 |

DEKP improves macro worst-side NMAE by 47.2% versus D2 and 11.0% versus PILC. However, it is 3.64% worse than the uncertainty-shuffled control, and its macro mean NMAE is worse than PILC.

| Held-out episode | D2 worst | PILC worst | DEKP worst | DEKP wins both? |
|---|---:|---:|---:|:---:|
| Box a | 0.283389 | **0.096524** | 0.130077 | No |
| Box b | 0.200845 | 0.148346 | **0.103377** | Yes |
| Box c | 0.162948 | **0.124550** | 0.139386 | No |
| Box d | 0.183468 | 0.123185 | **0.065585** | Yes |

The expert mean weights are `[0.288649 D2, 0.711351 PILC]`, so the model does not collapse. Exact-span error is `2.22e-16`; end-to-end state-swap error is `4.44e-16`.

## Gate

- PASS: at least 10% better than frozen D2 and PILC.
- PASS: better than fixed/projected average, coordinate-only, and both collapsed experts.
- PASS: both experts retain mean weight at least 0.05; exact span and state swap hold.
- **FAIL:** uncertainty shuffle scores 0.105756, better than DEKP's 0.109606.
- **FAIL:** DEKP beats both experts on only 2/4 episodes, below the required 3/4.

The overall gate fails; no Box e/f confirmation is authorized.

## Insight

The gain comes mainly from smooth kinematic averaging/projection, not from the proposed uncertainty semantics. Seven of eight posterior-entropy features are effectively zero and the shuffled feature alignment improves the primary metric. With only four correlated episodes from one object, the learned evidence-to-reliability mapping is not identifiable. The result supports keeping exact kinematic projection as a useful primitive, but rejects these D2 dispersion/entropy features as a defensible second innovation under the current evidence.
