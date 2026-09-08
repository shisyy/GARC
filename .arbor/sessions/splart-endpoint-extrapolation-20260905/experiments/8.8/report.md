# Node 8.8 — C-SMARC source preflight

## Outcome

**Pruned at the frozen source-only matching preflight.** The v1.3 protocol
requires at least 90% perturbed-object coverage separately for every domain,
split, and causal shuffle null. Articraft and both held-out domains pass, but
NJC training coverage is only `9/11 = 0.818182` for semantic shuffle and
`8/11 = 0.727273` for mechanical shuffle. Therefore the required matched
nulls are unavailable.

This is a partial-permutation feasibility failure, not a C-SMARC method score.
The two-step smoke and formal 1200-step tasks were not launched. Preflight did
not load DINO or train any model. No Box, exact36, e/f, old sealed 18/9/9,
B_test, or Full22 data was read.

## Changes

- Implemented same-domain partial permutations with a hard `0.5` robust-z
  caliper. The Hungarian objective first maximizes non-self coverage and then
  minimizes donor distance; unmatched tails remain unchanged.
- Mechanical shuffle replaces geometry only and retains the recipient's
  observed displacement.
- Added per-domain train/held coverage and held-donor diversity gates.
- Added historical row-uniform and hierarchy-weighted analytic controls; the
  planned method gate uses the lower-error control.
- Added exact six-hash and three-partition LOFO validation, source/render
  payload hashes, strict source manifests, raw forward/reverse state and signed
  displacement checks, and measured endpoint-swap logic.
- Added immutable `final`, `lofo_0`, `lofo_1`, and `lofo_2` task formats plus a
  deterministic receipt-validating aggregator. They were not executed after
  the preflight failure.

Local verification:
`PYTHONPATH=src python -m pytest tests/test_smarc.py tests/test_smarc_source.py -q`
→ `16 passed`.

## Analytic control audit

Object-macro range MARE:

| Domain/control | Historical row-uniform | Hierarchical | Stronger |
|---|---:|---:|---:|
| Articraft global | 0.375035611 | 0.426598188 | 0.375035611 |
| Articraft displacement | 0.144753706 | 0.152897543 | 0.144753706 |
| Articraft category | 0.253942188 | 0.248405656 | 0.248405656 |
| NJC global | 0.240420857 | 0.240420857 | 0.240420857 |
| NJC displacement | 0.134169719 | 0.134169719 | 0.134169719 |
| NJC category fallback | 0.240420857 | 0.240420857 | 0.240420857 |

The historical figures reproduce the frozen controls within `7e-7`.

## Partial-permutation preflight

| Null/split/domain | Coverage | Max non-self distance | Donor diversity | Result |
|---|---:|---:|---:|:---:|
| Semantic/train/Articraft | 1.000000 | 0.430613 | n/a | pass |
| Semantic/train/NJC | **0.818182** | 0.314831 | n/a | **fail** |
| Semantic/held/Articraft | 1.000000 | 0.164730 | 0.9167, max load 2 | pass |
| Semantic/held/NJC | 1.000000 | 0.224916 | 1.0000, max load 1 | pass |
| Mechanical/train/Articraft | 0.979381 | 0.319763 | n/a | pass |
| Mechanical/train/NJC | **0.727273** | 0.363700 | n/a | **fail** |
| Mechanical/held/Articraft | 1.000000 | 0.112585 | 1.0000, max load 1 | pass |
| Mechanical/held/NJC | 1.000000 | 0.143903 | 1.0000, max load 1 | pass |

All actual non-self edges obey the hard 0.5 caliper. The failure comes from
the one-to-one permutation constraint on the 11-object NJC train domain: too
many objects cannot participate in admissible cycles, despite having locally
close candidates.

## Score

No absolute method score exists because the preregistered null-availability
gate failed before DINO loading and training. The result is
`PRUNE_PARTIAL_PERMUTATION_COVERAGE_UNAVAILABLE`.

## Provenance

- Frozen implementation commit: `0365640` (following matching commit
  `71f7534`).
- Frozen config SHA-256:
  `c46852b019012d68d6413c2fe15f6938948dac34e6a1ea71360cec84560ea1bc`.
- Remote preflight:
  `/data1/public/yptang/splart-node88-csmarc/results/source-preflight-v1/preflight.json`,
  SHA-256 `5ed58c38dc5c081d6fc430ce15afefa3f898a64fffaba2645cb0330cc4512fc8`.
- Remote receipt SHA-256:
  `7be016ae988f2df1fd153849374d31b86a6f1e41ce513afe5d605e52a9360f7e`.
- `formal_training_started=false`, `dino_loaded_by_preflight=false`,
  `box_labels_read=[]`, `protected_splits_read=[]`.

## Insight

A hard caliper is feasible for nearest neighbours but not necessarily for a
full permutation: small-domain cycle closure can force several identities even
when individual close donors exist. A new preregistered node can use
same-domain calipered nearest-other many-to-one matching, while retaining the
90% coverage and donor-diversity gates to prevent collapse. Node 8.8 must stay
pruned rather than changing its matching rule after observing coverage.
