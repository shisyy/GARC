# Node 8.7 — SMARC source preflight

## Outcome

**Pruned before formal training and before any Box-label access.** The frozen
v1.2 source protocol requires both object-level matched-shuffle nulls to obey a
robust-z caliper. The semantic-shuffle matching is available, but the
mechanical-shuffle Hungarian derangement has maximum train donor distance
`1.562084944`, exceeding the preregistered maximum `1.5`. Consequently the
required matched null is unavailable and Phase A cannot produce an admissible
method score.

This is a protocol-executability failure, not evidence that DINO-SMARC itself
underperforms. No 1200-step model was trained, no source method metric was
recorded, and no public Box a-f label, exact36 target, old sealed 18/9/9,
B_test, or Full22 artifact was read.

## Implemented and verified

- Frozen DINO ViT-B/16 strict loading and deterministic pair encoding.
- Neutral-material triangle-face z-buffer rendering of the complete moving
  subtree and static complement. Articraft v2 overflowed and was invalidated;
  v3 uses `bin_size=0`, has zero overflow log hits, and covers all 109 objects
  and 169 joints. NJC covers all 15 objects and 15 joints.
- Exact source joins, 32 gauges per joint, 184 joints and 5,888 rows. Visual
  semantics remain cached once per joint; only the compact mechanical rows are
  expanded. Weights are explicitly domain → object → joint → gauge.
- Train-only PCA, analytic no-seed initialization, semantic-gate-only K=4
  experts, and exact range projection.
- Real forward/reverse source pairs and executable range/projection/endpoint
  swap-audit code; a negative unit test verifies that broken paired predictions
  are detected.
- Source-only `--preflight` that validates cache schema, DINO hash, renderer,
  coverage, tensor view shape, overflow logs, split/object hashes, analytic
  controls, and matched-null construction without loading DINO or training.

Core tests: `PYTHONPATH=src python -m pytest tests/test_smarc.py tests/test_smarc_source.py -q`
→ `13 passed`.

## Reproduced source controls

These are public-source validation object-macro range MARE values. They exactly
recover the independently frozen row-uniform controls (rounding to six
decimals), demonstrating that the 32-gauge join is correct.

| Domain | Global | Displacement-only | Category/fallback |
|---|---:|---:|---:|
| Articraft | 0.375035611 | 0.144753706 | 0.253942188 |
| NJC | 0.240420857 | 0.134169719 | 0.240420857 |

No SMARC result is shown because the frozen matched-null prerequisite failed
before formal training. Reporting a method score would violate the protocol.

## Matched-null audit

| Null | train median | train p90 | train max | regret p90 | held p90 | held max | Available |
|---|---:|---:|---:|---:|---:|---:|:---:|
| Semantic shuffle | 0.000397227 | 0.231715514 | 1.303895238 | 0.031151569 | 0.149865538 | 0.224916132 | yes |
| Mechanical shuffle | 0.027560493 | 0.146747816 | **1.562084944** | 0.036300655 | 0.074049094 | 0.143903397 | **no** |

The failed maximum is an Articraft tail case. Across all 108 training objects,
the nearest-other robust-z distance has p95 `0.335009068` and maximum
`1.562084944`; the failing assigned recipient's nearest-other distance is
`1.347953352`. This diagnosis used covariates only and did not change the
frozen threshold.

## Provenance

- Implementation/preflight commit: `4982284`.
- Frozen v1.2 config SHA-256:
  `90027389e4548931538f654011ba9fcd3ba3d1d2e98b9cf877a4a1588c355121`.
- Remote preflight:
  `/data1/public/yptang/splart-node87-smarc/results/source-preflight-v1/preflight.json`,
  SHA-256 `8bece64579f77ef1f269c63c7207df4495bdd63913e9290e7be29fe128a83f57`.
- DINO checkpoint SHA-256:
  `bf34ad0f424b9029b593e8dc3ed553bf26e88bcba0d32bf3e62a6209cb64c85e`.
- Valid Articraft shard manifest SHA-256 values:
  `9c95da2e...b774da8`, `67f7543c...a7cb4`, `07c05699...0641f`.
- NJC manifest SHA-256:
  `e3abd737feac4a4bff0ece57f417004399c6d3f9555dcf236e484f52b51daae5`.
- `box_labels_read=[]`; `protected_splits_read=[]`.

## Insight

The full source representation is now auditable, but a strict one-to-one
derangement is a poor null for the sparse tail of a one-dimensional matching
covariate: it can force an otherwise close population to pay one large tail
edge. A future preregistered node should preserve the caliper and donor
marginals while allowing a partial permutation with identity for unmatched
outliers; node 8.7 itself must remain pruned rather than relaxing its threshold
after observing occupancy.
