# Publication snapshot validation

- Tested code commit: `2c87d19` (core geometry/semantic code from `23f19b1`,
  view-reliability ablation integrated as `668fe32`).
- Date:2026-09-19.
- Environment: Linux server, existing SplArt Python3.11 Conda environment,
  `python-fcl==0.7.0.11`; CPU-only test execution with empty
  `CUDA_VISIBLE_DEVICES`, `OMP_NUM_THREADS=2`, `MKL_NUM_THREADS=2`.
- Test command: the ten-file pytest command in `GEOMETRY_SEMANTICS_RELEASE.md`.
- Result: **158 passed in44.65s**, exit0.
- Source checkout came from the committed Git bundle, with optional LFS media
  smudging disabled. No benchmark dataset, endpoint label or model checkpoint
  was required by these synthetic tests.
- Covered: relative search, view reliability, part interaction, geometry-first
  proposals/gate, observed surfaces, native FCL surface contacts, midpoint
  calibration, bounded semantic routing and four-mode prediction sealing.
- The final release commit additionally adds this documentation only; no
  implementation was changed after this test run.

This is regression evidence for the packaged code, not a new benchmark result.
The legacy source-supervised CLIP training pipelines are included for research
traceability but were not retrained in this publication task. The old node9.5
development table is historical; the separate observation-only19 rerun has not
been substituted into it.

Before publication, tracked files were scanned for common GitHub/OpenAI token
and private-key signatures, with no matches. The source bundle was about1.15MB;
datasets, weights, private label files and large caches were not added.
