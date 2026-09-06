## Codebase

Working directory: D:\workspace\projects\2026\09\05_splart_endpoint_extrapolation\.arbor\worktrees\pilc

## Git Isolation

Work in the assigned experiment branch/worktree. Do not switch back to the main repository for implementation or evaluation.

## Research Idea

**ID**: 4
**Hypothesis**:
Mechanism: Paired-Interior Limit Calibrator (PILC) learns a state-swap-equivariant dual-endpoint and selective closed-side posterior from canonicalized static/mobile surface features plus the observed relative screw transform; object-disjoint training may use limit labels, but runtime receives no URDF, limits, or absolute state fractions.
Hypothesis: A learned geometric support prior can resolve the upper-end bias left by D2-CEA and provide calibrated nonzero closed-side coverage without changing the 3DGS reconstruction.
Observable: On pre-registered held-out episodes, endpoint NMAE falls below 0.070, closed-side coverage is at least 0.5 with reported risk, terminal validity is positive, and state-swap/order consistency tests pass.
Conflicts: Category/range memorization, reversible metadata leakage, and genuinely unidentifiable partial surfaces could create false confidence; require object-disjoint splits, metadata audit, geometry-only/statistical ablations, and unknown abstention.

## Evaluation Info

- **Evaluation command (B_dev)**: `python D:\workspace\projects\2026\09\05_splart_endpoint_extrapolation\.arbor\worktrees\pilc/endpoint_render_eval.py --prediction D:\workspace\projects\2026\09\05_splart_endpoint_extrapolation\.arbor\worktrees\pilc/prediction.json --sealed-evaluator-record <sealed-dev-record> --postbuild-seal <postbuild-seal> --output D:\workspace\projects\2026\09\05_splart_endpoint_extrapolation\.arbor\worktrees\pilc/score.json`
- **Evaluation command (B_test, do not use for routine experiments)**: `UNAVAILABLE_PROTECTED_B_TEST`
- **Dataset info**: Box v4 proxy B_dev; public middle states only; sealed endpoints evaluator-only; B_test and Full22 protected/unavailable
- **Baseline score**: 0.31088139
- **Current trunk score**: 0.31088139

Use B_dev for final experiment scoring. Do NOT use B_test.

## Insights From Prior Experiments

- ROOT: Children findings: [1, done, score=0.3109] Original scratch SplArt accurately recovers axis and pivot but treats interior observations as endpoints: endpoint NMAE 0.310881, closed coverage 0, terminal validity 0. | [2, done, score=0.08573] D2-CEA reduced endpoint NMAE 72.42%, raised PSNR 3.72 dB and mobile IoU 46.74% with the base model hash unchanged. However all RMS/PCFG/OEC posthoc contact certificates abstained: closed coverage and terminal validity remain zero, showing the bottleneck is contact-surface representation rather than scalar search. | [3, done, score=0.08573] Children findings: [3.1, done, score=0.08573] View-balanced voxel fusion removed order/view-density bias, but observable surface contact and closure topology vote opposite sides; posthoc frozen-geometry terminal certification remains unidentifiable on Box.

## Instructions

1. Understand the code before editing.
2. Implement the idea faithfully.
3. Run quick checks to ensure the new logic is active.
4. Iterate on implementation bugs.
5. Run the B_dev evaluation when credible.
6. Report Changes, Baseline vs Result, Score, and Insight. The score must be the absolute primary metric, not a delta.

Save results to `results/4-<brief-description>/`.
