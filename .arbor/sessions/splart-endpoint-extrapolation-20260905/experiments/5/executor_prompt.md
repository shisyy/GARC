## Codebase

Working directory: D:\workspace\projects\2026\09\05_splart_endpoint_extrapolation\.arbor\worktrees\fecf

## Git Isolation

Work in the assigned experiment branch/worktree. Do not switch back to the main repository for implementation or evaluation.

## Research Idea

**ID**: 5
**Hypothesis**:
Mechanism: Factorized Endpoint-Closure Fusion (FECF) preserves the frozen D2-CEA physical endpoint scalars while attaching only PILC's independently trained state-swap-equivariant closed-side posterior with its fixed 0.80 abstention threshold; neither endpoint nor reconstruction parameters are updated.
Hypothesis: D2's analytic endpoint calibration and PILC's learned closure semantics are complementary, so factorization should retain D2 endpoint accuracy while converting closed coverage from zero to calibrated nonzero values.
Observable: On newly pre-registered Box c/d, endpoint predictions and reconstruction hashes are bit-identical to D2, closed coverage and accuracy are each at least 0.8 with reported selective risk, and constant-lower/category-prior baselines are explicitly compared.
Conflicts: PILC may simply learn an always-lower shortcut and terminal contact can remain invalid even when the side label is correct; reject the node if it does not beat the constant-side prior or if any D2 scalar/hash changes.

## Evaluation Info

- **Evaluation command (B_dev)**: `python D:\workspace\projects\2026\09\05_splart_endpoint_extrapolation\.arbor\worktrees\fecf/endpoint_render_eval.py --prediction D:\workspace\projects\2026\09\05_splart_endpoint_extrapolation\.arbor\worktrees\fecf/prediction.json --sealed-evaluator-record <sealed-dev-record> --postbuild-seal <postbuild-seal> --output D:\workspace\projects\2026\09\05_splart_endpoint_extrapolation\.arbor\worktrees\fecf/score.json`
- **Evaluation command (B_test, do not use for routine experiments)**: `UNAVAILABLE_PROTECTED_B_TEST`
- **Dataset info**: Box v4 proxy B_dev; public middle states only; sealed endpoints evaluator-only; B_test and Full22 protected/unavailable
- **Baseline score**: 0.31088139
- **Current trunk score**: 0.31088139

Use B_dev for final experiment scoring. Do NOT use B_test.

## Insights From Prior Experiments

- ROOT: Children findings: [1, done, score=0.3109] Original scratch SplArt accurately recovers axis and pivot but treats interior observations as endpoints: endpoint NMAE 0.310881, closed coverage 0, terminal validity 0. | [2, done, score=0.08573] D2-CEA reduced endpoint NMAE 72.42%, raised PSNR 3.72 dB and mobile IoU 46.74% with the base model hash unchanged. However all RMS/PCFG/OEC posthoc contact certificates abstained: closed coverage and terminal validity remain zero, showing the bottleneck is contact-surface representation rather than scalar search. | [3, done, score=0.08573] Children findings: [3.1, done, score=0.08573] View-balanced voxel fusion removed order/view-density bias, but observable surface contact and closure topology vote opposite sides; posthoc frozen-geometry terminal certification remains unidentifiable on Box. | [4, done, score=0.09515] Children findings: [4.1, done, score=0.1027] Deterministic visibility-censored Gaussianization slightly improved endpoint transfer but did not beat the NJC statistical prior; mesh-to-Gaussian appearance shift is not the sole bottleneck, and semantic/domain range priors remain dominant.

## Instructions

1. Understand the code before editing.
2. Implement the idea faithfully.
3. Run quick checks to ensure the new logic is active.
4. Iterate on implementation bugs.
5. Run the B_dev evaluation when credible.
6. Report Changes, Baseline vs Result, Score, and Insight. The score must be the absolute primary metric, not a delta.

Save results to `results/5-<brief-description>/`.
