## Codebase

Working directory: D:\workspace\projects\2026\09\05_splart_endpoint_extrapolation\.arbor\worktrees\vcgsa

## Git Isolation

Work in the assigned experiment branch/worktree. Do not switch back to the main repository for implementation or evaluation.

## Research Idea

**ID**: 4.1
**Hypothesis**:
Mechanism: Visibility-Censored Gaussian Surface Adaptation (VCGSA) trains PILC on deterministic partial-view, opacity-weighted Gaussianized surfaces generated from Articraft/NJC meshes, matching the same 47D runtime extractor while preserving the state-swap-equivariant endpoint and selective-closure heads.
Hypothesis: The remaining Box error is caused primarily by full-mesh-to-partial-3DGS feature shift; matching visibility, density, and opacity statistics during training should improve cross-domain endpoint calibration while retaining PILC's closed-side signal.
Observable: Articraft validation and NJC calibration both beat their frozen statistical priors, zero-geometry remains worse, swap error stays below 1e-6, and a newly pre-registered Box episode set reaches macro endpoint NMAE below 0.070 with closed coverage/accuracy above 0.8.
Conflicts: Surface censoring may erase the contact geometry needed for limits or merely simulate Box-specific artifacts; use fixed Fibonacci views/hash-derived censoring, object-disjoint splits, no Box-score-guided tuning, and stop if validation does not improve before any new Box evaluation.

## Evaluation Info

- **Evaluation command (B_dev)**: `python D:\workspace\projects\2026\09\05_splart_endpoint_extrapolation\.arbor\worktrees\vcgsa/endpoint_render_eval.py --prediction D:\workspace\projects\2026\09\05_splart_endpoint_extrapolation\.arbor\worktrees\vcgsa/prediction.json --sealed-evaluator-record <sealed-dev-record> --postbuild-seal <postbuild-seal> --output D:\workspace\projects\2026\09\05_splart_endpoint_extrapolation\.arbor\worktrees\vcgsa/score.json`
- **Evaluation command (B_test, do not use for routine experiments)**: `UNAVAILABLE_PROTECTED_B_TEST`
- **Dataset info**: Box v4 proxy B_dev; public middle states only; sealed endpoints evaluator-only; B_test and Full22 protected/unavailable
- **Baseline score**: 0.31088139
- **Current trunk score**: 0.31088139

Use B_dev for final experiment scoring. Do NOT use B_test.

## Insights From Prior Experiments

- ROOT: Children findings: [1, done, score=0.3109] Original scratch SplArt accurately recovers axis and pivot but treats interior observations as endpoints: endpoint NMAE 0.310881, closed coverage 0, terminal validity 0. | [2, done, score=0.08573] D2-CEA reduced endpoint NMAE 72.42%, raised PSNR 3.72 dB and mobile IoU 46.74% with the base model hash unchanged. However all RMS/PCFG/OEC posthoc contact certificates abstained: closed coverage and terminal validity remain zero, showing the bottleneck is contact-surface representation rather than scalar search. | [3, done, score=0.08573] Children findings: [3.1, done, score=0.08573] View-balanced voxel fusion removed order/view-density bias, but observable surface contact and closure topology vote opposite sides; posthoc frozen-geometry terminal certification remains unidentifiable on Box. | [4, done, score=0.09515] PILC learns genuine geometry-conditioned endpoint signal on object-disjoint Articraft and perfect selective closed-side classification on Box, but endpoint transfer to NJC/Box remains weak and terminal contact remains invalid; geometry-domain alignment is now the limiting factor.
- 4: PILC learns genuine geometry-conditioned endpoint signal on object-disjoint Articraft and perfect selective closed-side classification on Box, but endpoint transfer to NJC/Box remains weak and terminal contact remains invalid; geometry-domain alignment is now the limiting factor.

## Instructions

1. Understand the code before editing.
2. Implement the idea faithfully.
3. Run quick checks to ensure the new logic is active.
4. Iterate on implementation bugs.
5. Run the B_dev evaluation when credible.
6. Report Changes, Baseline vs Result, Score, and Insight. The score must be the absolute primary metric, not a delta.

Save results to `results/4.1-<brief-description>/`.
