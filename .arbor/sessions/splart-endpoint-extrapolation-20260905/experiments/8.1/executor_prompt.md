## Codebase

Working directory: D:\workspace\projects\2026\09\05_splart_endpoint_extrapolation\worktrees\node81_consensus

## Git Isolation

Work in the assigned experiment branch/worktree. Do not switch back to the main repository for implementation or evaluation.

## Research Idea

**ID**: 8.1
**Hypothesis**:
Mechanism: Cross-State Radius-Consensus Boundary preserves each reconstructed state's three-radius trajectory fields and estimates endpoints by robust consensus over their free-to-penetrating change points.
Hypothesis: The current geometry-mean profile lets a biased state or radius move the soft minimum; state-wise change-point agreement should suppress reconstruction-specific artifacts and directly reduce the dominant upper-end error.
Observable: On public Box a-f leave-two-episode-out evaluation, lower worst-side NMAE than frozen D2, single-radius, mean-profile, and geometry-shuffled controls, with exact observation-swap residual below 1e-10.
Conflicts: Pruned [3.1] found observable surface contact ambiguous after averaging; this counters by retaining intervention-indexed state/radius evidence and requiring consensus rather than claiming contact semantics.

## Evaluation Info

- **Evaluation command (B_dev)**: `cd D:\workspace\projects\2026\09\05_splart_endpoint_extrapolation\worktrees\node81_consensus && python evaluate_public_endpoint_profiles.py --split public_box_af --run-name 8.1`
- **Evaluation command (B_test, do not use for routine experiments)**: `UNAVAILABLE_PROTECTED_B_TEST`
- **Dataset info**: Public Box a-f development episodes plus target-free first36 profiles; old sealed 18/9/9, B_test, and Full22 prohibited
- **Baseline score**: 0.154877
- **Current trunk score**: 0.154877

Use B_dev for final experiment scoring. Do NOT use B_test.

## Insights From Prior Experiments

- ROOT: Children findings: [1, done, score=0.3109] Original scratch SplArt accurately recovers axis and pivot but treats interior observations as endpoints: endpoint NMAE 0.310881, closed coverage 0, terminal validity 0. | [2, done, score=0.08573] D2-CEA reduced endpoint NMAE 72.42%, raised PSNR 3.72 dB and mobile IoU 46.74% with the base model hash unchanged. However all RMS/PCFG/OEC posthoc contact certificates abstained: closed coverage and terminal validity remain zero, showing the bottleneck is contact-surface representation rather than scalar search. | [3, done, score=0.08573] Children findings: [3.1, done, score=0.08573] View-balanced voxel fusion removed order/view-density bias, but observable surface contact and closure topology vote opposite sides; posthoc frozen-geometry terminal certification remains unidentifiable on Box. | [4, done, score=0.09515] Children findings: [4.1, done, score=0.1027] Deterministic visibility-censored Gaussianization slightly improved endpoint transfer but did not beat the NJC statistical prior; mesh-to-Gaussian appearance shift is not the sole bottleneck, and semantic/domain range priors remain dominant. | [5, done, score=0.1576] Children findings:...

## Additional Context

Use only public Box a-f development episodes and target-free profiles. Preserve the old sealed 18/9/9, B_test, and Full22. First build a target-free per-state/per-radius profile exporter from frozen D2; do not average reconstructed states. Freeze a-d development choices before e-f evaluation. Pre-set CUBLAS_WORKSPACE_CONFIG before any CUDA process. Compare full D2, mean-profile, single-radius, state/radius shuffle, and the proposed robust consensus. Commit code and produce aggregate report.

## Instructions

1. Understand the code before editing.
2. Implement the idea faithfully.
3. Run quick checks to ensure the new logic is active.
4. Iterate on implementation bugs.
5. Run the B_dev evaluation when credible.
6. Report Changes, Baseline vs Result, Score, and Insight. The score must be the absolute primary metric, not a delta.

Save results to `results/8.1-<brief-description>/`.
