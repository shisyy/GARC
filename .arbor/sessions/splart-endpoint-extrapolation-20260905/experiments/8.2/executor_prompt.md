## Codebase

Working directory: D:\workspace\projects\2026\09\05_splart_endpoint_extrapolation\worktrees\node82_hazard

## Git Isolation

Work in the assigned experiment branch/worktree. Do not switch back to the main repository for implementation or evaluation.

## Research Idea

**ID**: 8.2
**Hypothesis**:
Mechanism: Monotone Endpoint Hazard converts penetration, contact mass, support rise, and energy curvature along each outward scan into a constrained survival/change-point model whose first stable hazard transition is the endpoint.
Hypothesis: Softargmin selects low-energy points even when no physical boundary exists, whereas monotone hazard accumulates ordered evidence for the free-to-terminal transition and should be less sensitive to absolute range priors.
Observable: On public Box a-f leave-two-episode-out evaluation, improve worst-side NMAE over frozen D2 and an energy-only changepoint while retaining gains after scalar-coordinate removal and failing under channel shuffle.
Conflicts: Pruned [6] showed naive mesh collision is penetrated everywhere; this uses relative transitions in learned Gaussian trajectory channels rather than absolute proxy collision labels.

## Evaluation Info

- **Evaluation command (B_dev)**: `cd D:\workspace\projects\2026\09\05_splart_endpoint_extrapolation\worktrees\node82_hazard && python evaluate_public_endpoint_profiles.py --split public_box_af --run-name 8.2`
- **Evaluation command (B_test, do not use for routine experiments)**: `UNAVAILABLE_PROTECTED_B_TEST`
- **Dataset info**: Public Box a-f development episodes plus target-free first36 profiles; old sealed 18/9/9, B_test, and Full22 prohibited
- **Baseline score**: 0.154877
- **Current trunk score**: 0.154877

Use B_dev for final experiment scoring. Do NOT use B_test.

## Insights From Prior Experiments

- ROOT: Children findings: [1, done, score=0.3109] Original scratch SplArt accurately recovers axis and pivot but treats interior observations as endpoints: endpoint NMAE 0.310881, closed coverage 0, terminal validity 0. | [2, done, score=0.08573] D2-CEA reduced endpoint NMAE 72.42%, raised PSNR 3.72 dB and mobile IoU 46.74% with the base model hash unchanged. However all RMS/PCFG/OEC posthoc contact certificates abstained: closed coverage and terminal validity remain zero, showing the bottleneck is contact-surface representation rather than scalar search. | [3, done, score=0.08573] Children findings: [3.1, done, score=0.08573] View-balanced voxel fusion removed order/view-density bias, but observable surface contact and closure topology vote opposite sides; posthoc frozen-geometry terminal certification remains unidentifiable on Box. | [4, done, score=0.09515] Children findings: [4.1, done, score=0.1027] Deterministic visibility-censored Gaussianization slightly improved endpoint transfer but did not beat the NJC statistical prior; mesh-to-Gaussian appearance shift is not the sole bottleneck, and semantic/domain range priors remain dominant. | [5, done, score=0.1576] Children findings:...

## Additional Context

Use only public Box a-f and public target-free D2 evidence; never open old sealed 18/9/9, B_test, or Full22. Implement monotone survival/change-point boundary inference with fixed a-d development and e-f held-out confirmation. Include energy-only, coordinate-only, channel-shuffle, and frozen D2 baselines. Pre-set deterministic CUDA environment before process launch. Commit code and report absolute aggregate metrics.

## Instructions

1. Understand the code before editing.
2. Implement the idea faithfully.
3. Run quick checks to ensure the new logic is active.
4. Iterate on implementation bugs.
5. Run the B_dev evaluation when credible.
6. Report Changes, Baseline vs Result, Score, and Insight. The score must be the absolute primary metric, not a delta.

Save results to `results/8.2-<brief-description>/`.
