## Codebase

Working directory: D:\workspace\projects\2026\09\05_splart_endpoint_extrapolation\worktrees\node83_projective

## Git Isolation

Work in the assigned experiment branch/worktree. Do not switch back to the main repository for implementation or evaluation.

## Research Idea

**ID**: 8.3
**Hypothesis**:
Mechanism: Virtual-Subinterval Projective Consistency reparameterizes one counterfactual trajectory under many synthetic interior observation intervals and trains a shared boundary functional whose physical endpoint predictions must agree after inverse gauge mapping.
Hypothesis: Range-prior heads exploit the fixed 0/1 observation gauge; projective consistency removes that shortcut and forces the model to use profile shape that survives changes of the observed interior interval.
Observable: Pretraining on the 36 target-free profiles plus public Box a-d labels improves held-out Box e/f worst-side NMAE over frozen D2 and the same architecture without projective consistency, with low cross-gauge variance.
Conflicts: Pruned [2.2] used only state-swap equivariance and could still learn a fixed range prior; this adds a continuous family of observation-gauge interventions and a direct no-consistency ablation.

## Evaluation Info

- **Evaluation command (B_dev)**: `cd D:\workspace\projects\2026\09\05_splart_endpoint_extrapolation\worktrees\node83_projective && python evaluate_public_endpoint_profiles.py --split public_box_af --run-name 8.3`
- **Evaluation command (B_test, do not use for routine experiments)**: `UNAVAILABLE_PROTECTED_B_TEST`
- **Dataset info**: Public Box a-f development episodes plus target-free first36 profiles; old sealed 18/9/9, B_test, and Full22 prohibited
- **Baseline score**: 0.154877
- **Current trunk score**: 0.154877

Use B_dev for final experiment scoring. Do NOT use B_test.

## Insights From Prior Experiments

- ROOT: Children findings: [1, done, score=0.3109] Original scratch SplArt accurately recovers axis and pivot but treats interior observations as endpoints: endpoint NMAE 0.310881, closed coverage 0, terminal validity 0. | [2, done, score=0.08573] D2-CEA reduced endpoint NMAE 72.42%, raised PSNR 3.72 dB and mobile IoU 46.74% with the base model hash unchanged. However all RMS/PCFG/OEC posthoc contact certificates abstained: closed coverage and terminal validity remain zero, showing the bottleneck is contact-surface representation rather than scalar search. | [3, done, score=0.08573] Children findings: [3.1, done, score=0.08573] View-balanced voxel fusion removed order/view-density bias, but observable surface contact and closure topology vote opposite sides; posthoc frozen-geometry terminal certification remains unidentifiable on Box. | [4, done, score=0.09515] Children findings: [4.1, done, score=0.1027] Deterministic visibility-censored Gaussianization slightly improved endpoint transfer but did not beat the NJC statistical prior; mesh-to-Gaussian appearance shift is not the sole bottleneck, and semantic/domain range priors remain dominant. | [5, done, score=0.1576] Children findings:...

## Additional Context

Develop projective/gauge consistency using only the 36 target-free profiles for self-supervision and public Box a-d labels; reserve public Box e-f as untouched confirmation. Do not access old sealed 18/9/9, B_test, or Full22. Compare same architecture without projective consistency, zero-geometry/range prior, coordinate-only, and frozen D2. Pre-set CUBLAS_WORKSPACE_CONFIG before CUDA. Commit implementation and provide aggregate metrics or a fail-closed feasibility report.

## Instructions

1. Understand the code before editing.
2. Implement the idea faithfully.
3. Run quick checks to ensure the new logic is active.
4. Iterate on implementation bugs.
5. Run the B_dev evaluation when credible.
6. Report Changes, Baseline vs Result, Score, and Insight. The score must be the absolute primary metric, not a delta.

Save results to `results/8.3-<brief-description>/`.
