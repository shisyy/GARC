## Codebase

Working directory: D:\workspace\projects\2026\09\05_splart_endpoint_extrapolation\worktrees\node811_rzaacwt

## Git Isolation

Work in the assigned experiment branch/worktree. Do not switch back to the main repository for implementation or evaluation.

## Research Idea

**ID**: 8.11
**Hypothesis**:
Mechanism: Raw-z Anchored Anisotropic Conditional Whitening Transport (RZA-ACWT) uses a raw-z-anchored minimal nonlinear location basis and a single learned conditional covariance direction with separate rank-quadratic axial and orthogonal-bulk scales.
Hypothesis: Node 8.10's low Spearman but high NJC distance correlation indicates direction-dependent heteroscedasticity rather than a remaining radial trend; exact raw-z anchoring removes the full-fit failure while rank-one whitening is the smallest identifiable covariance correction for NJC n=11.
Observable: Before any labels are read, all four domain-by-null cells meet the unchanged gates, including full-fit raw-z correlation <=1e-10 and OOF Spearman <=0.35 / dCor <=0.5, while exact reconstruction, energy, shuffle, coverage, recipient-u/d, repeatability and row-order gates do not regress.
Conflicts: Nodes 8.9 and 8.10 assumed linear or isotropic conditional residual structure; this node keeps every threshold and donor contract fixed and targets the observed anisotropic covariance failure with exactly one preregistered rank-one model, pruning without rank/degree/threshold changes if any label-free cell fails.

## Evaluation Info

- **Evaluation command (B_dev)**: `cd D:\workspace\projects\2026\09\05_splart_endpoint_extrapolation\worktrees\node811_rzaacwt && python evaluate_public_endpoint_profiles.py --split public_box_af --run-name 8.11`
- **Evaluation command (B_test, do not use for routine experiments)**: `UNAVAILABLE_PROTECTED_B_TEST`
- **Dataset info**: Public Box a-f development episodes plus target-free first36 profiles; old sealed 18/9/9, B_test, and Full22 prohibited
- **Baseline score**: 0.154877
- **Current trunk score**: 0.154877

Use B_dev for final experiment scoring. Do NOT use B_test.

## Insights From Prior Experiments

- ROOT: Children findings: [1, done, score=0.3109] Original scratch SplArt accurately recovers axis and pivot but treats interior observations as endpoints: endpoint NMAE 0.310881, closed coverage 0, terminal validity 0. | [2, done, score=0.08573] D2-CEA reduced endpoint NMAE 72.42%, raised PSNR 3.72 dB and mobile IoU 46.74% with the base model hash unchanged. However all RMS/PCFG/OEC posthoc contact certificates abstained: closed coverage and terminal validity remain zero, showing the bottleneck is contact-surface representation rather than scalar search. | [3, done, score=0.08573] Children findings: [3.1, done, score=0.08573] View-balanced voxel fusion removed order/view-density bias, but observable surface contact and closure topology vote opposite sides; posthoc frozen-geometry terminal certification remains unidentifiable on Box. | [4, done, score=0.09515] Children findings: [4.1, done, score=0.1027] Deterministic visibility-censored Gaussianization slightly improved endpoint transfer but did not beat the NJC statistical prior; mesh-to-Gaussian appearance shift is not the sole bottleneck, and semantic/domain range priors remain dominant. | [5, pruned, score=0.1576] FECF only establ...
- 8: Children findings: [8.1, pruned, score=0.6882] Cross-state multi-radius consensus fails on representative Box-a: all 12 curves lack signed-gap zero crossings, consensus loses to D2 and a shuffled-geometry null, and uncertainty gating can only fall back to D2. [Pruned: Representative Box-a disproved cross-state/radius consensus on frozen penetrated profiles; the geometry shuffle null was better and gating yielded no gain.] | [8.2, pruned, score=0.343] The monotone hazard is ordered and semantics-sensitive on 36 target-free profiles, but Box-a has 12/12 signed-gap curves entirely nonpositive; hazard loses to frozen D2 and a channel-shuffle null, falsifying the free-to-terminal premise. [Pruned: Representative public Box-a falsified the mechanism: all 12 curves lack a free/contact crossing and the channel-shuffle null beats the proposed hazard.] | [8.3, pruned, score=0.4118] Projective consistency reduces exact36 target-domain cross-gauge variance by 37.34%, but fails domain transfer: on Box-a its worst-side NMAE and variance are worse than no-consistency, frozen D2, and coordinate controls. [Pruned: Target-free gauge stability improved in-domain but reversed under Box transfer and...

## Additional Context

Implement the unique preregistered RZA-ACWT label-free feasibility candidate only. Base is exact node8.10 protocol commit f7bdfa1; retain its strict double-recompute receipts, fold-local transforms, raw-z gates, no-label boundary, mapping and all thresholds. Add a new config/module/runner/tests rather than silently changing RQ-LSOT history. Fixed algorithm: location basis B=[1,z_c,q_perp] where q=P2(train ECDF x) and q_perp is q projected off [1,z_c] then normalized; positive QR OLS, guaranteeing full-fit mean/raw-z orthogonality. From residual R apply inherited scalar radial scale s0(x), g=R/s0. Build A1=mean[x_i(ggT-C0)], A2=mean[P2(x_i)(ggT-C0)], H=A1^2+A2^2; fix sign of top eigenvector v, require eigengap >100*eps64*||H||2. Fit separate log axial |a| and orthogonal bulk RMS scales on [1,x,P2(x)] with inherited epsilon/no clip; E=(a/s_parallel)v+b/s_perp. Reconstruct recipients with recipient location/scales, donor E, exact recipient u; mechanical d bitwise append. Rank=1 only, no alternate rank/degree/threshold/seed/sweep. Crossfit every fold refits joint-domain PCA/mechanical scaling and all domain-specific z/ECDF/location/s0/covariance axis/block scales. Semantic OOF back to raw DINO coordinates; mechanical keep-mask zero-fill. Preserve all unchanged gates and add eigengap/axis/scale/reconstruction/receipt audits. Fix row identity to be globally unique (domain + stable row key) if node8.10 diagnosis confirms cross-domain gauge collision. No remote, labels, source scores, Box, or training. Run focused tests, freeze one score-free commit, worktree clean, report hashes.

## Instructions

1. Understand the code before editing.
2. Implement the idea faithfully.
3. Run quick checks to ensure the new logic is active.
4. Iterate on implementation bugs.
5. Run the B_dev evaluation when credible.
6. Report Changes, Baseline vs Result, Score, and Insight. The score must be the absolute primary metric, not a delta.

Save results to `results/8.11-<brief-description>/`.
