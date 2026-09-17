## Codebase

Working directory: D:\workspace\projects\2026\09\05_splart_endpoint_extrapolation\worktrees\node817_se_mdc

## Git Isolation

Work in the assigned experiment branch/worktree. Do not switch back to the main repository for implementation or evaluation.

## Research Idea

**ID**: 8.17
**Hypothesis**:
Mechanism: Swap-Equivariant Minimax Dual Coupler (SE-MDC) augments CR-FPL with a symmetric two-side attention step whose shared dual slack tokens estimate relative endpoint risk and jointly allocate each loop's convex log-distance update, trained by a smooth object-level worst-side saddle objective.
Hypothesis: CR-FPL's independent shared-side loops optimize the mean but cannot move correction capacity toward the harder endpoint, causing p99 regressions despite convergent residuals; a permutation-equivariant minimax dual lets the two sides compete through relative evidence without introducing side identities or handcrafted thresholds.
Observable: On the unchanged Articraft 13/6 and NJC 11/4 source gates, four-loop SE-MDC preserves CR-FPL's mean gain while lowering worst-side NMAE and target-relative p99 versus CR-FPL and its own same-parameter one-loop output; swap error remains <=1e-6 and loop residuals remain decreasing.
Conflicts: Validated [8.16] proves local re-query and fixed-point convergence but shows Articraft worst-side and both-domain p99 tail regressions; this node changes cross-side credit allocation rather than loop count, width, seed, thresholds, or data splits, and avoids the marginal-transport tail failures of pruned [8.11]-[8.13].

## Evaluation Info

- **Evaluation command (B_dev)**: `cd D:\workspace\projects\2026\09\05_splart_endpoint_extrapolation\worktrees\node817_se_mdc && python evaluate_public_endpoint_profiles.py --split public_box_af --run-name 8.17`
- **Evaluation command (B_test, do not use for routine experiments)**: `UNAVAILABLE_PROTECTED_B_TEST`
- **Dataset info**: Public Box a-f development episodes plus target-free first36 profiles; old sealed 18/9/9, B_test, and Full22 prohibited
- **Baseline score**: 0.154877
- **Current trunk score**: 0.154877

Use B_dev for final experiment scoring. Do NOT use B_test.

## Insights From Prior Experiments

- ROOT: Children findings: [1, done, score=0.3109] Original scratch SplArt accurately recovers axis and pivot but treats interior observations as endpoints: endpoint NMAE 0.310881, closed coverage 0, terminal validity 0. | [2, done, score=0.08573] D2-CEA reduced endpoint NMAE 72.42%, raised PSNR 3.72 dB and mobile IoU 46.74% with the base model hash unchanged. However all RMS/PCFG/OEC posthoc contact certificates abstained: closed coverage and terminal validity remain zero, showing the bottleneck is contact-surface representation rather than scalar search. | [3, done, score=0.08573] Children findings: [3.1, done, score=0.08573] View-balanced voxel fusion removed order/view-density bias, but observable surface contact and closure topology vote opposite sides; posthoc frozen-geometry terminal certification remains unidentifiable on Box. | [4, done, score=0.09515] Children findings: [4.1, done, score=0.1027] Deterministic visibility-censored Gaussianization slightly improved endpoint transfer but did not beat the NJC statistical prior; mesh-to-Gaussian appearance shift is not the sole bottleneck, and semantic/domain range priors remain dominant. | [5, pruned, score=0.1576] FECF only establ...
- 8: Children findings: [8.1, pruned, score=0.6882] Cross-state multi-radius consensus fails on representative Box-a: all 12 curves lack signed-gap zero crossings, consensus loses to D2 and a shuffled-geometry null, and uncertainty gating can only fall back to D2. [Pruned: Representative Box-a disproved cross-state/radius consensus on frozen penetrated profiles; the geometry shuffle null was better and gating yielded no gain.] | [8.2, pruned, score=0.343] The monotone hazard is ordered and semantics-sensitive on 36 target-free profiles, but Box-a has 12/12 signed-gap curves entirely nonpositive; hazard loses to frozen D2 and a channel-shuffle null, falsifying the free-to-terminal premise. [Pruned: Representative public Box-a falsified the mechanism: all 12 curves lack a free/contact crossing and the channel-shuffle null beats the proposed hazard.] | [8.3, pruned, score=0.4118] Projective consistency reduces exact36 target-domain cross-gauge variance by 37.34%, but fails domain transfer: on Box-a its worst-side NMAE and variance are worse than no-consistency, frozen D2, and coordinate controls. [Pruned: Target-free gauge stability improved in-domain but reversed under Box transfer and...

## Additional Context

Implement SE-MDC immediately on CR-FPL commit 7742819. Preserve exact four tied loops, width64 base representation, seed2202, 4000 steps, Articraft 13/6 and NJC 11/4 splits. Add a swap-equivariant two-side minimax dual coupling module and a smooth worst-side saddle/deep-supervision objective without hard thresholds or quantile tuning. Keep all existing source/NJC loaders and diagnostics. Report both-domain mean, worst-side, p99, exact swap, same-parameter one-loop, residuals, and matched independent-side CR-FPL reference. Local tests and commit only; coordinator launches GPUs.

## Instructions

1. Understand the code before editing.
2. Implement the idea faithfully.
3. Run quick checks to ensure the new logic is active.
4. Iterate on implementation bugs.
5. Run the B_dev evaluation when credible.
6. Report Changes, Baseline vs Result, Score, and Insight. The score must be the absolute primary metric, not a delta.

Save results to `results/8.17-<brief-description>/`.
