## Codebase

Working directory: D:\workspace\projects\2026\09\05_splart_endpoint_extrapolation\worktrees\node816_cr_fpl

## Git Isolation

Work in the assigned experiment branch/worktree. Do not switch back to the main repository for implementation or evaluation.

## Research Idea

**ID**: 8.16
**Hypothesis**:
Mechanism: Counterfactual Re-query Fixed-Point Loop (CR-FPL) turns each ALD-PDL log-distance estimate back into a differentiable query of the ordered D2 profile, injects the sampled local geometry and query residual as the next loop's primal-dual feedback token, and applies a shared convex fixed-point update with the same four tied pre-norm loops.
Hypothesis: Node 8.15's static recalled input makes later loops accumulate stale corrections, so one loop beats four; state-dependent re-query makes every later loop observe a new counterfactual constraint while convex fixed-point interpolation prevents the last-loop rebound.
Observable: On the unchanged 13/6 source B_dev, the four-loop prediction beats both node 8.15 and its own parameter-matched one-loop output in mean-side and worst-side NMAE, exact swap error is <=1e-6, target-relative p99 does not regress, and prediction/state residuals decrease through loop four.
Conflicts: Pruned [8.1]/[8.2] showed no explicit signed-gap crossing, so CR-FPL does not threshold or search for a crossing; it uses the continuously sampled local profile only as iterative feedback, directly countering validated [8.15]'s static-input overshoot without changing seed, width, loop count, or split.

## Evaluation Info

- **Evaluation command (B_dev)**: `cd D:\workspace\projects\2026\09\05_splart_endpoint_extrapolation\worktrees\node816_cr_fpl && python evaluate_public_endpoint_profiles.py --split public_box_af --run-name 8.16`
- **Evaluation command (B_test, do not use for routine experiments)**: `UNAVAILABLE_PROTECTED_B_TEST`
- **Dataset info**: Public Box a-f development episodes plus target-free first36 profiles; old sealed 18/9/9, B_test, and Full22 prohibited
- **Baseline score**: 0.154877
- **Current trunk score**: 0.154877

Use B_dev for final experiment scoring. Do NOT use B_test.

## Insights From Prior Experiments

- ROOT: Children findings: [1, done, score=0.3109] Original scratch SplArt accurately recovers axis and pivot but treats interior observations as endpoints: endpoint NMAE 0.310881, closed coverage 0, terminal validity 0. | [2, done, score=0.08573] D2-CEA reduced endpoint NMAE 72.42%, raised PSNR 3.72 dB and mobile IoU 46.74% with the base model hash unchanged. However all RMS/PCFG/OEC posthoc contact certificates abstained: closed coverage and terminal validity remain zero, showing the bottleneck is contact-surface representation rather than scalar search. | [3, done, score=0.08573] Children findings: [3.1, done, score=0.08573] View-balanced voxel fusion removed order/view-density bias, but observable surface contact and closure topology vote opposite sides; posthoc frozen-geometry terminal certification remains unidentifiable on Box. | [4, done, score=0.09515] Children findings: [4.1, done, score=0.1027] Deterministic visibility-censored Gaussianization slightly improved endpoint transfer but did not beat the NJC statistical prior; mesh-to-Gaussian appearance shift is not the sole bottleneck, and semantic/domain range priors remain dominant. | [5, pruned, score=0.1576] FECF only establ...
- 8: Children findings: [8.1, pruned, score=0.6882] Cross-state multi-radius consensus fails on representative Box-a: all 12 curves lack signed-gap zero crossings, consensus loses to D2 and a shuffled-geometry null, and uncertainty gating can only fall back to D2. [Pruned: Representative Box-a disproved cross-state/radius consensus on frozen penetrated profiles; the geometry shuffle null was better and gating yielded no gain.] | [8.2, pruned, score=0.343] The monotone hazard is ordered and semantics-sensitive on 36 target-free profiles, but Box-a has 12/12 signed-gap curves entirely nonpositive; hazard loses to frozen D2 and a channel-shuffle null, falsifying the free-to-terminal premise. [Pruned: Representative public Box-a falsified the mechanism: all 12 curves lack a free/contact crossing and the channel-shuffle null beats the proposed hazard.] | [8.3, pruned, score=0.4118] Projective consistency reduces exact36 target-domain cross-gauge variance by 37.34%, but fails domain transfer: on Box-a its worst-side NMAE and variance are worse than no-consistency, frozen D2, and coordinate controls. [Pruned: Target-free gauge stability improved in-domain but reversed under Box transfer and...

## Additional Context

Implement CR-FPL on node8.15 immediately. Preserve seed2202, 4000 steps, exact 13/6 split, width64 and exactly four tied loops. Add differentiable state-dependent profile re-query feedback and a convex fixed-point update; do not use endpoint thresholds/crossing search or tune any scalar. Retain exact side sharing and log-distance positivity. Report parameter-matched one-loop from the same trained weights, mean/worst NMAE, target-relative p99, swap error and per-loop residuals. Local tests and commit only; remote launch is coordinator-owned.

## Instructions

1. Understand the code before editing.
2. Implement the idea faithfully.
3. Run quick checks to ensure the new logic is active.
4. Iterate on implementation bugs.
5. Run the B_dev evaluation when credible.
6. Report Changes, Baseline vs Result, Score, and Insight. The score must be the absolute primary metric, not a delta.

Save results to `results/8.16-<brief-description>/`.
