## Codebase

Working directory: D:\workspace\projects\2026\09\05_splart_endpoint_extrapolation\worktrees\node88_csmarc

## Git Isolation

Work in the assigned experiment branch/worktree. Do not switch back to the main repository for implementation or evaluation.

## Research Idea

**ID**: 8.8
**Hypothesis**:
Mechanism: Calipered Semantic-Gated Range Completion (C-SMARC) couples frozen DINO part semantics to bias-free mechanical residual experts, predicts an authored revolute range inside [|d|,2pi], and exactly projects the two terminal extensions; its causal null uses same-domain partial permutations with hard 0.5 robust-z calipers and leaves unmatched tails unchanged.
Hypothesis: Functional appearance carries authored hard-stop information absent from two-state geometry/displacement, while conservative partial permutations, black-image/mechanical-only controls, and family-disjoint refits prevent category lookup or deliberately weak nulls from explaining gains.
Observable: Before Box access, one frozen 1200-step configuration must beat the stronger of historical and hierarchy-weighted global/displacement/category controls plus mechanical-only by at least 15% on both Articraft and NJC; matched shuffles must worsen at least 20%, each-domain perturbation coverage must be at least 90%, Articraft LOFO aggregate must pass 15%, and measured state-swap error must be at most 1e-6.
Conflicts: Unlike CARC's failed geometry-only range cue, DEKP's averaging-only gain, and node 8.7's infeasible full derangement, semantics only selects mechanical laws and the null preserves close-domain covariates while keeping unmatched outliers unperturbed.

## Evaluation Info

- **Phase-A evaluation**: source-only Articraft object-disjoint validation, NJC object-disjoint validation, and three Articraft leave-one-family-out replicas.
- **Primary metric**: object-macro physical-range MARE; lower is better.
- **Advancement rule**: only an admissible Phase-A pass may unlock public Box a-d. Box e/f, exact36, old sealed 18/9/9, B_test, and Full22 remain prohibited.
- **Frozen source controls**: historical row-uniform controls plus newly reported hierarchy-weighted controls; compare against the stronger lower-error value.

Do not invoke the generic target-domain evaluation command during Phase A.

## Insights From Prior Experiments

- ROOT: Children findings: [1, done, score=0.3109] Original scratch SplArt accurately recovers axis and pivot but treats interior observations as endpoints: endpoint NMAE 0.310881, closed coverage 0, terminal validity 0. | [2, done, score=0.08573] D2-CEA reduced endpoint NMAE 72.42%, raised PSNR 3.72 dB and mobile IoU 46.74% with the base model hash unchanged. However all RMS/PCFG/OEC posthoc contact certificates abstained: closed coverage and terminal validity remain zero, showing the bottleneck is contact-surface representation rather than scalar search. | [3, done, score=0.08573] Children findings: [3.1, done, score=0.08573] View-balanced voxel fusion removed order/view-density bias, but observable surface contact and closure topology vote opposite sides; posthoc frozen-geometry terminal certification remains unidentifiable on Box. | [4, done, score=0.09515] Children findings: [4.1, done, score=0.1027] Deterministic visibility-censored Gaussianization slightly improved endpoint transfer but did not beat the NJC statistical prior; mesh-to-Gaussian appearance shift is not the sole bottleneck, and semantic/domain range priors remain dominant. | [5, pruned, score=0.1576] FECF only establ...
- 8: Children findings: [8.1, pruned, score=0.6882] Cross-state multi-radius consensus fails on representative Box-a: all 12 curves lack signed-gap zero crossings, consensus loses to D2 and a shuffled-geometry null, and uncertainty gating can only fall back to D2. [Pruned: Representative Box-a disproved cross-state/radius consensus on frozen penetrated profiles; the geometry shuffle null was better and gating yielded no gain.] | [8.2, pruned, score=0.343] The monotone hazard is ordered and semantics-sensitive on 36 target-free profiles, but Box-a has 12/12 signed-gap curves entirely nonpositive; hazard loses to frozen D2 and a channel-shuffle null, falsifying the free-to-terminal premise. [Pruned: Representative public Box-a falsified the mechanism: all 12 curves lack a free/contact crossing and the channel-shuffle null beats the proposed hazard.] | [8.3, pruned, score=0.4118] Projective consistency reduces exact36 target-domain cross-gauge variance by 37.34%, but fails domain transfer: on Box-a its worst-side NMAE and variance are worse than no-consistency, frozen D2, and coordinate controls. [Pruned: Target-free gauge stability improved in-domain but reversed under Box transfer and...

## Additional Context

Start from the cherry-picked node 8.7 infrastructure at ae5b37b, but this is a fresh result-free node. Freeze config v1.3/8.8 before any method score. Implement calipered same-domain partial permutations: robust-z within train domain; non-self donor edges only if distance <=0.5; lexicographically maximize non-self coverage then minimize distance while preserving a permutation on train objects; unmatched train tails map to self and remain unperturbed; held uses nearest same-domain train donor within 0.5 else preserves its own field. Require and report >=90% perturbed object coverage separately for Articraft and NJC for each null, and gate per-domain. Mechanical shuffle swaps only geometry dimensions and preserves recipient displacement. Compute both historical row-uniform and hierarchy-weighted analytic controls; gate against the lower error/stronger control for global, displacement, and category. Fix audit P0/P1: assert all six frozen LOFO hashes, family-id keys and exact three-fold partition; strict PILC manifest schema/protected/Box/source SHA provenance; exact renderer angles/resolution/topology/coverage; true raw forward/reverse state input swap plus signed displacement and endpoint audit; formal config hash receipt. Add parallel task mode and deterministic aggregator so variants and three LOFO folds run across free GPUs 5,6,7. Use exactly 1200 steps, analytic deterministic initialization, no seed selection/sweep. Run preflight, 2-step smoke, then formal source Phase A only if audits pass. Do not read Box until the source gate passes. Preserve all partial outputs, logs, commits, hashes, and report honestly.

## Instructions

1. Implement and commit the frozen 8.8 protocol before running any method score.
2. Run unit and negative tests, then a source-only preflight and two-step plumbing smoke.
3. Dispatch the single formal 1200-step configuration across free GPUs 5, 6, and 7 with immutable task outputs and a deterministic aggregator.
4. Stop before Box if any source gate fails. If every source gate passes, freeze the exact candidate and request the coordinator's Box a-d unlock.
5. Record a full report with Changes, Baseline vs Result, Score, Analysis, Insight, provenance hashes, and explicit protected-read receipts.

Save artifacts under `results/8.8-csmarc-*` in the experiment worktree and the designated remote result root.
