## Codebase

Working directory: D:\workspace\projects\2026\09\05_splart_endpoint_extrapolation\worktrees\node84_source_transfer

## Git Isolation

Work in the assigned experiment branch/worktree. Do not switch back to the main repository for implementation or evaluation.

## Research Idea

**ID**: 8.4
**Hypothesis**:
Mechanism: Source-Supervised Projective Boundary Transfer learns a shared ordered boundary head from complete source meshes with true finite joint-limit trajectories, then adapts only its representation on 36 unlabeled target-domain 3DGS trajectories through virtual-subinterval projective consistency and rank-normalized physics channels.
Hypothesis: Posthoc Box failures arise because frozen Gaussian profiles never contain a calibrated free/contact phase, while complete source meshes do; supervised source transitions plus target-domain gauge consistency can transfer the physical boundary law without using Box endpoints or absolute range priors.
Observable: On object-disjoint source validation the full model beats zero-geometry/range and coordinate-only controls, reduces target-free cross-gauge variance by at least 20%, and after freezing beats frozen D2 on public Box a-d worst-side NMAE before a single untouched e/f confirmation.
Conflicts: Node 4.1 showed generic mesh-to-Gaussian descriptor alignment is insufficient, node 5.1 showed 47D summaries collapse to range priors, and node 6 showed naive visual-mesh collision proxies are already penetrated; this node counters them with dense full-trajectory source supervision, per-channel rank normalization, and an explicit geometry-shuffle null, and must be pruned if the null ties or Box-a loses.

## Evaluation Info

- **Evaluation command (B_dev)**: `cd D:\workspace\projects\2026\09\05_splart_endpoint_extrapolation\worktrees\node84_source_transfer && python evaluate_public_endpoint_profiles.py --split public_box_af --run-name 8.4`
- **Evaluation command (B_test, do not use for routine experiments)**: `UNAVAILABLE_PROTECTED_B_TEST`
- **Dataset info**: Public Box a-f development episodes plus target-free first36 profiles; old sealed 18/9/9, B_test, and Full22 prohibited
- **Baseline score**: 0.154877
- **Current trunk score**: 0.154877

Use B_dev for final experiment scoring. Do NOT use B_test.

## Insights From Prior Experiments

- ROOT: Children findings: [1, done, score=0.3109] Original scratch SplArt accurately recovers axis and pivot but treats interior observations as endpoints: endpoint NMAE 0.310881, closed coverage 0, terminal validity 0. | [2, done, score=0.08573] D2-CEA reduced endpoint NMAE 72.42%, raised PSNR 3.72 dB and mobile IoU 46.74% with the base model hash unchanged. However all RMS/PCFG/OEC posthoc contact certificates abstained: closed coverage and terminal validity remain zero, showing the bottleneck is contact-surface representation rather than scalar search. | [3, done, score=0.08573] Children findings: [3.1, done, score=0.08573] View-balanced voxel fusion removed order/view-density bias, but observable surface contact and closure topology vote opposite sides; posthoc frozen-geometry terminal certification remains unidentifiable on Box. | [4, done, score=0.09515] Children findings: [4.1, done, score=0.1027] Deterministic visibility-censored Gaussianization slightly improved endpoint transfer but did not beat the NJC statistical prior; mesh-to-Gaussian appearance shift is not the sole bottleneck, and semantic/domain range priors remain dominant. | [5, pruned, score=0.1576] FECF only establ...
- 8: Children findings: [8.1, pruned, score=0.6882] Cross-state multi-radius consensus fails on representative Box-a: all 12 curves lack signed-gap zero crossings, consensus loses to D2 and a shuffled-geometry null, and uncertainty gating can only fall back to D2. [Pruned: Representative Box-a disproved cross-state/radius consensus on frozen penetrated profiles; the geometry shuffle null was better and gating yielded no gain.] | [8.2, pruned, score=0.343] The monotone hazard is ordered and semantics-sensitive on 36 target-free profiles, but Box-a has 12/12 signed-gap curves entirely nonpositive; hazard loses to frozen D2 and a channel-shuffle null, falsifying the free-to-terminal premise. [Pruned: Representative public Box-a falsified the mechanism: all 12 curves lack a free/contact crossing and the channel-shuffle null beats the proposed hazard.] | [8.3, done, score=0.4118] Projective consistency reduces exact36 target-domain cross-gauge variance by 37.34%, but fails domain transfer: on Box-a its worst-side NMAE and variance are worse than no-consistency, frozen D2, and coordinate controls.

## Additional Context

Use existing node-4 PILC Articraft/NJC source materialization and mesh assets by read-only git/artifact inspection; do not repeat descriptor MLP. Implement dense full-trajectory mesh energy profiles with true source joint-limit supervision, rank-normalize channels, then target-adapt on exact36 full-q target-free artifacts at /data1/public/yptang/splart-projective-node83/full-trajectories-v1. Match initialization across ablations; no seed search. Public Box a-d is development and e/f is untouched one-shot confirmation. Old sealed 18/9/9, B_test, Full22 are forbidden. Use GPU5 for source profile generation, GPU6 for training, GPU7 for independent ablations if memory permits; pre-set CUBLAS_WORKSPACE_CONFIG=:4096:8 before Python.

## Instructions

1. Understand the code before editing.
2. Implement the idea faithfully.
3. Run quick checks to ensure the new logic is active.
4. Iterate on implementation bugs.
5. Run the B_dev evaluation when credible.
6. Report Changes, Baseline vs Result, Score, and Insight. The score must be the absolute primary metric, not a delta.

Save results to `results/8.4-<brief-description>/`.
