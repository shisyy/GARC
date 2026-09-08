## Codebase

Working directory: D:\workspace\projects\2026\09\05_splart_endpoint_extrapolation\worktrees\node86_dekp

## Git Isolation

Work in the assigned experiment branch/worktree. Do not switch back to the main repository for implementation or evaluation.

## Research Idea

**ID**: 8.6
**Hypothesis**:
Mechanism: Dual-Expert Kinematic Projection (DEKP) fuses the frozen D2 analytic endpoint and frozen PILC learned-prior endpoint with per-side reliabilities computed from expert disagreement, D2 multi-gauge dispersion, and posterior entropy, then applies the exact span projection retained from CARC.
Hypothesis: D2 is accurate when its counterfactual energy is well conditioned whereas PILC repairs missing-contact cases; their complementary failures on frozen Box development episodes can be selected by target-side evidence without transferring the failed source-geometry range head.
Observable: Under leave-one-episode-out evaluation on public Box a-d, DEKP lowers macro worst-side NMAE by at least 10% versus both frozen D2 and frozen PILC, wins at least 3/4 episodes, and beats fixed averaging, coordinate-only, uncertainty-shuffled, D2-only, and PILC-only controls before a single untouched e/f confirmation.
Conflicts: Node 8.3 showed projective variance alone can be confidently wrong and node 8.5 showed source geometry hurts range transfer; DEKP counters them with disagreement between independently derived experts, monotone low-capacity reliability calibration, exact kinematic projection, and mandatory shuffled-uncertainty/collapsed-weight nulls, and is pruned if weights collapse or the shuffle ties.

## Evaluation Info

- **Evaluation command (B_dev)**: `cd D:\workspace\projects\2026\09\05_splart_endpoint_extrapolation\worktrees\node86_dekp && python evaluate_public_endpoint_profiles.py --split public_box_af --run-name 8.6`
- **Evaluation command (B_test, do not use for routine experiments)**: `UNAVAILABLE_PROTECTED_B_TEST`
- **Dataset info**: Public Box a-f development episodes plus target-free first36 profiles; old sealed 18/9/9, B_test, and Full22 prohibited
- **Baseline score**: 0.154877
- **Current trunk score**: 0.154877

Use B_dev for final experiment scoring. Do NOT use B_test.

## Insights From Prior Experiments

- ROOT: Children findings: [1, done, score=0.3109] Original scratch SplArt accurately recovers axis and pivot but treats interior observations as endpoints: endpoint NMAE 0.310881, closed coverage 0, terminal validity 0. | [2, done, score=0.08573] D2-CEA reduced endpoint NMAE 72.42%, raised PSNR 3.72 dB and mobile IoU 46.74% with the base model hash unchanged. However all RMS/PCFG/OEC posthoc contact certificates abstained: closed coverage and terminal validity remain zero, showing the bottleneck is contact-surface representation rather than scalar search. | [3, done, score=0.08573] Children findings: [3.1, done, score=0.08573] View-balanced voxel fusion removed order/view-density bias, but observable surface contact and closure topology vote opposite sides; posthoc frozen-geometry terminal certification remains unidentifiable on Box. | [4, done, score=0.09515] Children findings: [4.1, done, score=0.1027] Deterministic visibility-censored Gaussianization slightly improved endpoint transfer but did not beat the NJC statistical prior; mesh-to-Gaussian appearance shift is not the sole bottleneck, and semantic/domain range priors remain dominant. | [5, pruned, score=0.1576] FECF only establ...
- 8: Children findings: [8.1, pruned, score=0.6882] Cross-state multi-radius consensus fails on representative Box-a: all 12 curves lack signed-gap zero crossings, consensus loses to D2 and a shuffled-geometry null, and uncertainty gating can only fall back to D2. [Pruned: Representative Box-a disproved cross-state/radius consensus on frozen penetrated profiles; the geometry shuffle null was better and gating yielded no gain.] | [8.2, pruned, score=0.343] The monotone hazard is ordered and semantics-sensitive on 36 target-free profiles, but Box-a has 12/12 signed-gap curves entirely nonpositive; hazard loses to frozen D2 and a channel-shuffle null, falsifying the free-to-terminal premise. [Pruned: Representative public Box-a falsified the mechanism: all 12 curves lack a free/contact crossing and the channel-shuffle null beats the proposed hazard.] | [8.3, done, score=0.4118] Projective consistency reduces exact36 target-domain cross-gauge variance by 37.34%, but fails domain transfer: on Box-a its worst-side NMAE and variance are worse than no-consistency, frozen D2, and coordinate controls.

## Additional Context

Implement node 8.6 in D:\workspace\projects\2026\09\05_splart_endpoint_extrapolation\worktrees\node86_dekp only. This is real full execution. Frozen experts: D2 commit dd78dcc5355fd0f41fb00541d6a18dc36d87ef91 and PILC checkpoint bbd109b66ed854400c47571945a625938a80dede80b827053236231b8108c38a. Public dev episodes are exactly box-extra-a, box-extra-b, box-c, box-d; exclude v4 from the primary four and never access e/f. D2 public scores are known: a worst .283389, b .200845, c .162948, d .183468; macro .207662. PILC a/b predictions/scores exist and c/d frozen comparator is being generated under /data1/public/yptang/splart-node85-carc/pilc-cd-comparator-v1. Export or reuse target-free D2 full trajectories for a-d before fitting reliability; a/b profiles exist at /data1/public/yptang/splart-projective-node83/box-trajectories-v1, c/d may be exported with the node83 frozen exporter on free GPUs. Build a no-seed, low-capacity monotone reliability rule using expert disagreement, per-side multi-gauge variance/entropy, and PILC confidence; all leave-one-episode-out folds must fit on exactly three episodes and score the fourth. Also evaluate fixed 0.5 average, projected average, D2-only, PILC-only, coordinate-only, shuffled-uncertainty, and collapsed-weight nulls. Exact state-swap equivariance and projection span identity required. Do not claim generalization from four same-object episodes; promotion is development evidence only and requires >=10% macro worst-side gain vs both experts plus >=3/4 wins. Freeze method/config before any e/f, and do not open old sealed18/9/9, B_test, or Full22. Use verified-free GPU5/6/7, CUBLAS env before Python, durable tmux, exact PID/log/output.

## Instructions

1. Understand the code before editing.
2. Implement the idea faithfully.
3. Run quick checks to ensure the new logic is active.
4. Iterate on implementation bugs.
5. Run the B_dev evaluation when credible.
6. Report Changes, Baseline vs Result, Score, and Insight. The score must be the absolute primary metric, not a delta.

Save results to `results/8.6-<brief-description>/`.
