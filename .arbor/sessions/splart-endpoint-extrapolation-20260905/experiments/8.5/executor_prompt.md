## Codebase

Working directory: D:\workspace\projects\2026\09\05_splart_endpoint_extrapolation\worktrees\node85_carc

## Git Isolation

Work in the assigned experiment branch/worktree. Do not switch back to the main repository for implementation or evaluation.

## Research Idea

**ID**: 8.5
**Hypothesis**:
Mechanism: Closure-Anchored Range Completion (CARC) factorizes the two extrapolated endpoints into one geometry-supported closed-contact coordinate and one learned physical motion span; an exactly swap-equivariant kinematic layer places the opposite hard stop from the predicted span and the observed relative screw displacement.
Hypothesis: Surface evidence can identify closure but not an authored open stop, while source supervision can estimate mechanism span; separating these evidence types avoids forcing both endpoints through the same corrupted contact minimum and prevents independent-extension range collapse.
Observable: With object-disjoint Articraft/NJC source validation, the full factorized model beats range-only, zero/shuffled-geometry, and the old independent PILC endpoint head by at least 15% worst-side NMAE; after target-free freeze it lowers public Box a-d macro worst-side NMAE at least 10% versus frozen D2 and wins at least 3/4 objects before any e/f access.
Conflicts: Node 5.1 found that 47D geometry harmed transfer and node 8.4 proved URDF hard stops are not contact minima; CARC counters them by using geometry only for closure/evidence gating, supervising only physical span in the range head, enforcing span closure algebraically, and pruning if a range-only or geometry-shuffled null ties.

## Evaluation Info

- **Evaluation command (B_dev)**: `cd D:\workspace\projects\2026\09\05_splart_endpoint_extrapolation\worktrees\node85_carc && python evaluate_public_endpoint_profiles.py --split public_box_af --run-name 8.5`
- **Evaluation command (B_test, do not use for routine experiments)**: `UNAVAILABLE_PROTECTED_B_TEST`
- **Dataset info**: Public Box a-f development episodes plus target-free first36 profiles; old sealed 18/9/9, B_test, and Full22 prohibited
- **Baseline score**: 0.154877
- **Current trunk score**: 0.154877

Use B_dev for final experiment scoring. Do NOT use B_test.

## Insights From Prior Experiments

- ROOT: Children findings: [1, done, score=0.3109] Original scratch SplArt accurately recovers axis and pivot but treats interior observations as endpoints: endpoint NMAE 0.310881, closed coverage 0, terminal validity 0. | [2, done, score=0.08573] D2-CEA reduced endpoint NMAE 72.42%, raised PSNR 3.72 dB and mobile IoU 46.74% with the base model hash unchanged. However all RMS/PCFG/OEC posthoc contact certificates abstained: closed coverage and terminal validity remain zero, showing the bottleneck is contact-surface representation rather than scalar search. | [3, done, score=0.08573] Children findings: [3.1, done, score=0.08573] View-balanced voxel fusion removed order/view-density bias, but observable surface contact and closure topology vote opposite sides; posthoc frozen-geometry terminal certification remains unidentifiable on Box. | [4, done, score=0.09515] Children findings: [4.1, done, score=0.1027] Deterministic visibility-censored Gaussianization slightly improved endpoint transfer but did not beat the NJC statistical prior; mesh-to-Gaussian appearance shift is not the sole bottleneck, and semantic/domain range priors remain dominant. | [5, pruned, score=0.1576] FECF only establ...
- 8: Children findings: [8.1, pruned, score=0.6882] Cross-state multi-radius consensus fails on representative Box-a: all 12 curves lack signed-gap zero crossings, consensus loses to D2 and a shuffled-geometry null, and uncertainty gating can only fall back to D2. [Pruned: Representative Box-a disproved cross-state/radius consensus on frozen penetrated profiles; the geometry shuffle null was better and gating yielded no gain.] | [8.2, pruned, score=0.343] The monotone hazard is ordered and semantics-sensitive on 36 target-free profiles, but Box-a has 12/12 signed-gap curves entirely nonpositive; hazard loses to frozen D2 and a channel-shuffle null, falsifying the free-to-terminal premise. [Pruned: Representative public Box-a falsified the mechanism: all 12 curves lack a free/contact crossing and the channel-shuffle null beats the proposed hazard.] | [8.3, done, score=0.4118] Projective consistency reduces exact36 target-domain cross-gauge variance by 37.34%, but fails domain transfer: on Box-a its worst-side NMAE and variance are worse than no-consistency, frozen D2, and coordinate controls.

## Additional Context

Implement node 8.5 only in isolated worktree D:\workspace\projects\2026\09\05_splart_endpoint_extrapolation\worktrees\node85_carc. Reuse read-only node4 PILC code/artifacts and source data on CASIA_98. Effective units are joints/objects, not 32 grid rows. Start with a feasibility runner on existing Articraft 97/12 and NJC 11/4: global range, displacement-only, geometry-only, geometry+displacement, geometry-shuffle, old PILC, and a factorized range-projection head with matched deterministic initialization/no seed search. Geometry may inform closure/reliability; physical_range supervision informs span. Enforce the exact span identity R=|d|(1+e0+e1) and state-swap equivariance. Because Articraft lacks closed labels, report oracle scalar-anchor feasibility there, and use NJC closed_index+extensions for end-to-end closure anchoring; never pretend a binary side label identifies distance. Gate on joint/object macro worst-side; require full to beat displacement-only and old independent head, not only global prior. Public Box a-d is permitted only after freezing source/target-free choice; e/f, old sealed18/9/9, B_test and Full22 are forbidden. Target exact36 profiles may be used without labels only for projective/noncollapse diagnostics. Use server98 GPUs 5/6/7 only after verifying free; pre-set CUBLAS_WORKSPACE_CONFIG before Python; durable tmux; report PID/log/output. First commit tests + smoke runner and launch full/null in parallel.

## Full Experiment Mode

This is an authorized real implementation and experiment run, not an orchestration smoke. Implement the node in the assigned worktree, first run a cheap plumbing smoke, then launch the preregistered full and null experiments on verified-free GPUs. Do not substitute mocked or cached scores for new evidence.

## Instructions

1. Implement tests, a deterministic feasibility runner, and the CARC factorized projection without reading protected targets.
2. Treat the 32 fixed gauges per joint as augmentations; score by joint/object macro and add leave-one-factor-out plus continuous virtual-gauge tests.
3. Run a short plumbing smoke, then full and null experiments with matched initialization and no seed search.
4. Freeze the source/target-free selection before any public Box a-d evaluation; do not read e/f, old sealed 18/9/9, B_test, or Full22.
5. Record a full report with Changes, Baseline vs Result, Score, Analysis, Insight, provenance, exact PID/log/output paths, and all failed gates.

Save experiment artifacts under `.arbor/sessions/<run>/experiments/8.5/` and immutable remote outputs under `/data1/public/yptang/splart-node85-carc/`.
