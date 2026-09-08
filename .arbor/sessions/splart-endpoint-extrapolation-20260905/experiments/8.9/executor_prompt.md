## Codebase

Working directory: D:\workspace\projects\2026\09\05_splart_endpoint_extrapolation\worktrees\node89_ocrsmarc

## Git Isolation

Work in the assigned experiment branch/worktree. Do not switch back to the main repository for implementation or evaluation.

## Research Idea

**ID**: 8.9
**Hypothesis**:
Mechanism: Orthogonalized Conditional-Residual SMARC (OCR-SMARC) predicts authored articulation range with frozen DINO part semantics gated by geometry/displacement, and audits semantic/mechanical dependence using train-only same-domain object-residual permutations that preserve recipient conditional means plus within-object joint/gauge structure.
Hypothesis: The semantic-gated mechanical predictor can recover object-specific physical limits beyond domain/category priors; unlike failed 8.7/8.8 hard-caliper nulls, unregularized train-only OLS/SVD residual swaps provide 100% perturbation coverage with auditable zero-mean and linear orthogonality, without seed or threshold search.
Observable: Freeze all mappings and provenance before scores; require finite/nondegenerate residuals, cross-fit nonlinear-dependence diagnostics, exact state-swap and bit-identical displacement, then require full to beat the stronger analytic and mechanical-only controls by 15% on both source domains while both conditional-residual nulls worsen by 20%, with three LOFO folds stable at 15%.
Conflicts: CARC [8.5] showed geometry-only span evidence is harmful, DEKP [8.6] was explained by averaging, and SMARC nulls [8.7/8.8] were infeasible; this retains the semantic-mechanical interaction but replaces cross-object proximity matching with a fully covered conditional-residual ablation, and opens public Box a-d only after the source gate while leaving sealed e/f and B_test untouched.

## Evaluation Info

- **Phase A only**: Articraft and NJC object-disjoint source validation, final fit plus three frozen LOFO folds.
- **Primary metric**: source object-macro worst-side endpoint NMAE; lower is better.
- **Controls**: stronger of historical row-uniform and hierarchy-weighted analytic controls, mechanical-only, semantic conditional-residual ablation, and mechanical conditional-residual ablation.
- **Forbidden until Phase A passes**: every Box label/metric, old sealed 18/9/9, B_test, and Full22.
- **After a passing source gate only**: evaluate the exact frozen candidate on public Box a-d. Box e/f remain untouched.

## Insights From Prior Experiments

- ROOT: Children findings: [1, done, score=0.3109] Original scratch SplArt accurately recovers axis and pivot but treats interior observations as endpoints: endpoint NMAE 0.310881, closed coverage 0, terminal validity 0. | [2, done, score=0.08573] D2-CEA reduced endpoint NMAE 72.42%, raised PSNR 3.72 dB and mobile IoU 46.74% with the base model hash unchanged. However all RMS/PCFG/OEC posthoc contact certificates abstained: closed coverage and terminal validity remain zero, showing the bottleneck is contact-surface representation rather than scalar search. | [3, done, score=0.08573] Children findings: [3.1, done, score=0.08573] View-balanced voxel fusion removed order/view-density bias, but observable surface contact and closure topology vote opposite sides; posthoc frozen-geometry terminal certification remains unidentifiable on Box. | [4, done, score=0.09515] Children findings: [4.1, done, score=0.1027] Deterministic visibility-censored Gaussianization slightly improved endpoint transfer but did not beat the NJC statistical prior; mesh-to-Gaussian appearance shift is not the sole bottleneck, and semantic/domain range priors remain dominant. | [5, pruned, score=0.1576] FECF only establ...
- 8: Children findings: [8.1, pruned, score=0.6882] Cross-state multi-radius consensus fails on representative Box-a: all 12 curves lack signed-gap zero crossings, consensus loses to D2 and a shuffled-geometry null, and uncertainty gating can only fall back to D2. [Pruned: Representative Box-a disproved cross-state/radius consensus on frozen penetrated profiles; the geometry shuffle null was better and gating yielded no gain.] | [8.2, pruned, score=0.343] The monotone hazard is ordered and semantics-sensitive on 36 target-free profiles, but Box-a has 12/12 signed-gap curves entirely nonpositive; hazard loses to frozen D2 and a channel-shuffle null, falsifying the free-to-terminal premise. [Pruned: Representative public Box-a falsified the mechanism: all 12 curves lack a free/contact crossing and the channel-shuffle null beats the proposed hazard.] | [8.3, pruned, score=0.4118] Projective consistency reduces exact36 target-domain cross-gauge variance by 37.34%, but fails domain transfer: on Box-a its worst-side NMAE and variance are worse than no-consistency, frozen D2, and coordinate controls. [Pruned: Target-free gauge stability improved in-domain but reversed under Box transfer and...

## Additional Context

Implement node 8.9 as a fresh, result-free successor to C-SMARC infrastructure commit 35df7dc in the isolated worktree. Freeze configuration and null mappings before any score. Replace caliper matching with train-only, per-domain, object-level conditional residual ablations. For semantic: train-only PCA16 frozen DINO field; conditioning scalar z is each object's mean observed displacement. For mechanical: standardized nonconstant geometry field only; z is semantic distance to the train-domain mean; original recipient displacement d must remain bit-identical. Preserve recipient within-object offsets u_ok and reconstruct f'_ok = beta0 + beta1*z_o + r_donor + u_ok. Fit [1,z] by unregularized object-equal OLS/SVD, fail closed for Var(z)<=1e-12. Train donors are opaque-ID sorted cyclic shift one; held donors are same-domain train residuals in deterministic balanced round-robin, with mappings frozen and SHA-bound before scoring. Do not call this a strict CRT.
P0 preflight gates: train-only/domain-specific fit/preprocessing; no Box/protected/target reads; all finite; exact residual reconstruction; per-dimension normalized weighted mean and z-correlation errors <=1e-10; train/held coverage 1; train self rate 0, donor marginal exactly 1; held donors all train, load <=ceil(nheld/ntrain), effective donors/held=1; mechanical d bitwise unchanged; true raw state-swap invariance <=1e-10. Add target-free nondegeneracy gates: residual energy ratio >=0.05; shuffle RMS/original field SD >=0.1; shuffled latent norm p99 <=1.25x original train p99. Freeze a 5-fold opaque-hash cross-fit diagnostic and report residual-vs-z Spearman plus distance correlation with preregistered finite thresholds; failure makes the null unavailable, never tune after results.
Fix prior runner P0/P1 before any formal run: aggregate output directory exactly once and atomically; bind receipt task/steps/source provenance/config/code hash to each partial and current checked-out code; revalidate exact family IDs, all six frozen LOFO hashes, 109-object union and disjointness, null gates, and render-log content hashes; add end-to-end four-partial aggregate tests, permutation-order invariance, and mutation rejection tests. Require two commits before execution: protocol/null/tests first, hardened runner/tests second. Run local tests and source-only preflight. Only if preflight passes, launch 2-step smoke. Only then launch exactly 1200 steps final plus three LOFO tasks across free server98 A6000s, with immutable partial directories and separate logs. No seed sweep or selection. Source gate: full beats stronger analytic and mechanical-only controls by >=15% on both Articraft and NJC; semantic and mechanical conditional-residual ablations each worsen >=20%; each LOFO fold stable at >=15%; exact swap. Box a-d only after this gate; e/f and B_test untouched.

## Execution Mode

This is a real executor run, not an orchestration mock. Implement and commit the result-free protocol, run its source-only preflight, then use the remote A6000s for smoke/formal work only when the preceding gates pass. Never fabricate or mock a score.

## Instructions

1. Make a first commit containing only the frozen OCR-SMARC config, conditional-residual implementation, and focused tests; do not score before this commit.
2. Make a second commit fixing and testing the parallel producer/aggregator/provenance protocol. Aggregate must reject stale code, mutated receipts, wrong task/steps, wrong provenance, incomplete or overlapping LOFO partitions, weakened null gates, and partial-order changes.
3. Run local tests and a source-only remote preflight. Preserve its JSON, receipt, hashes, and logs in a new immutable result directory.
4. If and only if preflight passes, run a two-step smoke in a separate immutable directory. If and only if smoke passes, launch the four 1200-step source tasks across free GPUs.
5. Report actual metrics and all gate outcomes. A protocol/nondegeneracy failure is a prune-before-training result, not a method score.

Save local reports under `.arbor/sessions/splart-endpoint-extrapolation-20260905/experiments/8.9/` and remote artifacts under a new `/data1/public/yptang/splart-node89-ocrsmarc/` root.
