## Codebase

Working directory: D:\workspace\projects\2026\09\05_splart_endpoint_extrapolation\worktrees\node812_jsaect

## Git Isolation

Work in the assigned experiment branch/worktree. Do not switch back to the main repository for implementation or evaluation.

## Research Idea

**ID**: 8.12
**Hypothesis**:
Mechanism: Joint-domain Shared-Axis Energy-Conserving Transport (JSA-ECT) learns one pooled rank-one conditional axis and shared anisotropy from Articraft+NJC, while retaining domain-specific exact raw-z anchors and constant block-scale intercepts.
Hypothesis: NJC failures arise from estimating a conditional axis and free scale curve with only 11 objects; common-principal-component pooling raises the axis effective sample size to 108, and fixed-energy anisotropy prevents the tail inflation seen on Articraft semantic.
Observable: With the frozen five folds and unchanged gates, all four label-free cells pass, full-fit raw-z mean/correlation remain <=1e-10, Articraft semantic p99<=1.25, and NJC semantic/mechanical dCor<=0.5 without loss of recipient-u or bitwise mechanical-d preservation.
Conflicts: Pruned node8.11 showed per-domain rank-one anisotropic transport overfits NJC; this node shares only identifiable conditional structure across domains while preserving domain nuisance anchors and scales, and is pruned without changing rank, basis, folds or thresholds if any cell fails.

## Evaluation Info

- **Evaluation command (B_dev)**: `cd D:\workspace\projects\2026\09\05_splart_endpoint_extrapolation\worktrees\node812_jsaect && python evaluate_public_endpoint_profiles.py --split public_box_af --run-name 8.12`
- **Evaluation command (B_test, do not use for routine experiments)**: `UNAVAILABLE_PROTECTED_B_TEST`
- **Dataset info**: Public Box a-f development episodes plus target-free first36 profiles; old sealed 18/9/9, B_test, and Full22 prohibited
- **Baseline score**: 0.154877
- **Current trunk score**: 0.154877

Use B_dev for final experiment scoring. Do NOT use B_test.

## Insights From Prior Experiments

- ROOT: Children findings: [1, done, score=0.3109] Original scratch SplArt accurately recovers axis and pivot but treats interior observations as endpoints: endpoint NMAE 0.310881, closed coverage 0, terminal validity 0. | [2, done, score=0.08573] D2-CEA reduced endpoint NMAE 72.42%, raised PSNR 3.72 dB and mobile IoU 46.74% with the base model hash unchanged. However all RMS/PCFG/OEC posthoc contact certificates abstained: closed coverage and terminal validity remain zero, showing the bottleneck is contact-surface representation rather than scalar search. | [3, done, score=0.08573] Children findings: [3.1, done, score=0.08573] View-balanced voxel fusion removed order/view-density bias, but observable surface contact and closure topology vote opposite sides; posthoc frozen-geometry terminal certification remains unidentifiable on Box. | [4, done, score=0.09515] Children findings: [4.1, done, score=0.1027] Deterministic visibility-censored Gaussianization slightly improved endpoint transfer but did not beat the NJC statistical prior; mesh-to-Gaussian appearance shift is not the sole bottleneck, and semantic/domain range priors remain dominant. | [5, pruned, score=0.1576] FECF only establ...
- 8: Children findings: [8.1, pruned, score=0.6882] Cross-state multi-radius consensus fails on representative Box-a: all 12 curves lack signed-gap zero crossings, consensus loses to D2 and a shuffled-geometry null, and uncertainty gating can only fall back to D2. [Pruned: Representative Box-a disproved cross-state/radius consensus on frozen penetrated profiles; the geometry shuffle null was better and gating yielded no gain.] | [8.2, pruned, score=0.343] The monotone hazard is ordered and semantics-sensitive on 36 target-free profiles, but Box-a has 12/12 signed-gap curves entirely nonpositive; hazard loses to frozen D2 and a channel-shuffle null, falsifying the free-to-terminal premise. [Pruned: Representative public Box-a falsified the mechanism: all 12 curves lack a free/contact crossing and the channel-shuffle null beats the proposed hazard.] | [8.3, pruned, score=0.4118] Projective consistency reduces exact36 target-domain cross-gauge variance by 37.34%, but fails domain transfer: on Box-a its worst-side NMAE and variance are worse than no-consistency, frozen D2, and coordinate controls. [Pruned: Target-free gauge stability improved in-domain but reversed under Box transfer and...

## Additional Context

Implement exactly one score-free JSA-ECT candidate on base 00c8cbd; add new config/module/runner/tests and leave RZA/RQ history untouched. No remote/data/labels/source scores/Box/training or candidate sweep.
Fixed math per field in the existing joint fold-local feature coordinates: for each domain d compute own-domain train ECDF u and q0=6u^2-6u+1; residualize q0 by positive-QR against [1,z_centered] within d and divide by positive infinity norm (natural q0 sign, no candidate choice). Build one pooled OLS design with domain-specific intercept and raw-z slope columns plus one shared q column, object weight exactly 1 so Art97 stabilizes shared structure. This design must guarantee each domain raw-residual mean/raw-z covariance <=1e-10. Let residual r.
For each domain compute covariance intercept C_d=mean(rrT). Form pooled symmetric contrast H=sum_i q_i(rrT-C_di)/sum_i q_i^2, then G=H@H symmetrized and take its unique top eigenvector as the one shared axis; eigengap on G uses unchanged numeric multiplier, sign largest-absolute coordinate positive. No rank alternative. Let axial a=r dot v, bulk b=r-av. Domain baseline sigma_parallel^2=mean(a^2), sigma_perp^2=mean(||b||^2)/(p-1), both positive/fail-closed. Define c_i=a_i^2/sigma_parallel_d^2 - ||b_i||^2/((p-1)sigma_perp_d^2); theta=(sum q_i c_i)/sqrt(sum q_i^2 sum c_i^2), the fixed cosine/Rayleigh modulation in [-1,1] by Cauchy-Schwarz; fail if nonfinite or abs(theta)>1+numeric tolerance, never clamp. With |q|<=1 set v_i=exp(theta q_i), D_i=(sigma_parallel_d^2 v_i+(p-1)sigma_perp_d^2)/(sigma_parallel_d^2+(p-1)sigma_perp_d^2), s_parallel^2=sigma_parallel_d^2 v_i/D_i, s_perp^2=sigma_perp_d^2/D_i. Audit exact per-row total block energy conservation; no learned free scale, clipping, ridge or refit. Standardized E=(a/s_parallel)v+b/s_perp; exact reconstruction location+s_parallel E_parallel+s_perp E_bulk.
Transport uses unchanged domain sorted shift1/RR, recipient offsets u and mechanical d bitwise. Five-fold crossfit refits joint PCA/mechanical scaling, both domain ECDF/q anchors, pooled mean, pooled H/G/axis/theta, and domain energy intercepts on complement; other-domain held objects excluded. Apply held domain transform. Keep canonical composite row reductions and independent double recompute/no-alias receipts from node8.11. Preserve every gate exactly: Spearman .35, dCor .5, raw mean/corr and orth/recon 1e-10, clip .25, energy .05, shuffle .1, p99 1.25, coverage/mapping/u/d/repeat/row-order; add pooled design, H/G symmetry, theta bound, energy conservation, shared-axis equality and fold provenance audits. Formal runner output remote=true/CASIA_98/label_free_feasibility while config snapshot remote=false. Freeze config before result; local synthetic tests only; one score-free commit clean.

## Instructions

1. Understand the code before editing.
2. Implement the idea faithfully.
3. Run quick checks to ensure the new logic is active.
4. Iterate on implementation bugs.
5. Run the B_dev evaluation when credible.
6. Report Changes, Baseline vs Result, Score, and Insight. The score must be the absolute primary metric, not a delta.

Save results to `results/8.12-<brief-description>/`.
