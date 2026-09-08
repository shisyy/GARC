## Codebase

Working directory: D:\workspace\projects\2026\09\05_splart_endpoint_extrapolation\worktrees\node810_rqlsot

## Git Isolation

Work in the assigned experiment branch/worktree. Do not switch back to the main repository for implementation or evaluation.

## Research Idea

**ID**: 8.10
**Hypothesis**:
Mechanism: Rank-Quadratic Location-Scale Orthogonal Transport SMARC (RQ-LSOT-SMARC) replaces the failed linear residual null with a per-domain train-only empirical-rank quadratic location model and linear log-radial-scale model, then transports standardized object residuals through the same exact opaque-ID shift-one/balanced donor maps while preserving recipient joint/gauge offsets and mechanical displacement.
Hypothesis: Node 8.9 failed only the frozen nonlinear-dependence diagnostics while passing coverage, exact reconstruction, linear orthogonality, energy, perturbation and norm gates; a fixed three-term rank-quadratic mean plus two-term log-scale correction should remove the remaining nonlinear location and heteroscedastic dependence without changing thresholds, seeds, the predictor, or target data.
Observable: Before any label read, freeze QR sign/rank/condition rules, ECDF held clipping, cross-fit-local preprocessing, exact hashes and the unchanged Spearman<=0.35/dCor<=0.5 and prior gates; require all four domain-by-null cells to pass twice bit-identically with row-order invariance, boundary-clipped held fraction<=25%, exact quadratic-basis orthogonality, 100% donor coverage, and no regression on any 8.9 gate.
Conflicts: Unlike 8.7/8.8 donor-distance matching and 8.9 linear OLS residualization, RQ-LSOT changes the conditional nuisance model rather than relaxing feasibility thresholds; it remains a deterministic falsification ablation rather than a permutation p-value, and if label-free feasibility fails the node is pruned without source labels, training, Box, degree search, or threshold adjustment.

## Evaluation Info

- **Current phase**: label-free nuisance-null feasibility only.
- **Inputs allowed**: frozen source feature caches, opaque object/split/family sidecars, renderer logs, and frozen DINO checkpoint hash.
- **Forbidden**: opening, hashing, or scoring any source label payload; every Box episode; old sealed 18/9/9; B_test; Full22.
- **Decision**: all four Articraft/NJC × semantic/mechanical RQ-LSOT cells and every unchanged gate pass, or prune before scoring.

## Insights From Prior Experiments

- ROOT: Children findings: [1, done, score=0.3109] Original scratch SplArt accurately recovers axis and pivot but treats interior observations as endpoints: endpoint NMAE 0.310881, closed coverage 0, terminal validity 0. | [2, done, score=0.08573] D2-CEA reduced endpoint NMAE 72.42%, raised PSNR 3.72 dB and mobile IoU 46.74% with the base model hash unchanged. However all RMS/PCFG/OEC posthoc contact certificates abstained: closed coverage and terminal validity remain zero, showing the bottleneck is contact-surface representation rather than scalar search. | [3, done, score=0.08573] Children findings: [3.1, done, score=0.08573] View-balanced voxel fusion removed order/view-density bias, but observable surface contact and closure topology vote opposite sides; posthoc frozen-geometry terminal certification remains unidentifiable on Box. | [4, done, score=0.09515] Children findings: [4.1, done, score=0.1027] Deterministic visibility-censored Gaussianization slightly improved endpoint transfer but did not beat the NJC statistical prior; mesh-to-Gaussian appearance shift is not the sole bottleneck, and semantic/domain range priors remain dominant. | [5, pruned, score=0.1576] FECF only establ...
- 8: Children findings: [8.1, pruned, score=0.6882] Cross-state multi-radius consensus fails on representative Box-a: all 12 curves lack signed-gap zero crossings, consensus loses to D2 and a shuffled-geometry null, and uncertainty gating can only fall back to D2. [Pruned: Representative Box-a disproved cross-state/radius consensus on frozen penetrated profiles; the geometry shuffle null was better and gating yielded no gain.] | [8.2, pruned, score=0.343] The monotone hazard is ordered and semantics-sensitive on 36 target-free profiles, but Box-a has 12/12 signed-gap curves entirely nonpositive; hazard loses to frozen D2 and a channel-shuffle null, falsifying the free-to-terminal premise. [Pruned: Representative public Box-a falsified the mechanism: all 12 curves lack a free/contact crossing and the channel-shuffle null beats the proposed hazard.] | [8.3, pruned, score=0.4118] Projective consistency reduces exact36 target-domain cross-gauge variance by 37.34%, but fails domain transfer: on Box-a its worst-side NMAE and variance are worse than no-consistency, frozen D2, and coordinate controls. [Pruned: Target-free gauge stability improved in-domain but reversed under Box transfer and...

## Additional Context

Implement only the single preregistered RQ-LSOT conditional nuisance model on top of clean node8.9 commit 1c3d1c9. Do not change the SMARC predictor, optimizer, source gates, seeds, or any existing threshold. Per domain and field: convert train z to empirical mid-CDF rank x in [-1,1]; held uses train ECDF clipped to [0.5/n,1-0.5/n], report clipped fraction and fail >0.25. Fit object means with fixed Legendre-style basis [1,x,(3x^2-1)/2] using float64 thin QR with positive R diagonal; rank<3 or condition>1e6 fails. Fit log radial residual scale with [1,x] QR-OLS, eps=max(1e-12,1e-6*RMS(centered object means)), no clipping. Standardized residual E=(Ybar-M(x))/s(x). Transport exact sorted opaque-ID shift1 on train and sorted-held to sorted-train balanced RR on held; reconstruct M_recipient+s_recipient*E_donor+u_recipient. Preserve mechanical d bitwise. Crossfit diagnostics must recompute ECDF, QR mean, and scale wholly inside each balanced hash fold and evaluate OOF standardized E. Keep original Spearman<=.35, dCor<=.5, reconstruction/basis-orthogonality<=1e-10, residual energy>=.05, shuffle RMS/SD>=.1, p99 ratio<=1.25 and all coverage/mapping gates. Add rank/condition/finite/no-scale-clip, row-order invariance, repeated-run bit identity, mapping/hash/mutation tests. First commit result-free config+implementation+tests before any remote run. Then independent audit. Remote feasibility must be labels-never-opened and output both frozen node8.9 linear historical diagnostics and the unique RQ-LSOT diagnostics without candidate sweep/selection; all four domain-null cells must pass and no old gate regress. If feasibility fails, prune and never read source labels or Box. If it passes, stop and request coordinator authorization for source score/smoke/formal using the exact frozen code.

## Execution Mode

This is a real result-free implementation and feature-feasibility run, not an orchestration mock. Implement, test, commit, independently audit, then run the single frozen candidate remotely without labels.

## Instructions

1. Freeze config, implementation, and focused tests in one result-free commit before any remote feasibility execution.
2. Demonstrate row-order invariance, repeat bit identity, cross-fit locality, exact transport reconstruction, mapping/provenance mutation rejection, and that source label files are neither opened nor hashed.
3. Obtain independent code audit, then run one immutable remote label-free feasibility task for the unique RQ-LSOT candidate.
4. Do not run model training, smoke, formal scoring, or Box evaluation in this node. If feasibility passes, return control for a separately authorized scoring phase; if it fails, record and prune without tuning.
5. Preserve exact config/code/input/output hashes and a report under `results/8.10-rqlsot-*`.
