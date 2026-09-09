## Codebase

Working directory: D:\workspace\projects\2026\09\05_splart_endpoint_extrapolation\worktrees\node813_pksrt

## Git Isolation

Work in the assigned experiment branch/worktree. Do not switch back to the main repository for implementation or evaluation.

## Research Idea

**ID**: 8.13
**Hypothesis**:
Mechanism: Pooled Kernel Sliced Rosenblatt Transport (PKSRT) uses domain marginal quantile bijections, an Articraft-stabilized shared kernel conditional CDF, a fixed identity-plus-DCT slice sequence, and a final per-domain raw-z QR anchor to form an invertible conditional distribution transport.
Hypothesis: Node8.12's remaining errors come from finite-moment models failing on nonlinear marginals and copula dependence; a shared nonparametric conditional probability-integral transform can use 97 Articraft objects to stabilize NJC while domain marginals and anchors prevent cross-domain shift.
Observable: Under the unchanged five folds and gates, all four cells pass: mechanical/Articraft Spearman<=0.35, mechanical/NJC and semantic/NJC dCor<=0.5, actual E mean/raw-z<=1e-10, inverse error<=1e-10 and p99<=1.25 with exact recipient-u/mechanical-d preservation.
Conflicts: Nodes8.9-8.12 show linear, quadratic, per-domain rank-one and shared rank-one moment transports leave different dependence forms; this node changes to a complete conditional-distribution transport and is pruned without adding bases, directions, bandwidth choices or thresholds if it fails.

## Evaluation Info

- **Evaluation command (B_dev)**: `cd D:\workspace\projects\2026\09\05_splart_endpoint_extrapolation\worktrees\node813_pksrt && python evaluate_public_endpoint_profiles.py --split public_box_af --run-name 8.13`
- **Evaluation command (B_test, do not use for routine experiments)**: `UNAVAILABLE_PROTECTED_B_TEST`
- **Dataset info**: Public Box a-f development episodes plus target-free first36 profiles; old sealed 18/9/9, B_test, and Full22 prohibited
- **Baseline score**: 0.154877
- **Current trunk score**: 0.154877

Use B_dev for final experiment scoring. Do NOT use B_test.

## Insights From Prior Experiments

- ROOT: Children findings: [1, done, score=0.3109] Original scratch SplArt accurately recovers axis and pivot but treats interior observations as endpoints: endpoint NMAE 0.310881, closed coverage 0, terminal validity 0. | [2, done, score=0.08573] D2-CEA reduced endpoint NMAE 72.42%, raised PSNR 3.72 dB and mobile IoU 46.74% with the base model hash unchanged. However all RMS/PCFG/OEC posthoc contact certificates abstained: closed coverage and terminal validity remain zero, showing the bottleneck is contact-surface representation rather than scalar search. | [3, done, score=0.08573] Children findings: [3.1, done, score=0.08573] View-balanced voxel fusion removed order/view-density bias, but observable surface contact and closure topology vote opposite sides; posthoc frozen-geometry terminal certification remains unidentifiable on Box. | [4, done, score=0.09515] Children findings: [4.1, done, score=0.1027] Deterministic visibility-censored Gaussianization slightly improved endpoint transfer but did not beat the NJC statistical prior; mesh-to-Gaussian appearance shift is not the sole bottleneck, and semantic/domain range priors remain dominant. | [5, pruned, score=0.1576] FECF only establ...
- 8: Children findings: [8.1, pruned, score=0.6882] Cross-state multi-radius consensus fails on representative Box-a: all 12 curves lack signed-gap zero crossings, consensus loses to D2 and a shuffled-geometry null, and uncertainty gating can only fall back to D2. [Pruned: Representative Box-a disproved cross-state/radius consensus on frozen penetrated profiles; the geometry shuffle null was better and gating yielded no gain.] | [8.2, pruned, score=0.343] The monotone hazard is ordered and semantics-sensitive on 36 target-free profiles, but Box-a has 12/12 signed-gap curves entirely nonpositive; hazard loses to frozen D2 and a channel-shuffle null, falsifying the free-to-terminal premise. [Pruned: Representative public Box-a falsified the mechanism: all 12 curves lack a free/contact crossing and the channel-shuffle null beats the proposed hazard.] | [8.3, pruned, score=0.4118] Projective consistency reduces exact36 target-domain cross-gauge variance by 37.34%, but fails domain transfer: on Box-a its worst-side NMAE and variance are worse than no-consistency, frozen D2, and coordinate controls. [Pruned: Target-free gauge stability improved in-domain but reversed under Box transfer and...

## Additional Context

Implement exactly one score-free PKSRT candidate atop exact 7a57da0, adding new config/module/runner/tests and leaving JSA/RZA/RQ history untouched. No remote/real data/labels/source scores/Box/training, no seed/direction/layer/bandwidth/gate sweep.
Fixed first stage per domain: object means y, raw z; own train ECDF u=(x+1)/2. Positive-QR OLS A_d=[1,z_centered]B_d, residual r=(y-A_d)/s_d where s_d=RMS residual scalar; require s_d^2 > rank_relative_tolerance*centered-field energy, no floor. Mechanical d excluded.
Fixed slice sequence exactly the p coordinate basis rows then p canonical orthonormal DCT-II rows, each row sign fixed by largest-absolute coordinate positive, order 0..p-1 in each basis. For each slice v, current t=vT r. Domain marginal G_d is a strict real-line bijection: sort unique t (tie/nonpositive gap fail), map knots t_(j) to a_j=logit((j+0.5)/n_d), piecewise linear internally and linear endpoint extrapolation with first/last positive slopes; inverse is exact piecewise linear. This is the fixed implementation of the quantile-tail contract.
Shared conditional bijection at recipient u: pool all train objects with weight1. Store each training a_j=G_domain(t_j) and u_j. Group samples having exactly equal a into ordered unique knots a_g using canonical domain/object order and canonical reductions; this is required because the frozen 97/11 odd-sized domain rank grids share the median a=0. h_N=(1/sqrt(3))*(4/(3N))^(1/5), no selection. For query u, w_j=exp(-(u-u_j)^2/(2h_N^2)); n_eff=(sumw)^2/sumw^2>=2. Let W_g=sum_{j:a_j=a_g}w_j and p_g(u)=(0.5 + sum_{h<g}W_h + 0.5*W_g)/(1+sumw); eta_g=logit(p_g). Require at least two unique knots. The grouped knots are strict; define H_u:a->eta by piecewise linear internal + positive linear endpoint tails, exact inverse eta->a. Record raw pooled count, unique-knot count, tie-group count, maximum multiplicity and grouping hash. Forward eta=H_u(G_d(t)); update r+=(eta-t)v. Train sequentially on updated r. Inverse for recipient uses same stored slice model in exact reverse order: eta=vTr, a=H_u^{-1}(eta), t=G_d^{-1}(a), update r+=(t-eta)v.
After all 2p slices, per domain positive-QR fit C_d on [1,z_centered] to final latent eta-vector. Actual transport E=eta-XC_d and gate E per-domain mean/raw-z <=1e-10. Reconstruction adds recipient XC_d, inverse slices with recipient domain/u, multiplies recipient s_d, adds A_d(z). Self inverse <=1e-10. Null maps unchanged domain shift1/RR donor E, preserves recipient within-object offsets and mechanical d bitwise.
Crossfit frozen5: each fold removes both domains' held objects, refits joint PCA/mech mask, both domain ECDF/A/s, all 2p G/H slice models, final C. Held only forward. Semantic E backproject raw DINO; mechanical zero-fill. Keep unchanged gates .35/.5/p991.25/etc plus inverse, strict monotonicity, bandwidth, min n_eff, slice count/order/basis hashes, all knots/inputs/provenance. Canonical reductions/row-order, independent double recompute with frozen expected SHA and full graph-disjoint, global structure + per-domain metric gates. Formal runner must inherit hardened source/predecessor/object/log/code provenance; required predecessor config is JSA config SHA a89543..., source 454f...; module invocation; result remote=true/CASIA_98/label_free_feasibility, config snapshot false. Output may be large but atomic. Synthetic tests cover exact forward/inverse, tails, ties, n_eff, reverse/repeat, held exclusion, coherent receipt attacks. Freeze one clean score-free commit after broad tests.

### Preregistration erratum before implementation

The original draft required every pooled `a_j` to be unique. Before code was implemented or any remote payload was opened, the already recorded source object counts (Articraft 97, NJC 11) proved this impossible: both domain ECDF grids contain the exact median `a=0`. The fixed grouped-weight definition above replaces that impossible precondition with the canonical weighted empirical-CDF definition. It adds no parameter, jitter, direction, threshold or data-dependent choice, and must be covered by an odd/odd-domain shared-median unit test.

## Instructions

1. Understand the code before editing.
2. Implement the idea faithfully.
3. Run quick checks to ensure the new logic is active.
4. Iterate on implementation bugs.
5. Run the B_dev evaluation when credible.
6. Report Changes, Baseline vs Result, Score, and Insight. The score must be the absolute primary metric, not a delta.

Save results to `results/8.13-<brief-description>/`.
