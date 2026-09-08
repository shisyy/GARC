# Idea Tree

**Baseline**: 0.3109 | **Trunk**: 0.3109

## ROOT: Given two interior articulation states relabelled 0 and 1, predict the fully closed endpoint and the opposite physical joint limit, and reconstruct held-out endpoint views without access to absolute interior fractions or true limits. [DONE]

**Insight**: Children findings: [1, done, score=0.3109] Original scratch SplArt accurately recovers axis and pivot but treats interior observations as endpoints: endpoint NMAE 0.310881, closed coverage 0, terminal validity 0. | [2, done, score=0.08573] D2-CEA reduced endpoint NMAE 72.42%, raised PSNR 3.72 dB and mobile IoU 46.74% with the base model hash unchanged. However all RMS/PCFG/OEC posthoc contact certificates abstained: closed coverage and terminal validity remain zero, showing the bottleneck is contact-surface representation rather than scalar search. | [3, done, score=0.08573] Children findings: [3.1, done, score=0.08573] View-balanced voxel fusion removed order/view-density bias, but observable surface contact and closure topology vote opposite sides; posthoc frozen-geometry terminal certification remains unidentifiable on Box. | [4, done, score=0.09515] Children findings: [4.1, done, score=0.1027] Deterministic visibility-censored Gaussianization slightly improved endpoint transfer but did not beat the NJC statistical prior; mesh-to-Gaussian appearance shift is not the sole bottleneck, and semantic/domain range priors remain dominant. | [5, done, score=0.1576] Children findings:...

### 1: Mechanism: Middle-State Episode Reparameterization builds two interior multi-view states, erases their absolute joint fractions from model-facing metadata, and evaluates predicted closed/limit endpoints on held-out endpoint cameras.
Hypothesis: The current task appears successful because training states equal the URDF limits; moving both observations strictly inside the range will expose true extrapolation error and establish an honest new baseline.
Observable: A sealed Dev3 proxy passes leakage and geometry audits, and unmodified scratch SplArt produces endpoint-limit, endpoint-render, depth, IoU, and articulation baseline metrics.
Conflicts: none - this changes the benchmark assumption rather than retreading a failed optimization mechanism. [DONE] (score: 0.3109)

**Insight**: Original scratch SplArt accurately recovers axis and pivot but treats interior observations as endpoints: endpoint NMAE 0.310881, closed coverage 0, terminal validity 0.

**Result**: Box v4 baseline trained from scratch for 25k iterations and was evaluated on 20 sealed endpoint cameras; score SHA da7ea787.

**Branch**: arbor/splart-middle-baseline

### 2: Mechanism: Contact-Feasible Endpoint Field scans the learned screw trajectory with differentiable static/mobile contact, penetration, and terminal-support energies, predicting two latent endpoint scalars plus a closed-end label and uncertainty interval.
Hypothesis: Physical endpoints are extrema of the collision-free configuration interval; broad closure contact and localized stop contact provide different signatures that can extrapolate beyond two interior observations.
Observable: On sealed middle-state Dev3, normalized endpoint error and closed-end accuracy improve over scratch SplArt while endpoint PSNR, SSIM, LPIPS, depth, IoUs, and articulation remain noninferior.
Conflicts: prior IKC point-to-point experiments optimized consistency between observed endpoint states rather than inferring unseen limits; this node uses them only as implementation lessons and adds a latent feasible interval with no limit labels. [DONE] (score: 0.08573)

**Insight**: D2-CEA reduced endpoint NMAE 72.42%, raised PSNR 3.72 dB and mobile IoU 46.74% with the base model hash unchanged. However all RMS/PCFG/OEC posthoc contact certificates abstained: closed coverage and terminal validity remain zero, showing the bottleneck is contact-surface representation rather than scalar search.

**Result**: Best fixed multi-radius counterfactual commit cdb62c6: q=(-0.80130,1.38924), NMAE 0.085727, PSNR 26.8976, mIoU 0.81979. Independently audited PASS receipt f65f28f9; not merge-eligible because closed and terminal gates failed.

**Branch**: arbor/node-2-contact-endpoint-field

#### 2.1: Mechanism: Endpoint Extrapolation Null Suite implements symmetric linear extension, an object-disjoint global range prior, and single-factor D2 ablations for contact, penetration, terminal support, radius aggregation, and axis-pivot quality.
Hypothesis: Scratch SplArt is not a sufficient comparator for an endpoint-extrapolation method; D2 is credible only if its counterfactual geometric terms outperform non-geometric range priors and each claimed component has measurable causal value.
Observable: On frozen development episodes and then the object-level balanced suite, report endpoint NMAE, per-end error, win rate, bootstrap confidence intervals, and runtime for every baseline and ablation under one pre-registered evaluation budget without score-selected hyperparameters.
Conflicts: Node 2 showed a large canonical Box gain but only about 2.3 percent macro improvement on c/d and lacked strong null models; this node tests mechanism necessity and stability rather than proposing another endpoint estimator. [DONE] (score: 0.1549)

**Insight**: Across six preregistered Box development episodes, full D2 retains contact: macro NMAE 0.154877 vs 0.155311 without contact, no-contact wins only 2/6 and slightly worsens every reconstruction/penetration macro; both have terminal validity zero. Strong symmetric and range priors plus v4 component ablations support multi-radius profiling and penetration avoidance, not physical terminal contact. Superseding independent P0 PASS receipt 559138980c579fbfa6df4dbd17b16773dae520f98974f3b0c23305a5abb130ea.

**Result**: Frozen full-vs-no-contact a-f evaluation and all hashes/provenance independently reproduced after a permission-only 0755-to-0700 repair. Delete-contact gate failed; full D2 remains candidate. Receipt SHA256 559138980c579fbfa6df4dbd17b16773dae520f98974f3b0c23305a5abb130ea.

**Branch**: b610f01a7f3e5a85cc568d0872749d2f460403bd

#### 2.2: Mechanism: Gauge-Equivariant Dual-Boundary Energy Profile Head maps the full multi-radius D2 trajectory energy curve to two nonnegative endpoint distances and split-conformal feasible intervals, with shared weights and an exact observation-order swap operator.
Hypothesis: D2's scalar optimizer discards profile shape and produces a 13.8x lower-versus-upper error imbalance; retaining the bidirectional profile lets an object-disjoint head correct asymmetric range bias while calibrated intervals expose intrinsically ambiguous endpoints.
Observable: On sealed B_dev objects, reduce object-macro endpoint NMAE versus frozen full D2 and global/range-prior heads, preserve swap error below 1e-10, and achieve preregistered 90% object-level interval coverage with narrower width than marginal/global conformal controls.
Conflicts: pruned [5] showed an equivariant closure head learned only the canonical-side prior; this predicts geometric boundary distances from shuffled-audited energy profiles and must beat zero-geometry, permuted-profile, and order-only nulls before promotion. [PRUNED]

**Insight**: The independently authorized sealed v2 recovery terminated at shared-head step 0 on the first backward because deterministic CUDA execution required CUBLAS_WORKSPACE_CONFIG to be set before process launch. Targets had been consumed and the optimizer initialized, so the frozen no-retry/no-v3 rule applies. No model, checkpoint, calibration statistic, prediction, RESULT, or metric was produced; node 2.2 has no confirmatory performance evidence.

**Result**: FAILED_TERMINAL environment abort; ABORTED_ENVIRONMENT SHA256 2eb45b2af72743deb213ab928dda59a2cd430e6a79c89cf77662b3d178cc5ae9. V1 and V2 failures are retained; B_test and Full22 remain unread.

**Branch**: commit 25119c2; runner 97c12d076faa3dbfa307b70a1d5feabcf09080600e309babcc1f98612cb0ff5a

### 3: Mechanism: Counterfactual Closure Topology renders static, mobile, and interface-only alpha-depth over all public cameras plus a fixed virtual orbit and jointly scores aperture leakage, interface exposure, enclosure, and distributed contact coverage.
Hypothesis: Global functional topology distinguishes sealed closure from a local mechanical stop even when point-contact mass is noisy, while uncertainty gates prevent forced labels.
Observable: Relative to D2-CEA on B_dev, closed coverage and selective accuracy improve without increasing endpoint NMAE or degrading reconstruction and articulation; symmetric or infeasible cases remain unknown.
Conflicts: Node 2 showed all frozen Gaussian contact proxies abstain; this attacks global rendered topology rather than retuning their local thresholds. [DONE] (score: 0.08573)

**Insight**: Children findings: [3.1, done, score=0.08573] View-balanced voxel fusion removed order/view-density bias, but observable surface contact and closure topology vote opposite sides; posthoc frozen-geometry terminal certification remains unidentifiable on Box.

**Result**: Commits 8e31e8b and 54c08a3; 29 tests passed. Closed remained unknown and D2 scalars were preserved; no score claim or threshold relaxation.

**Branch**: arbor/node-3-closure-topology

#### 3.1: Mechanism: Joint Topology-Contact Surface Refinement learns an explicit static-mobile interface SDF from public RGB-D and alternates endpoint-scalar projection with topology-contact consistency while distilling the frozen SplArt renders.
Hypothesis: The current failure is caused by optimizing endpoints against an inaccurate frozen Gaussian collision surface; a learned interface surface and alternating feasibility projection can create physically meaningful terminal support without sacrificing the successful D2 endpoint geometry.
Observable: On Box B_dev, endpoint NMAE stays at or below 0.085727 while closed coverage becomes nonzero with correct selective prediction, terminal validity becomes one, penetration does not increase, and endpoint rendering/articulation remain noninferior.
Conflicts: Nodes 2 and 3 showed posthoc RMS, PCFG, OEC, topology, and self-rendered certificates all abstain; this counters by changing the surface representation and training objective rather than relaxing any certificate gate. [DONE] (score: 0.08573)

**Insight**: View-balanced voxel fusion removed order/view-density bias, but observable surface contact and closure topology vote opposite sides; posthoc frozen-geometry terminal certification remains unidentifiable on Box.

**Result**: Formal-v2 kept D2 endpoint NMAE 0.08572745 but closed coverage and terminal validity remained zero. Frozen certificate rejected both scalar updates; node stopped without sealed evaluator or threshold tuning.

**Branch**: arbor/node-3-1-jtcsr@4df5e8bf4036c8f4665c43abba1e269269e3c0e1

### 4: Mechanism: Paired-Interior Limit Calibrator (PILC) learns a state-swap-equivariant dual-endpoint and selective closed-side posterior from canonicalized static/mobile surface features plus the observed relative screw transform; object-disjoint training may use limit labels, but runtime receives no URDF, limits, or absolute state fractions.
Hypothesis: A learned geometric support prior can resolve the upper-end bias left by D2-CEA and provide calibrated nonzero closed-side coverage without changing the 3DGS reconstruction.
Observable: On pre-registered held-out episodes, endpoint NMAE falls below 0.070, closed-side coverage is at least 0.5 with reported risk, terminal validity is positive, and state-swap/order consistency tests pass.
Conflicts: Category/range memorization, reversible metadata leakage, and genuinely unidentifiable partial surfaces could create false confidence; require object-disjoint splits, metadata audit, geometry-only/statistical ablations, and unknown abstention. [DONE] (score: 0.09515)

**Insight**: Children findings: [4.1, done, score=0.1027] Deterministic visibility-censored Gaussianization slightly improved endpoint transfer but did not beat the NJC statistical prior; mesh-to-Gaussian appearance shift is not the sole bottleneck, and semantic/domain range priors remain dominant.

**Result**: Frozen triplet macro endpoint NMAE 0.08396856, closed coverage/accuracy 1/1, terminal validity 0. Canonical v4 NMAE 0.09515394 failed the 0.070 gate and was worse than D2 0.08572745; node not merge eligible.

**Branch**: arbor/node-4-pilc@fadc2ceff3c2d8c36cf69f48b706be927e839c34

#### 4.1: Mechanism: Visibility-Censored Gaussian Surface Adaptation (VCGSA) trains PILC on deterministic partial-view, opacity-weighted Gaussianized surfaces generated from Articraft/NJC meshes, matching the same 47D runtime extractor while preserving the state-swap-equivariant endpoint and selective-closure heads.
Hypothesis: The remaining Box error is caused primarily by full-mesh-to-partial-3DGS feature shift; matching visibility, density, and opacity statistics during training should improve cross-domain endpoint calibration while retaining PILC's closed-side signal.
Observable: Articraft validation and NJC calibration both beat their frozen statistical priors, zero-geometry remains worse, swap error stays below 1e-6, and a newly pre-registered Box episode set reaches macro endpoint NMAE below 0.070 with closed coverage/accuracy above 0.8.
Conflicts: Surface censoring may erase the contact geometry needed for limits or merely simulate Box-specific artifacts; use fixed Fibonacci views/hash-derived censoring, object-disjoint splits, no Box-score-guided tuning, and stop if validation does not improve before any new Box evaluation. [DONE] (score: 0.1027)

**Insight**: Deterministic visibility-censored Gaussianization slightly improved endpoint transfer but did not beat the NJC statistical prior; mesh-to-Gaussian appearance shift is not the sole bottleneck, and semantic/domain range priors remain dominant.

**Result**: Articraft validation NMAE 0.113945 beat 0.240316 prior, but NJC 0.102659 failed against 0.098711 prior. Internal gate stopped all Box evaluation; no second censor adjustment.

**Branch**: arbor/node-4-1-vcgsa@7938ef9fd9624cbefd81f974bc4a9e93d596318d

### 5: Mechanism: Factorized Endpoint-Closure Fusion (FECF) preserves the frozen D2-CEA physical endpoint scalars while attaching only PILC's independently trained state-swap-equivariant closed-side posterior with its fixed 0.80 abstention threshold; neither endpoint nor reconstruction parameters are updated.
Hypothesis: D2's analytic endpoint calibration and PILC's learned closure semantics are complementary, so factorization should retain D2 endpoint accuracy while converting closed coverage from zero to calibrated nonzero values.
Observable: On newly pre-registered Box c/d, endpoint predictions and reconstruction hashes are bit-identical to D2, closed coverage and accuracy are each at least 0.8 with reported selective risk, and constant-lower/category-prior baselines are explicitly compared.
Conflicts: PILC may simply learn an always-lower shortcut and terminal contact can remain invalid even when the side label is correct; reject the node if it does not beat the constant-side prior or if any D2 scalar/hash changes. [PRUNED] (score: 0.1576)

**Insight**: FECF only establishes exact coordinate equivariance on current Box episodes, not learned closure semantics. All canonical labels are identical; a zero-geometry equivariant prior ties decisions and beats calibration, while shuffled geometry is indistinguishable. The second-innovation claim is withdrawn pending a canonical-label-balanced object-level benchmark.

**Related work**: Conditionally novel after two-round 2023-2026 search: closest are LARM (two endpoint-state interpolation), CVPR24 Digital Twins (two-state articulation reconstruction), CARTO (factorized shape/articulation), Articulate-Anything and ArtLLM (joint-limit prediction), and learned-prior plus analytic factor-graph fusion. Safe novelty is the strict combination of two endpoint-free interior observations, bidirectional limit extrapolation, exact observation-order equivariance for limits and closed-side, and factorized analytic magnitude plus learned selective closure semantics; do not claim first joint-limit prediction, first two-state reconstruction, or terminal certification.

**Result**: Withdrawn after node5.3 preregistered null audit. D2 endpoint scalars remain valid and unchanged; FECF method artifacts are preserved as negative evidence but are not merge-eligible or publication-claim eligible.

**Branch**: arbor/node-5-fecf@9decf84456bf7efb05a0a9599da4578220cf60f4 (method core 60da6d86fc36f3d7228aa1715632c51a14051f56)

#### 5.1: Mechanism: Counterfactual Terminal Field learns a query-conditioned static-mobile clearance, contact-support, and penetration field from object-disjoint physically swept trajectories, then selects each limit at the feasible-to-supported-contact boundary while FECF supplies the closed-side semantics.
Hypothesis: Terminal validity is zero because frozen partial 3DGS surfaces contain no supervision for the unobserved transition from free motion to supported contact; dense counterfactual sweep supervision changes both the representation and objective so that the transition itself, rather than a posthoc proximity threshold, is learned.
Observable: On pre-registered c/d/e/f Dev episodes, terminal validity becomes positive on both ends, endpoint NMAE is no worse than frozen D2, FECF paired closed accuracy remains 1.0 under state swap, and object-disjoint prior plus zero-geometry ablations show geometric conditioning is necessary.
Conflicts: Pruned 3.1 failed with a per-instance RGB-D interface SDF and 4.1 failed with visibility-censored endpoint regression; this counters them using intervention-conditioned full-trajectory contact states and explicit feasible/contact/penetration transitions rather than observable-surface self-contact or global range priors. [DONE]

**Insight**: The preregistered query-conditioned CTF over 47D geometry summaries failed its object-disjoint internal gate: zero-geometry outperformed full geometry, so dataset range priors dominated and the summary representation harmed transfer despite exact swap equivariance.

**Result**: Formal 2000-step commit 35115a3: full transition accuracy 0.673344 and endpoint NMAE 0.165419 versus zero-geometry 0.712607 and 0.098718; swap residual 0; 33 tests passed; protected_splits_read empty. Stopped before c/d/e/f or sealed evaluation and did not tune.

**Branch**: arbor/node-5-1-counterfactual-terminal-field@35115a332fa22e78abb66a5f17bbb19cc3e60d07

#### 5.2: Mechanism: Equivariant Interface Token Field constructs query-specific cross-part tokens from transformed mobile and static point pairs, encoding relative displacement, normal agreement, opacity, and visibility in a permutation-invariant local operator that predicts free, supported-contact, and penetration states.
Hypothesis: CTF failed because a global 47D summary destroyed the local geometry that changes at a terminal event; retaining point-level pair interactions makes the feasible-to-contact transition identifiable and prevents a zero-geometry range prior from winning.
Observable: On the same frozen object-disjoint internal split, the point-token field beats both the zero-geometry and label-prior baselines in transition accuracy and endpoint NMAE with exact swap residual, after which a pre-registered c/d/e/f evaluation must improve terminal validity without worsening frozen D2 NMAE or FECF closed-side accuracy.
Conflicts: Pruned 3.1 used an unsupervised per-instance fused interface SDF and 5.1 used a supervised but globally compressed 47D trajectory summary; this node counters both with supervised counterfactual trajectories represented by local cross-part interaction tokens rather than a threshold or scalar feature retune. [PRUNED]

**Insight**: Point-level interaction tokens are implementable and swap-exact, but the current corpus cannot support a physically supervised terminal claim: only one NJC object has complete collision geometry and zero Articraft assets are collision-validated.
[Pruned: Physical-supervision data gate failed: 1 collision-complete object is below the frozen 10 train + 2 validation minimum, so any formal terminal-contact model would be unvalidated.]

**Result**: Preregistered commit 203c764, implementation 21e7748, final clean commit a1ced23, 36 tests. Frozen data gate required 10 train plus 2 validation collision-complete objects but found 1 plus 0 validated Articraft; formal GPU training and all sealed evaluation were NOT_RUN_BY_DATA_GATE.

**Branch**: arbor/node-5-2-equivariant-interface-token-field@a1ced23159558784cb4fa99ced31bd84f9c990cd

#### 5.3: Mechanism: Equivariant Null-Model Audit evaluates FECF against observation-order-aware canonical-side priors, a zero-geometry closed head, shuffled geometry, and label-frequency controls while counting only original objects as independent samples.
Hypothesis: The prior constant-local-lower control was not a valid group-equivariant null model; if FECF learned closure evidence rather than dataset ordering, it must outperform null predictors that transform labels correctly under state swap.
Observable: On frozen c/d/e/f and retrospective v4/a/b without retraining or threshold changes, report object-level balanced accuracy, Brier score, NLL, coverage-risk, and exact swap residual; revoke the FECF innovation claim if any zero-geometry equivariant control ties its decisions or calibration.
Conflicts: Node 5 passed only against a non-equivariant constant-side baseline and all original episodes used local-lower closure; this node directly tests that hidden shortcut instead of adding more deterministic swap copies. [DONE]

**Insight**: The current FECF evidence is invalid as a geometry-dependent closure claim: all seven original Box episodes share canonical outside_state_0, an order-aware zero-geometry prior ties 100% decisions with better Brier/NLL, and cyclically shuffled geometry preserves FECF metrics.

**Result**: Preregister 597bf4d, final e65829b, 40 tests. Fresh c/d/e/f FECF object BA 1.0 but canonical prior BA 1.0/Brier 0/NLL about 1e-12 and shuffled geometry exact; primary fail rule triggered. No B_test/Full22.

**Branch**: arbor/node-5-3-equivariant-null-audit@e65829b

### 6: Mechanism: Proxy-to-Verified Collision Curriculum deterministically converts articulated visual meshes into per-link watertight collision proxies, rejects geometrically invalid proxies with fixed QA, and generates object-disjoint quasi-static sweeps that supervise a query-conditioned terminal field.
Hypothesis: Terminal learning is blocked by absent collision supervision rather than model capacity; constructing and validating the missing feasible, supported-contact, and penetration trajectories should make point-level geometry useful and remove the zero-geometry prior advantage.
Observable: At least ten training and two validation objects pass frozen watertightness, self-collision, and sweep-continuity QA; on the held-out internal split a geometry-conditioned field beats zero-geometry and label-prior controls before any c/d/e/f evaluation, and only then may terminal validity be tested while D2/FECF outputs stay frozen.
Conflicts: Pruned 5.2 found only one collision-complete object and forbade training on authored limits as fake contact; this node directly creates the missing collision representation with explicit rejection and physics-sweep receipts instead of reusing incomplete sidecars. [PRUNED]

**Insight**: Deterministic visual-mesh collision proxies are reproducible and watertight, but naive proxy geometry does not recover a terminal transition: 11 of 12 objects are penetrated for all 65 queries and one is in contact for all queries; two joints also have zero authored span.
[Pruned: Frozen proxy QA found zero usable class transitions across 12 fixed objects; watertight proxies remained permanently penetrating/contacting, so they cannot supervise terminal physics without a new collision-part representation.]

**Result**: Clean commit ac532a2, manifest SHA 15676b8a, audit SHA f392c689. Twelve fixed objects produced 24 byte-identical proxies; continuity passed 10/12, but zero objects had a free/contact/penetration class transition, so accepted train/val was 0/10 and 0/2. Training handoff blocked; no sealed evaluation.

**Branch**: arbor/node-6-proxy-verified-collision-curriculum@ac532a2

### 7: Mechanism: Object-Level Balanced Endpoint-Free Suite builds pre-registered interior-state episodes on disjoint articulated objects with canonical closed labels balanced between local lower and upper, stratified by revolute versus prismatic motion and observation span.
Hypothesis: Same-object Box episodes cannot distinguish physical extrapolation from object or ordering priors; object-disjoint balanced sampling makes generalization and closed-side evidence statistically identifiable.
Observable: A public manifest contains at least twelve eligible objects with no identity leakage, both joint types where available, balanced canonical closed sides, fixed train/dev/test partitions, and runnable scratch, symmetric-prior, D2, PILC, and FECF evaluations whose confidence intervals use objects as the sampling unit.
Conflicts: Nodes 1 through 5 evaluated one Box object and node 6 showed existing visual assets lack reliable collision transitions; this suite targets kinematic authored endpoints and functional closure labels only, explicitly excluding terminal-contact claims. [DONE]

**Insight**: The available 144-object corpus is heavily authoring-biased: q=0 closure maps to lower for 119 objects and upper for only 2, so native-label 6/6 balancing is impossible without selection or convention manipulation.

**Result**: Clean commit 52ee81f. Inventory 83bc4aba, preregister 29dd386c, manifest 89b61a11. Frozen 12-slot builder validated 6 lower plus 2 upper and blocked training; no score, sealed, B_test, or Full22 read.

**Branch**: arbor/node-7-object-balanced-suite@52ee81fbbe37f735941f8c3bfddd20d40ccbf17d

#### 7.1: Mechanism: Sealed Per-Object Order Randomization assigns each object exactly one hash-determined presentation of its two physical interior states, hiding the order bit and balancing whether the functional closed endpoint lies beyond state 0 or state 1 without changing geometry, joint axis, or endpoint labels.
Hypothesis: The closure shortcut comes from a corpus-wide input-order convention rather than the physical task; randomizing presentation once per object removes this nuisance variable so a zero-geometry prior is forced to chance while geometry-aware closure evidence remains learnable.
Observable: A pre-evaluation manifest contains at least twelve unique objects with one episode each, outside-state-0 versus outside-state-1 balance within one object, no paired swap counted as a sample, and an order-only or zero-geometry baseline at approximately 50 percent before any learned method is evaluated.
Conflicts: Node 7 proved native authored lower-versus-upper labels are 119 to 2 and cannot be honestly balanced; this node counters by intervening only on model-facing observation order with a sealed hash bit, preserving the physical closed endpoint and avoiding axis flips, label rewrites, or object selection. [DONE]

**Insight**: The sealed order-randomized 12-object suite completed materialization and true scratch 25k training for all objects. Every checkpoint is fresh step24999, worker return codes are zero, config/checkpoint/dataparser/log hashes are frozen, and error/leakage scans pass. This suite remains target-free and cannot by itself support the preregistered 90% conformal plus independent confirmation split.

**Result**: 12/12 fresh scratch checkpoints complete. Checkpoint manifest SHA256 bb5c78aeafb0d60899ebdf0d1d6ff9f5428cf7522b004d6620465ef1e36c9141; completion receipt SHA256 a0e6f0747487292bf71854c652ceb1a299e1d7c1d10c409db21a180f97415f6e.

**Branch**: 47d18a4060875e3ec3ee58c05f652169674af3ad

#### 7.2: Mechanism: Prospective Hash-Split 36-Object Extension expands the frozen target-free suite by the original selection hash, then uses an independent sealed salt to assign 18 train, 9 calibration, and 9 confirmatory objects before targets or scores are read.
Hypothesis: A sufficiently large object-disjoint evaluator-owned protocol removes the post-hoc split and finite-sample conformal failures that block the 12-object suite, enabling an honest test of the gauge-equivariant profile head.
Observable: All 36 public profiles pass leakage/provenance gates, calibration rank ceil((9+1)*0.9)=9 is valid, and one frozen head can be calibrated and scored once on nine untouched confirmatory B_dev objects.
Conflicts: Pruned [5] showed closure labels and equivariance can be solved by a zero-geometry prior; this extension counters via label-unread object splitting and mandatory zero-geometry/permuted-profile confirmatory nulls rather than reusing the biased closure claim. [DONE] (score: 1)

**Insight**: All 36 target-free profiles and the frozen first36 roster passed independent Gate A. The sealed salt and 18/9/9 split were validated in isolation without exposing membership; the evaluator is now authorized while B_test and Full22 remain unread.

**Result**: AUTHORIZED_FOR_EVALUATOR receipt SHA256 ec7c93e6a9743b2958c6c6cf3f4d30406c2bed503b95af6abb470b26cf9f106b; 36/36 profile/provenance/denylist and sealed split checks PASS.

**Branch**: bfb25bb

### 8: Mechanism: Projective Boundary Inference treats each endpoint as a gauge-invariant phase boundary in a counterfactual trajectory rather than a directly regressed range scalar.
Hypothesis: D2's 13.8x lower/upper error imbalance and the prior-dominated learned heads arise because averaging and scalar regression discard whether geometric evidence is stable across states, radii, and observation gauges.
Observable: On a new public development protocol, beat frozen D2 object-macro worst-side NMAE while retaining exact state-swap and virtual-subinterval consistency and defeating zero-geometry/range-prior controls.
Conflicts: Pruned [2.2] produced no performance evidence and pruned [5]/[5.1] exposed range-prior shortcuts; this counters them by using within-object phase structure and intervention consistency instead of category-level scalar regression. [PENDING]

#### 8.1: Mechanism: Cross-State Radius-Consensus Boundary preserves each reconstructed state's three-radius trajectory fields and estimates endpoints by robust consensus over their free-to-penetrating change points.
Hypothesis: The current geometry-mean profile lets a biased state or radius move the soft minimum; state-wise change-point agreement should suppress reconstruction-specific artifacts and directly reduce the dominant upper-end error.
Observable: On public Box a-f leave-two-episode-out evaluation, lower worst-side NMAE than frozen D2, single-radius, mean-profile, and geometry-shuffled controls, with exact observation-swap residual below 1e-10.
Conflicts: Pruned [3.1] found observable surface contact ambiguous after averaging; this counters by retaining intervention-indexed state/radius evidence and requiring consensus rather than claiming contact semantics. [PENDING]

#### 8.2: Mechanism: Monotone Endpoint Hazard converts penetration, contact mass, support rise, and energy curvature along each outward scan into a constrained survival/change-point model whose first stable hazard transition is the endpoint.
Hypothesis: Softargmin selects low-energy points even when no physical boundary exists, whereas monotone hazard accumulates ordered evidence for the free-to-terminal transition and should be less sensitive to absolute range priors.
Observable: On public Box a-f leave-two-episode-out evaluation, improve worst-side NMAE over frozen D2 and an energy-only changepoint while retaining gains after scalar-coordinate removal and failing under channel shuffle.
Conflicts: Pruned [6] showed naive mesh collision is penetrated everywhere; this uses relative transitions in learned Gaussian trajectory channels rather than absolute proxy collision labels. [PENDING]

#### 8.3: Mechanism: Virtual-Subinterval Projective Consistency reparameterizes one counterfactual trajectory under many synthetic interior observation intervals and trains a shared boundary functional whose physical endpoint predictions must agree after inverse gauge mapping.
Hypothesis: Range-prior heads exploit the fixed 0/1 observation gauge; projective consistency removes that shortcut and forces the model to use profile shape that survives changes of the observed interior interval.
Observable: Pretraining on the 36 target-free profiles plus public Box a-d labels improves held-out Box e/f worst-side NMAE over frozen D2 and the same architecture without projective consistency, with low cross-gauge variance.
Conflicts: Pruned [2.2] used only state-swap equivariance and could still learn a fixed range prior; this adds a continuous family of observation-gauge interventions and a direct no-consistency ablation. [PENDING]
