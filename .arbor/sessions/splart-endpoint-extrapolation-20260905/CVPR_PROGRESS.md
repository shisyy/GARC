# SplArt Endpoint Extrapolation: CVPR Evidence Ledger

## Current verdict

The defensible paper target is **kinematic endpoint extrapolation from two
strictly interior states**, not physical stopper/contact prediction.

One method contribution is validated on the current Box development object:
multi-radius, penetration-aware D2 endpoint search. The former FECF
closed-side head is withdrawn as a second innovation because a zero-geometry,
observation-order-aware prior matches its decisions and improves its
calibration. A replacement second contribution, a gauge-equivariant
dual-boundary energy-profile head with object-level conformal intervals, is
preregistered as Arbor node 2.2 but has not yet passed confirmatory evaluation.

B_test and Full22 remain protected and unread.

## Development result: canonical Box v4

| Metric | Scratch SplArt | D2 full | D2 no-contact | Symmetric linear | Global range prior |
|---|---:|---:|---:|---:|---:|
| Endpoint NMAE down | 0.310881 | 0.085727 | **0.084910** | 0.121763 | 0.127391 |
| Lower NMAE down | 0.314698 | 0.011616 | **0.011621** | 0.125579 | 0.131207 |
| Upper NMAE down | 0.307065 | 0.159839 | **0.158198** | 0.117947 | 0.123575 |
| PSNR up | 23.1807 | 26.8976 | **26.9066** | 25.3263 | 25.2318 |
| SSIM up | 0.808770 | **0.829077** | 0.829070 | 0.821725 | 0.821425 |
| LPIPS down | 0.219408 | 0.165690 | **0.165440** | 0.183776 | 0.185119 |
| Depth MAE down | 0.429544 | 0.174547 | **0.173214** | 0.201093 | 0.208748 |
| mIoU up | 0.705265 | 0.819786 | **0.820359** | 0.790505 | 0.786964 |
| Closed coverage | 0 | 0 | 0 | 0 | 0 |
| Terminal validity | 0 | 0 | 0 | 0 | 0 |

Against the two meaningful non-learning baselines, D2 full lowers NMAE by
29.6% relative to symmetric linear and 32.7% relative to the independent
global range prior. Although no-contact was slightly better on Box v4, the
preregistered Box a-f stability test rejected deleting contact: episode-macro
NMAE was `0.154877` for full versus `0.155311` for no-contact, with no-contact
winning only 2/6 episodes and slightly degrading all six reconstruction/
penetration macro metrics. Full D2 therefore remains the frozen candidate.

Independent P0 receipt for D2 full:
`f65f28f9d80f5057d7ce0dc374982dc931418d151f2652cff7884dd0fe6e0a30`.

## Component attribution

| Variant | Endpoint NMAE down | PSNR up | Depth MAE down | mIoU up |
|---|---:|---:|---:|---:|
| Full D2 | 0.085727 | 26.8976 | 0.174547 | 0.819786 |
| Single radius | 0.122296 | 25.5546 | 0.216294 | 0.787295 |
| No contact | **0.084910** | **26.9066** | **0.173214** | **0.820359** |
| No penetration | 0.127866 | 25.3819 | 0.221466 | 0.783604 |
| No terminal support | 0.085638 | 26.8979 | 0.174395 | 0.819850 |

The causal mechanism supported by this ablation is multi-scale geometric
profiling plus penetration avoidance. The explicit contact term is slightly
detrimental on canonical v4 but weakly beneficial across the preregistered a-f
stability set; terminal-support is numerically neutral on v4. Paper naming and
claims must reflect this mixed but correctly attributed evidence.

Box a-f full/no-contact summary artifact SHA256:
`042aa620df04a8648cf090bbc7eaf822a9b438159127f3730d19def6da6d42a2`.
The first independent P0 audit reproduced every metric and provenance check
but failed one operational assertion: the new evaluation root was mode `0755`
while its report claimed `0700`. A permission-only repair changed only the
root directory mode, preserving its inode, mtime, complete content-tree hash,
and path-plus-mtime manifest. The superseding independent P0 re-ran the full
numerical, provenance, checkpoint, render-count, leakage, and filesystem audit
and passed. Immutable receipt SHA256:
`559138980c579fbfa6df4dbd17b16773dae520f98974f3b0c23305a5abb130ea`.

## FECF null audit: claim withdrawn

All seven original Box episodes use the same canonical closed side. On fresh
c/d/e/f objects-as-episodes, FECF has balanced accuracy 1.0, but a
zero-geometry swap-equivariant canonical-side prior also has balanced accuracy
1.0 and achieves better Brier/NLL. Cyclically shuffling the geometry preserves
FECF predictions and calibration.

Therefore FECF currently demonstrates coordinate equivariance only, not
geometry-dependent closure understanding, and is not counted as an innovation.
Audit artifact SHA256:
`43f126baaed78f136f278f3d6e6bf34e4552517f3c461c5efe807c94c327420b`.

## Object-level protocol gate

The native 144-object inventory is heavily authoring-biased: q=0 maps to the
lower limit for 119 objects, upper for 2, interior for 3, and is unbounded for
20. Honest native 6/6 lower/upper balancing is impossible without selection.

Node 7.1 therefore freezes 12 unique objects by a label-independent hash, then
uses an independent secret-salt ranking and rank parity to assign presentation
order. Every object appears once; swaps are transformations, not extra samples.
The sealed targets are exactly 6 outside_state0 and 6 outside_state1, while
constant-index/order-only controls are exactly 50%. Public paths and metadata
leak none of the order bit, rank, near/far side, canonical side, limits,
fractions, or outside target. The selected set happens to contain 12 revolute
objects; it was not redrawn to manufacture joint-type coverage.

- Protocol commit: `f781a40193c1ee5ff6b7a421a33345e7d8da4208`
- Public manifest SHA256: `b10b61f7022f81a0b6e62c7fc03a7b37f376b40dd6dc67b12baec1d664e41274`
- Audit SHA256: `1163b08863ed56ee659a3e390be394c4932a802aeb5b1d904d8b0ea7679720c6`
- Sealed mapping SHA256: `8ddedecb2c86b19bb443976728dc59a4f0397f33d0e3d9db18c0c7d559424c6a`
- Independent P0 receipt SHA256: `a54d197246a4dc6af1d9ebbab8d6c353c427f0c2d05dd8486333b1e19a4d5ebc`

This balances presentation order only. It does not create physical terminal
labels or repair canonical lower/upper authoring bias.

## Replacement second contribution under test

Arbor node 2.2 retains the full bidirectional, multi-radius D2 energy profile
instead of collapsing it immediately to a scalar optimum. A shared-weight
gauge-equivariant head predicts both nonnegative boundary distances, while an
object-disjoint split-conformal stage emits feasible intervals.

Promotion requires all of the following on sealed B_dev objects:

- lower object-macro NMAE than frozen D2, symmetric linear, and global/range priors;
- exact observation-order swap residual below `1e-10`;
- preregistered simultaneous two-endpoint object coverage (the nine-object
  confirmatory sample requires `9/9` empirical coverage to meet 90%);
- wins over coordinate-only, fixed-permutation, D2-scalar-only, pooled-profile,
  and unshared-head controls.

A target-unread protocol audit found that the original head preregistration did
not completely freeze optimization, duplicated its zero-geometry/order-only
nulls, and compared intervals with mismatched coverage semantics. Before any
split membership or target access, a superseding addendum froze AdamW training,
one seed and one final checkpoint, train-only normalization, all strong nulls,
and a profile-conditioned scale with object-level joint conformal scores. Width
is now descriptive rather than a promotion gate. The platform-independent P0
binds canonical JSON hashes and passes 4/4 tests at trunk commits `94fa7ce` and
`9268db9`; evaluator launch deliberately remains unauthorized.
The matching executable runtime is frozen at `a857374`: it implements the
shared distance-plus-scale head, all preregistered comparators, deterministic
train18-only fitting, joint conformal calibration, aggregate-only output,
recursive private-field denial, and terminal rerun guards. Its synthetic and
public-interface suite passes 16/16 tests without opening any private data.

The head and real-data adapter are implemented at commits `bbc03b1` and
`c071a05`; 52/52 tests pass. The adapter independently verified all 12 public
objects, 864 frames, and 2,604 files, checks exact clean D2 provenance and
base-state hashes, and fails closed on target/sealed/B_test fields. Adapter
preflight SHA256:
`e9dadd3818ce51870ad2ffb8982f19a5ffe248f260cb6c265f19bf8780d63b69`.
Target-free profile export has passed for all twelve finished 25k checkpoints:
12/12 artifacts have shape `[2,3,257,9]`, independently
recomputed hashes, unchanged base-model state before/after export, and frozen
step-24999 checkpoint/config/dataparser provenance. Batch index SHA256 values:
`c5f5e0284e8e6385e974bfd8f42989c01590697330522dae31eae910ffb2ab28`
for the first six,
`d991c24f4e84231a877449586a51da192b5d0e70c234cbf53b317b73efa137ee`
and `78250ad6a75922004407845ce149bdb9d870bf3b5e288d4a445c6395fa7d046a`
for the next two-object batches, and
`706645124e165ea440ae6e2a2744b6e531bc41c751fff967a658eabbfdfe55b0`
for the final two. The immutable, reference-only 12-object merged index SHA256
is `328e9dffd2e8d9ad5c7372160bf3ca3ab7f2f076d72520a2fc5cc00d1ced3985`.
These profiles remain explicitly `unassigned`; no head was trained and no
targets, sealed mapping, B_test, or Full22 were read.

An independent protocol review blocks head training on these 12 objects. The
original preregistration did not freeze an object-level train/calibration/
confirmatory assignment, and exact 90% split conformal requires at least nine
calibration objects, leaving too few objects for honest head training and
confirmation. No target was exposed. The prospective repair is a label- and
score-unread dataset extension with a deterministic object split frozen before
materialization or target service; the existing 12-object profiles remain
target-free development artifacts only until that gate is satisfied.
The immutable independent BLOCKED receipt SHA256 is
`d2c6a9dbe686becb35b2d3813dd166193eed79bd73d058b2000e20301b18d8e5`.

Prospective node 7.2 is now frozen before any target or score access. It uses
the original node-7.1 selection hash to extend the public suite to the first 36
objects, then an independent sealed salt to assign 18 head-training, 9
calibration, and 9 confirmatory objects without redraw or label-based
stratification. The evaluator alone owns targets, training, calibration, and
the one-shot confirmatory score. Its finite-sample calibration rank is
`ceil((9+1)*0.9)=9`. Launch remains unauthorized until all 36 target-free
profiles and the sealed split manifest pass independent P0.

- Preregister commit: `b7a2a3658816fc2f0a93ab046e8091a8542da448`
- Immutable preregister SHA256:
  `8d00e654a990e2d96b912164bf4c5aaaefc7159304aeceaf3544e0f4ea6d85cc`

The first-36 public roster is now frozen and passes the prefix gate: the
original 12 objects are exactly the first-12 prefix, while the extension adds
24 objects without target-dependent redraw. Its target-free manifest SHA256 is
`79c0f575135fed75d92d6837ec78e7aeb7ebd56b19f827d2880639802f4eb61f`.
After preserving three failed infrastructure preflights, the fourth preflight
passes 24/24. Four NJC assets require visual-only renderer URDFs because their
published URDFs reference absent collision-proxy directories; this is allowed
only for RGB-D/segmentation generation and does not restore a terminal-contact
claim. Preflight SHA256:
`a05dcc634cdef8051702d3aec81da9b513c5b27330e87cc7abb433ba0b329a72`.
A 128x128 smoke materialized 24/24 objects. Full 512x512 materialization then
completed all 24 fixed objects with 72 frames each (32 training plus 4
validation views per state), 217 files per object, and 24/24 image/depth/
segmentation/hash checks passing. Materialization receipt SHA256:
`97b9df75919c85ce214e22c781624e3a77df864a89d451ee459a9b04f9008c2d`;
independent audit SHA256:
`c44f7756298ca01808f93d677ba2214b087033b8c7a96c7153b7a17f30cb3949`.
Fresh 10-step smoke training passed independently on GPU2 and GPU3, each
writing a step-9 checkpoint. After 16/24 fresh step-24999 checkpoints completed,
the remaining eight objects were redistributed without redraw across GPUs
2/3/5/6 using explicit disjoint lists and `O_CREAT|O_EXCL` claims. The queue
coverage audit proves `16 complete + 8 delta = 24`, with no duplicate or
omitted object (SHA256
`30cef6583af23fa38f7d518c0905f7526d25c1e6cbd043b003d75591864fb1e5`).
All active logs confirm fresh random initialization and no checkpoint load.
GPU4 is not used because it hosts an unrelated VLLM process, and GPU7 is
reserved for profile export. No split, target, sealed mapping, score, B_test,
or Full22 file has been read.

GPU7 subsequently became genuinely idle and is used for incremental
target-free profile export without interrupting training. All first-sixteen
profiles pass shape
`[2,3,257,9]`, step-24999, artifact-rehash, exact clean D2 provenance, and
base-state-before/after equality checks. Incremental index SHA256 values are
`2b794007ccd31e0e7cd6c064275497782801dbcfbd6ca5ee9858037dfbd8ab55`
for the first six and
`8800bfcb68add185f421eae5af320628e93ac473f359f06ab48865a57839b34d`
`90d671d5c825a7f3e5430fd7ed02ccb72eb3c25bdec8eb7328ffd63ca38ac627`,
`b5604d5ac202921bf17e55af495f568ca5adcdab841dde30bea7a916dd6c85db`,
`82faad8386eff9e14256608cd0649eff21285bb88995477b7d2e9741fca49925`,
and `05e7b9a81c968c32e25b8d19033ebea80ef78ce47167fa3403fb4365e03202be`
for the subsequent two-object batches.
The immutable reference-only cumulative 16-object index SHA256 is
`3ef628cef1d118e0646a3c8e5e0797d11e44ef85aee7e5bd98cbee7eec83eaae`.
No head training or protected-data access occurred.

## Remaining blockers before a complete CVPR evidence loop

The immutable decision rules, evaluator sequence, required baselines, and
allowed claim boundary are consolidated in `CVPR_FREEZE_CHECKLIST.md` before
private evaluation.

1. Finish multi-object RGB-D/multi-view materialization and true scratch SplArt
   training. The 12/12 asset preflight now passes after a SAPIEN 3 camera API
   compatibility fix and visual-scale-aware camera placement: every archive
   hash matches, each URDF/mesh graph resolves, both strict interior probes
   have nonempty depth, and no target is exposed to the model. Full 512x512
   materialization (32 train plus 4 validation views per state) finished for
   12/12 objects with 0 blocked and 2,604 files. Preflight receipt SHA256:
   `69182f56c88d45d0ac257217cfb39fb81cf10364786024f03605bd8d58c8e31b`;
   materialization receipt SHA256:
   `980c11183929aca7a96b6019e0aa1ed379baa086e18857e64711eb8fa37591d0`.
   The first two scratch workers failed before training because the deployed
   runner still contained an unsupported Nerfstudio downscale CLI option.
   Their failure receipts/logs are retained; a CLI-only addendum removed that
   option. A second 10-iteration smoke exposed missing public articulation type;
   that failure is also retained. Fresh v3 materialization then passed 12/12
   (SHA prefix `97731bdf`) and its public integrity preflight passed 12/12
   (SHA prefix `444255e6`). Both v3 smoke jobs completed 10 iterations and
   wrote fresh checkpoints. The first 25k v3 attempts genuinely trained from
   random initialization to approximately step 2,000, then both failed at the
   first checkpoint write because `/data` had only about 4.2 GiB free; one log
   records `ENOSPC` explicitly. No later object was started and no result is
   claimed from these failed runs. Their logs and receipts are retained. The
   disk-only v4 recovery redirects new checkpoints to the 4.2 TiB-free
   `/data1` filesystem, requires at least 32 GiB free before launch, and still
   forbids checkpoint loading/resume. The v3 receipts are now explicitly
   marked `failed_enospc`, with stale originals preserved under the v4 evidence
   directory. Both disk-gated v4 smoke runs completed 10 iterations and wrote
   729,905,478-byte checkpoints while logging `No Nerfstudio checkpoint to
   load`. Both full 25k v4 workers completed successfully: all 12 unique
   objects have fresh step-24999 checkpoints, each worker completed six with
   return code zero, and all config/checkpoint/dataparser/log hashes are
   frozen. The full error scan is clean and no checkpoint load/resume occurred.
   Checkpoint-manifest SHA256:
   `bb5c78aeafb0d60899ebdf0d1d6ff9f5428cf7522b004d6620465ef1e36c9141`;
   completion-receipt SHA256:
   `a0e6f0747487292bf71854c652ceb1a299e1d7c1d10c409db21a180f97415f6e`.
2. Freeze and execute a label-unread multi-object extension with enough
   object-disjoint train, calibration, and confirmatory objects for exact 90%
   split conformal; do not retrofit a split to the completed 12 profiles.
3. Train and evaluate scratch SplArt, frozen D2, and node 2.2 on object-disjoint
   B_dev; use objects, not swaps or views, as the statistical unit.
4. Freeze the selected method and evaluation code before opening B_test.
5. Run one blind B_test/Full22 milestone and report object-level confidence
   intervals, failures, runtime, and all null controls.

Until those gates pass, the project has a credible task definition and one
validated development method, but not yet a completed two-innovation CVPR
submission.
