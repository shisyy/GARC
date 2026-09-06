# Executor Report — Node 3.1 JTC-SR

## Idea

Joint Topology-Contact Surface Refinement (JTC-SR) replaces the preceding
unsigned, frozen-Gaussian collision proxy with an explicit signed interface
field.  The field is learned only from public middle-state RGB-D first
surfaces, oriented camera rays, and frozen SplArt static/mobile alpha.  It
alternates SDF fitting with a tightly bounded projection of the two frozen D2
endpoint scalars.  The SplArt renderer is excluded from both optimizers.

## Changes

- Added `src/splart/joint_surface_refinement.py`:
  - oriented free/surface/occupied SDF supervision;
  - continuous learned RBF interface SDF;
  - revolute/prismatic canonical kinematics;
  - alternating SDF and endpoint projection;
  - counterfactual blocking, distributed contact, penetration, and robust
    topology evidence;
  - selective closed-end prediction;
  - state-balanced metric-voxel fusion.
- Added `run_joint_surface_refinement.py`, a model-facing runner with no sealed
  evaluator arguments.  It validates fixed D2/checkpoint/source provenance,
  uses every public view in formal mode, enforces an 8192 MiB allocator cap,
  hashes the frozen model before/after, and verifies exact public-view RGB and
  depth distillation.
- Added eight CPU tests, including finite SDF gradients, sign orientation,
  exact joint motion, D2 trust-region preservation, inference-tensor
  materialization, frozen-renderer isolation, input-view permutation
  invariance, and repeated-view invariance.

## Implementation Choices

- No endpoint image, physical interior fraction, URDF limit, closed label,
  B_test, Full22, measurement, or sealed evaluator record is accepted by the
  model-facing process.
- No seed, checkpoint, or view selection is performed.  Formal mode consumes
  all 200 public training cameras.
- The D2 scalar trust region is fixed at 0.015 observed-span units.  A scalar
  update is published only if the public physical objective improves by the
  predeclared amount without increasing penetration.
- The existing certificate thresholds were never changed.  After the first
  full-view run exposed order/density bias, the one permitted mechanism repair
  fused repeated observations in metric voxels and allocated equal capacity
  to state 0 and state 1.  Thresholds and loss weights remained frozen.

## Validation

- Local adjacent regression: 14/14 passed before the balancing repair.
- Final local core/invariance suite: 8/8 passed.
- Final server98 Conda core/invariance suite: 8/8 passed.
- GPU formal run: clean exit on GPU2 under tmux, observed peak 794 MiB, cap
  8192 MiB.
- Formal public input: all 200 views; 4096 balanced static and 4096 balanced
  mobile interface samples; 2048 samples from each observed state and group.
- Frozen reconstruction audit:
  - maximum public RGB delta: `0.0`;
  - maximum public depth delta: `0.0`;
  - model SHA256 before/after:
    `12b928d3dfc86ad86936df5a1fa489016c26fc121fa4504c152db0e590cc37ae`.

## Baseline vs Result

The final JTC-SR safety gate failed, so no sealed candidate-bound evaluator was
launched.  Endpoint scalars and the renderer are byte-for-byte the frozen best
D2 candidate; consequently its endpoint/reconstruction numbers below are
inherited exactly, not presented as a fresh evaluation.

| Box B_dev metric | SplArt-middle | Frozen D2 | JTC-SR final |
|---|---:|---:|---:|
| Endpoint NMAE down | 0.310881 | 0.085727 | 0.085727 (exact scalar identity) |
| Lower scalar | 0.0 | -0.80129993 | -0.80129993 |
| Upper scalar | 1.0 | 1.38924360 | 1.38924360 |
| Closed coverage up | 0 | 0 | 0 (`unknown`) |
| Terminal valid up | 0 | 0 | 0 (not claimed) |
| Penetration down | 0.065374 | 0.06473 | 0.06473 (frozen render identity) |
| PSNR up | 23.1807 | 26.8976 | 26.8976 (frozen render identity) |
| SSIM up | 0.80877 | 0.82908 | 0.82908 (frozen render identity) |
| LPIPS down | 0.21941 | 0.16569 | 0.16569 (frozen render identity) |
| Depth MAE down | 0.42954 | 0.17455 | 0.17455 (frozen render identity) |
| Mobile IoU up | 0.52876 | 0.77591 | 0.77591 (frozen render identity) |
| mIoU up | 0.70526 | 0.81979 | 0.81979 (frozen render identity) |
| Rotation error down | 74.57 deg | 20.49 deg | 20.49 deg (frozen articulation) |

Final public-only evidence after balanced voxel fusion:

- lower/upper distributed contact: `0.391881 / 0.322193`;
- lower/upper counterfactual blocking: `-0.000791 / +0.000076`;
- contact-plus-blocking margin (lower minus upper): `+0.068821`, voting
  **lower**;
- robust topology log-volume margin (upper minus lower): `-0.142610`, voting
  **upper**;
- scalar proposals: `-0.80130154 / 1.38924205`; both rejected by the fixed
  improvement/penetration gate;
- closed label: `unknown` because contact and topology directions conflict.

## Score

Absolute endpoint NMAE: **0.085727** by exact identity with the frozen D2
scalars.  No new sealed score was produced because the method's own public
gate failed before evaluation.

## Analysis

The explicit SDF fixes the earlier surface locality problem: it produces
substantially separated endpoint contact support while keeping all render and
articulation parameters frozen.  However, the signed contact direction and
global compactness direction disagree after removing view-density bias.  This
is a substantive ambiguity rather than missing evidence: both margins are
large and point to opposite endpoints.  Publishing either label would amount
to choosing which proxy to trust after observing the scene, violating the
selective-prediction contract.

The endpoint optimizer also correctly declined tiny proposals that did not
meet the predeclared physical-improvement gate.  Therefore JTC-SR preserves
D2's endpoint/reconstruction performance but does not improve closed-end
coverage or terminal validity on Box.

## Insights

Single-surface contact plus global compactness is not sufficient to identify
semantic closure from two interior states.  The useful next direction is not
another threshold or post-hoc certificate: learn an object-category-neutral
*enclosed free-space / aperture-flow* field whose sign is constrained across
the observed motion trajectory, then test its direction on multiple Dev3
scenes before any sealed evaluation.

## Result

JTC-SR is a valid negative result.  It preserves the best D2 NMAE and every
reconstruction metric but fails its frozen public consistency gate, leaving
closed coverage and terminal validity at zero.  Node 3.1 should stop and must
not merge as a performance improvement.

## Artifacts

- Branch: `arbor/node-3-1-jtcsr`
- Code commit: `bfed3b4e8ba1ec0acb8fffd636e4592872b4d524`
- Final remote prediction:
  `/home/yptang/arbor-runs/splart-endpoint-jtcsr-box/formal-v2-bfed3b4/prediction.json`
- Final remote SDF:
  `/home/yptang/arbor-runs/splart-endpoint-jtcsr-box/formal-v2-bfed3b4/prediction-interface-sdf.pt`
- Final remote log:
  `/home/yptang/arbor-runs/splart-endpoint-jtcsr-box/formal-v2-bfed3b4/run.log`
- Exit status:
  `/home/yptang/arbor-runs/splart-endpoint-jtcsr-box/formal-v2-bfed3b4/exit.status`

```json
{
  "score": 0.085727,
  "insight": "A learned signed interface field separates endpoint contact but still conflicts with global topology; two interior states need an enclosed-free-space or aperture-flow cue, not threshold relaxation.",
  "result": "Preserved frozen D2 endpoint/reconstruction performance, but the fixed public gate abstained, so closed coverage and terminal validity remained zero and no sealed evaluator was launched.",
  "code_ref": "arbor/node-3-1-jtcsr@bfed3b4e8ba1ec0acb8fffd636e4592872b4d524"
}
```
