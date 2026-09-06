# CVPR Evidence Closure Checklist

Status at freeze: 24/24 added-object scratch checkpoints and public profiles
complete under atomically claimed GPU queues; independent Gate A is pending. No
endpoint target, split membership, score, B_test, or Full22 data has been read.

## Claim boundary

The task is **kinematic endpoint extrapolation from exactly two strictly
interior observations**. The current 36-object suite is revolute-only. It does
not support claims about physical stoppers, terminal contact, true functional
closure, prismatic joints, Full22, or state of the art.

## Gate A: complete the public target-free package

- The roster must equal the frozen first-36 selection-hash prefix, with the
  original 12 objects as its exact prefix and no redraw, duplicate, or omission.
- Every object must have a from-random-initialization step-24999 checkpoint;
  loading, resuming, checkpoint selection, view selection, and seed selection
  are forbidden.
- Rehash checkpoint, config, dataparser, episode manifest, and profile payload.
- Every profile must be finite and exactly `[2,3,257,9]`, use the frozen channel
  order and valid two-sided scalar grids, and preserve the SplArt state hash
  before and after export.
- Every profile must report `split=unassigned` and contain no target, limit,
  fraction, presentation bit, canonical side, salt, rank, score, or evaluator
  path.
- Independent P0 must fail closed on any mismatch and publish only aggregate
  counts and hashes.

## Gate B: freeze the evaluator before private access

- Bind the method commit, evaluator tree, Conda/dependency lock, request schema,
  metric definitions, output schema, and protected-path denylist.
- Use the superseding node2.2 addendum: AdamW, learning rate `1e-3`, 4,000
  full-batch steps, seed 2202, train-only normalization, deterministic fp32,
  one final checkpoint, and no result/checkpoint/seed selection.
- Run full head and every learned baseline exactly once on the same train18.
- Infrastructure retry is allowed only before target read and optimizer
  initialization; after either event, failure is terminal.
- Launch remains unauthorized until the 36-profile receipt, sealed 18/9/9 split
  receipt, and independent source/request P0 all pass.

## Gate C: evaluator-only 18/9/9 execution

1. The producer submits only 36 unassigned profiles and the frozen method.
2. The evaluator verifies the sealed salt hash and 18/9/9 partition internally;
   object membership is never returned to the executor.
3. Train all learned methods once on train18 and freeze their hashes.
4. Use calibration9 only to compute the preregistered object-level joint
   conformal quantile; do not update models or choose methods.
5. Score confirmatory9 once, simultaneously for all methods.
6. Return only aggregate metrics, object-level win count, coverage/width, and
   provenance. State swaps and views are not independent samples.

For nine calibration objects the 90% finite-sample rank is 9. On nine
confirmatory objects, empirical coverage must be 9/9 to be at least 90%.

## Contribution 1: D2 energy profile/search

Required rows: Scratch SplArt, symmetric linear extension, train18-only global
and range priors, and full D2. Required ablations: single radius, no
penetration, no contact, and no terminal support.

Pass only if full D2 improves confirmatory object-macro endpoint NMAE over all
non-learning baselines, reports gains on both lower and upper sides, preserves
exact state swap and the frozen SplArt state, and its claimed components beat
their ablations. If no-contact is not worse, remove contact from the claimed
mechanism and describe D2 as a multi-radius penetration-aware profile.

## Contribution 2: gauge-equivariant dual-boundary profile head

Required rows: frozen D2 scalar output, train18-only priors, coordinate-only,
fixed reverse permutation, D2-scalar-only MLP, pooled-profile MLP, unshared
two-head ablation, and the full sequential shared head.

Pass only if the full head has lower confirmatory object-macro endpoint NMAE
than every required comparator, wins against D2 on at least 8/9 objects, has
swap residual below `1e-10`, and achieves 9/9 simultaneous interval coverage.
Interval width is descriptive only and must be shown beside coverage. If the
coordinate-only or permuted-profile null ties the full head, contribution 2 is
rejected as a learned range prior.

Both contributions survive only if D2 beats the non-learning baselines and the
full head then beats D2 plus the learned/null controls. If only the combined
method succeeds, present it as one method contribution rather than two.

## Frozen confirmatory table

| Method | #Obj | Endpoint NMAE down | Lower down | Upper down | Worst-side down | Median down | Wins vs D2 | Swap max down | Joint coverage up | Mean width down | Runtime |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|

Endpoint-render PSNR, SSIM, LPIPS, depth, and mIoU belong in a separate table
only if the sealed evaluator supplies genuine endpoint renders. Report
object-level bootstrap intervals; never inflate sample size with views or
deterministic state swaps.
