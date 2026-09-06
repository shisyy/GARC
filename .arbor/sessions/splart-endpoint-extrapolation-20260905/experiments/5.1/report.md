# Arbor executor report: node 5.1 CTF

## Idea

Counterfactual Terminal Field (CTF) is a shared, query-conditioned classifier
for `feasible -> supported_contact -> penetration`. Inputs are the two observed
surface descriptors, relative screw displacement, query extension, and side
ordering. The shared side head gives exact endpoint swap equivariance. FECF
closed-side semantics remain frozen and were not trained or changed.

## Preregistration and data gate

The immutable protocol was committed before training at `60817ae`. It fixes 13
trajectory offsets, support half-width `0.0625`, a 257-point inference grid,
hidden width 96, AdamW LR `0.003`, 2000 steps, and every gate.

- Articraft: 109 QC-passing object-disjoint objects; 97 train and 12 validation.
  No closed-side labels are present.
- NJC: 15 objects and 480 episodes; 352 train and 128 calibration, all 26/26 QC.
  Inputs and label sidecars are separate.
- Existing receipts state `object_disjoint=true`, protected access `[]`, and
  bind the input hashes recorded in `ctf_preregister.json`.
- FECF c/d/e/f semantics stayed untouched. Ancestor evidence has swap residual
  0 and paired side accuracy 1.0, but terminal validity 0.

## Changes

- `src/splart/counterfactual_terminal_field.py`: shared three-class field,
  paired inference, outward-sweep endpoint extraction, and swap residual.
- `train_ctf.py`: deterministic training, held-out metrics, zero-geometry
  ablation, frozen gates, and checkpoint provenance.
- Tests cover query gradients, exact swap identity, and zero-geometry input.
- Full repository tests: **33 passed**.

Implementation commit before this report: `6308a0c`.

## Runs

Both runs used server98 GPU2 and the existing conda environment. No external
process was stopped or modified.

### Smoke (2 steps; plumbing only)

- PID `2171019` (completed)
- output `/home/yptang/arbor-runs/splart-endpoint-ctf/smoke-6308a0c`
- checkpoint SHA `4d61c5754ce01a87aa3ed8a7aca336ee37ed1e259435a2d462bf8388079bafdf`
- geometry NMAE `0.26746148`; zero-geometry NMAE `0.67360026`; swap `0.0`

The smoke established working data/model/checkpoint plumbing. Its absolute
gates were not interpreted after only two steps.

### Formal (frozen 2000 steps)

- PID `2171826` (completed)
- output `/home/yptang/arbor-runs/splart-endpoint-ctf/formal-6308a0c`
- checkpoint SHA `cb5ec2d9dca24b5fb83bdc6c8ede16b580d489bc9fbfe160897f816260447389`
- metrics SHA `cc989b417cb9c2c8522753fbe7e485846617940ee2dc08dce07b25fd61a3a579`

| Internal metric | Frozen gate | Geometry | Zero geometry | Pass |
|---|---:|---:|---:|---:|
| object-disjoint prior | required | true | n/a | yes |
| transition accuracy | >= 0.90 | 0.67334402 | 0.71260685 | no |
| endpoint NMAE | < 0.12 | 0.16541883 | 0.09871782 | no |
| geometry NMAE advantage | >= 0.02 | -0.06670102 | reference | no |
| swap residual | <= 1e-6 | 0.0 | 0.0 | yes |

The formal internal result is **FAIL**. Therefore no c/d/e/f Dev evaluation,
terminal-validity evaluation, candidate-bound evaluator, B_test, or Full22 was
launched.

## Analysis

Training cross entropy fell from `2.31676` to `0.15457`, but the model did not
learn a transferable geometry-to-limit relation. Removing both surface
descriptors improves held-out endpoint NMAE by `0.06670` and transition
accuracy by `0.03926`. Predictions are therefore dominated by the observed-
displacement/range prior; surface features add object-specific correlations
that fail to transfer. Exact swap behavior is architectural and does not imply
physical validity.

This is a method failure rather than plumbing failure: tests pass, loss falls,
artifacts are complete, and the negative control is stronger. No threshold,
seed, split, or hyperparameter changed after results were observed.

## Score

No eligible Dev score. Internal endpoint NMAE is `0.16541883` (minimize), but
the node is rejected by its preregistered internal gate.

## Insight

Query conditioning and swap symmetry are insufficient with global moment/contact
descriptors: CTF reduces to a range prior and geometry harms object-disjoint
transfer. A future node needs genuinely local swept-contact representation and
independently varied geometry/range pairs, not threshold tuning.

## Provenance and protections

- `protected_splits_read`: `[]`
- B_test / Full22 / c-d-e-f sealed evaluator accessed: no
- seed, checkpoint, or view selection: no
- FECF side modified: no
- branch: `arbor/node-5-1-counterfactual-terminal-field`
