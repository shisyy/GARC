# Arbor executor report: node 4 PILC

## Idea

Paired-Interior Limit Calibrator (PILC) applies one shared state encoder to the
two interior observations.  Its two positive endpoint extensions and two
closed-side logits swap by construction when the input state order is swapped.
Runtime input is restricted to canonicalized static/mobile surface features
and the observed relative screw displacement.

## Changes

- Added a 47-dimensional, scale-normalized static/mobile surface descriptor and
  an exactly state-swap-equivariant shared endpoint network.
- Added recursive inference-metadata rejection for URDF, limit, fraction,
  endpoint, closed, sealed, object/category identity, and ground-truth keys.
- Replaced initially reversible row identifiers with opaque domain-separated
  SHA-256 joins and kept inputs/labels in separate sidecars.
- Pre-registered NJC 11/4 object-disjoint training/calibration and a fixed
  Articraft 128-object hash-selected 112/16 endpoint pretrain/validation split.
- Added statistical-prior and zero-geometry ablations, swap checks, QC receipts,
  guarded GPU3 launchers, and 19 passing CPU/GPU tests.

## Implementation choices

The first formal model used 15 CC-BY-4.0 NJC assets.  All 15 published QC files
passed 26/26 checks.  Because this model overfit, the one authorized scale-up
kept architecture (47D input, hidden dimension 96), confidence threshold 0.80,
optimizer settings, and episode grid fixed.  It used Articraft only for
endpoint-extension supervision.  Of 128 pre-registered Articraft objects, 109
contained at least one usable finite-limit revolute parent/child visual pair
(97 pretrain, 12 validation; 169 local joint pairs).  The 19 rejected objects
had no usable finite-limit revolute pair.  No Articraft closed label exists.

Staged training first optimized the encoder and endpoint head on Articraft,
then froze both and optimized only the linear closed-side head on NJC.  Box v4,
Box extra-a/b, B_test, and Full22 were not read or used for selection.

## Baseline vs result

| Evaluation | PILC endpoint NMAE | Statistical prior | Zero geometry | Closed coverage/risk | Swap max error |
|---|---:|---:|---:|---:|---:|
| NJC formal-v1 calibration | 0.227354 | 0.129210 | 0.138582 | 1.00 / 0.00 | 1.19e-7 |
| Articraft staged validation | 0.114299 | 0.240316 | 0.682025 | n/a | 1.19e-7 |
| NJC staged calibration | 0.106854 | 0.098711 | 0.191094 | 1.00 / 0.00 | 1.19e-7 |

The frozen model then ran exactly once on the pre-declared public-only Box
triplet and all three predictions were sent together to the clean evaluator:

| Episode | Endpoint NMAE | Lower / upper | Closed coverage / accuracy | PSNR | mIoU | Terminal valid |
|---|---:|---:|---:|---:|---:|---:|
| Box v4 | 0.095154 | 0.022370 / 0.167938 | 1.00 / 1.00 | 26.356 | 0.81769 | 0 |
| Box extra-a | 0.062686 | 0.028848 / 0.096524 | 1.00 / 1.00 | 27.227 | 0.85093 | 0 |
| Box extra-b | 0.094066 | 0.039786 / 0.148346 | 1.00 / 1.00 | 26.479 | 0.82997 | 0 |
| Fixed macro | **0.083969** | — | 1.00 / 1.00 | 26.687 | 0.83286 | 0 |

The data-scale mechanism cut the NJC result from 0.227354 to 0.106854 and beat
the Articraft statistical prior by 52.4%, while the zero-geometry ablation
confirmed the surface branch was active.  Nevertheless, its NJC calibration
endpoint error is 8.25% worse than the statistical prior and above the
pre-registered 0.070 advancement gate.

## Score

The absolute original Box v4 B_dev score is `0.0951539351`; the fixed
three-episode macro is `0.0839685613`.  Box v4 is worse than node 2's
`0.08572745`, two of three episodes miss the `0.070` endpoint gate, and terminal
validity is zero on all three.  PILC is therefore not merge eligible despite
perfect selective closed-side accuracy.

## Analysis

The exact swap constraint and selective closed classifier work, and the
Articraft validation ablation shows that surface geometry contains useful
endpoint information.  The remaining failure is cross-source calibration:
full authored URDF geometry and partial SplArt/NJC-style surfaces do not share a
stable enough endpoint-distance representation.  More fitting on the 15 NJC
objects would violate the authorized staged policy and risks category/range
memorization, so the node stops rather than tuning on evaluation episodes.

## Insights

Large endpoint-only pretraining is materially better than a one-dataset MLP,
but exact equivariance and closed-side accuracy are insufficient evidence for
physical-limit prediction.  A future node would need a domain-aligned partial
surface pretraining objective or explicit visibility simulation, not more
threshold/seed fitting.

## Provenance

- Branch: `arbor/node-4-pilc`
- Frozen staged training commit: `d31e4821dedcca5f6864d56004ac2f615db0536a`.
- Public inference/provenance commit:
  `ac029b0f45fb2953d4ec930f3c1afeacb093c4b1`.
- Independent-evaluation orchestration commit:
  `1fc996f189e385176a7bb9c58cf4d9d46d973fd7`.
- Articraft pre-registration SHA-256:
  `070c7d3287d418280b24decf32f10c5827ec5c2953df069755c27d667fbb0b1f`.
- Staged checkpoint SHA-256:
  `bbd109b66ed854400c47571945a625938a80dede80b827053236231b8108c38a`.
- Server run:
  `/home/yptang/arbor-runs/splart-endpoint-pilc/staged-formal-v1`.
- Data receipt SHA-256:
  `3da37a99cd35f11f9996e98893405c9464743a59dbcbed2680e4423252cfcea8`.
- Local staged metrics SHA-256:
  `230cc5011095a7ba90f67eb1104757a625581a32e9afb67cd19de64be256bbac`.
- Independent triplet receipt SHA-256:
  `e212ce2a0ccb66bdddb9c62bab0e5ff8b89e939e84ff0ad346a24ae13d3c5974`.
- Evaluator commit: `cd8d3da2fb72e21992af50ee2e73fdba26f2ae1c`.
