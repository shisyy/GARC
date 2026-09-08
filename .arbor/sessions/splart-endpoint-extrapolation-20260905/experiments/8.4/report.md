# Node 8.4 — Source-Supervised Projective Boundary Transfer

## Outcome

**Pruned at the preregistered source feasibility gate.** The proposed shared
head was restricted to a `±0.10q` residual around the frozen D2 posterior
anchor. On the object-disjoint Articraft smoke validation split, the anchor was
`1.088138q` away from the authored finite limits in mean worst-side error and
no validation object had both endpoints reachable by the allowed residual.
Training therefore could not possibly satisfy source supervision, independent
of optimizer, seed, or model capacity.

No Box a-d profile, endpoint label, e/f confirmation record, old sealed 18/9/9,
B_test, or Full22 was read. The exact36 target adaptation stage was not entered.

## Changes

- Added a dense source mesh trajectory exporter with deterministic vertex
  subsampling, finite-limit coordinate construction, three fixed radius
  gauges, and D2-compatible nine-channel energy fields.
- Added domain-local ECDF normalization and a shared, swap-equivariant bounded
  residual head implementation for the preregistered mechanism.
- Added an explicit anchor-feasibility audit that verifies artifact hashes and
  reports truth, D2 anchor, signed required residual, posterior entropy,
  signed gap, and energy for every source object.
- Added tests for posterior recomputation, finite ECDF output, exact side-swap
  equivariance, and the architectural residual bound.

## Source smoke and quality control

The plumbing smoke requested 16 preregistered source-train and 8
source-validation Articraft archives. It exported 19 finite profiles (13 train,
6 validation), each shaped `[2,2,3,513,9]`; five USB assets were rejected
because they exposed no finite revolute joint. All saved tensors were finite.
This is deliberately not presented as the full 109-object source benchmark.
The failed reachability gate is mathematical—`0/13` train and `0/6`
validation objects are representable under `tau=0.10`—so additional source
objects or training steps cannot rescue this preregistered model family.

## Absolute metrics

| Split | Objects | D2 mean-side absolute error (q) | D2 mean worst-side absolute error (q) | Max object worst-side (q) | Mean posterior entropy | Reachable with ±0.10q |
|---|---:|---:|---:|---:|---:|---:|
| Source train | 13 | 0.847889 | 1.047778 | 1.700000 | 0.162873 | 0/13 |
| Source validation | 6 | 0.813660 | **1.088138** | 1.749910 | 0.151998 | **0/6** |

The absolute primary feasibility score is therefore **1.088138q source
validation worst-side error**. A public Box score was not produced because the
source gate failed before Box-a, as required by the fail-fast protocol.

## Mechanistic diagnosis

The failure is not merely a large numerical error. The posterior entropy is
low while the endpoint error is large, meaning the D2 field is often
confidently concentrated at the wrong location. Authored limits also do not
consistently coincide with a well-calibrated spherical-envelope contact
minimum: the mean absolute signed gap at the true validation limits is
`0.057887` scene units, while the total-energy minima frequently occur at the
scan boundary. A head allowed to move an anchor by only `0.10q` cannot express
the required corrections of up to `1.749910q`.

This falsifies the specific node-8.4 decomposition “both endpoints = frozen D2
anchor + small transferred residual.” A follow-up must be preregistered as a
new node. The most defensible next mechanism is asymmetric: keep only a
demonstrably contact-identifiable closure anchor and learn total physical range
or the opposite hard-stop extension, with the second endpoint closed
analytically by kinematics. That is a new hypothesis and was not retrofitted
into node 8.4.

## Verification

`PYTHONPATH=src python -m pytest tests/test_source_transfer.py -q` → `2 passed`.

The complete per-object audit is in `metrics.json`; immutable remote paths and
SHA-256 receipts are in `provenance.json`.
