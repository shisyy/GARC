# Node 2 executor report: Contact-Feasible Endpoint Field

## Idea

Scan the learned screw trajectory outside the two relabelled interior states
and infer the collision-free interval from contact, penetration, and terminal
support evidence. Predict both endpoint scalars, the closed-end label, and an
uncertainty/identifiability certificate without limit supervision.

## Changes

- Added `src/splart/contact_endpoint_field.py` with revolute and prismatic
  screw transforms, four differentiable physical energies, posterior endpoint
  estimates, uncertainty intervals, broad-versus-localized closure evidence,
  and fail-closed behavior.
- Added deterministic analytic CPU coverage in
  `tests/test_contact_endpoint_field.py`.
- Added `CONTACT_ENDPOINT_FIELD.md` documenting the leakage boundary,
  mechanism, certificate, and current prototype scope.

## Implementation choices

- Coordinates are strictly the observed-state convention: state 0 is scalar
  0 and state 1 is scalar 1.
- The public API has no input for absolute interior fractions, URDF limits,
  endpoint images, endpoint labels, scene identity, seeds, or checkpoints.
- The terminal-support term requires collision evidence to increase just
  beyond a proposed endpoint while the adjacent inward configuration remains
  collision-free.
- A physical stop is rejected when the best candidate lies at the scan
  boundary, lacks contact/support evidence, has excessive inside penetration,
  or has a diffuse posterior.
- The closed label is rejected as `unknown` when contact-area evidence is
  symmetric, even when both endpoint locations are identifiable.
- Dense contact pairs are capped. Large real scenes must supply a deterministic
  geometry-only broad-phase pair list rather than allocate all pairs.

## Baseline vs result

The new from-scratch extrapolation baseline is not yet sealed, so no B_dev or
baseline metric was read and no candidate comparison was performed. This
executor stage is deliberately limited to mechanism implementation and
analytic CPU validation.

## Score

N/A (pre-baseline analytic stage; no performance claim).

## Analysis

Command:

```text
python -m pytest -q tests/test_contact_endpoint_field.py
```

Result: `7 passed`.

Covered cases:

- prismatic broad-closure plus localized opposite stop;
- revolute broad-closure plus localized opposite stop;
- free space with no physical stop (fail closed);
- symmetric terminal contacts (closed label remains unknown);
- finite gradients through prismatic and revolute geometry/screw parameters;
- explicit broad-phase requirement for unsafe dense pair counts.

Formatting and syntax checks also pass:

```text
python -m py_compile src/splart/contact_endpoint_field.py tests/test_contact_endpoint_field.py tests/conftest.py
python -m black --check src/splart/contact_endpoint_field.py tests/test_contact_endpoint_field.py tests/conftest.py
git diff --check
```

## Insights

Physical limit location and semantic closure are separate identifiability
problems. Contact can identify both stops while symmetric geometry still gives
no defensible closed-end label; explicitly returning `unknown` is necessary to
avoid manufacturing supervision. The next valid step is a sealed-baseline
candidate/control run, not further threshold fitting on benchmark outcomes.
