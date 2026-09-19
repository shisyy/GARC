# Geometry + semantics: reproducible exploration snapshot

Version: `geometry-semantics-20260919`; branch `codex/geometry-semantics-20260919`.
Repository: <https://github.com/shisyy/GARC> (previous GAR/ALEx URLs redirect here).
Core frozen implementation: `23f19b19802271791faa26b56b640602bfc4bd2d`.
The independently explored view-reliability ablation is included from `f877e5a`.

This is a code/research snapshot, **not a claim that semantics improves the
latest geometry-only method**. It preserves both useful geometry results and
negative semantic results. No datasets, model weights, CLIP caches, native
asset archives or private endpoint labels are distributed in this release.

## What is included

| Exploration | Entry point | Core implementation | Interpretation |
|---|---|---|---|
| Frozen CLIP limit features / C-CLIP-LD | `build_clip_limit_cache.py`, `run_clip_limit_source_gate.py` | `clip_limit.py`, `clip_limit_gate.py` | Legacy source-supervised head; NOT observation-only training |
| Semantic tangent residual / CSTR | Same CLIP gate runners, CSTR config | `clip_limit.py` | Legacy source-supervised comparison; see `CSTR_DEPLOYMENT.md` |
| Relative geometry, semantics and joint search | `run_relative_search.py` | `relative_search.py` | No trained endpoint head; includes geometry/semantic/donor controls |
| Observed-motion view weighting | `run_view_reliability.py` | `view_reliability.py` | Frozen weighting, donor/shuffle controls; not a validated gain |
| Visible-part deletion interaction | `run_part_interaction.py` | `part_interaction.py` | Same-RGB binding and matched area-null controls |
| Geometry-first bracket verification | `run_geometry_first_gate.py` | `geometry_first.py` | Earlier gate; semantics did not change validation positions |
| Observed triangle contact atlas | `run_surface_contact.py` | `surface_contact.py` | Exact triangle-pair enumeration, not a solid collision certificate |
| Held-out midpoint calibration | `run_contact_calibration.py` | `contact_calibration.py` | Check the frozen atlas; do not update it with calibration observations |
| Contact-bounded closure semantics | `run_contact_bounded_gate.py` | `contact_bounded_semantics.py` | Latest four-mode experiment, with sealed predictions and controls |

Core modules are under `src/splart/`. Rendering and acquisition are provided by
`renderer/renderer_clip_stream.py`, `build_relative_clip_cache.py`,
`build_shared_rgb_part_cache.py`, `rebase_shared_rgb_inputs.py` and
`build_observed_surfaces.py`. See [RELATIVE_SEARCH.md](RELATIVE_SEARCH.md),
[SHARED_RGB_INPUTS.md](SHARED_RGB_INPUTS.md),
[PART_INTERACTION.md](PART_INTERACTION.md) and
[VIEW_RELIABILITY.md](VIEW_RELIABILITY.md) for their exact boundaries.

## Latest method, in plain terms

1. Keep the original observation coordinate system: the two inputs are q=0
   and q=1. Candidate states extend on either side; no virtual .1/.9 gauge.
2. Record triangle contacts across 65 poses within the observed interval.
   This distinguishes pre-existing mesh intersections from new contact events.
3. Check 64 interstitial poses against that fixed atlas. New unexplained
   events fail the calibration check; the atlas is not enlarged to pass it.
4. Obtain per-side contact-based candidate bounds. The tested semantic module
   searches only within eligible closure-side prefixes using frozen CLIP.
5. Run raw, bounded, donor-semantics and shuffled-verification modes. Each has
   coordinate-only, geometry-only, semantic-only, joint and object-shuffle arms.
   Seal all 380 predictions before independent endpoint scoring.

This snapshot's no-contact fallback preserves its original experiment. The
next research direction will examine semantic priors on missing-contact sides
after the observation-only baseline/main-method reruns; it is NOT implemented
or validated by relabelling this release.

## Conda environment

Use the existing SplArt Python3.11 Conda environment, or follow upstream SplArt
installation instructions in the README to create a separate environment.
Do not delete or replace an existing environment to use this snapshot.
Additional contact/testing dependencies in the verified environment are:

```bash
conda activate splart
python -m pip install 'python-fcl==0.7.0.11' pytest
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
```

PyTorch and NumPy are required. Full CLIP acquisition additionally needs the
renderer/CLIP dependencies documented in [C_CLIP_LD_DEPLOYMENT.md](C_CLIP_LD_DEPLOYMENT.md).
Frozen-cache prediction itself does not download a CLIP model. The surface
engine records installed FCL binary hashes: caches generated with another
backend must not be silently mixed. CUDA is needed for full SplArt/CLIP GPU
work; the synthetic contact/gating tests below can run CPU-only on Linux.

## Test the published code

Run from this branch's repository root:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
python -m pytest -q \
  tests/test_relative_search.py tests/test_view_reliability.py \
  tests/test_part_interaction.py tests/test_geometry_first.py \
  tests/test_geometry_first_gate.py tests/test_observed_surfaces.py \
  tests/test_surface_contact.py tests/test_contact_calibration.py \
  tests/test_contact_bounded_semantics.py tests/test_contact_bounded_gate.py
```

These tests use synthetic fixtures, exercise geometry/semantic routing, frozen
controls, swap symmetry and sealing, and do not open real endpoint labels.
The original node9.5 implementation passed 127 remote tests. Publication-branch
regression results, including the added view-weighting ablation, are recorded
separately in `release/geometry-semantics-20260919/VALIDATION.md`.

## Reproduce the latest frozen-cache experiment

Obtain the approved source assets under their original license. A trusted
preparation process owns the native render plan and evaluator source data;
the predictor must receive only the exported, hash-bound input/cache indices.
The source indices must retain the fixed 19-object join (historical 13/6
subgroups). Commands below accept your local artifact paths; all output
directories must be new.

```bash
# Trusted preparation: exact asset/plan bindings are required by this exporter.
python build_observed_surfaces.py --help
# Prepare relative CLIP and shared-RGB inputs as documented in SHARED_RGB_INPUTS.md.

python run_surface_contact.py \
  --surfaces "$SURFACE_INDEX" --base-inputs "$INPUT_INDEX" \
  --output-dir "$RUN/contact"

python run_contact_calibration.py \
  --surfaces "$SURFACE_INDEX" --contacts "$RUN/contact/index.json" \
  --base-inputs "$INPUT_INDEX" --output-dir "$RUN/calibration"

# Predictor: no endpoint/evaluator source index is an argument.
python run_contact_bounded_gate.py \
  --inputs "$INPUT_INDEX" --contacts "$RUN/contact/index.json" \
  --calibration "$RUN/calibration/index.json" --output-dir "$RUN/predictions"

# Separate trusted evaluator, only AFTER all_modes.seal.json exists and verifies.
for mode in raw bounded donor_semantics shuffled_verification; do
  python evaluate_relative_search.py \
    --predictions "$RUN/predictions/$mode/predictions.json" \
    --source-index "$EVALUATOR_SOURCE_INDEX" \
    --output "$RUN/predictions/$mode/report.json"
done
```

Do not invoke a legacy source-supervised CLIP/LOOP trainer and call its result
endpoint-label-free. Do not substitute reconstructed Gaussian centers for
triangle surfaces without a separately validated reconstruction adapter.

## Verified results

All19 objects were predicted; the preregistered primary comparison used the
6-object development subgroup (12 directions). These repeatedly inspected
objects are not an independent final test. Errors are unconditional mean
absolute endpoint errors in **observed-span units**, not GT-full-range NMAE.
Abstentions remain in the errors. Lower is better.

| Method | Development mean endpoint error |
|---|---:|
| Raw geometry | 0.815414190 |
| Raw semantics | 0.591369003 |
| Raw joint | 0.706820458 |
| Contact-bounded geometry | **0.405518351** |
| Contact-bounded own semantics | 0.426041663 |
| Contact-bounded donor semantics | 0.441232640 |
| Contact-bounded shuffled role | 0.425607640 |

Contact geometry improves about50.3% over raw geometry. Adding own semantics
worsens that geometry by about5.1% and loses to the shuffled-role control.
All3 semantic position changes in the development subgroup worsen error.
Therefore **the geometry improvement is supported in this diagnostic, while
the claimed additional benefit of the tested semantic selector is not**.

Frozen provenance:

- Core commit: `23f19b19802271791faa26b56b640602bfc4bd2d`.
- All-mode seal: `845220c880bfa999bb84f906f2e09006a83419ffedddee4ae6e13321ddee32a2`.
- Evaluator SHA256: `e7e4003d41209f3c67834646495c73ff709830be72e562851356a1dacc001689`.
- Renderer SHA256: `136e5658ed45e5c7c13aa4f10191ddc10978e9e14d181adb4cd31209c873a374`.
- Calibration:19/19; exact observation-swap error0; all380 predictions sealed.

## Limitations and ongoing work

These are **ideal full-asset surface diagnostics**, with privileged geometry
and motion information relative to a two-RGB-state reconstruction task.
The source renderings are stylized. This release does not establish the same
gains using reconstructed SplArt geometry. Triangle contact and midpoint
checks do not certify solid collision freedom, a mechanical hard stop or
semantic closedness. Authored URDF limits define benchmark targets, not
independently measured physical stops.

A separate new round independently trains SplArt from scratch on19 objects
using two native-rendered interior observations per object. Numerical GT
kinematics and endpoint imagery are isolated from optimization. Its D2/LOOP
adaptations, fresh checkpoints and eventual results are not included in this
historical snapshot and must not be conflated with the table above.
