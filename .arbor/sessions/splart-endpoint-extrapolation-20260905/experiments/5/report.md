# Arbor executor report: node 5 FECF

## Idea

Factorized Endpoint and Closure Fusion (FECF) preserves the frozen D2 endpoint
scalars bit-for-bit and uses only the frozen PILC closed-side posterior to add a
selective semantic answer: which extrapolated endpoint is physically closed.
The posterior is evaluated in both the original ordering and the mandatory
state-swapped ordering, where `q' = 1 - q` and the local endpoint name flips.
This makes a constant-local-lower predictor exactly 50%, rather than allowing
the URDF convention to masquerade as a learned result.

## Changes

- Added a factorized fusion adapter with threshold `0.80`, exact D2 scalar/hash
  identity checks, and a strictly swap-equivariant closed-side posterior.
- Added original, state-swapped, constant-lower-original, and
  constant-lower-state-swapped outputs in a fixed order for every episode.
- Added fail-closed handoff validation: public-only prediction hashes are
  cross-checked against the atomic producer receipt and its original execution
  receipt; score-bearing handoff metadata is rejected.
- Added a durable watcher, fixed clean-evaluator provenance checks, unbuffered
  logging, atomic receipts, and failure-preserving evaluator-only recovery.
- Added an independent e/f confirmatory addendum bound before evaluation to
  pre-eval manifest SHA-256
  `7cfcf3453dc0531771eeb5352829649044573345c83ab7aea189cce093ea09e4`.
- Added 30 passing CPU tests. No seed, checkpoint, threshold, view selection,
  endpoint scalar, or model parameter was changed after seeing evaluation
  results. `B_test` and `Full22` were not read.

## Frozen contract

- Frozen FECF method commit:
  `1ed4b7dedf287781b6138d8bab1d72680400131f`.
- Frozen preregistration canonical SHA-256:
  `bcc160040cf9b8b815b925c65f5d553b3f6b2372cbcc48297241d6c886490ca6`.
- Frozen PILC checkpoint SHA-256:
  `bbd109b66ed854400c47571945a625938a80dede80b827053236231b8108c38a`.
- Frozen D2 source commit:
  `cdb62c61d2d6ff06ad0b78c3add7a60df08ab867`.
- Frozen confidence threshold: `0.80`.
- Clean evaluator commit:
  `cd8d3da2fb72e21992af50ee2e73fdba26f2ae1c`.
- e/f confirmatory addendum canonical SHA-256:
  `28db5b9eac9138370bae6dbd0316ebb7dfe9ed138ce02aa6ed25bf55f6b0f447`.

The original preregistration and the four method files remain byte-identical
to commit `1ed4b7d`; later commits change watcher/evaluator plumbing only.

## c/d plumbing history

1. `box-c-d-v1` failed closed before inference because the watcher expected the
   obsolete receipt schema. Exit status was `1`; watcher log SHA-256 was
   `403e9d5303592b7c45da510b4cede1b509f205cee6d990f0109a306644d31aba`.
2. Commit `49c4bcf513e1c3f828bd8d45a12a343890c86b69` adapted the watcher to the
   actual `splart-fecf-d2-handoff/v1` schema and verified original execution
   receipt hashes. `box-c-d-v2` then completed all public inference and produced
   the eight frozen predictions plus two identity receipts, but its evaluator
   process failed with `ModuleNotFoundError: splart.endpoint_baselines`. The
   failure was preserved with exit status `1`; log SHA-256 was
   `21ec92efb34a54821d547ab60e4b3777c716cfaecfc1b47da734ba0fd682ad46`.
3. Commit `f9eefaf78da5eb4f6c7f13d88b7466cd0cb1c89c` fixed only import-path
   plumbing. `box-c-d-v3-evaluator-only` reused the exact v2 predictions and
   identity receipts, invoked the clean evaluator once, and finished with exit
   status `0`. No inference or selection was repeated.
4. Commit `301760810422c6630ab5968405d30eef1b90ab07` generalized only the
   validated episode/config plumbing and added the pre-frozen e/f addendum.
   `box-e-f-confirmatory-v1` ran end-to-end once and finished with status `0`.

## All 16 paired results

Every row has closed coverage `1.0`. Endpoint NMAE and its lower/upper terms
are identical across all four variants of an episode, confirming that FECF did
not change D2 endpoint estimation.

| Episode | Variant | Endpoint NMAE | Lower | Upper | Closed acc. | Terminal valid |
|---|---|---:|---:|---:|---:|---:|
| Box-c | FECF original | 0.162144 | 0.161341 | 0.162948 | 1.0 | 0.0 |
| Box-c | FECF swapped | 0.162144 | 0.161341 | 0.162948 | 1.0 | 0.0 |
| Box-c | Constant lower original | 0.162144 | 0.161341 | 0.162948 | 1.0 | 0.0 |
| Box-c | Constant lower swapped | 0.162144 | 0.161341 | 0.162948 | 0.0 | 0.0 |
| Box-d | FECF original | 0.153365 | 0.183468 | 0.123262 | 1.0 | 0.0 |
| Box-d | FECF swapped | 0.153365 | 0.183468 | 0.123262 | 1.0 | 0.0 |
| Box-d | Constant lower original | 0.153365 | 0.183468 | 0.123262 | 1.0 | 0.0 |
| Box-d | Constant lower swapped | 0.153365 | 0.183468 | 0.123262 | 0.0 | 0.0 |
| Box-e | FECF original | 0.146500 | 0.065544 | 0.227457 | 1.0 | 0.0 |
| Box-e | FECF swapped | 0.146500 | 0.065544 | 0.227457 | 1.0 | 0.0 |
| Box-e | Constant lower original | 0.146500 | 0.065544 | 0.227457 | 1.0 | 0.0 |
| Box-e | Constant lower swapped | 0.146500 | 0.065544 | 0.227457 | 0.0 | 0.0 |
| Box-f | FECF original | 0.168209 | 0.085328 | 0.251091 | 1.0 | 0.0 |
| Box-f | FECF swapped | 0.168209 | 0.085328 | 0.251091 | 1.0 | 0.0 |
| Box-f | Constant lower original | 0.168209 | 0.085328 | 0.251091 | 1.0 | 0.0 |
| Box-f | Constant lower swapped | 0.168209 | 0.085328 | 0.251091 | 0.0 | 0.0 |

The c/d endpoint-NMAE macro is `0.1577548412`; the independent e/f
confirmatory macro is `0.1573547441`; the four-episode macro is
`0.1575547927`. Across eight original/swapped decisions, FECF is `8/8 = 1.0`
while constant-local-lower is `4/8 = 0.5`.

## D2 identity and equivariance

All four outputs for each episode have bit-identical D2 projections:

| Episode | D2 projection SHA-256 | Four-way identity | Posterior swap error |
|---|---|---:|---:|
| Box-c | `a6c76140ebd6f85676005b88b1db1d9e4dfad1773538de33339d5009e7e098d9` | true | 0.0 |
| Box-d | `d9a1520e5c7cc4b6de1cac3a0410b79e19886f5a7507103a82c0a36bdbaca219` | true | 0.0 |
| Box-e | `2953314ecf8068886af6a67972fcbfbeba8d52c8841bb6f0cfcc861808b59618` | true | 0.0 |
| Box-f | `0db7de0ccb2abda93d315c74261f62d75e5e4f4d32c4b50b5e653b8324977a62` | true | 0.0 |

## Receipt SHA-256 audit

| Receipt | SHA-256 |
|---|---|
| Box-c D2 atomic handoff | `1ab9cb2b2d55816ba0c9de7b0c73ffbc88eb179b8f44a404663be1cc7b6e037a` |
| Box-d D2 atomic handoff | `33b2a261fa9f141b08bd61fd7e7933fa729d5c5efc295461612cb990c826cfb4` |
| Box-c original execution | `99d37e5eb255346fc3268ebcff0a7ef0cb6c1c2deb794dece4df4bd4d792196d` |
| Box-d original execution | `75450ac036341746beb12b046e7759a548352351ed1fa4edf851aaff70694ec0` |
| c/d v2 readiness | `61db3d19ec67cdc9e8c0e80103465788e322d2db332d60df370ae0c48acff28e` |
| Box-c identity | `53bd80f073d398926eeef2dd231fe2b152ae3f54782e32e5cee7dd59065761b9` |
| Box-d identity | `5edd620482558b1e20e650ac29602178effc50c333a41a126a041fb57d8eb9c1` |
| c/d v3 independent evaluation | `7c8a26e32ebd4ba4fdf922efdd9e627cc057cdd8f4215a7e1adf1d270baf6393` |
| Box-e D2 atomic handoff | `c4a9a64a811c30961786b3aac670b722bf9faa0522af66d5f31cb579edee8ee4` |
| Box-f D2 atomic handoff | `b8c02ffe0c46b9a892f68c0c0ff1fc22f00216131ff3d4cb0b4a0a55eccc3e6d` |
| Box-e original execution | `1a5ad146597b1ba91d4d177b1acc81c275e6ac21ff75efca88797602af980aa6` |
| Box-f original execution | `b6d01262c031f82409e3809b332f63cffb47a58c5535f7aeedd41e7ffb770703` |
| e/f confirmatory readiness | `cb04b6d66c7d033511353f883e97023a2d431d5342562e507b955cbaba75bd67` |
| Box-e identity | `a3efa0f7db3ba00dc9b0b046458e2655d330f25a22ff701b83ca3bf0a37adeac` |
| Box-f identity | `a67591973d14e0efb660bc413c5ce9ab590d625271ebea42e0c9728afd99a0b2` |
| e/f independent evaluation | `127a0dba9cd11556a5ebc2930d6ea134dd78d35e1a2f86843f2e854bd02d3b45` |

Both independent-evaluation receipts state
`intermediate_scores_parsed=false`, `selection_by_result=false`, and
`protected_splits_read=[]`.

## Score and interpretation

The node's absolute paired closed-side accuracy is **1.0**, versus **0.5** for
the mandatory constant-side baseline, on both the c/d confirmation and the
independent e/f stress confirmation. This is valid evidence that the frozen
public-geometry posterior supplies state-swap-consistent closed-side semantics.

It is not evidence that the physical endpoint values or terminal configuration
are solved. FECF deliberately preserves D2 endpoint scalars, the four-episode
endpoint NMAE remains `0.1575547927`, and `terminal_contact_valid` is `0.0` in
all 16 evaluations. The second innovation therefore solves only selective
closed-side semantics; it does not provide terminal-contact certification or
improve the extrapolated endpoint locations.

## Result

FECF passes its closed-side semantic objective robustly across c/d and the
independent e/f stress episodes, with exact swap equivariance and a 50-point
absolute gain over the constant-side baseline. It remains ineligible for any
claim of complete physical-limit prediction because terminal validity is zero
and endpoint NMAE is unchanged. The branch is frozen and intentionally not
merged, pending independent P0 review.

## Provenance

- Branch: `arbor/node-5-fecf`.
- Frozen method commit: `1ed4b7dedf287781b6138d8bab1d72680400131f`.
- Schema adapter commit: `49c4bcf513e1c3f828bd8d45a12a343890c86b69`.
- Clean evaluator import fix: `f9eefaf78da5eb4f6c7f13d88b7466cd0cb1c89c`.
- e/f confirmatory orchestration commit:
  `301760810422c6630ab5968405d30eef1b90ab07`.
- c/d evaluation run:
  `/home/yptang/arbor-runs/splart-endpoint-fecf/box-c-d-v3-evaluator-only`.
- e/f confirmatory run:
  `/home/yptang/arbor-runs/splart-endpoint-fecf/box-e-f-confirmatory-v1`.
- Local score/receipt snapshots: `results/5-fecf/`.
