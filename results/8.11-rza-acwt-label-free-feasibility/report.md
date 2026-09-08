# Node 8.11 — RZA-ACWT label-free feasibility

## Outcome

**PRUNE_LABEL_FREE.** Only the mechanical Articraft cell passes every frozen
gate. The other three cells fail without opening source labels, computing
source scores, training, or reading Box/protected data. RZA-ACWT therefore
does not advance to source scoring.

The intended raw-z correction itself worked: full-fit normalized raw-z
correlations are between `2.7564e-16` and `4.1167e-16` across all four cells
(mean `3.4435e-16`, threshold `1e-10`). The failure is instead OOF dependence
on small NJC and an Articraft semantic norm-tail expansion.

## Frozen decision table

| Transport cell | OOF abs-d Spearman (`<=0.35`) | OOF raw-z dCor (`<=0.5`) | Shuffled p99 ratio (`<=1.25`) | Decision |
|---|---:|---:|---:|---|
| Semantic / Articraft | 0.104549 | 0.289016 | **1.506940** | Fail: p99 |
| Semantic / NJC | 0.031891 | **0.635188** | 1.063196 | Fail: dCor |
| Mechanical / Articraft | 0.031914 | 0.282532 | 1.122491 | **Pass** |
| Mechanical / NJC | **0.500000** | **0.711675** | 0.955979 | Fail: Spearman and dCor |

The NJC OOF folds contain only `[3,2,2,2,2]` objects. Rank-one anisotropic
whitening is identifiable by its eigengap audits, but does not generalize its
conditional independence to those held objects. No alternate rank, degree,
threshold, seed, or sweep was attempted.

## Unchanged gates

Every cell satisfies the following unchanged gates:

- exact reconstruction (`8.33e-17` to `3.59e-14`, threshold `1e-10`);
- location, rank-quadratic, radial-scale, and block-scale orthogonality
  (`<=1.60e-15`, threshold `1e-10`);
- residual energy (`0.5851` to `0.9187`, minimum `0.05`);
- shuffle magnitude (`0.8228` to `1.1445`, minimum `0.1`);
- train/held coverage `1.0`, zero train self-rate, exact train donor marginal,
  effective held donors/object `1.0`, and donor loads within their bounds;
- positive unclipped radial, axial, and orthogonal scales;
- recipient within-object offset errors below `1.78e-15` and bit-identical
  mechanical displacement;
- held ECDF boundary clipping at `0` except semantic NJC at exactly the allowed
  `0.25` boundary.

All full-fit and fold-local eigengaps pass by large margins. Full-fit gaps are
`3.3382`, `4.0743`, `11.1291`, and `467.0430`, versus numerical thresholds
from `9.75e-14` to `1.74e-11`. Every fitted axis has unit norm to floating-point
precision and satisfies the preregistered deterministic sign rule.

## Repeatability and row identity

The node 8.10 row-order defect is closed. Object reductions now use canonical
global identities formed from domain plus stable gauge key. Both semantic and
mechanical outputs are bit-identical under independent recomputation and row
reversal. Candidate and gate-recompute prediction hashes match exactly:

- semantic: `ae6ecd4294b68eb7cb45d33d9259e75ef0abbdf345ea677b3ab59eeb8031f286`;
- mechanical: `d9344de3484f0233a4422401f9fea121e571e65be926f15e326af724bd824403`.

The global row-identity digest is
`1395c6c8a76175302e1c0fa46fe41c4493834d27d5146c9fc41597c15f37ae82`.

## Provenance and boundary

- method commit: `00c8cbd9e5dc6c07927f40fb2a77bb273cff7a50e`;
- frozen config SHA-256:
  `779874c52dc3c64b3e6606ad11c4b0a2e3d07fe5b814054b7d2ce8a0ff852be5`;
- remote output:
  `/data1/public/yptang/splart-node811-rzaacwt/results/label-free-feasibility-v1`;
- feasibility SHA-256:
  `67181bee2c9785a5a5349dcc832312de14e07ff494f528ef9e108e6e9aa023e1`;
- remote receipt SHA-256:
  `8870bfd6c6bdb3ff37c770bd266a9dea68df8e7f9aa5506a2431be5705cc23d1`;
- log: `/data1/public/yptang/splart-node811-rzaacwt/feasibility-v1.log`,
  SHA-256 `694e69b7872fb5111f9956352a48d848628770c1cea7a6f32e0be2a671f12fd8`;
- code/feature provenance digests:
  `023d6545f3000b46c0e4017be37e6fa54ce6cf02659ec365ee3f8d958880cf62` /
  `cbf5bae9888d46ef3f820daa26ebd3d6a2718842974f3365064f0a79d558fb3b`.

The remote receipt records `remote_execution_started=true`,
`execution_environment=CASIA_98`, and
`execution_phase=label_free_feasibility`. It also records all of
`source_labels_opened`, `source_labels_hashed`, `source_scores_computed`, and
`training_started` as false, with empty Box and protected-split lists.

The 10,336,078-byte feasibility payload remains at its hashed remote path and
is deliberately not committed. `metrics.json` contains the complete compact
four-cell gate values, scales, eigengaps, runtime audits, and provenance.
