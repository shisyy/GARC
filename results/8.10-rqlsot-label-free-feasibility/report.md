# Node 8.10 — RQ-LSOT label-free feasibility

## Decision

**PRUNE_LABEL_FREE.** The preregistered rank-quadratic location/scale orthogonal transport (RQ-LSOT) null fails before any source-label access, source scoring, training, or Box evaluation. No follow-on experiment is authorized from this node.

The executed code commit was `f7bdfa1a4a112c96c0067db19ead8c1fe1716a2e`; the frozen configuration SHA-256 remained `e3d6c1b3d59e2fd4c155fcff7e818cce0af95343da81c9d2f01bb2990c0a5acc`.

## Label-free gate results

All errors/correlations below are target-free diagnostics. “Raw corr” is the preregistered in-sample maximum normalized residual correlation with original raw-z (threshold `<=1e-10`). OOF diagnostics compare fold-local standardized residuals with one unified original raw-z per domain; rank-x diagnostics were recorded only as secondary evidence.

| Transport cell | Norm. mean `<=1e-10` | Raw corr `<=1e-10` | OOF Spearman `<=.35` | OOF dCor `<=.50` | Energy `>=.05` | Shuffle RMS `>=.10` | Norm p99 `<=1.25` | Held clip `<=.25` | Max recipient-u `<=1e-12` | Cell |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Semantic / Articraft | 4.59e-16 ✓ | 0.08134 ✗ | 0.06790 ✓ | 0.31496 ✓ | 0.81992 ✓ | 1.11359 ✓ | 1.13443 ✓ | 0.000 ✓ | 1.67e-16 ✓ | Fail |
| Semantic / NJC | 1.79e-16 ✓ | 0.16475 ✗ | 0.15490 ✓ | 0.58467 ✗ | 0.73258 ✓ | 1.16482 ✓ | 1.14794 ✓ | 0.250 ✓ | 1.11e-16 ✓ | Fail |
| Mechanical / Articraft | 4.30e-16 ✓ | 0.08318 ✗ | 0.01837 ✓ | 0.26071 ✓ | 0.92040 ✓ | 1.07258 ✓ | 1.05105 ✓ | 0.000 ✓ | 2.72e-15 ✓ | Fail |
| Mechanical / NJC | 1.38e-15 ✓ | 0.10253 ✗ | 0.12727 ✓ | 0.63356 ✗ | 0.58149 ✓ | 0.82588 ✓ | 0.93120 ✓ | 0.000 ✓ | 6.66e-16 ✓ | Fail |

The decisive common failure is the unchanged raw-z correlation gate: all four values are many orders of magnitude above `1e-10`. NJC additionally fails the OOF distance-correlation gate for both transports. Reconstruction and basis orthogonality remained within `1e-10`; coverage was 1.0, train self-rate was 0, held effective donors per object was 1.0, and donor load equaled its bound in every cell.

## Runtime audits

- Mechanical: repeat bit-identical `true`; row-order invariant `true`; recipient displacement bit-identical `true`.
- Semantic: repeat bit-identical `true`; row-order invariant `false`.
- Full-receipt pass: semantic `false`, mechanical `false`; every per-domain receipt and cell pass was `false`.
- Cross-fit fold counts: Articraft `[20,20,19,19,19]`; NJC `[3,2,2,2,2]`.

The semantic row-order failure is retained as a real runtime negative result. It is not needed to reach the prune decision because each semantic domain already fails the raw-z gate.

## Provenance and access boundary

- Remote output: `/data1/public/yptang/splart-node810-rqlsot/results/label-free-feasibility-v1`
- `feasibility.json` SHA-256: `86ec72ef6d814cc06eedf78cbdc4541b42254958e7a3758b6a91123ed4523ef1`
- `receipt.json` SHA-256: `ac53cf8f4594438be15b2abaa968b54602cc2fdd5d9f62342d548003b68583e2`
- Run log SHA-256: `9e5893f43571b20ec31fcec7d0b169e309151124f37d88e0c072bfadc03980d0`
- Code provenance SHA-256: `70049c08e8dce527be157959ca6488cd4da05e11c9b228efaca9a73dba6a51fb`
- Feature provenance SHA-256: `cbf5bae9888d46ef3f820daa26ebd3d6a2718842974f3365064f0a79d558fb3b`
- Source config SHA-256: `454f4e16af01e4986b9e4534168b6be7bc96eb4617a63bcc988dfdb7841aaca8`

The immutable receipt records `source_labels_opened=false`, `source_labels_hashed=false`, `source_scores_computed=false`, `training_started=false`, `box_labels_read=[]`, and `protected_splits_read=[]`. Consequently, NMAE is not applicable and no label-derived model comparison was computed.

