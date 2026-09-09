# Node 8.13: PKSRT label-free feasibility

Decision: **PRUNE_LABEL_FREE**. Only semantic/Articraft passed the preregistered four-cell gate. No source scoring, training, or Box evaluation is authorized.

## Frozen identity and artifacts

- Final method commit: `431426b538abb942c4f9c650e68d231398fd974e`
- Config SHA-256: `62c69b18a3c2171bf57fa71acdcb4fa43b186377212a719ff118b09f913f8570`
- Canonical feasibility SHA-256: `6df3b8249463a47830432bffd71b39e1d3be5af6566e8df03fcd6c030015b68b`
- Raw feasibility file SHA-256: `e176f72a24278dc14a41391c6d04cd6a64ff757f11e6b6f76a9b23dd35cdff96`
- Raw receipt file SHA-256: `9820f6966d593ed234e3ad92436addd5e7ce9514def476474af6387bcc9ad9f5`
- Feature provenance SHA-256: `cbf5bae9888d46ef3f820daa26ebd3d6a2718842974f3365064f0a79d558fb3b`
- Code provenance SHA-256: `7143a37d72bd30f1093e080ae0618035224172a73a14a33a03f26fd630974bc1`

The first formal launch completed its computation but failed the final provenance check because the validator replayed a producer-side row projection through a differently dispatched matrix kernel. This was a receipt-runtime defect, not a model, configuration, threshold, or data change. Commit `431426b` canonicalized both sides to the same fixed scalar reduction; all 84 lineage tests passed locally and on server 98. The v1 failure log has SHA-256 `5189c269bd2b8ce48fe64eb96f0e76d4ab2490840aa2f069011feaeaed89a688`. The unchanged frozen configuration was rerun as v2, and an independent full replay returned `verify_provenance_binding=true` and reproduced the signed decision.

## Four-cell gate comparison

The frozen limits were absolute Spearman ≤ 0.35, distance correlation ≤ 0.50, and shuffled p99 ratio ≤ 1.25.

| Field/domain | Spearman | dCor | p99 ratio | Result |
|---|---:|---:|---:|---|
| semantic / Articraft | 0.063798 | 0.354233 | 1.066533 | pass |
| semantic / NJC | 0.501140 | 0.509976 | 1.756696 | fail: Spearman, dCor, p99 |
| mechanical / Articraft | 0.226265 | 0.257444 | 1.837604 | fail: p99 |
| mechanical / NJC | 0.327273 | 0.591941 | 1.540105 | fail: dCor, p99 |

Both global structural receipts passed. Repeat determinism and row-order invariance passed for both fields, and mechanical recipient displacement remained bitwise unchanged. Residual means and raw-z correlations were below `2e-16` in every cell, so the raw-z anchor worked exactly. The rejection is substantive: pooled conditional transport removed first-order dependence but inflated tails on three cells and retained cross-fitted dependence on NJC.

## Research conclusion

PKSRT falsifies the hypothesis that a complete pooled conditional distribution transform alone resolves the endpoint residual nuisance. Articraft semantic benefits, but pooling does not stabilize the small NJC domain and the nonlinear map amplifies residual norms. The next credible direction should constrain transport by geometric/physical invariants or use domain-adaptive partial pooling, with any new node preregistered before execution; bandwidth, slice count, thresholds, and seeds must not be swept on these results.

## Access boundary

`source_labels_opened=false`, `source_labels_hashed=false`, `source_scores_computed=false`, and `training_started=false`. `box_labels_read=[]` and `protected_splits_read=[]`. The 999,928,940-byte feasibility payload remains on server 98; this directory stores its immutable hashes, compact metrics, and signed receipt.
