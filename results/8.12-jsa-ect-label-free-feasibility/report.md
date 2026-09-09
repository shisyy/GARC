# Node 8.12: JSA-ECT label-free feasibility

Decision: **PRUNE_LABEL_FREE**. Only the semantic/Articraft cell passed. The method therefore did not satisfy the preregistered requirement that all four domain/null cells pass, and no source scoring, training, or Box evaluation is authorized.

## Frozen identity and artifacts

- Method commit: `7a57da0a0a9a438d3305c7dd1ea081346cd528b7`
- Config SHA-256: `a89543b1cf954cfc2de191fb50dda265ca8182dcac9e1a7d14a8faed90f853b2`
- Remote feasibility SHA-256: `bae89006521d3fe1b573f4d6edfc81a1987d86d1ff160706c94c1304e191d978`
- Remote receipt SHA-256: `06e189cdfc1297a88d884dd1e7494ce72df75558f5e8f11cb96c700a772fe9a5`
- Feature provenance SHA-256: `cbf5bae9888d46ef3f820daa26ebd3d6a2718842974f3365064f0a79d558fb3b`
- Code provenance SHA-256: `aefe92007582cc69b6000bc78e192405efde8af18bc6c4e99e7b083c8ebcdee5`

The first launch was a deployment error: the frozen source copy omitted `scripts/build_smarc_njc_cache.py`. It produced no method output; its error log is SHA-256 `4c405f09fc35c725be9687223830fbcdadbda0785a8d77c3c443400ef2f52e1a`. The corrected v2 launch completed; its deliberately empty stdout log is SHA-256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.

## Four-cell gate comparison

The frozen limits were absolute Spearman ≤ 0.35, distance correlation ≤ 0.50, and shuffled p99 ratio ≤ 1.25.

| Field/domain | Spearman | dCor | p99 ratio | Result |
|---|---:|---:|---:|---|
| semantic / Articraft | 0.139905 | 0.367139 | 1.183848 | pass |
| semantic / NJC | 0.287017 | 0.679309 | 1.154967 | fail: dCor |
| mechanical / Articraft | 0.436803 | 0.341980 | 1.035548 | fail: Spearman |
| mechanical / NJC | 0.009091 | 0.591673 | 0.967328 | fail: dCor |

Both global structural receipts passed. Repeat determinism and row-order invariance passed for both fields, and the mechanical recipient displacement remained bitwise unchanged. All p99/tail structural checks passed. The post-scale residual mean and raw-z correlation were approximately `1e-15` or smaller in every cell, comfortably below `1e-10`; thus the remaining failures are cross-fitted dependence failures, not failures of the exact FWL algebra.

## Access boundary

`source_labels_opened=false`, `source_labels_hashed=false`, `source_scores_computed=false`, and `training_started=false`. `box_labels_read=[]` and `protected_splits_read=[]`. The large feasibility payload is intentionally not committed; this directory records its immutable hash, compact metrics, and the copied signed-off receipt.
