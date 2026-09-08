# Node 8.9 OCR-SMARC source preflight

Decision: **PRUNE_BEFORE_TRAINING** (`PRUNE_NULL_PREFLIGHT`). The preregistered conditional-residual controls did not pass their target-free conditional-independence diagnostics. Per contract, no 2-step smoke, 1200-step source training, Box evaluation, or protected-split read was performed.

## Result boundary

- Frozen implementation commit: `1c3d1c974a350de79d561bd21501539354a9ab74`.
- Frozen normalized config SHA256: `454f4e16af01e4986b9e4534168b6be7bc96eb4617a63bcc988dfdb7841aaca8`.
- Remote immutable artifact: `/data1/public/yptang/splart-node89-ocrsmarc/results/source-preflight-v2`.
- Remote `preflight.json` SHA256: `05e6e8fa6c40bbff49e6f7cc35cd225c6bf56ebcc8f7160cb57874aaf86fa6a3`.
- Remote `receipt.json` SHA256: `bb32098ee59d833c9194e1ffd34a13a5428e3f737d18e6333b582cee5f45bca9`.
- Target-free feature provenance SHA256: `b600bc9469579832ef7d09d1ffa586c4d4d565c302cab1ca1d0109e5db5664d9`.
- Conditional-null receipt SHA256: `2bdff7b18bbc414a05c8136ca2db2dae2a430b7beb15b2728a7f5963fe9c676f`.
- Receipt states `source_labels_opened=false`, `source_scores_computed=false`, `box_labels_read=[]`, `protected_splits_read=[]`, and `target_payloads_read=[]`.

No source NMAE exists for this node: the null gate failed before source labels were opened. Reporting or comparing an endpoint NMAE here would violate the frozen order.

## Target-free null diagnostics

Frozen thresholds were: reconstruction, normalized residual mean, and normalized residual-z correlation at most `1e-10`; residual energy ratio at least `0.05`; shuffle RMS/original SD at least `0.10`; shuffled latent p99/original train p99 at most `1.25`; cross-fit absolute Spearman at most `0.35`; cross-fit distance correlation at most `0.50`; train/held coverage `1.0`, train self-rate `0`, and one effective held donor per held object.

| Null | Domain | Recon max | Residual energy | Shuffle RMS/SD | p99 ratio | Spearman | dCor | Verdict |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Semantic conditional residual | Articraft | 2.78e-17 | 0.8686 | 1.1591 | 1.1601 | 0.1338 | 0.3792 | Pass |
| Semantic conditional residual | NJC | 2.78e-17 | 0.8905 | 1.2934 | 1.0840 | 0.0228 | **0.6566** | Fail: dCor > 0.50 |
| Mechanical conditional residual | Articraft | 1.78e-15 | 0.9342 | 1.0734 | 1.0345 | **0.3589** | 0.3036 | Fail: Spearman > 0.35 |
| Mechanical conditional residual | NJC | 4.44e-16 | 0.6653 | 0.9297 | 0.9796 | 0.0545 | **0.5954** | Fail: dCor > 0.50 |

For all four null/domain combinations, normalized residual mean and z-correlation were below `8.49e-15`, coverage was exactly `1.0`, train self-rate was `0`, donor marginal was exact, held effective-donors/object was `1.0`, and held maximum donor load equaled its bound (`1`). Balanced five-fold counts were `[20,20,19,19,19]` for Articraft and `[3,2,2,2,2]` for NJC. The failure is therefore specifically the predeclared conditional-independence diagnostic, not coverage or numerical plumbing.

## Execution record

The first launch (`source-preflight-v1.log`, SHA256 `77e0c2716748fe7263bac5777ee2a5a702642e7530783c4b06bc7e8db21cfbe7`) used a nonexistent relative config path and exited before config/data access, without an output directory. This is an execution-path error, not a scientific result.

The corrected v2 launch used the absolute synced config path and completed with log SHA256 `6eed03ab6b6231eb59ffc5178c2c9c2b1e3a4cb5354aa8a1ab86f0f785b2f466`. The immutable receipt decision is `PRUNE_NULL_PREFLIGHT`. No GPU training process was started after this decision.
