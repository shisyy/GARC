#!/usr/bin/env python3
"""Independent endpoint evaluator: check sealed predictions BEFORE opening labels."""
import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from run_glpdt_source_gate import SPLITS, validate_index
from splart.clip_limit_gate import _side_nmae, _target_relative_p99_side
from splart.relative_search import METHODS, file_sha256, opaque_object_id, write_json_new


def run(predictions: Path, source_index: Path, output: Path, allow_train_smoke2: bool = False) -> dict:
    seal = json.loads(predictions.with_suffix(".seal.json").read_text(encoding="utf-8"))
    digest = file_sha256(predictions)
    if seal.get("schema") != "relative-search-seal/v1" or seal.get("prediction_sha256") != digest:
        raise ValueError("prediction seal mismatch; refuse endpoint evaluation")
    pred = json.loads(predictions.read_text(encoding="utf-8"))
    source_hash = file_sha256(source_index)
    if source_hash != pred["source_index_sha256"] or source_hash != seal["source_index_sha256"]:
        raise ValueError("sealed source binding mismatch")
    if pred["schema"] != "relative-search-predictions/v1" or pred["methods"] != list(METHODS):
        raise ValueError("all fixed controls are required")
    if pred["implementation_sha256"] != seal["implementation_sha256"]:
        raise ValueError("implementation seal binding mismatch")
    if pred["endpoint_labels_used"] or pred["trained_parameters"] != 0 or pred["exact_swap_error"] != 0:
        raise ValueError("training-free invariant failed")
    by_key = {}
    for row in pred["rows"]:
        key = (row["split"], row["object_id"], row["method"])
        if key in by_key or key[0] not in SPLITS or key[2] not in METHODS:
            raise ValueError("duplicate or unsupported prediction")
        by_key[key] = row
    identities = sorted({key[:2] for key in by_key})
    if not identities or any((s, o, m) not in by_key for s, o in identities for m in METHODS):
        raise ValueError("missing control rows")
    # This is the first operation that opens target-bearing source artifacts.
    source_rows = validate_index(source_index.resolve())
    verified = {(r["split"], opaque_object_id(r["object_id"])): p for r, p in source_rows}
    if allow_train_smoke2:
        smoke = sorted((r for r, _ in source_rows if r["split"] == "source_train"), key=lambda r: r["object_id"])[:2]
        required = {(r["split"], opaque_object_id(r["object_id"])) for r in smoke}
    else:
        required = set(verified)
    if set(identities) != required:
        raise ValueError("evaluation requires all 13/6 identities, or explicitly fixed first-two training smoke")
    targets = {}
    for identity in identities:
        raw = torch.load(verified[identity], map_location="cpu", weights_only=True)
        ep = torch.as_tensor(raw["true_endpoints"], dtype=torch.float32)
        target = torch.stack((-ep[0], ep[1] - 1.0))
        if target.shape != (2,) or not torch.isfinite(target).all() or (target < 0).any():
            raise ValueError("original q=0/1 observed states are not bracketed by endpoints")
        targets[identity] = target
    reports = {}
    for split in SPLITS:
        selected = [identity for identity in identities if identity[0] == split]
        if not selected:
            continue
        target = torch.stack([targets[key] for key in selected])
        reports[split] = {}
        for method in METHODS:
            rows = [by_key[(*key, method)] for key in selected]
            point = torch.tensor([r["distance"] for r in rows])
            if point.shape != target.shape or not torch.isfinite(point).all() or (point < 0).any():
                raise ValueError("invalid point prediction")
            side = _side_nmae(point, target)
            p99 = _target_relative_p99_side(point, target)
            abstain = torch.tensor([[s["abstain"] for s in r["sides"]] for r in rows])
            intervals = torch.tensor([[s["evidence_interval"] for s in r["sides"]] for r in rows])
            interval_contains = (target >= intervals[..., 0]) & (target <= intervals[..., 1])
            error = (point - target).abs()
            reports[split][method] = {
                "objects": len(rows), "side_nmae": side, "mean_side_nmae": sum(side) / 2,
                "worst_side_nmae": max(side), "target_relative_p99_side": p99, "target_relative_p99_max_side": max(p99),
                "unconditional_all_object_score": True, "abstention_fraction": float(abstain.float().mean()),
                "accepted_side_coverage": float((~abstain).float().mean()),
                "accepted_side_nmae_secondary": float(error[~abstain].mean()) if (~abstain).any() else None,
                "heuristic_interval_empirical_containment": float(interval_contains.float().mean()),
                "heuristic_interval_mean_width": float((intervals[..., 1] - intervals[..., 0]).mean()),
                "per_object": [{"object_id": key[1], "prediction": rows[i]["distance"], "target_distance": target[i].tolist(),
                                "absolute_error": error[i].tolist(), "abstain": abstain[i].tolist()} for i, key in enumerate(selected)],
            }
    payload = {"schema": "relative-search-evaluation/v1", "prediction_sha256": digest,
               "source_index_sha256": source_hash, "diagnostic": pred["diagnostic"], "metrics": reports,
               "complete_fixed_source_gate": {s: sum(k[0] == s for k in identities) for s in SPLITS} == SPLITS,
               "gauge": "original observed q0=0 q1=1; no virtual (0.1,0.9) gauge",
               "primary_metric": "source_validation.joint.mean_side_nmae", "metric_direction": "minimize",
               "uncertainty_calibrated": False, "physical_certification": False,
               "protected_splits_read": [], "box_labels_read": False}
    write_json_new(output, payload)
    return payload


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--source-index", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--allow-train-smoke2", action="store_true")
    a = p.parse_args()
    print(json.dumps(run(a.predictions, a.source_index, a.output, a.allow_train_smoke2), sort_keys=True))
