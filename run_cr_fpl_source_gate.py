#!/usr/bin/env python3
"""Train/evaluate CR-FPL on the authorized 13/6 source split only."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import sys
from typing import Any

import torch
from torch import Tensor

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from run_glpdt_source_gate import CONFIG, load_source, sha256_file
from splart.cr_fpl import (
    CR_FPL_EPSILON,
    CRFPLHead,
    cr_fpl_deep_supervision_loss,
    exact_cr_fpl_swap_error,
)


FORBIDDEN_MARKERS = ("sealed", "b_test", "full22", "box_e", "box_f")


def normalize_source_for_requery(data: dict[str, tuple]) -> dict[str, tuple]:
    """Normalize geometry channels while retaining physical query coordinates."""

    train_features = data["source_train"][0]
    feature_mean = train_features.mean(dim=(0, 1, 2, 3))
    feature_std = train_features.var(
        dim=(0, 1, 2, 3), unbiased=False
    ).add(1e-6).sqrt()
    normalized = {}
    for split, (features, coordinates, anchor, target, identifiers) in data.items():
        normalized[split] = (
            (features - feature_mean) / feature_std,
            coordinates,
            anchor,
            target,
            identifiers,
        )
    return normalized


def _nmae(prediction: Tensor, target: Tensor) -> float:
    return float((prediction - target).abs().mean())


def _side_nmae(prediction: Tensor, target: Tensor) -> list[float]:
    return [float(value) for value in (prediction - target).abs().mean(dim=0)]


def _target_relative_p99(prediction: Tensor, target: Tensor) -> float:
    relative = (prediction - target).abs() / target.clamp_min(CR_FPL_EPSILON)
    return float(torch.quantile(relative.flatten(), 0.99))


def summarize_metrics(output: Any, anchor: Tensor, target: Tensor) -> dict[str, Any]:
    """Compare the final fixed point with its same-weights first iterate."""

    anchor_side = _side_nmae(anchor, target)
    one_loop = output.loop_distances[..., 0]
    one_loop_side = _side_nmae(one_loop, target)
    final_side = _side_nmae(output.distance, target)
    return {
        "d2_anchor_nmae": _nmae(anchor, target),
        "d2_anchor_side_nmae": anchor_side,
        "d2_anchor_mean_side_nmae": float(sum(anchor_side) / 2),
        "d2_anchor_worst_side_nmae": max(anchor_side),
        "one_loop_same_parameters_nmae": _nmae(one_loop, target),
        "one_loop_same_parameters_side_nmae": one_loop_side,
        "one_loop_same_parameters_mean_side_nmae": float(sum(one_loop_side) / 2),
        "one_loop_same_parameters_worst_side_nmae": max(one_loop_side),
        "cr_fpl_nmae": _nmae(output.distance, target),
        "cr_fpl_side_nmae": final_side,
        "cr_fpl_mean_side_nmae": float(sum(final_side) / 2),
        "cr_fpl_worst_side_nmae": max(final_side),
        "d2_anchor_target_relative_p99_error": _target_relative_p99(anchor, target),
        "one_loop_target_relative_p99_error": _target_relative_p99(one_loop, target),
        "cr_fpl_target_relative_p99_error": _target_relative_p99(output.distance, target),
        "prediction_p99_anchor_ratio": float(
            torch.quantile(
                (output.distance / anchor.clamp_min(CR_FPL_EPSILON)).flatten(), 0.99
            )
        ),
        "loop_prediction_residual_mean": output.loop_prediction_residuals.mean(
            (0, 1)
        ).cpu().tolist(),
        "loop_state_residual_l2_mean": output.loop_state_residual_l2.mean(
            (0, 1)
        ).cpu().tolist(),
        "loop_fixed_point_residual_mean": output.loop_fixed_point_residuals.mean(
            (0, 1)
        ).cpu().tolist(),
        "loop_query_residual_l2_mean": output.loop_query_residual_l2.mean(
            (0, 1)
        ).cpu().tolist(),
        "loop_convex_weight_mean": output.loop_convex_weights.mean(
            (0, 1)
        ).cpu().tolist(),
        "loop_log_energy_mean": output.loop_log_energy.mean((0, 1)).cpu().tolist(),
        "loop_dual_penalty_mean": output.loop_dual_penalty.mean((0, 1)).cpu().tolist(),
    }


def _load_node815_reference(path: Path, source_index_sha256: str) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "splart-ald-pdl-source-gate/v1":
        raise ValueError("unexpected node 8.15 report schema")
    if payload.get("source_index", {}).get("sha256") != source_index_sha256:
        raise ValueError("node 8.15 report used a different source index")
    metrics = payload.get("metrics", {})
    return {
        "ald_pdl_mean_side_nmae": float(metrics["ald_pdl_mean_side_nmae"]),
        "ald_pdl_worst_side_nmae": float(metrics["ald_pdl_worst_side_nmae"]),
        "ald_pdl_target_relative_p99_error": float(
            metrics["ald_pdl_target_relative_p99_error"]
        ),
        "ald_pdl_one_loop_mean_side_nmae": float(
            metrics["one_loop_same_parameters_mean_side_nmae"]
        ),
    }


def run(
    index_path: Path,
    output_dir: Path,
    device_name: str,
    node815_report: Path | None = None,
) -> dict[str, Any]:
    if output_dir.exists():
        raise ValueError("output directory already exists")
    source_index_sha256 = sha256_file(index_path.resolve())
    data = normalize_source_for_requery(load_source(index_path))
    device = torch.device(device_name)
    random.seed(CONFIG.seed)
    torch.manual_seed(CONFIG.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(CONFIG.seed)

    model = CRFPLHead().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=CONFIG.learning_rate, weight_decay=CONFIG.weight_decay
    )
    train = tuple(
        value.to(device) if isinstance(value, Tensor) else value
        for value in data["source_train"]
    )
    for _ in range(CONFIG.steps):
        optimizer.zero_grad(set_to_none=True)
        output = model(train[0], train[1], train[2])
        log_disagreement = (
            torch.log(train[2].clamp_min(CR_FPL_EPSILON)) - torch.log(train[3])
        ).abs()
        violation_target = 1.0 - torch.exp(-log_disagreement)
        loss = cr_fpl_deep_supervision_loss(output, train[3], violation_target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG.gradient_clip_norm)
        optimizer.step()

    validation = tuple(
        value.to(device) if isinstance(value, Tensor) else value
        for value in data["source_validation"]
    )
    model.eval()
    with torch.no_grad():
        output = model(validation[0], validation[1], validation[2])
        metrics = summarize_metrics(output, validation[2], validation[3])
        metrics["exact_swap_error"] = exact_cr_fpl_swap_error(
            model, validation[0], validation[1], validation[2]
        )
        reference = None
        if node815_report is not None:
            reference = _load_node815_reference(
                node815_report.resolve(), source_index_sha256
            )
            metrics.update(reference)
        report = {
            "schema": "splart-cr-fpl-source-gate/v1",
            "config": asdict(CONFIG),
            "architecture": {
                "loops": 4,
                "width": 64,
                "action_space": "counterfactual_requery_log_ratio_fixed_point",
                "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            },
            "source_index": {
                "path": str(index_path.resolve()),
                "sha256": source_index_sha256,
            },
            "node815_reference": (
                {"path": str(node815_report.resolve()), "metrics": reference}
                if node815_report is not None
                else None
            ),
            "objects": {"train": 13, "validation": 6},
            "metrics": metrics,
            "protected_splits_read": [],
            "box_labels_read": False,
            "per_object_predictions_emitted": False,
        }

    output_dir.mkdir(parents=True, exist_ok=False)
    torch.save(model.state_dict(), output_dir / "cr_fpl.pt")
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--node815-report", type=Path)
    args = parser.parse_args()
    launch = "\n".join(
        str(value)
        for value in (args.index, args.output_dir, args.node815_report)
        if value is not None
    ).lower()
    if any(marker in launch for marker in FORBIDDEN_MARKERS):
        raise ValueError("protected path marker in launch")
    print(
        json.dumps(
            run(args.index, args.output_dir, args.device, args.node815_report),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
