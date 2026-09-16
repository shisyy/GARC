#!/usr/bin/env python3
"""Train/evaluate ALD-PDL on the authorized 13/6 source split only."""

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

from run_glpdt_source_gate import CONFIG, load_source, normalize_source, sha256_file
from splart.ald_pdl import (
    ALD_PDL_EPSILON,
    ALDPDLHead,
    ald_pdl_deep_supervision_loss,
    exact_ald_pdl_swap_error,
)


FORBIDDEN_MARKERS = ("sealed", "b_test", "full22", "box_e", "box_f")


def _nmae(prediction: Tensor, target: Tensor) -> float:
    return float((prediction - target).abs().mean())


def _side_nmae(prediction: Tensor, target: Tensor) -> list[float]:
    return [float(value) for value in (prediction - target).abs().mean(dim=0)]


def _target_relative_p99(prediction: Tensor, target: Tensor) -> float:
    relative = (prediction - target).abs() / target.clamp_min(ALD_PDL_EPSILON)
    return float(torch.quantile(relative.flatten(), 0.99))


def summarize_metrics(output: Any, anchor: Tensor, target: Tensor) -> dict[str, Any]:
    """Build the frozen-anchor, four-loop, and same-parameter one-loop comparison."""

    anchor_side = _side_nmae(anchor, target)
    one_loop = output.loop_distances[..., 0]
    one_loop_side = _side_nmae(one_loop, target)
    final_side = _side_nmae(output.distance, target)
    target_log_ratio = torch.log(target) - torch.log(anchor.clamp_min(ALD_PDL_EPSILON))
    representable = torch.isfinite(target_log_ratio)
    return {
        "d2_anchor_nmae": _nmae(anchor, target),
        "d2_anchor_side_nmae": anchor_side,
        "d2_anchor_mean_side_nmae": float(sum(anchor_side) / len(anchor_side)),
        "d2_anchor_worst_side_nmae": max(anchor_side),
        "one_loop_same_parameters_nmae": _nmae(one_loop, target),
        "one_loop_same_parameters_side_nmae": one_loop_side,
        "one_loop_same_parameters_mean_side_nmae": float(sum(one_loop_side) / len(one_loop_side)),
        "one_loop_same_parameters_worst_side_nmae": max(one_loop_side),
        "ald_pdl_nmae": _nmae(output.distance, target),
        "ald_pdl_side_nmae": final_side,
        "ald_pdl_mean_side_nmae": float(sum(final_side) / len(final_side)),
        "ald_pdl_worst_side_nmae": max(final_side),
        "d2_anchor_target_relative_p99_error": _target_relative_p99(anchor, target),
        "one_loop_target_relative_p99_error": _target_relative_p99(one_loop, target),
        "ald_pdl_target_relative_p99_error": _target_relative_p99(output.distance, target),
        "prediction_p99_anchor_ratio": float(
            torch.quantile(
                (output.distance / anchor.clamp_min(ALD_PDL_EPSILON)).flatten(), 0.99
            )
        ),
        "representable_endpoint_count": int(representable.sum()),
        "endpoint_count": int(representable.numel()),
        "loop_prediction_residual_mean": output.loop_prediction_residuals.mean((0, 1)).cpu().tolist(),
        "loop_state_residual_l2_mean": output.loop_state_residual_l2.mean((0, 1)).cpu().tolist(),
        "loop_log_energy_mean": output.loop_log_energy.mean((0, 1)).cpu().tolist(),
        "loop_dual_penalty_mean": output.loop_dual_penalty.mean((0, 1)).cpu().tolist(),
        "loop_primal_update_abs_mean": output.loop_primal_updates.abs().mean((0, 1)).cpu().tolist(),
    }


def _load_node814_reference(path: Path, source_index_sha256: str) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "splart-glpdt-source-gate/v1":
        raise ValueError("unexpected node 8.14 report schema")
    if payload.get("source_index", {}).get("sha256") != source_index_sha256:
        raise ValueError("node 8.14 report used a different source index")
    metrics = payload.get("metrics", {})
    return {
        "glpdt_nmae": float(metrics["glpdt_nmae"]),
        "glpdt_prediction_p99_anchor_ratio": float(metrics["prediction_p99_anchor_ratio"]),
    }


def run(
    index_path: Path,
    output_dir: Path,
    device_name: str,
    node814_report: Path | None = None,
) -> dict[str, Any]:
    if output_dir.exists():
        raise ValueError("output directory already exists")
    source_index_sha256 = sha256_file(index_path.resolve())
    data = normalize_source(load_source(index_path))
    device = torch.device(device_name)
    random.seed(CONFIG.seed)
    torch.manual_seed(CONFIG.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(CONFIG.seed)

    model = ALDPDLHead().to(device)
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
            torch.log(train[2].clamp_min(ALD_PDL_EPSILON)) - torch.log(train[3])
        ).abs()
        violation_target = 1.0 - torch.exp(-log_disagreement)
        loss = ald_pdl_deep_supervision_loss(output, train[3], violation_target)
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
        metrics["exact_swap_error"] = exact_ald_pdl_swap_error(
            model, validation[0], validation[1], validation[2]
        )
        reference = None
        if node814_report is not None:
            reference = _load_node814_reference(node814_report.resolve(), source_index_sha256)
            metrics.update(reference)
        report = {
            "schema": "splart-ald-pdl-source-gate/v1",
            "config": asdict(CONFIG),
            "architecture": {"loops": 4, "width": 64, "action_space": "anchor_log_ratio"},
            "source_index": {
                "path": str(index_path.resolve()),
                "sha256": source_index_sha256,
            },
            "node814_reference": (
                {"path": str(node814_report.resolve()), "metrics": reference}
                if node814_report is not None
                else None
            ),
            "objects": {"train": 13, "validation": 6},
            "metrics": metrics,
            "protected_splits_read": [],
            "box_labels_read": False,
            "per_object_predictions_emitted": False,
        }

    output_dir.mkdir(parents=True, exist_ok=False)
    torch.save(model.state_dict(), output_dir / "ald_pdl.pt")
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--node814-report", type=Path)
    args = parser.parse_args()
    launch = "\n".join(
        str(value)
        for value in (args.index, args.output_dir, args.node814_report)
        if value is not None
    ).lower()
    if any(marker in launch for marker in FORBIDDEN_MARKERS):
        raise ValueError("protected path marker in launch")
    print(
        json.dumps(
            run(args.index, args.output_dir, args.device, args.node814_report),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
