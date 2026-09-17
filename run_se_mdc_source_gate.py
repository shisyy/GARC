#!/usr/bin/env python3
"""Train/evaluate SE-MDC on the frozen Articraft 13/6 source gate."""

from __future__ import annotations

import argparse
import copy
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

from run_cr_fpl_source_gate import normalize_source_for_requery  # noqa: E402
from run_glpdt_source_gate import CONFIG, load_source, sha256_file  # noqa: E402
from splart.cr_fpl import CRFPLHead  # noqa: E402
from splart.se_mdc import (  # noqa: E402
    SE_MDC_EPSILON,
    SEMDCHead,
    fixed_point_loss_terms,
)


FORBIDDEN_MARKERS = ("sealed", "b_test", "full22", "box_e", "box_f")
SOURCE_CR_FPL_REPORT_SHA256 = "647f846262e9a46d0882c21f91a259bb94e5afec98370cd224f90355c5092fc7"
SOURCE_CR_FPL_CHECKPOINT_SHA256 = "7d81987c6aad190922d95d183ab823811db76afb2670adc945fd9dfba84fb3d7"
ARCHITECTURES = ("cr_fpl", "se_mdc", "se_mdc_independent")
LOSSES = ("original", "minimax")


def _nmae(prediction: Tensor, target: Tensor) -> float:
    return float((prediction - target).abs().mean())


def _side_nmae(prediction: Tensor, target: Tensor) -> list[float]:
    return [float(value) for value in (prediction - target).abs().mean(dim=0)]


def _target_relative_p99(prediction: Tensor, target: Tensor) -> float:
    relative = (prediction - target).abs() / target.clamp_min(SE_MDC_EPSILON)
    return float(torch.quantile(relative.flatten(), 0.99))


def _object_groups(object_ids: tuple[str, ...]) -> list[list[int]]:
    return [
        [index for index, value in enumerate(object_ids) if value == object_id]
        for object_id in sorted(set(object_ids))
    ]


def tail_metrics(
    prediction: Tensor, target: Tensor, object_ids: tuple[str, ...], prefix: str
) -> dict[str, Any]:
    signed_relative = (prediction - target) / target.clamp_min(SE_MDC_EPSILON)
    absolute_relative = signed_relative.abs()
    absolute_log_ratio = (
        torch.log(prediction.clamp_min(SE_MDC_EPSILON))
        - torch.log(target.clamp_min(SE_MDC_EPSILON))
    ).abs()
    per_side_p99 = [
        float(torch.quantile(absolute_relative[:, side], 0.99)) for side in range(2)
    ]
    log_q99 = [
        float(torch.quantile(absolute_log_ratio[:, side], 0.99)) for side in range(2)
    ]
    log_max = [float(absolute_log_ratio[:, side].max()) for side in range(2)]
    signed_q01 = [
        float(torch.quantile(signed_relative[:, side], 0.01)) for side in range(2)
    ]
    signed_q99 = [
        float(torch.quantile(signed_relative[:, side], 0.99)) for side in range(2)
    ]
    object_p99 = []
    object_log_q99 = []
    for indices in _object_groups(object_ids):
        object_p99.append(
            torch.stack(
                [torch.quantile(absolute_relative[indices, side], 0.99) for side in range(2)]
            )
        )
        object_log_q99.append(
            torch.stack(
                [torch.quantile(absolute_log_ratio[indices, side], 0.99) for side in range(2)]
            )
        )
    macro_p99 = torch.stack(object_p99).mean(0)
    macro_log_q99 = torch.stack(object_log_q99).mean(0)
    return {
        f"{prefix}_target_relative_p99_error": float(
            torch.quantile(absolute_relative.flatten(), 0.99)
        ),
        f"{prefix}_target_relative_p99_per_side": per_side_p99,
        f"{prefix}_target_relative_p99_max_side": max(per_side_p99),
        f"{prefix}_absolute_log_ratio_q99_per_side": log_q99,
        f"{prefix}_absolute_log_ratio_q99_max_side": max(log_q99),
        f"{prefix}_absolute_log_ratio_max_per_side": log_max,
        f"{prefix}_absolute_log_ratio_max": max(log_max),
        f"{prefix}_object_macro_target_relative_p99_per_side": [
            float(value) for value in macro_p99
        ],
        f"{prefix}_object_macro_target_relative_p99_max_side": float(
            macro_p99.max()
        ),
        f"{prefix}_object_macro_absolute_log_ratio_q99_per_side": [
            float(value) for value in macro_log_q99
        ],
        f"{prefix}_signed_relative_q01_per_side": signed_q01,
        f"{prefix}_signed_relative_q99_per_side": signed_q99,
    }


def paired_object_wins(
    candidate: Tensor,
    reference: Tensor,
    target: Tensor,
    object_ids: tuple[str, ...],
) -> dict[str, int]:
    wins = {"candidate": 0, "reference": 0, "ties": 0}
    for indices in _object_groups(object_ids):
        candidate_error = float((candidate[indices] - target[indices]).abs().mean())
        reference_error = float((reference[indices] - target[indices]).abs().mean())
        if abs(candidate_error - reference_error) <= 1e-12:
            wins["ties"] += 1
        elif candidate_error < reference_error:
            wins["candidate"] += 1
        else:
            wins["reference"] += 1
    return wins


def dual_risk_diagnostics(
    output: Any, target: Tensor, object_ids: tuple[str, ...]
) -> dict[str, Any]:
    """Measure whether held-out dual mass selects the empirically harder side."""

    loop_error = (
        torch.log(output.loop_distances.clamp_min(SE_MDC_EPSILON))
        - torch.log(target.clamp_min(SE_MDC_EPSILON)).unsqueeze(-1)
    ).abs()
    grouped_error, grouped_allocation = [], []
    for indices in _object_groups(object_ids):
        grouped_error.append(loop_error[indices].mean(dim=0))
        grouped_allocation.append(output.loop_dual_allocation[indices].mean(dim=0))
    error = torch.stack(grouped_error)
    allocation = torch.stack(grouped_allocation)
    hit_rate, correlation = [], []
    for loop in range(error.shape[-1]):
        hit_rate.append(
            float(
                (allocation[..., loop].argmax(dim=1) == error[..., loop].argmax(dim=1))
                .float()
                .mean()
            )
        )
        x = allocation[..., loop].flatten()
        y = error[..., loop].flatten()
        x = x - x.mean()
        y = y - y.mean()
        denominator = x.square().sum().sqrt() * y.square().sum().sqrt()
        correlation.append(
            float((x * y).sum() / denominator) if float(denominator) > 0 else 0.0
        )
    return {
        "heldout_harder_side_allocation_argmax_hit_rate": hit_rate,
        "heldout_allocation_error_pearson": correlation,
    }


def summarize_metrics(
    output: Any,
    anchor: Tensor,
    target: Tensor,
    object_ids: tuple[str, ...],
    prefix: str = "candidate",
) -> dict[str, Any]:
    """Report final, same-parameter first-loop, tail, and loop diagnostics."""

    anchor_side = _side_nmae(anchor, target)
    one_loop = output.loop_distances[..., 0]
    one_loop_side = _side_nmae(one_loop, target)
    final_side = _side_nmae(output.distance, target)
    metrics = {
        "d2_anchor_nmae": _nmae(anchor, target),
        "d2_anchor_side_nmae": anchor_side,
        "d2_anchor_mean_side_nmae": sum(anchor_side) / 2.0,
        "d2_anchor_worst_side_nmae": max(anchor_side),
        "one_loop_same_parameters_nmae": _nmae(one_loop, target),
        "one_loop_same_parameters_side_nmae": one_loop_side,
        "one_loop_same_parameters_mean_side_nmae": sum(one_loop_side) / 2.0,
        "one_loop_same_parameters_worst_side_nmae": max(one_loop_side),
        f"{prefix}_nmae": _nmae(output.distance, target),
        f"{prefix}_side_nmae": final_side,
        f"{prefix}_mean_side_nmae": sum(final_side) / 2.0,
        f"{prefix}_worst_side_nmae": max(final_side),
        "d2_anchor_target_relative_p99_error": _target_relative_p99(anchor, target),
        "one_loop_target_relative_p99_error": _target_relative_p99(one_loop, target),
        "prediction_p99_anchor_ratio": float(
            torch.quantile(
                (output.distance / anchor.clamp_min(SE_MDC_EPSILON)).flatten(), 0.99
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
    metrics.update(tail_metrics(output.distance, target, object_ids, prefix))
    metrics.update(tail_metrics(one_loop, target, object_ids, "one_loop"))
    if hasattr(output, "loop_dual_allocation"):
        metrics.update(
            {
                "loop_dual_allocation_side_mean": output.loop_dual_allocation.mean(0)
                .cpu()
                .tolist(),
                "loop_dual_allocation_entropy_mean": (
                    -(
                        output.loop_dual_allocation
                        * output.loop_dual_allocation.clamp_min(SE_MDC_EPSILON).log()
                    ).sum(dim=1)
                )
                .mean(0)
                .cpu()
                .tolist(),
                "loop_update_budget_mean": output.loop_update_budget.mean(0)
                .cpu()
                .tolist(),
                "loop_coupled_slack_residual_l2_mean": output.loop_coupled_slack_residual_l2.mean(
                    (0, 1)
                )
                .cpu()
                .tolist(),
            }
        )
        metrics.update(dual_risk_diagnostics(output, target, object_ids))
    return metrics


def load_cr_fpl_reference(
    path: Path,
    source_index_sha256: str,
    expected_schema: str,
    expected_report_sha256: str,
) -> dict[str, Any]:
    if sha256_file(path) != expected_report_sha256:
        raise ValueError("matched CR-FPL report digest mismatch")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != expected_schema:
        raise ValueError("unexpected matched CR-FPL report schema")
    if payload.get("source_index", {}).get("sha256") != source_index_sha256:
        raise ValueError("matched CR-FPL report used a different source index")
    if payload.get("protected_splits_read") != [] or payload.get("box_labels_read") not in (
        False,
        0,
        [],
    ):
        raise ValueError("matched CR-FPL report records protected access")
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("matched CR-FPL report has no metrics")
    return metrics


def make_model(architecture: str) -> torch.nn.Module:
    if architecture == "cr_fpl":
        return CRFPLHead()
    if architecture == "se_mdc":
        return SEMDCHead(coupling_mode="coupled")
    if architecture == "se_mdc_independent":
        return SEMDCHead(coupling_mode="independent")
    raise ValueError(f"unknown architecture: {architecture}")


def _output_max_difference(left: Any, right: Any) -> float:
    fields = tuple(type(left).__dataclass_fields__)
    return float(
        max((getattr(left, field) - getattr(right, field)).abs().max().item() for field in fields)
    )


def invariance_diagnostics(
    model: torch.nn.Module,
    features: Tensor,
    coordinates: Tensor,
    anchor: Tensor,
) -> dict[str, float]:
    """Check structural invariance on an isolated deterministic double copy.

    Training and reported predictions remain on the requested CUDA dtype.  The
    audit itself uses a CPU float64 clone so changing only the GEMM batch shape
    cannot create a float32 rounding delta large enough to masquerade as
    cross-example coupling.  A real dependency on another batch member is
    preserved by the clone and still fails the unchanged ``1e-6`` threshold.
    """

    audit_model = copy.deepcopy(model).to(device="cpu", dtype=torch.float64).eval()
    features = features.detach().to(device="cpu", dtype=torch.float64)
    coordinates = coordinates.detach().to(device="cpu", dtype=torch.float64)
    anchor = anchor.detach().to(device="cpu", dtype=torch.float64)
    model = audit_model

    reference = model(features, coordinates, anchor)
    fields = tuple(type(reference).__dataclass_fields__)
    swapped = model(features.flip(1), coordinates.flip(1), anchor.flip(1))
    side_flip = max(
        (getattr(swapped, field) - getattr(reference, field).flip(1)).abs().max().item()
        for field in fields
    )
    permutation = torch.arange(features.shape[0] - 1, -1, -1, device=features.device)
    permuted = model(features[permutation], coordinates[permutation], anchor[permutation])
    batch_permutation = max(
        (getattr(permuted, field) - getattr(reference, field)[permutation]).abs().max().item()
        for field in fields
    )

    split = max(1, features.shape[0] // 3)
    chunks = []
    for start in range(0, features.shape[0], split):
        chunks.append(
            model(
                features[start : start + split],
                coordinates[start : start + split],
                anchor[start : start + split],
            )
        )
    chunking = max(
        (
            torch.cat([getattr(chunk, field) for chunk in chunks], dim=0)
            - getattr(reference, field)
        )
        .abs()
        .max()
        .item()
        for field in fields
    )
    singles = [
        model(features[index : index + 1], coordinates[index : index + 1], anchor[index : index + 1])
        for index in range(features.shape[0])
    ]
    single_vs_batched = max(
        (
            torch.cat([getattr(single, field) for single in singles], dim=0)
            - getattr(reference, field)
        )
        .abs()
        .max()
        .item()
        for field in fields
    )
    diagnostics = {
        "side_flip_max_error": float(side_flip),
        "batch_permutation_max_error": float(batch_permutation),
        "chunking_max_error": float(chunking),
        "single_vs_batched_max_error": float(single_vs_batched),
    }
    if max(diagnostics.values()) > 1e-6:
        raise ValueError(f"strict invariance check failed: {diagnostics}")
    return diagnostics


def load_cr_fpl_checkpoint(
    path: Path,
    device: torch.device,
    validation: tuple,
    reference: dict[str, Any],
    object_macro: bool = False,
    expected_checkpoint_sha256: str | None = None,
) -> tuple[Any, str]:
    checkpoint_sha256 = sha256_file(path.resolve())
    if (
        expected_checkpoint_sha256 is not None
        and checkpoint_sha256 != expected_checkpoint_sha256
    ):
        raise ValueError("CR-FPL checkpoint digest mismatch")
    model = CRFPLHead().to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.eval()
    with torch.no_grad():
        output = model(validation[0], validation[1], validation[2])
    observed_metrics = summarize_metrics(
        output,
        validation[2],
        validation[3],
        validation[4],
        prefix="cr_fpl",
    )
    if object_macro:
        values, one_loop_values = [], []
        for indices in _object_groups(validation[4]):
            values.append(
                (output.distance[indices] - validation[3][indices]).abs().mean(dim=0)
            )
            one_loop_values.append(
                (
                    output.loop_distances[indices, :, 0] - validation[3][indices]
                ).abs().mean(dim=0)
            )
        macro = torch.stack(values).mean(0)
        one_loop_macro = torch.stack(one_loop_values).mean(0)
        observed_metrics.update(
            {
                "cr_fpl_object_macro_endpoint_nmae": float(macro.mean()),
                "cr_fpl_object_macro_side_nmae": [float(value) for value in macro],
                "cr_fpl_object_macro_worst_side_nmae": float(macro.max()),
                "one_loop_same_parameters_object_macro_endpoint_nmae": float(
                    one_loop_macro.mean()
                ),
            }
        )
    required = [
        "cr_fpl_side_nmae",
        "cr_fpl_target_relative_p99_error",
        "one_loop_same_parameters_mean_side_nmae",
        "loop_prediction_residual_mean",
        "loop_state_residual_l2_mean",
        "loop_fixed_point_residual_mean",
        "loop_query_residual_l2_mean",
    ]
    if object_macro:
        required.extend(
            [
                "cr_fpl_object_macro_endpoint_nmae",
                "cr_fpl_object_macro_side_nmae",
                "cr_fpl_object_macro_worst_side_nmae",
                "one_loop_same_parameters_object_macro_endpoint_nmae",
            ]
        )
    else:
        required.extend(["cr_fpl_mean_side_nmae", "cr_fpl_worst_side_nmae"])
    for key in required:
        if key not in reference or key not in observed_metrics:
            raise ValueError(f"CR-FPL binding metric missing: {key}")
        observed_value = torch.as_tensor(observed_metrics[key], dtype=torch.float64)
        reference_value = torch.as_tensor(reference[key], dtype=torch.float64)
        if observed_value.shape != reference_value.shape or not torch.allclose(
            observed_value, reference_value, atol=1e-5, rtol=1e-5
        ):
            raise ValueError(f"CR-FPL checkpoint does not reproduce {key}")
    return output, checkpoint_sha256


def run(
    index_path: Path,
    output_dir: Path,
    device_name: str,
    cr_fpl_report: Path,
    cr_fpl_checkpoint: Path,
    architecture: str,
    loss_name: str,
) -> dict[str, Any]:
    if output_dir.exists():
        raise ValueError("output directory already exists")
    source_index_sha256 = sha256_file(index_path.resolve())
    reference = load_cr_fpl_reference(
        cr_fpl_report.resolve(),
        source_index_sha256,
        "splart-cr-fpl-source-gate/v1",
        SOURCE_CR_FPL_REPORT_SHA256,
    )
    data = normalize_source_for_requery(load_source(index_path))
    device = torch.device(device_name)
    random.seed(CONFIG.seed)
    torch.manual_seed(CONFIG.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(CONFIG.seed)

    model = make_model(architecture).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=CONFIG.learning_rate, weight_decay=CONFIG.weight_decay
    )
    train = tuple(
        value.to(device) if isinstance(value, Tensor) else value
        for value in data["source_train"]
    )
    loss_term_sums: dict[str, Tensor] = {}
    gradient_norm_sum = torch.zeros((), device=device)
    gradient_clip_count = torch.zeros((), device=device)
    for _ in range(CONFIG.steps):
        optimizer.zero_grad(set_to_none=True)
        output = model(train[0], train[1], train[2])
        log_disagreement = (
            torch.log(train[2].clamp_min(SE_MDC_EPSILON)) - torch.log(train[3])
        ).abs()
        violation_target = 1.0 - torch.exp(-log_disagreement)
        loss_terms = fixed_point_loss_terms(
            output,
            train[3],
            violation_target,
            train[4],
            minimax=loss_name == "minimax",
        )
        loss = torch.stack(tuple(loss_terms.values())).sum()
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), CONFIG.gradient_clip_norm
        )
        gradient_norm_sum += gradient_norm
        gradient_clip_count += (gradient_norm > CONFIG.gradient_clip_norm).to(
            gradient_clip_count.dtype
        )
        for name, value in loss_terms.items():
            loss_term_sums[name] = loss_term_sums.get(
                name, torch.zeros((), device=device)
            ) + value.detach()
        optimizer.step()

    validation = tuple(
        value.to(device) if isinstance(value, Tensor) else value
        for value in data["source_validation"]
    )
    model.eval()
    with torch.no_grad():
        output = model(validation[0], validation[1], validation[2])
        metrics = summarize_metrics(
            output, validation[2], validation[3], validation[4]
        )
        metrics["invariance"] = invariance_diagnostics(
            model, validation[0], validation[1], validation[2]
        )
        metrics["exact_swap_error"] = metrics["invariance"]["side_flip_max_error"]
        cr_output, cr_checkpoint_sha256 = load_cr_fpl_checkpoint(
            cr_fpl_checkpoint.resolve(),
            device,
            validation,
            reference,
            expected_checkpoint_sha256=SOURCE_CR_FPL_CHECKPOINT_SHA256,
        )
        metrics.update(
            tail_metrics(
                cr_output.distance,
                validation[3],
                validation[4],
                "matched_cr_fpl",
            )
        )
        metrics["paired_object_wins_vs_matched_cr_fpl"] = paired_object_wins(
            output.distance, cr_output.distance, validation[3], validation[4]
        )
        metrics["paired_object_wins_four_loop_vs_own_one_loop"] = paired_object_wins(
            output.distance,
            output.loop_distances[..., 0],
            validation[3],
            validation[4],
        )
        metrics.update(
            {
                "matched_cr_fpl_mean_side_nmae": float(
                    reference["cr_fpl_mean_side_nmae"]
                ),
                "matched_cr_fpl_worst_side_nmae": float(
                    reference["cr_fpl_worst_side_nmae"]
                ),
                "matched_cr_fpl_target_relative_p99_error": float(
                    reference["cr_fpl_target_relative_p99_error"]
                ),
                "matched_cr_fpl_one_loop_mean_side_nmae": float(
                    reference["one_loop_same_parameters_mean_side_nmae"]
                ),
            }
        )
        report = {
            "schema": "splart-se-mdc-source-gate/v1",
            "config": asdict(CONFIG),
            "experiment_variant": {"architecture": architecture, "loss": loss_name},
            "architecture": {
                "implementation": type(model).__name__,
                "loops": 4,
                "width": 64,
                "coupling": architecture,
                "objective": loss_name,
                "parameter_count": sum(
                    parameter.numel() for parameter in model.parameters()
                ),
            },
            "source_index": {
                "path": str(index_path.resolve()),
                "sha256": source_index_sha256,
            },
            "matched_cr_fpl_reference": {
                "path": str(cr_fpl_report.resolve()),
                "sha256": sha256_file(cr_fpl_report.resolve()),
                "checkpoint_path": str(cr_fpl_checkpoint.resolve()),
                "checkpoint_sha256": cr_checkpoint_sha256,
            },
            "objects": {"train": 13, "validation": 6},
            "primary_metric": "candidate_mean_side_nmae",
            "metrics": metrics,
            "training_diagnostics": {
                "unweighted_loss_term_means": {
                    name: float(value / CONFIG.steps)
                    for name, value in loss_term_sums.items()
                },
                "gradient_norm_mean_before_clip": float(
                    gradient_norm_sum / CONFIG.steps
                ),
                "gradient_clip_rate": float(gradient_clip_count / CONFIG.steps),
                "gradient_clip_norm": CONFIG.gradient_clip_norm,
            },
            "protected_splits_read": [],
            "box_labels_read": False,
            "per_object_predictions_emitted": False,
        }

    output_dir.mkdir(parents=True, exist_ok=False)
    torch.save(model.state_dict(), output_dir / "model.pt")
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cr-fpl-report", type=Path, required=True)
    parser.add_argument("--cr-fpl-checkpoint", type=Path, required=True)
    parser.add_argument("--architecture", choices=ARCHITECTURES, required=True)
    parser.add_argument("--loss", dest="loss_name", choices=LOSSES, required=True)
    args = parser.parse_args()
    launch = "\n".join(str(value) for value in vars(args).values()).lower()
    if any(marker in launch for marker in FORBIDDEN_MARKERS):
        raise ValueError("protected path marker in launch")
    print(
        json.dumps(
            run(
                args.index,
                args.output_dir,
                args.device,
                args.cr_fpl_report,
                args.cr_fpl_checkpoint,
                args.architecture,
                args.loss_name,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
