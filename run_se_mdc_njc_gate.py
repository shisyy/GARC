#!/usr/bin/env python3
"""Run SE-MDC on the frozen leakage-safe NJC 11/4 source protocol."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import sys
from typing import Any, Iterable

import torch
from torch import Tensor

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from build_njc_full_trajectory_profiles import (  # noqa: E402
    EXPECTED_SPLIT_EPISODES,
    EXPECTED_SPLIT_OBJECTS,
    FORBIDDEN_PATH_MARKERS,
)
from run_cr_fpl_njc_gate import load_njc  # noqa: E402
from run_cr_fpl_source_gate import CONFIG, normalize_source_for_requery  # noqa: E402
from run_glpdt_source_gate import sha256_file  # noqa: E402
from run_se_mdc_source_gate import (  # noqa: E402
    ARCHITECTURES,
    LOSSES,
    invariance_diagnostics,
    load_cr_fpl_checkpoint,
    load_cr_fpl_reference,
    make_model,
    paired_object_wins,
    summarize_metrics,
    tail_metrics,
)
from splart.se_mdc import (  # noqa: E402
    SE_MDC_EPSILON,
    fixed_point_loss_terms,
)


NJC_CR_FPL_REPORT_SHA256 = "2e07f767c6deba1da83c766c5d7aeb6ae09f69fb4fc91779188a15239f7f9c55"
NJC_CR_FPL_CHECKPOINT_SHA256 = "22d38b52aa77766d775d36c773c664f7690c5807945d2aa3c4ecf2326bd0b4c5"


def _object_macro_side_nmae(
    prediction: Tensor, target: Tensor, object_ids: Iterable[str]
) -> list[float]:
    identifiers = tuple(object_ids)
    if prediction.shape != target.shape or prediction.shape != (len(identifiers), 2):
        raise ValueError("NJC predictions/targets/object IDs are not aligned")
    values = []
    for object_id in sorted(set(identifiers)):
        indices = [index for index, value in enumerate(identifiers) if value == object_id]
        values.append((prediction[indices] - target[indices]).abs().mean(dim=0))
    macro = torch.stack(values).mean(dim=0)
    return [float(value) for value in macro]


def add_object_macro_metrics(
    metrics: dict[str, Any],
    output: Any,
    anchor: Tensor,
    target: Tensor,
    object_ids: Iterable[str],
) -> None:
    predictions = {
        "d2_anchor": anchor,
        "one_loop_same_parameters": output.loop_distances[..., 0],
        "candidate": output.distance,
    }
    for name, prediction in predictions.items():
        side = _object_macro_side_nmae(prediction, target, object_ids)
        metrics[f"{name}_object_macro_side_nmae"] = side
        metrics[f"{name}_object_macro_endpoint_nmae"] = sum(side) / 2.0
        metrics[f"{name}_object_macro_worst_side_nmae"] = max(side)


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
        "splart-cr-fpl-njc-gate/v1",
        NJC_CR_FPL_REPORT_SHA256,
    )
    data = normalize_source_for_requery(load_njc(index_path))
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
    loss_term_sums: dict[str, float] = {}
    gradient_norm_sum = 0.0
    gradient_clip_count = 0
    for _ in range(CONFIG.steps):
        optimizer.zero_grad(set_to_none=True)
        output = model(train[0], train[1], train[2])
        log_disagreement = (
            torch.log(train[2].clamp_min(SE_MDC_EPSILON)) - torch.log(train[3])
        ).abs()
        violation_target = 1.0 - torch.exp(-log_disagreement)
        # train[4] contains 352 repeated object IDs spanning exactly 11 source
        # objects; fixed_point_loss_terms aggregates before the minimax saddle.
        loss_terms = fixed_point_loss_terms(
            output,
            train[3],
            violation_target,
            train[4],
            minimax=loss_name == "minimax",
        )
        loss = torch.stack(tuple(loss_terms.values())).sum()
        loss.backward()
        gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), CONFIG.gradient_clip_norm
            )
        )
        gradient_norm_sum += gradient_norm
        gradient_clip_count += int(gradient_norm > CONFIG.gradient_clip_norm)
        for name, value in loss_terms.items():
            loss_term_sums[name] = loss_term_sums.get(name, 0.0) + float(value.detach())
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
        add_object_macro_metrics(
            metrics, output, validation[2], validation[3], validation[4]
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
            object_macro=True,
            expected_checkpoint_sha256=NJC_CR_FPL_CHECKPOINT_SHA256,
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
                "matched_cr_fpl_object_macro_endpoint_nmae": float(
                    reference["cr_fpl_object_macro_endpoint_nmae"]
                ),
                "matched_cr_fpl_object_macro_worst_side_nmae": float(
                    reference["cr_fpl_object_macro_worst_side_nmae"]
                ),
                "matched_cr_fpl_target_relative_p99_error": float(
                    reference["cr_fpl_target_relative_p99_error"]
                ),
                "matched_cr_fpl_one_loop_object_macro_endpoint_nmae": float(
                    reference["one_loop_same_parameters_object_macro_endpoint_nmae"]
                ),
            }
        )
        report = {
            "schema": "splart-se-mdc-njc-gate/v1",
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
            "objects": dict(EXPECTED_SPLIT_OBJECTS),
            "episodes": dict(EXPECTED_SPLIT_EPISODES),
            "primary_metric": "candidate_object_macro_endpoint_nmae",
            "metrics": metrics,
            "training_diagnostics": {
                "unweighted_loss_term_means": {
                    name: value / CONFIG.steps for name, value in loss_term_sums.items()
                },
                "gradient_norm_mean_before_clip": gradient_norm_sum / CONFIG.steps,
                "gradient_clip_rate": gradient_clip_count / CONFIG.steps,
                "gradient_clip_norm": CONFIG.gradient_clip_norm,
                "minimax_aggregation_objects": 11,
                "minimax_aggregation_episodes": 352,
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
    launch = "\n".join(str(value).lower() for value in vars(args).values())
    if any(marker in launch for marker in FORBIDDEN_PATH_MARKERS):
        raise ValueError("NJC source gate launch contains a protected path marker")
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
