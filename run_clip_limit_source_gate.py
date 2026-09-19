#!/usr/bin/env python3
"""Train/evaluate CSTR on the unchanged Articraft 13/6 source gate."""

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

from run_cr_fpl_source_gate import CONFIG, normalize_source_for_requery
from run_glpdt_source_gate import FORBIDDEN_MARKERS, SPLITS, load_source, sha256_file, validate_index
from splart.clip_limit import CSTRHead, c_clip_ld_deep_supervision_loss, exact_cstr_swap_error
from splart.clip_limit_cache import load_semantic_cache
from splart.clip_limit_gate import (
    CONTROLS,
    apply_semantic_control,
    normalize_semantic_by_train_rms,
    semantic_endpoint_auroc,
    summarize_candidate,
    summarize_cr_fpl_control,
)
from splart.cr_fpl import CR_FPL_EPSILON, CRFPLHead, exact_cr_fpl_swap_error


def _load_cr_fpl_reference(path: Path, index_sha256: str) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "splart-cr-fpl-source-gate/v1":
        raise ValueError("unexpected CR-FPL source report schema")
    if payload.get("source_index", {}).get("sha256") != index_sha256:
        raise ValueError("CR-FPL reference used a different source index")
    metrics = payload.get("metrics", {})
    return {
        "mean_side_nmae": float(metrics["cr_fpl_mean_side_nmae"]),
        "worst_side_nmae": float(metrics["cr_fpl_worst_side_nmae"]),
        "target_relative_p99": float(metrics["cr_fpl_target_relative_p99_error"]),
        "one_loop_mean_side_nmae": float(metrics["one_loop_same_parameters_mean_side_nmae"]),
    }


def _joined_data(geometry_index: Path, semantic_index: Path, control: str) -> dict[str, tuple[Any, ...]]:
    index_sha256 = sha256_file(geometry_index.resolve())
    rows = validate_index(geometry_index.resolve())
    expected = {(row["split"], row["object_id"], row["object_id"]) for row, _ in rows}
    geometry = normalize_source_for_requery(load_source(geometry_index))
    semantic = load_semantic_cache(
        semantic_index, source_index_sha256=index_sha256, expected_split_counts=SPLITS, expected_identities=expected
    )
    controlled = {}
    for split, (evidence, coordinates, object_ids, episode_ids) in semantic.items():
        evidence, coordinates = apply_semantic_control(evidence, coordinates, object_ids, episode_ids, control)
        controlled[split] = (evidence, coordinates, object_ids, episode_ids)
    semantic = normalize_semantic_by_train_rms(controlled)
    joined = {}
    for split, values in geometry.items():
        evidence, semantic_coordinates, object_ids, episode_ids = semantic[split]
        if values[4] != object_ids or values[4] != episode_ids:
            raise ValueError("Articraft semantic trajectories are not aligned to geometry")
        joined[split] = (*values, evidence, semantic_coordinates, episode_ids)
    return joined


def run(
    index_path: Path,
    semantic_index: Path,
    output_dir: Path,
    device_name: str,
    control: str,
    cr_fpl_report: Path,
    cr_fpl_checkpoint: Path,
) -> dict[str, Any]:
    if output_dir.exists():
        raise ValueError("output directory already exists")
    if control not in CONTROLS:
        raise ValueError(f"control must be one of {CONTROLS}")
    if cr_fpl_checkpoint.is_symlink() or not cr_fpl_checkpoint.is_file():
        raise ValueError("CR-FPL checkpoint must be a regular non-symlink file")
    index_sha256 = sha256_file(index_path.resolve())
    reference = _load_cr_fpl_reference(cr_fpl_report.resolve(), index_sha256)
    data = _joined_data(index_path, semantic_index, control)
    device = torch.device(device_name)
    random.seed(CONFIG.seed)
    torch.manual_seed(CONFIG.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(CONFIG.seed)
    model = CSTRHead().to(device)
    baseline = CRFPLHead().to(device)
    baseline.load_state_dict(
        torch.load(cr_fpl_checkpoint.resolve(), map_location=device, weights_only=True), strict=True
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG.learning_rate, weight_decay=CONFIG.weight_decay)
    train = tuple(value.to(device) if isinstance(value, Tensor) else value for value in data["source_train"])
    for _ in range(CONFIG.steps):
        optimizer.zero_grad(set_to_none=True)
        output = model(train[0], train[1], train[2], train[5], train[6])
        log_disagreement = (torch.log(train[2].clamp_min(CR_FPL_EPSILON)) - torch.log(train[3])).abs()
        violation_target = 1.0 - torch.exp(-log_disagreement)
        loss = c_clip_ld_deep_supervision_loss(output, train[3], violation_target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG.gradient_clip_norm)
        optimizer.step()

    validation = tuple(value.to(device) if isinstance(value, Tensor) else value for value in data["source_validation"])
    model.eval()
    with torch.no_grad():
        output = model(validation[0], validation[1], validation[2], validation[5], validation[6])
        metrics = summarize_candidate(output, validation[2], validation[3])
        baseline_output = baseline.eval()(validation[0], validation[1], validation[2])
        metrics.update(summarize_cr_fpl_control(baseline_output, validation[3]))
        metrics["cr_fpl_control_exact_swap_error"] = exact_cr_fpl_swap_error(
            baseline, validation[0], validation[1], validation[2]
        )
        metrics["exact_swap_error"] = exact_cstr_swap_error(
            model, validation[0], validation[1], validation[2], validation[5], validation[6]
        )
        metrics["cstr_residual_endpoint_auroc"] = semantic_endpoint_auroc(validation[5], validation[6], validation[3])
        report = {
            "schema": "splart-cstr-source-gate/v1",
            "config": asdict(CONFIG),
            "control": control,
            "architecture": {
                "implementation": "splart.clip_limit.CSTRHead",
                "loops": 4,
                "width": 64,
                "semantic_channels": 3,
                "semantic_query": "sign_invariant_CSTR_channels_at_current_distance",
                "tangent_fit": "per_object_side_first_17_samples_distance_le_0.25",
                "residual_energy_window": 9,
                "semantic_gate": "bias_free_zero_preserving_residual",
                "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            },
            "source_index": {"path": str(index_path.resolve()), "sha256": index_sha256},
            "semantic_index": {"path": str(semantic_index.resolve()), "sha256": sha256_file(semantic_index.resolve())},
            "cr_fpl_reference": {
                "path": str(cr_fpl_report.resolve()),
                "metrics": reference,
                "checkpoint": str(cr_fpl_checkpoint.resolve()),
                "checkpoint_sha256": sha256_file(cr_fpl_checkpoint.resolve()),
            },
            "objects": {"train": 13, "validation": 6},
            "primary_metric": "candidate_mean_side_nmae",
            "metrics": metrics,
            "protected_splits_read": [],
            "box_labels_read": False,
            "per_object_predictions_emitted": False,
        }
    output_dir.mkdir(parents=True, exist_ok=False)
    torch.save(model.state_dict(), output_dir / "cstr.pt")
    (output_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--semantic-index", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--control", choices=CONTROLS, default="semantic")
    parser.add_argument("--cr-fpl-report", type=Path, required=True)
    parser.add_argument("--cr-fpl-checkpoint", type=Path, required=True)
    args = parser.parse_args()
    launch = "\n".join(str(value).lower() for value in vars(args).values())
    if any(marker in launch for marker in FORBIDDEN_MARKERS):
        raise ValueError("CSTR source launch contains a protected path marker")
    print(
        json.dumps(
            run(
                args.index,
                args.semantic_index,
                args.output_dir,
                args.device,
                args.control,
                args.cr_fpl_report,
                args.cr_fpl_checkpoint,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
