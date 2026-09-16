#!/usr/bin/env python3
"""Run unchanged CR-FPL on the frozen 11/4 NJC source-only protocol."""

from __future__ import annotations

import argparse
from collections import Counter
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

from build_njc_full_trajectory_profiles import (
    ARTIFACT_SCHEMA,
    AUTHORIZED_HASHES,
    EXPECTED_SPLIT_EPISODES,
    EXPECTED_SPLIT_OBJECTS,
    FORBIDDEN_PATH_MARKERS,
    PROFILE_CHANNELS,
    SCHEMA,
)
from run_cr_fpl_source_gate import CONFIG, normalize_source_for_requery, summarize_metrics
from run_glpdt_source_gate import ARTIFACT_KEYS, sha256_file, virtual_profile
from splart.cr_fpl import CR_FPL_EPSILON, CRFPLHead, cr_fpl_deep_supervision_loss, exact_cr_fpl_swap_error


ROW_KEYS = {"artifact", "artifact_sha256", "episode_id", "object_id", "shape", "split"}
NJC_ARTIFACT_KEYS = ARTIFACT_KEYS | {"episode_id"}


def _artifact_path(index_path: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = index_path.parent / path
    if path.is_symlink():
        raise ValueError(f"symlinks are not allowed: {path}")
    resolved = path.resolve()
    if any(marker in str(resolved).lower() for marker in FORBIDDEN_PATH_MARKERS):
        raise ValueError("protected path marker in NJC source artifact")
    if not resolved.is_file():
        raise ValueError(f"NJC artifact is not a regular file: {resolved}")
    return resolved


def validate_njc_index(index_path: Path, expected_samples: int = 513) -> list[tuple[dict[str, Any], Path]]:
    if index_path.is_symlink() or not index_path.is_file():
        raise ValueError("NJC index must be a regular non-symlink file")
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA:
        raise ValueError("unexpected NJC trajectory index schema")
    if payload.get("protected_splits_read") != [] or payload.get("box_labels_read") not in (False, 0, []):
        raise ValueError("NJC index records protected access")
    if payload.get("source_hashes") != AUTHORIZED_HASHES:
        raise ValueError("NJC index is not bound to the authorized source hashes")
    if payload.get("channels") != list(PROFILE_CHANNELS):
        raise ValueError("NJC index channel contract mismatch")
    if payload.get("trajectory") != {"samples": expected_samples, "radii": 3, "channels": 9}:
        raise ValueError("NJC index is not the required trajectory shape")
    objects = payload.get("objects")
    if not isinstance(objects, dict):
        raise ValueError("NJC index has no object split")
    if {key: len(value) for key, value in objects.items()} != EXPECTED_SPLIT_OBJECTS:
        raise ValueError("NJC index is not the frozen 11/4 object split")
    if set(objects["source_train"]) & set(objects["source_validation"]):
        raise ValueError("NJC train/validation objects overlap")
    if payload.get("episode_counts") != EXPECTED_SPLIT_EPISODES:
        raise ValueError("NJC index is not the frozen 352/128 episode split")
    rows = payload.get("rows")
    if not isinstance(rows, list) or len(rows) != sum(EXPECTED_SPLIT_EPISODES.values()):
        raise ValueError("NJC index must contain exactly 480 episodes")
    seen_episodes: set[str] = set()
    per_object: Counter[tuple[str, str]] = Counter()
    verified = []
    object_sets = {split: set(values) for split, values in objects.items()}
    for row in rows:
        if not isinstance(row, dict) or set(row) != ROW_KEYS:
            raise ValueError("unexpected NJC trajectory row contract")
        split, object_id, episode_id = (row["split"], row["object_id"], row["episode_id"])
        if split not in EXPECTED_SPLIT_EPISODES or object_id not in object_sets[split]:
            raise ValueError("NJC episode is outside the frozen object split")
        if (
            not isinstance(episode_id, str)
            or len(episode_id) != 64
            or any(value not in "0123456789abcdef" for value in episode_id)
            or episode_id in seen_episodes
        ):
            raise ValueError("NJC episode IDs must be unique opaque SHA256 digests")
        seen_episodes.add(episode_id)
        if row["shape"] != [2, 2, 3, expected_samples, 9]:
            raise ValueError("NJC artifact row has the wrong shape")
        path = _artifact_path(index_path, row["artifact"])
        if sha256_file(path) != row["artifact_sha256"]:
            raise ValueError("NJC artifact digest mismatch")
        per_object[(split, object_id)] += 1
        verified.append((row, path))
    if set(per_object.values()) != {32} or len(per_object) != sum(EXPECTED_SPLIT_OBJECTS.values()):
        raise ValueError("each NJC object must contribute exactly 32 episodes")
    return verified


def load_njc(
    index_path: Path, expected_samples: int = 513
) -> dict[str, tuple[Tensor, Tensor, Tensor, Tensor, tuple[str, ...]]]:
    groups: dict[str, list[tuple[Tensor, Tensor, Tensor, Tensor, str, str]]] = {
        key: [] for key in EXPECTED_SPLIT_EPISODES
    }
    for row, path in validate_njc_index(index_path.resolve(), expected_samples):
        artifact = torch.load(path, map_location="cpu", weights_only=True)
        if set(artifact) != NJC_ARTIFACT_KEYS:
            raise ValueError("unexpected NJC artifact contract")
        if (
            artifact.get("schema") != ARTIFACT_SCHEMA
            or artifact.get("episode_id") != row["episode_id"]
            or artifact.get("object_id") != row["object_id"]
            or artifact.get("split") != row["split"]
        ):
            raise ValueError("NJC artifact identity mismatch")
        compatible = {key: artifact[key] for key in ARTIFACT_KEYS}
        features, coordinates, anchor, target = virtual_profile(compatible)
        groups[row["split"]].append((features, coordinates, anchor, target, row["object_id"], row["episode_id"]))
    result = {}
    for split, values in groups.items():
        values.sort(key=lambda value: value[-1])
        result[split] = (
            torch.stack([value[0] for value in values]),
            torch.stack([value[1] for value in values]),
            torch.stack([value[2] for value in values]),
            torch.stack([value[3] for value in values]),
            tuple(value[4] for value in values),
        )
    return result


def _object_macro_side_nmae(prediction: Tensor, target: Tensor, object_ids: Iterable[str]) -> list[float]:
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
    metrics: dict[str, Any], output: Any, anchor: Tensor, target: Tensor, object_ids: Iterable[str]
) -> None:
    predictions = {
        "d2_anchor": anchor,
        "one_loop_same_parameters": output.loop_distances[..., 0],
        "cr_fpl": output.distance,
    }
    for name, prediction in predictions.items():
        side = _object_macro_side_nmae(prediction, target, object_ids)
        metrics[f"{name}_object_macro_side_nmae"] = side
        metrics[f"{name}_object_macro_endpoint_nmae"] = sum(side) / 2.0
        metrics[f"{name}_object_macro_worst_side_nmae"] = max(side)


def run(index_path: Path, output_dir: Path, device_name: str) -> dict[str, Any]:
    if output_dir.exists():
        raise ValueError("output directory already exists")
    data = normalize_source_for_requery(load_njc(index_path))
    device = torch.device(device_name)
    random.seed(CONFIG.seed)
    torch.manual_seed(CONFIG.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(CONFIG.seed)
    model = CRFPLHead().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG.learning_rate, weight_decay=CONFIG.weight_decay)
    train = tuple(value.to(device) if isinstance(value, Tensor) else value for value in data["source_train"])
    for _ in range(CONFIG.steps):
        optimizer.zero_grad(set_to_none=True)
        output = model(train[0], train[1], train[2])
        log_disagreement = (torch.log(train[2].clamp_min(CR_FPL_EPSILON)) - torch.log(train[3])).abs()
        violation_target = 1.0 - torch.exp(-log_disagreement)
        loss = cr_fpl_deep_supervision_loss(output, train[3], violation_target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG.gradient_clip_norm)
        optimizer.step()
    validation = tuple(value.to(device) if isinstance(value, Tensor) else value for value in data["source_validation"])
    model.eval()
    with torch.no_grad():
        output = model(validation[0], validation[1], validation[2])
        metrics = summarize_metrics(output, validation[2], validation[3])
        add_object_macro_metrics(metrics, output, validation[2], validation[3], validation[4])
        metrics["exact_swap_error"] = exact_cr_fpl_swap_error(model, validation[0], validation[1], validation[2])
        report = {
            "schema": "splart-cr-fpl-njc-gate/v1",
            "config": asdict(CONFIG),
            "architecture": {
                "implementation": "splart.cr_fpl.CRFPLHead",
                "loops": 4,
                "width": 64,
                "action_space": "counterfactual_requery_log_ratio_fixed_point",
                "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            },
            "source_index": {"path": str(index_path.resolve()), "sha256": sha256_file(index_path.resolve())},
            "objects": dict(EXPECTED_SPLIT_OBJECTS),
            "episodes": dict(EXPECTED_SPLIT_EPISODES),
            "primary_metric": "cr_fpl_object_macro_endpoint_nmae",
            "metrics": metrics,
            "protected_splits_read": [],
            "box_labels_read": False,
            "per_object_predictions_emitted": False,
        }
    output_dir.mkdir(parents=True, exist_ok=False)
    torch.save(model.state_dict(), output_dir / "cr_fpl.pt")
    (output_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    launch = "\n".join(str(value).lower() for value in vars(args).values())
    if any(marker in launch for marker in FORBIDDEN_PATH_MARKERS):
        raise ValueError("NJC source gate launch contains a protected path marker")
    print(json.dumps(run(args.index, args.output_dir, args.device), sort_keys=True))


if __name__ == "__main__":
    main()
