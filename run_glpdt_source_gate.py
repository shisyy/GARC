#!/usr/bin/env python3
"""Train/evaluate GLPDT on the authorized source full-trajectory cache only."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import random
import sys
from typing import Any

import torch
from torch import Tensor

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from splart.gauge_energy_profile import PROFILE_CHANNELS
from splart.glpdt import GLPDTHead, exact_glpdt_swap_error, glpdt_deep_supervision_loss


INDEX_SCHEMA = "splart-source-full-trajectory-index/v1"
ARTIFACT_KEYS = {"channels", "fields", "object_id", "q", "schema", "split", "true_endpoints"}
SPLITS = {"source_train": 13, "source_validation": 6}
FORBIDDEN_MARKERS = ("sealed", "b_test", "full22", "box_e", "box_f")
OBSERVED_GAUGE = (0.1, 0.9)
PROFILE_SAMPLES = 129
POSTERIOR_TEMPERATURE = 0.035


@dataclass(frozen=True)
class SourceGateConfig:
    seed: int = 2202
    steps: int = 4000
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    gradient_clip_norm: float = 1.0


CONFIG = SourceGateConfig()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_path(index_path: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = index_path.parent / path
    resolved = path.resolve()
    if any(marker in str(resolved).lower() for marker in FORBIDDEN_MARKERS):
        raise ValueError("protected path marker in source artifact")
    if resolved.is_symlink() or not resolved.is_file():
        raise ValueError(f"source artifact is not a regular file: {resolved}")
    return resolved


def validate_index(index_path: Path) -> list[tuple[dict[str, Any], Path]]:
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    if payload.get("schema") != INDEX_SCHEMA:
        raise ValueError("unexpected source index schema")
    if payload.get("protected_splits_read") != [] or payload.get("box_labels_read") not in (False, 0):
        raise ValueError("source index records protected access")
    rows = payload.get("rows")
    if not isinstance(rows, list) or len(rows) != sum(SPLITS.values()):
        raise ValueError("source index must contain exactly 19 rows")
    counts = {split: 0 for split in SPLITS}
    identifiers: set[str] = set()
    verified = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"artifact", "artifact_sha256", "object_id", "shape", "split"}:
            raise ValueError("unexpected source row contract")
        split = row["split"]
        if split not in SPLITS:
            raise ValueError("non-source split in source index")
        if row["object_id"] in identifiers:
            raise ValueError("duplicate source object_id")
        identifiers.add(row["object_id"])
        counts[split] += 1
        path = _artifact_path(index_path, row["artifact"])
        if sha256_file(path) != row["artifact_sha256"]:
            raise ValueError("source artifact digest mismatch")
        verified.append((row, path))
    if counts != SPLITS:
        raise ValueError(f"source split counts must be {SPLITS}")
    return verified


def _interpolate_profile(fields: Tensor, q: Tensor, query: Tensor) -> Tensor:
    if q.ndim != 1 or not torch.all(q[1:] > q[:-1]):
        raise ValueError("q must be a strictly increasing vector")
    if query.min() < q[0] or query.max() > q[-1]:
        raise ValueError("canonical source query lies outside cached q")
    upper = torch.searchsorted(q, query).clamp(1, q.numel() - 1)
    lower = upper - 1
    weight = (query - q[lower]) / (q[upper] - q[lower])
    left = fields[:, :, lower, :]
    right = fields[:, :, upper, :]
    return left + weight.view(1, 1, -1, 1) * (right - left)


def virtual_profile(artifact: dict[str, Any]) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    if set(artifact) != ARTIFACT_KEYS:
        raise ValueError("unexpected source artifact contract")
    if artifact["split"] not in SPLITS or tuple(artifact["channels"]) != PROFILE_CHANNELS:
        raise ValueError("source artifact split/channels mismatch")
    q = torch.as_tensor(artifact["q"], dtype=torch.float32)
    fields = torch.as_tensor(artifact["fields"], dtype=torch.float32)
    endpoints = torch.as_tensor(artifact["true_endpoints"], dtype=torch.float32)
    if fields.ndim != 5 or fields.shape[1:3] != (2, 3) or fields.shape[3] != q.numel() or fields.shape[4] != 9:
        raise ValueError("fields must be [G,2,3,Q,9]")
    if endpoints.shape != (2,) or not torch.isfinite(fields).all() or not torch.isfinite(endpoints).all():
        raise ValueError("invalid source tensors")

    gauge_low, gauge_high = OBSERVED_GAUGE
    interval = gauge_high - gauge_low
    lower_grid = torch.linspace(-2.0, 0.0, PROFILE_SAMPLES + 1)[:-1]
    upper_grid = torch.linspace(1.0, 3.0, PROFILE_SAMPLES + 1)[1:]
    side_queries = (gauge_low + interval * lower_grid, gauge_low + interval * upper_grid)
    side_features, side_coordinates, anchor_scalars = [], [], []
    for side, query in enumerate(side_queries):
        sampled = _interpolate_profile(fields[:, side], q, query)
        posterior = torch.softmax(-sampled[..., 7] / POSTERIOR_TEMPERATURE, dim=-1)
        sampled = sampled.clone()
        sampled[..., 8] = posterior
        side_features.append(sampled.mean(0))
        outward = (gauge_low - query) / interval if side == 0 else (query - gauge_high) / interval
        side_coordinates.append(outward.expand(fields.shape[2], -1))
        anchor_scalars.append((posterior * query.view(1, 1, -1)).sum(-1).mean())
    features = torch.stack(side_features)
    coordinates = torch.stack(side_coordinates)
    anchor = torch.stack(((gauge_low - anchor_scalars[0]) / interval, (anchor_scalars[1] - gauge_high) / interval))
    target = torch.stack(((gauge_low - endpoints[0]) / interval, (endpoints[1] - gauge_high) / interval))
    if (anchor < 0).any() or (target < 0).any():
        raise ValueError("source endpoints do not enclose the observed gauge")
    return features, coordinates, anchor, target


def load_source(index_path: Path) -> dict[str, tuple[Tensor, Tensor, Tensor, Tensor, tuple[str, ...]]]:
    groups: dict[str, list[tuple[Tensor, Tensor, Tensor, Tensor, str]]] = {key: [] for key in SPLITS}
    for row, path in validate_index(index_path.resolve()):
        artifact = torch.load(path, map_location="cpu", weights_only=True)
        if artifact.get("object_id") != row["object_id"] or artifact.get("split") != row["split"]:
            raise ValueError("source artifact identity mismatch")
        groups[row["split"]].append((*virtual_profile(artifact), row["object_id"]))
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


def normalize_source(data: dict[str, tuple]) -> dict[str, tuple]:
    train_features, train_coordinates = data["source_train"][:2]
    feature_mean = train_features.mean(dim=(0, 1, 2, 3))
    feature_std = train_features.var(dim=(0, 1, 2, 3), unbiased=False).add(1e-6).sqrt()
    coordinate_mean = train_coordinates.mean()
    coordinate_std = train_coordinates.var(unbiased=False).add(1e-6).sqrt()
    normalized = {}
    for split, (features, coordinates, anchor, target, identifiers) in data.items():
        normalized[split] = (
            (features - feature_mean) / feature_std,
            (coordinates - coordinate_mean) / coordinate_std,
            anchor,
            target,
            identifiers,
        )
    return normalized


def _nmae(prediction: Tensor, target: Tensor) -> float:
    return float((prediction - target).abs().mean())


def run(index_path: Path, output_dir: Path, device_name: str) -> dict[str, Any]:
    if output_dir.exists():
        raise ValueError("output directory already exists")
    data = normalize_source(load_source(index_path))
    device = torch.device(device_name)
    random.seed(CONFIG.seed)
    torch.manual_seed(CONFIG.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(CONFIG.seed)
    model = GLPDTHead().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG.learning_rate, weight_decay=CONFIG.weight_decay)
    train = tuple(value.to(device) if isinstance(value, Tensor) else value for value in data["source_train"])
    for _ in range(CONFIG.steps):
        optimizer.zero_grad(set_to_none=True)
        output = model(train[0], train[1], train[2])
        violation_target = 1.0 - torch.exp(-(train[2] - train[3]).abs())
        loss = glpdt_deep_supervision_loss(output, train[3], violation_target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG.gradient_clip_norm)
        optimizer.step()

    validation = tuple(value.to(device) if isinstance(value, Tensor) else value for value in data["source_validation"])
    model.eval()
    with torch.no_grad():
        output = model(validation[0], validation[1], validation[2])
        ratio = output.distance.flatten() / validation[2].clamp_min(1e-6).flatten()
        report = {
            "schema": "splart-glpdt-source-gate/v1",
            "config": asdict(CONFIG),
            "source_index": {"path": str(index_path.resolve()), "sha256": sha256_file(index_path.resolve())},
            "objects": {"train": 13, "validation": 6},
            "metrics": {
                "d2_anchor_nmae": _nmae(validation[2], validation[3]),
                "glpdt_nmae": _nmae(output.distance, validation[3]),
                "prediction_p99_anchor_ratio": float(torch.quantile(ratio, 0.99)),
                "exact_swap_error": exact_glpdt_swap_error(model, validation[0], validation[1], validation[2]),
                "loop_prediction_residual_mean": output.loop_prediction_residuals.mean((0, 1)).cpu().tolist(),
                "loop_state_residual_l2_mean": output.loop_state_residual_l2.mean((0, 1)).cpu().tolist(),
            },
            "protected_splits_read": [],
            "box_labels_read": False,
            "per_object_predictions_emitted": False,
        }
    output_dir.mkdir(parents=True, exist_ok=False)
    torch.save(model.state_dict(), output_dir / "glpdt.pt")
    (output_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    launch = "\n".join((str(args.index), str(args.output_dir))).lower()
    if any(marker in launch for marker in FORBIDDEN_MARKERS):
        raise ValueError("protected path marker in launch")
    print(json.dumps(run(args.index, args.output_dir, args.device), sort_keys=True))


if __name__ == "__main__":
    main()
