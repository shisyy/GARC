#!/usr/bin/env python3
"""Materialize the pre-registered NJC episodes as dense D2-style profiles.

This is an offline, supervised source-data builder.  It consumes only the
published NJC assets and the already-authorized PILC input/label sidecars.  No
target-domain or sealed evaluator artifact is part of the interface.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any
import xml.etree.ElementTree as ET

import torch
from torch import Tensor
from torch.nn import functional as F


SCHEMA = "splart-njc-full-trajectory-index/v1"
ARTIFACT_SCHEMA = "splart-njc-full-trajectory/v1"
PROFILE_CHANNELS = (
    "signed_gap",
    "contact_energy",
    "penetration_energy",
    "support_energy",
    "inside_penetration_energy",
    "contact_mass",
    "support_rise",
    "total_energy",
    "posterior",
)
EXPECTED_SPLIT_OBJECTS = {"source_train": 11, "source_validation": 4}
EXPECTED_SPLIT_EPISODES = {"source_train": 352, "source_validation": 128}
FORBIDDEN_PATH_MARKERS = ("sealed", "b_test", "full22", "box_e", "box_f")
AUTHORIZED_HASHES = {
    "preregister": "1928eed04e666e3162d8306ace62a6b195c7202a0ddf84d939dc98d3b1db8f25",
    "inputs": "dcfb65e548ea08b0d71af9f52ebc68b615b1b3dea459605bbefd554393005c9e",
    "labels": "5cf81ecfefcf408e77782c2d65119319f38540d680e9f89783c9f07a6cb7d706",
    "manifest": "c750569d08a54474f821c45f484ea110c846e5f2262d4ec2aacf39c9b78ce91c",
    "qc_receipt": "46b756fac54687813e6ffd8ddc348e9b747e6f88d4d26218fe19aab9069a6f7e",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_regular_file(path: Path) -> Path:
    if path.is_symlink():
        raise ValueError(f"symlinks are not allowed: {path}")
    resolved = path.resolve()
    if not resolved.is_file():
        raise ValueError(f"expected a regular file: {resolved}")
    return resolved


def _verify_hash(path: Path, expected: str, name: str) -> Path:
    path = _require_regular_file(path)
    if sha256_file(path) != expected:
        raise ValueError(f"{name} SHA256 mismatch")
    return path


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _rpy(text: str) -> Tensor:
    roll, pitch, yaw = (float(value) for value in text.split())
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = torch.tensor(((1.0, 0.0, 0.0), (0.0, cr, -sr), (0.0, sr, cr)))
    ry = torch.tensor(((cp, 0.0, sp), (0.0, 1.0, 0.0), (-sp, 0.0, cp)))
    rz = torch.tensor(((cy, -sy, 0.0), (sy, cy, 0.0), (0.0, 0.0, 1.0)))
    return rz @ ry @ rx


def _obj_vertices(path: Path) -> Tensor:
    rows = []
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            if line.startswith("v "):
                rows.append(tuple(float(value) for value in line.split()[1:4]))
    if len(rows) < 4:
        raise ValueError(f"too few mesh vertices: {path}")
    return torch.tensor(rows, dtype=torch.float32)


def _visual_points(link: ET.Element, asset_root: Path) -> Tensor:
    clouds = []
    for visual in link.findall("visual"):
        mesh = visual.find("geometry/mesh")
        if mesh is None:
            continue
        filename = Path(mesh.attrib["filename"].removeprefix("package://")).name
        mesh_path = _require_regular_file(asset_root / filename)
        scale = torch.tensor([float(value) for value in mesh.attrib.get("scale", "1 1 1").split()], dtype=torch.float32)
        cloud = _obj_vertices(mesh_path) * scale
        origin = visual.find("origin")
        rotation = torch.eye(3) if origin is None else _rpy(origin.attrib.get("rpy", "0 0 0"))
        translation = (
            torch.zeros(3)
            if origin is None
            else torch.tensor([float(value) for value in origin.attrib.get("xyz", "0 0 0").split()])
        )
        clouds.append(cloud @ rotation.T + translation)
    if not clouds:
        raise ValueError(f"no usable visual geometry in {asset_root}")
    return torch.cat(clouds)


def _downsample(points: Tensor, count: int) -> Tensor:
    if len(points) <= count:
        return points
    indices = torch.linspace(0, len(points) - 1, count).round().long()
    return points[indices]


def parse_asset(asset_root: Path, points: int) -> dict[str, Any]:
    urdf_path = _require_regular_file(asset_root / "object.urdf")
    robot = ET.parse(urdf_path).getroot()
    links = {node.attrib["name"]: node for node in robot.findall("link")}
    joints = [
        node
        for node in robot.findall("joint")
        if node.attrib.get("type") == "revolute" and node.find("limit") is not None
    ]
    if len(joints) != 1:
        raise ValueError(f"expected one finite revolute joint: {asset_root.name}")
    joint = joints[0]
    parent_name = joint.find("parent").attrib["link"]
    child_name = joint.find("child").attrib["link"]
    origin = joint.find("origin")
    rotation = torch.eye(3) if origin is None else _rpy(origin.attrib.get("rpy", "0 0 0"))
    pivot = (
        torch.zeros(3)
        if origin is None
        else torch.tensor([float(value) for value in origin.attrib.get("xyz", "0 0 0").split()])
    )
    axis_node = joint.find("axis")
    axis = rotation @ torch.tensor(
        [float(value) for value in ("1 0 0" if axis_node is None else axis_node.attrib.get("xyz", "1 0 0")).split()]
    )
    limit = joint.find("limit")
    lower, upper = float(limit.attrib["lower"]), float(limit.attrib["upper"])
    if not upper > lower:
        raise ValueError(f"invalid authored range: {asset_root.name}")
    static = _visual_points(links[parent_name], asset_root)
    mobile_child = _visual_points(links[child_name], asset_root)
    mobile_closed = mobile_child @ rotation.T + pivot
    return {
        "static": _downsample(static, points),
        "mobile": _downsample(mobile_closed, points),
        "axis": axis,
        "pivot": pivot,
        "lower": lower,
        "upper": upper,
        "urdf_sha256": sha256_file(urdf_path),
    }


def _rotate(points: Tensor, axis: Tensor, pivot: Tensor, angle: Tensor) -> Tensor:
    axis = F.normalize(axis, dim=0)
    centered = points - pivot
    cross = torch.linalg.cross(axis.expand_as(centered), centered, dim=-1)
    projection = (centered * axis).sum(-1, keepdim=True) * axis
    cosine, sine = torch.cos(angle)[:, None, None], torch.sin(angle)[:, None, None]
    return centered[None] * cosine + cross[None] * sine + projection[None] * (1.0 - cosine) + pivot


def make_fields(
    static: Tensor,
    mobile: Tensor,
    axis: Tensor,
    pivot: Tensor,
    lower: float,
    upper: float,
    state0_fraction: float,
    state1_fraction: float,
    samples: int,
) -> tuple[Tensor, Tensor]:
    """Build the exact source-transfer 3-radius/9-channel trajectory field."""

    q = torch.linspace(-2.0, 3.0, samples, device=static.device)
    fraction = state0_fraction + q * (state1_fraction - state0_fraction)
    angles = torch.tensor(lower, device=q.device) + fraction * (upper - lower)
    moved = _rotate(mobile, axis, pivot, angles)
    distance = torch.cdist(moved, static).amin(dim=(-1, -2))
    combined = torch.cat((static, mobile))
    extent = torch.linalg.vector_norm(combined.amax(0) - combined.amin(0))
    base_radius = extent / math.sqrt(float(len(static) + len(mobile))) * 0.45
    outputs = []
    for radius_scale in (0.75, 1.0, 1.25):
        band = torch.maximum(base_radius * 0.35 * radius_scale, extent * 0.002)
        gap = distance - 2.0 * base_radius * radius_scale
        normalized_gap = gap / band
        contact = normalized_gap.square()
        penetration = F.relu(-normalized_gap).square()
        mass = torch.exp(-0.5 * normalized_gap.square())
        side_rows = []
        step = max(1, round(0.04 / float(q[1] - q[0])))
        for direction in (-1, 1):
            outside = torch.roll(gap, -direction * step)
            inside = torch.roll(gap, direction * step)
            support_rise = F.relu((gap - outside) / band)
            support = F.relu(0.15 - support_rise).square()
            inside_penetration = F.relu(-inside / band).square()
            total = contact + 6.0 * penetration + 2.0 * support + 5.0 * inside_penetration
            posterior = torch.softmax(-(total - total.min()) / 0.035, dim=0)
            side_rows.append(
                torch.stack(
                    (gap, contact, penetration, support, inside_penetration, mass, support_rise, total, posterior), -1
                )
            )
        outputs.append(torch.stack(side_rows))
    fields = torch.stack(outputs, 1).unsqueeze(0).repeat(2, 1, 1, 1, 1)
    if fields.shape != (2, 2, 3, samples, len(PROFILE_CHANNELS)):
        raise RuntimeError("unexpected trajectory tensor shape")
    return q, fields


def reconstruct_episode_rows(prereg: dict[str, Any], inputs: list[dict], labels: list[dict]) -> list[dict[str, Any]]:
    if len(inputs) != len(labels):
        raise ValueError("input/label sidecars differ in length")
    grid = prereg["episode_grid"]
    expected = []
    cursor = 0
    for prereg_split, output_split in (("train", "source_train"), ("calibration", "source_validation")):
        for object_id in prereg["object_split"][prereg_split]:
            object_count = 0
            for f0 in grid["state0_fraction"]:
                for f1 in grid["state1_fraction"]:
                    if f1 - f0 < grid["minimum_span"]:
                        continue
                    for orientation in grid["axis_orientation_augmentation"]:
                        base = f"{prereg_split}:{object_id}:{f0:.2f}:{f1:.2f}:" f"axis{orientation:+d}"
                        for order in grid["state_order_augmentation"]:
                            if cursor >= len(inputs):
                                raise ValueError("sidecars ended before preregistered grid")
                            input_row, label_row = inputs[cursor], labels[cursor]
                            key = hashlib.sha256(f"splart-pilc-v1:{base}:{order}".encode()).hexdigest()
                            if input_row.get("key") != key or label_row.get("key") != key:
                                raise ValueError("opaque episode key/order mismatch")
                            if input_row.get("split") != prereg_split:
                                raise ValueError("episode split differs from preregistration")
                            extension0 = f0 / (f1 - f0)
                            extension1 = (1.0 - f1) / (f1 - f0)
                            if order == "forward":
                                start, end = f0, f1
                                ext0, ext1, closed = extension0, extension1, 0
                            elif order == "reverse":
                                start, end = f1, f0
                                ext0, ext1, closed = extension1, extension0, 1
                            else:
                                raise ValueError(f"unsupported state order: {order}")
                            actual = (
                                float(label_row["extension0"]),
                                float(label_row["extension1"]),
                                int(label_row["closed_index"]),
                            )
                            if abs(actual[0] - ext0) > 1e-12 or abs(actual[1] - ext1) > 1e-12 or actual[2] != closed:
                                raise ValueError("label sidecar differs from preregistered grid")
                            expected.append(
                                {
                                    "episode_id": key,
                                    "object_id": object_id,
                                    "split": output_split,
                                    "axis_orientation": int(orientation),
                                    "state_order": order,
                                    "state0_fraction": float(start),
                                    "state1_fraction": float(end),
                                    "observed_displacement": float(input_row["observed_displacement"]),
                                    "physical_range": float(label_row["physical_range"]),
                                    "true_endpoints": (-ext0, 1.0 + ext1),
                                }
                            )
                            cursor += 1
                            object_count += 1
            if object_count != 32:
                raise ValueError("each NJC object must materialize exactly 32 episodes")
    if cursor != len(inputs) or len({row["episode_id"] for row in expected}) != cursor:
        raise ValueError("sidecars have extra or duplicate opaque episodes")
    return expected


def _validate_contract(
    asset_root: Path,
    prereg_path: Path,
    inputs_path: Path,
    labels_path: Path,
    manifest_path: Path,
    qc_receipt_path: Path,
    expected_hashes: dict[str, str],
) -> tuple[dict[str, Any], list[dict], list[dict], dict[str, Any]]:
    paths = {
        "preregister": prereg_path,
        "inputs": inputs_path,
        "labels": labels_path,
        "manifest": manifest_path,
        "qc_receipt": qc_receipt_path,
    }
    for name, path in paths.items():
        _verify_hash(path, expected_hashes[name], name)
    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    if prereg.get("schema") != "splart-pilc-preregister-v1":
        raise ValueError("unexpected preregistration schema")
    if set(prereg["object_split"]["train"]) & set(prereg["object_split"]["calibration"]):
        raise ValueError("NJC train/calibration objects overlap")
    if (
        len(prereg["object_split"]["train"]) != EXPECTED_SPLIT_OBJECTS["source_train"]
        or len(prereg["object_split"]["calibration"]) != EXPECTED_SPLIT_OBJECTS["source_validation"]
    ):
        raise ValueError("NJC object split is not the frozen 11/4 contract")
    if any(
        marker in str(value).lower()
        for value in prereg["object_split"].get("target_evaluation_only", [])
        for marker in FORBIDDEN_PATH_MARKERS
    ):
        # Target names are allowed to be declared but never dereferenced.  The
        # current preregistration contains Box-v4/a/b, not forbidden e/f.
        raise ValueError("forbidden target marker in source preregistration")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "splart-pilc-njc-features-v1":
        raise ValueError("unexpected NJC sidecar manifest schema")
    if manifest.get("preregister_sha256") != expected_hashes["preregister"]:
        raise ValueError("manifest does not bind preregistration")
    if (
        manifest.get("materialized", {}).get("train") != prereg["object_split"]["train"]
        or manifest.get("materialized", {}).get("calibration") != prereg["object_split"]["calibration"]
        or manifest.get("missing") != []
        or manifest.get("rows") != sum(EXPECTED_SPLIT_EPISODES.values())
        or manifest.get("feature_dim") != 47
        or manifest.get("separate_sidecars") is not True
        or set(manifest.get("input_fields", []))
        != {"key", "observed_displacement", "split", "state0_features", "state1_features"}
        or set(manifest.get("label_fields", []))
        != {"closed_index", "extension0", "extension1", "key", "physical_range"}
    ):
        raise ValueError("NJC sidecar manifest differs from the frozen contract")
    canonical = dict(manifest)
    recorded_canonical = canonical.pop("canonical_sha256", None)
    if recorded_canonical != _canonical_sha256(canonical):
        raise ValueError("manifest canonical digest mismatch")
    qc = json.loads(qc_receipt_path.read_text(encoding="utf-8"))
    if (
        qc.get("schema") != "splart-pilc-njc-audit-v1"
        or qc.get("status") != "PASS"
        or qc.get("protected_splits_read") != []
        or qc.get("box_extra_scores_read") != []
    ):
        raise ValueError("NJC QC receipt is not source-only PASS")
    for field in ("preregister", "inputs", "labels", "manifest"):
        key = f"{field}_sha256"
        if qc.get(key) != expected_hashes[field]:
            raise ValueError(f"QC receipt does not bind {field}")
    expected_objects = prereg["object_split"]["train"] + prereg["object_split"]["calibration"]
    receipt_assets = {row["name"]: row for row in qc.get("assets", [])}
    if set(receipt_assets) != set(expected_objects):
        raise ValueError("QC asset roster differs from preregistration")
    for object_id in expected_objects:
        root = asset_root / object_id
        receipt = receipt_assets[object_id]
        for filename, receipt_key in (
            ("object.urdf", "urdf_sha256"),
            ("base_final.obj", "base_sha256"),
            ("lid_final.obj", "mobile_sha256"),
            ("qc.json", "qc_sha256"),
        ):
            _verify_hash(root / filename, receipt[receipt_key], f"{object_id}/{filename}")
    inputs = torch.load(inputs_path, map_location="cpu", weights_only=True)
    labels = torch.load(labels_path, map_location="cpu", weights_only=True)
    if not isinstance(inputs, list) or not isinstance(labels, list):
        raise ValueError("NJC sidecars must be lists")
    return prereg, inputs, labels, qc


def build_dataset(
    *,
    asset_root: Path,
    prereg_path: Path,
    inputs_path: Path,
    labels_path: Path,
    manifest_path: Path,
    qc_receipt_path: Path,
    output: Path,
    device_name: str,
    expected_hashes: dict[str, str] = AUTHORIZED_HASHES,
    samples: int = 513,
    points: int = 64,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    if samples < 3 or points < 4:
        raise ValueError("insufficient trajectory samples or mesh points")
    prereg, inputs, labels, _ = _validate_contract(
        asset_root, prereg_path, inputs_path, labels_path, manifest_path, qc_receipt_path, expected_hashes
    )
    episodes = reconstruct_episode_rows(prereg, inputs, labels)
    counts = {split: sum(row["split"] == split for row in episodes) for split in EXPECTED_SPLIT_EPISODES}
    if counts != EXPECTED_SPLIT_EPISODES:
        raise ValueError("NJC episode split is not the frozen 352/128 contract")
    device = torch.device(device_name)
    if device.type == "cuda" and os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("set CUBLAS_WORKSPACE_CONFIG=:4096:8 before CUDA materialization")
    asset_cache = {
        object_id: parse_asset(asset_root / object_id, points)
        for object_id in sorted({row["object_id"] for row in episodes})
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix="njc-full-trajectory-stage-", dir=output.parent))
    try:
        rows = []
        field_cache: dict[tuple[str, float, float], tuple[Tensor, Tensor]] = {}
        for episode in episodes:
            asset = asset_cache[episode["object_id"]]
            if abs((asset["upper"] - asset["lower"]) - episode["physical_range"]) > 1e-5:
                raise ValueError("label physical range differs from authored URDF")
            expected_displacement = (
                (episode["state1_fraction"] - episode["state0_fraction"])
                * episode["physical_range"]
                * episode["axis_orientation"]
            )
            if abs(expected_displacement - episode["observed_displacement"]) > 1e-5:
                raise ValueError("input displacement differs from preregistered episode")
            cache_key = (episode["object_id"], episode["state0_fraction"], episode["state1_fraction"])
            if cache_key not in field_cache:
                field_cache[cache_key] = make_fields(
                    *(asset[name].to(device) for name in ("static", "mobile", "axis", "pivot")),
                    asset["lower"],
                    asset["upper"],
                    episode["state0_fraction"],
                    episode["state1_fraction"],
                    samples,
                )
            q, fields = field_cache[cache_key]
            filename = f"{episode['episode_id']}.pt"
            artifact_path = stage / filename
            torch.save(
                {
                    "schema": ARTIFACT_SCHEMA,
                    "episode_id": episode["episode_id"],
                    "object_id": episode["object_id"],
                    "split": episode["split"],
                    "q": q.cpu(),
                    "fields": fields.cpu(),
                    "true_endpoints": torch.tensor(episode["true_endpoints"]),
                    "channels": PROFILE_CHANNELS,
                },
                artifact_path,
            )
            rows.append(
                {
                    "episode_id": episode["episode_id"],
                    "object_id": episode["object_id"],
                    "split": episode["split"],
                    "artifact": filename,
                    "artifact_sha256": sha256_file(artifact_path),
                    "shape": list(fields.shape),
                }
            )
        index = {
            "schema": SCHEMA,
            "channels": list(PROFILE_CHANNELS),
            "trajectory": {"samples": samples, "radii": 3, "channels": 9},
            "objects": {
                "source_train": prereg["object_split"]["train"],
                "source_validation": prereg["object_split"]["calibration"],
            },
            "episode_counts": counts,
            "source_hashes": dict(expected_hashes),
            "rows": rows,
            "protected_splits_read": [],
            "box_labels_read": False,
        }
        (stage / "index.json").write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(stage, output)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return {
        "output": str(output.resolve()),
        "objects": EXPECTED_SPLIT_OBJECTS,
        "episodes": counts,
        "index_sha256": sha256_file(output / "index.json"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--preregister", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--qc-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    launch = "\n".join(str(value).lower() for value in vars(args).values())
    if any(marker in launch for marker in FORBIDDEN_PATH_MARKERS):
        raise ValueError("source materialization launch contains a protected path marker")
    print(
        json.dumps(
            build_dataset(
                asset_root=args.asset_root,
                prereg_path=args.preregister,
                inputs_path=args.inputs,
                labels_path=args.labels,
                manifest_path=args.manifest,
                qc_receipt_path=args.qc_receipt,
                output=args.output,
                device_name=args.device,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
