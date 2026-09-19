"""Strict observed-pose mesh consumer; no asset, renderer or exporter access."""
from __future__ import annotations

import json
import math
from pathlib import Path
import re

import torch

from splart.relative_search import file_sha256, load_inputs

SCHEMA = "observed-surfaces/v1"
INDEX_SCHEMA = "observed-surfaces-index/v1"
KEYS = {"schema", "object_id", "split", "static_vertices", "static_faces", "mobile_vertices", "mobile_faces",
        "axis", "pivot", "angular_increment", "coordinates", "provenance"}
HASH_KEYS = {"render_plan_sha256", "renderer_sha256", "base_input_index_sha256", "exporter_sha256", "consumer_sha256"}
ARTIFACT_HASH_KEYS = HASH_KEYS | {"asset_bundle_sha256", "base_input_artifact_sha256"}
COUNTS = {"source_train": 13, "source_validation": 6}


def _hashes(provenance, keys):
    if not isinstance(provenance, dict) or set(provenance) != keys:
        raise ValueError("provenance requires strict hash-only keys")
    if any(not isinstance(v, str) or re.fullmatch("[0-9a-f]{64}", v) is None for v in provenance.values()):
        raise ValueError("invalid SHA256 provenance")


def _identity(value):
    if not isinstance(value["object_id"], str) or re.fullmatch("[0-9a-f]{64}", value["object_id"]) is None:
        raise ValueError("identity must be opaque SHA256")
    if value["split"] not in COUNTS:
        raise ValueError("unauthorized split")


def rotate_vertices(vertices, axis, pivot, angle):
    """Float64 Rodrigues rotation in a common observed coordinate frame."""
    vertices, axis, pivot = (x.to(dtype=torch.float64, device="cpu") for x in (vertices, axis, pivot))
    axis = axis / axis.norm()
    offset = vertices - pivot
    c, s = math.cos(float(angle)), math.sin(float(angle))
    return pivot + offset * c + torch.linalg.cross(axis.expand_as(offset), offset, dim=-1) * s + (offset * axis).sum(-1, keepdim=True) * axis * (1 - c)


def mobile_at(value, side, distance):
    if side not in (0, 1) or not math.isfinite(float(distance)):
        raise ValueError("finite outward distance and side 0/1 required")
    return rotate_vertices(value["mobile_vertices"][side], value["axis"], value["pivot"],
                           value["angular_increment"] * (-float(distance) if side == 0 else float(distance)))


def swap_surface(value):
    """Exchange actual observations, without exposing a canonical/rest pose."""
    return dict(value, mobile_vertices=value["mobile_vertices"].flip(0), coordinates=value["coordinates"].flip(0),
                angular_increment=-value["angular_increment"])


def _vertices(vertices, dimensions):
    if not isinstance(vertices, torch.Tensor) or vertices.dtype != torch.float64 or vertices.device.type != "cpu":
        raise ValueError("vertices must be CPU float64 tensors")
    if vertices.ndim != dimensions or vertices.shape[-1] != 3 or vertices.shape[-2] < 3 or not torch.isfinite(vertices).all():
        raise ValueError("invalid/nonfinite vertex array")


def _faces(faces, count):
    if not isinstance(faces, torch.Tensor) or faces.dtype != torch.int64 or faces.device.type != "cpu":
        raise ValueError("faces must be CPU int64 tensors")
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) < 1 or int(faces.min()) < 0 or int(faces.max()) >= count:
        raise ValueError("invalid face indices")


def validate_surface(value):
    if not isinstance(value, dict) or set(value) != KEYS or value["schema"] != SCHEMA:
        raise ValueError("surface artifact must obey strict observed-only keys")
    _identity(value)
    _hashes(value["provenance"], ARTIFACT_HASH_KEYS)
    _vertices(value["static_vertices"], 2)
    _vertices(value["mobile_vertices"], 3)
    if value["mobile_vertices"].shape[0] != 2:
        raise ValueError("two actual observed mobile meshes required")
    _faces(value["static_faces"], len(value["static_vertices"]))
    _faces(value["mobile_faces"], value["mobile_vertices"].shape[1])
    for key in ("axis", "pivot"):
        x = value[key]
        if not isinstance(x, torch.Tensor) or x.dtype != torch.float64 or x.shape != (3,) or x.device.type != "cpu" or not torch.isfinite(x).all():
            raise ValueError("motion vectors must be finite CPU float64 triples")
    if abs(float(value["axis"].norm()) - 1.0) > 1e-10:
        raise ValueError("relative axis must be unit normalized")
    angle = value["angular_increment"]
    if isinstance(angle, bool) or not isinstance(angle, float) or not math.isfinite(angle) or angle == 0:
        raise ValueError("relative angular increment must be finite and nonzero")
    d = value["coordinates"]
    expected = torch.linspace(0, 2, 129, dtype=torch.float32).repeat(2, 1)
    if not isinstance(d, torch.Tensor) or d.dtype != torch.float32 or d.device.type != "cpu" or not torch.equal(d, expected):
        raise ValueError("exact frozen two-side 129-distance grid required")
    mesh = value["mobile_vertices"]
    scale = max(1.0, float((mesh.amax((0, 1)) - mesh.amin((0, 1))).norm()))
    for first, second, increment in ((0, 1, angle), (1, 0, -angle)):
        replay = rotate_vertices(mesh[first], value["axis"], value["pivot"], increment)
        if not torch.allclose(replay, mesh[second], rtol=0, atol=1e-9 * scale):
            raise ValueError("observed rigid motion replay failed")


def load_surfaces(index_path: Path, base_input_index: Path | None = None):
    index_path = Path(index_path)
    if index_path.is_symlink():
        raise ValueError("surface index may not be symlinked")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if set(index) != {"schema", "rows", "provenance", "split_counts"} or index["schema"] != INDEX_SCHEMA:
        raise ValueError("surface index must obey strict keys")
    _hashes(index["provenance"], HASH_KEYS)
    if index["split_counts"] != COUNTS or not isinstance(index["rows"], list) or len(index["rows"]) != 19:
        raise ValueError("full fixed 13/6 roster required")
    base_values, base_rows = None, None
    if base_input_index is not None:
        base_input_index = Path(base_input_index)
        if file_sha256(base_input_index) != index["provenance"]["base_input_index_sha256"]:
            raise ValueError("base input index hash mismatch")
        base, values = load_inputs(base_input_index)
        base_values = {(v["split"], v["object_id"]): v for v in values}
        base_rows = {(v["split"], v["object_id"]): v for v in base["rows"]}
    values, identities = [], set()
    for row in index["rows"]:
        if set(row) != {"object_id", "split", "artifact", "artifact_sha256"}:
            raise ValueError("surface index row has unapproved keys")
        _identity(row)
        identity = row["split"], row["object_id"]
        if identity in identities:
            raise ValueError("duplicate surface identity")
        identities.add(identity)
        relative = row["artifact"]
        if relative != f"artifacts/{row['object_id']}.pt":
            raise ValueError("surface artifact path must be canonical and relative")
        path = index_path.parent / relative
        if path.is_symlink() or not path.resolve().is_relative_to(index_path.parent.resolve()):
            raise ValueError("surface artifact path escapes export")
        if file_sha256(path) != row["artifact_sha256"]:
            raise ValueError("surface artifact hash mismatch")
        value = torch.load(path, map_location="cpu", weights_only=True)
        validate_surface(value)
        if (value["split"], value["object_id"]) != identity:
            raise ValueError("surface identity mismatch")
        if any(value["provenance"][key] != index["provenance"][key] for key in HASH_KEYS):
            raise ValueError("surface provenance mismatch")
        if base_values is not None:
            if identity not in base_values or value["provenance"]["base_input_artifact_sha256"] != base_rows[identity]["artifact_sha256"]:
                raise ValueError("surface/base roster or artifact binding mismatch")
            if not torch.equal(value["coordinates"], base_values[identity]["coordinates"]):
                raise ValueError("surface/base distance grid mismatch")
        values.append(value)
    if {s: sum(a == s for a, _ in identities) for s in COUNTS} != COUNTS:
        raise ValueError("full fixed 13/6 roster required")
    if base_values is not None and identities != set(base_values):
        raise ValueError("surface/base roster mismatch")
    return index, sorted(values, key=lambda v: (v["split"], v["object_id"]))
