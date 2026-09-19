"""Training-free, observed-gauge semantic/geometry search; no target loader."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import torch
from torch import Tensor


METHODS = ("coordinate_only", "geometry_only", "semantic_only", "joint", "object_shuffle")
INPUT_KEYS = {"schema", "object_id", "split", "coordinates", "geometry", "image_embeddings",
              "observed_embeddings", "text_direction", "provenance"}


@dataclass(frozen=True)
class SearchConfig:
    version: str = "relative-search/v1"
    semantic_slope_weight: float = 1.0
    joint_semantic_weight: float = 1.0
    near_optimal_tolerance: float = 0.1
    disagreement_fraction: float = 0.2
    minimum_direction_strength: float = 0.005


CONFIG = SearchConfig()


def opaque_object_id(original_id: str) -> str:
    return hashlib.sha256(("node91-object:" + original_id).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json_new(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")


def validate_input(value: dict) -> None:
    if set(value) != INPUT_KEYS or value["schema"] != "relative-search-input/v1":
        raise ValueError("input is not the strict target-free whitelist")
    if value["split"] not in ("source_train", "source_validation"):
        raise ValueError("only authorized source diagnostic splits are allowed")
    if not isinstance(value["object_id"], str) or len(value["object_id"]) != 64 or any(c not in "0123456789abcdef" for c in value["object_id"]):
        raise ValueError("object identity must be an opaque SHA256 digest")
    d, g, e, a, t = (value[k] for k in ("coordinates", "geometry", "image_embeddings", "observed_embeddings", "text_direction"))
    if d.ndim != 2 or d.shape[0] != 2 or d.shape[1] < 5 or not torch.equal(d[0], d[1]):
        raise ValueError("paired identical outward coordinate grids required")
    if not torch.all(d[:, 1:] > d[:, :-1]) or not torch.equal(d[:, 0], torch.zeros(2, dtype=d.dtype)):
        raise ValueError("grids must increase from observed anchors at d=0")
    if g.ndim != 5 or g.shape[1] != 2 or g.shape[3:] != (d.shape[1], 8):
        raise ValueError("geometry must be [G,2,R,S,8], without posterior")
    if e.ndim != 4 or e.shape[:2] != d.shape or a.shape != (2, *e.shape[2:]) or t.shape != e.shape[-1:]:
        raise ValueError("per-view embedding shapes disagree")
    if not torch.equal(e[:, 0], a):
        raise ValueError("d=0 must exactly equal observed embeddings")
    if not all(torch.isfinite(x).all() for x in (d, g, e, a, t)):
        raise ValueError("nonfinite input")
    if any(not torch.allclose(x.norm(dim=-1), torch.ones_like(x.norm(dim=-1)), atol=1e-4) for x in (e, a, t)):
        raise ValueError("CLIP embeddings/direction must be unit normalized")
    allowed = {"geometry_artifact_sha256", "semantic_artifact_sha256", "source_index_sha256"}
    if set(value["provenance"]) != allowed or any(len(x) != 64 for x in value["provenance"].values()):
        raise ValueError("provenance must contain only immutable hash bindings")


def unit_range(x: Tensor) -> Tensor:
    """Fixed 10/90% robust scale; does not fit labels or use other objects."""
    low = torch.quantile(x, 0.1, dim=-1, keepdim=True)
    high = torch.quantile(x, 0.9, dim=-1, keepdim=True)
    return ((x - low) / (high - low).clamp_min(1e-6)).clamp(-1, 2)


def evidence(value: dict) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    validate_input(value)
    d, g = value["coordinates"], value["geometry"]
    # Preserve geometry/radius disagreement; use pre-existing unsupervised D2
    # total energy. No endpoint-supervised checkpoint or posterior enters.
    geometry_cost = unit_range(g[..., 7]).permute(1, 0, 2, 3).flatten(1, 2)
    e, a, t = value["image_embeddings"], value["observed_embeddings"], value["text_direction"]
    # Opposite observed anchor defines outward semantic direction independently
    # per view. Full paired feature trajectories remain available in the cache.
    anchor_outward = ((a - a.flip(0)) * t).sum(-1)
    orientation = torch.where(anchor_outward >= 0, 1.0, -1.0)
    progress = ((e - a[:, None]) * t).sum(-1) * orientation[:, None]
    progress = progress.permute(0, 2, 1)
    slope = torch.empty_like(progress)
    slope[..., 1:-1] = (progress[..., 2:] - progress[..., :-2]) / (d[:, None, 2:] - d[:, None, :-2])
    slope[..., 0] = (progress[..., 1] - progress[..., 0]) / (d[:, 1:] - d[:, :-1])[:, :1]
    slope[..., -1] = (progress[..., -1] - progress[..., -2]) / (d[:, 1:] - d[:, :-1])[:, -1:]
    semantic_cost = -unit_range(progress) + CONFIG.semantic_slope_weight * unit_range(slope.abs())
    return geometry_cost, semantic_cost, anchor_outward, progress


def _summarize_side(cost: Tensor, components: Tensor, d: Tensor, reasons: list[str]) -> dict:
    index = int(cost.argmin())
    selected = cost <= cost.min() + CONFIG.near_optimal_tolerance
    positions = d[selected]
    component_positions = d[components.argmin(-1)]
    span = float(d[-1] - d[0])
    if index in (0, len(d) - 1):
        reasons.append("search_boundary_minimum")
    if float(cost.max() - cost.min()) < 1e-6:
        reasons.append("flat_evidence")
    if float(component_positions.max() - component_positions.min()) > CONFIG.disagreement_fraction * span:
        reasons.append("view_or_geometry_disagreement")
    if float(positions[-1] - positions[0]) > CONFIG.disagreement_fraction * span:
        reasons.append("broad_or_disconnected_optimum")
    abstain = bool(reasons)
    return {
        "distance": float(d[index]), "abstain": abstain, "reasons": sorted(set(reasons)),
        "evidence_interval": [float(d[0]), float(d[-1])] if abstain else [float(positions[0]), float(positions[-1])],
        "near_optimum_hull": [float(positions[0]), float(positions[-1])],
        "component_optima": component_positions.tolist(),
        "cost": cost.tolist(),
    }


def predict(value: dict, method: str, semantic_donor: dict | None = None) -> dict:
    if method not in METHODS:
        raise ValueError("unknown fixed control")
    geo, sem, direction, _ = evidence(value)
    if method == "object_shuffle":
        if semantic_donor is None or semantic_donor["object_id"] == value["object_id"]:
            raise ValueError("shuffle needs a distinct deterministic donor")
        if semantic_donor["split"] != value["split"] or not torch.equal(value["coordinates"], semantic_donor["coordinates"]):
            raise ValueError("donor must use same source split and exact grid")
        _, sem, direction, _ = evidence(semantic_donor)
    d = value["coordinates"]
    result = []
    for side in range(2):
        reasons: list[str] = []
        if method == "coordinate_only":
            cost = ((d[side] / d[side, -1]) - 0.5).abs()
            components = cost[None]
            reasons = ["coordinate_prior_has_no_boundary_evidence"]
        elif method == "geometry_only":
            cost, components = geo[side].mean(0), geo[side]
        elif method == "semantic_only":
            cost, components = sem[side].mean(0), sem[side]
        else:
            cost = geo[side].mean(0) + CONFIG.joint_semantic_weight * sem[side].mean(0)
            components = torch.cat((geo[side], sem[side]))
        if method in ("semantic_only", "joint", "object_shuffle"):
            if float(direction[side].abs().mean()) < CONFIG.minimum_direction_strength:
                reasons.append("weak_observed_semantic_direction")
            if float(direction[side].mean()) <= 0:
                reasons.append("opening_limit_not_identifiable_from_closure_semantics")
        result.append(_summarize_side(cost, components, d[side], reasons))
    return {"method": method, "object_id": value["object_id"], "split": value["split"],
            "distance": [r["distance"] for r in result], "sides": result,
            "uncertainty": "heuristic_not_calibrated_not_physical_certification"}


def swap_input(value: dict) -> dict:
    out = dict(value)
    for key in ("coordinates", "image_embeddings", "observed_embeddings"):
        out[key] = value[key].flip(0)
    out["geometry"] = value["geometry"].flip(1)
    return out


def load_inputs(index_path: Path) -> tuple[dict, list[dict]]:
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if set(index) != {"schema", "rows", "source_index_sha256", "diagnostic"} or index["schema"] != "relative-search-input-index/v1":
        raise ValueError("unexpected target-free index")
    if not isinstance(index["rows"], list) or not index["rows"]:
        raise ValueError("nonempty input rows required")
    values, identities = [], set()
    for row in index["rows"]:
        if set(row) != {"object_id", "split", "artifact", "artifact_sha256"}:
            raise ValueError("unexpected input row")
        relative = Path(row["artifact"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("input artifacts must remain within exported input directory")
        path = index_path.parent / relative
        if path.is_symlink() or not path.resolve().is_relative_to(index_path.parent.resolve()):
            raise ValueError("input artifact escapes export directory")
        if file_sha256(path) != row["artifact_sha256"]:
            raise ValueError("input digest mismatch")
        value = torch.load(path, map_location="cpu", weights_only=True)
        validate_input(value)
        identity = (value["split"], value["object_id"])
        if identity in identities or identity != (row["split"], row["object_id"]):
            raise ValueError("input identity mismatch")
        if value["provenance"]["source_index_sha256"] != index["source_index_sha256"]:
            raise ValueError("geometry source index binding mismatch")
        identities.add(identity)
        values.append(value)
    return index, sorted(values, key=lambda x: (x["split"], x["object_id"]))
