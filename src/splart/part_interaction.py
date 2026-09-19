"""Frozen visible-deletion interaction evidence; no endpoint labels or training."""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import Tensor

from splart.relative_search import CONFIG, METHODS, _summarize_side, evidence, file_sha256, predict, unit_range


MODES = ("raw", "part_interaction", "matched_partition", "donor_interaction")
FEATURE_KEYS = ("static_embeddings", "mobile_embeddings", "partition_a_embeddings", "partition_b_embeddings")
PROVENANCE_KEYS = {"render_plan_sha256", "renderer_sha256", "render_config_sha256", "encoder_checkpoint_sha256",
                   "open_clip_wheel_sha256", "builder_sha256", "mask_rule_sha256", "base_semantic_index_sha256"}


def _hash_dict(value: dict, keys: set[str]) -> None:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError("unexpected interaction provenance")
    if any(not isinstance(h, str) or len(h) != 64 or any(c not in "0123456789abcdef" for c in h) for h in value.values()):
        raise ValueError("provenance must contain only SHA256 hashes")


def _rows(index: dict) -> dict:
    if set(index) != {"schema", "rows", "provenance", "split_counts"} or not isinstance(index["rows"], list):
        raise ValueError("unexpected cache index keys")
    result = {}
    for row in index["rows"]:
        if set(row) != {"object_id", "split", "artifact", "artifact_sha256"}:
            raise ValueError("unexpected cache row")
        identity = (row["split"], row["object_id"])
        if identity in result:
            raise ValueError("duplicate cache identity")
        result[identity] = row
    counts = {s: sum(k[0] == s for k in result) for s in index["split_counts"]}
    if counts != index["split_counts"]:
        raise ValueError("cache split counts mismatch")
    return result


def load_interactions(cache_index: Path, base_semantic_index: Path, values: list[dict],
                      allow_train_smoke2: bool = False) -> dict:
    """Bind caches to exact whitelist semantic-artifact hashes, never raw labels."""
    index = json.loads(cache_index.read_text(encoding="utf-8"))
    base = json.loads(base_semantic_index.read_text(encoding="utf-8"))
    if index.get("schema") != "part-interaction-index/v1" or base.get("schema") != "relative-clip-index/v1":
        raise ValueError("unexpected interaction/base semantic schema")
    rows, base_rows = _rows(index), _rows(base)
    _hash_dict(index["provenance"], PROVENANCE_KEYS)
    if index["provenance"]["base_semantic_index_sha256"] != file_sha256(base_semantic_index):
        raise ValueError("interaction/base semantic index hash mismatch")
    identities = {(v["split"], v["object_id"]) for v in values}
    base_roster_ok = (identities <= set(base_rows) and len(identities) == 2 and
                      all(k[0] == "source_train" for k in identities)) if allow_train_smoke2 else set(base_rows) == identities
    if len(identities) != len(values) or set(rows) != identities or not base_roster_ok:
        raise ValueError("input/base/interaction rosters must match exactly")
    for key in ("render_plan_sha256", "renderer_sha256", "render_config_sha256", "encoder_checkpoint_sha256", "open_clip_wheel_sha256"):
        if index["provenance"][key] != base["provenance"].get(key):
            raise ValueError("interaction renderer/encoder differs from base semantic cache")
    output = {}
    for value in values:
        identity = (value["split"], value["object_id"])
        if base_rows[identity]["artifact_sha256"] != value["provenance"]["semantic_artifact_sha256"]:
            raise ValueError("whitelist semantic artifact hash mismatch")
        relative = Path(rows[identity]["artifact"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("interaction artifact must remain inside cache")
        path = cache_index.parent / relative
        if path.is_symlink() or not path.resolve().is_relative_to(cache_index.parent.resolve()):
            raise ValueError("interaction artifact escapes cache")
        if file_sha256(path) != rows[identity]["artifact_sha256"]:
            raise ValueError("interaction artifact digest mismatch")
        artifact = torch.load(path, map_location="cpu", weights_only=True)
        expected = {"schema", "object_id", "split", "coordinates", "blank_embedding", "provenance", *FEATURE_KEYS}
        if set(artifact) != expected or artifact["schema"] != "part-interaction/v1":
            raise ValueError("unexpected interaction artifact contract")
        if (artifact["split"], artifact["object_id"]) != identity:
            raise ValueError("interaction artifact identity mismatch")
        _hash_dict(artifact["provenance"], PROVENANCE_KEYS | {"asset_bundle_sha256"})
        if any(artifact["provenance"][k] != index["provenance"][k] for k in PROVENANCE_KEYS):
            raise ValueError("interaction artifact/index provenance mismatch")
        if not torch.equal(artifact["coordinates"], value["coordinates"]):
            raise ValueError("interaction and original observed-gauge grids differ")
        for key in (*FEATURE_KEYS, "blank_embedding"):
            tensor = artifact[key]
            shape = value["image_embeddings"].shape if key != "blank_embedding" else value["text_direction"].shape
            if tensor.shape != shape or tensor.dtype != torch.float32 or not torch.isfinite(tensor).all():
                raise ValueError("invalid interaction embedding shape/dtype/value")
            if not torch.allclose(tensor.norm(dim=-1), torch.ones_like(tensor.norm(dim=-1)), atol=1e-4, rtol=0):
                raise ValueError("interaction component embeddings must be unit length")
        output[identity] = artifact
    return output


def residual(value: dict, artifact: dict, mode: str) -> Tensor:
    first, second = (FEATURE_KEYS[2:] if mode == "matched_partition" else FEATURE_KEYS[:2])
    # Every component is unit-normalized upstream; the residual is intentionally NOT.
    return value["image_embeddings"] - artifact[first] - artifact[second] + artifact["blank_embedding"]


def interaction_evidence(value: dict, artifact: dict, mode: str) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    r = residual(value, artifact, mode)
    a, t, d = r[:, 0], value["text_direction"], value["coordinates"]
    direction = ((a - a.flip(0)) * t).sum(-1)
    orientation = torch.where(direction >= 0, 1.0, -1.0)
    progress = (((r - a[:, None]) * t).sum(-1) * orientation[:, None]).permute(0, 2, 1)
    slope = torch.empty_like(progress)
    slope[..., 1:-1] = (progress[..., 2:] - progress[..., :-2]) / (d[:, None, 2:] - d[:, None, :-2])
    slope[..., 0] = (progress[..., 1] - progress[..., 0]) / (d[:, 1:] - d[:, :-1])[:, :1]
    slope[..., -1] = (progress[..., -1] - progress[..., -2]) / (d[:, 1:] - d[:, :-1])[:, -1:]
    cost = -unit_range(progress) + CONFIG.semantic_slope_weight * unit_range(slope.abs())
    return cost, direction, progress, r


def interaction_predict(value: dict, method: str, mode: str, artifact: dict,
                        semantic_owner: dict | None = None) -> dict:
    if method not in METHODS or mode not in MODES:
        raise ValueError("unknown preregistered mode/method")
    if mode == "raw":
        return predict(value, method, semantic_owner)
    if method in ("coordinate_only", "geometry_only"):
        return predict(value, method)
    owner = value if semantic_owner is None else semantic_owner
    if owner["split"] != value["split"] or not torch.equal(owner["coordinates"], value["coordinates"]):
        raise ValueError("semantic owner must share split and original grid")
    if (artifact["split"], artifact["object_id"]) != (owner["split"], owner["object_id"]):
        raise ValueError("interaction cache belongs to another semantic owner")
    geo = evidence(value)[0]
    sem, direction, progress, _ = interaction_evidence(owner, artifact, mode)
    results = []
    for side in range(2):
        reasons = []
        if method == "semantic_only":
            cost, components = sem[side].mean(0), sem[side]
        else:
            cost = geo[side].mean(0) + CONFIG.joint_semantic_weight * sem[side].mean(0)
            components = torch.cat((geo[side], sem[side]))
        if float(direction[side].abs().mean()) < CONFIG.minimum_direction_strength:
            reasons.append("weak_observed_semantic_direction")
        if float(direction[side].mean()) <= 0:
            reasons.append("opening_limit_not_identifiable_from_closure_semantics")
        if not torch.any(progress[side] != 0):
            reasons.append("zero_interaction_signal")
        results.append(_summarize_side(cost, components, value["coordinates"][side], reasons))
    return {"method": method, "object_id": value["object_id"], "split": value["split"],
            "distance": [x["distance"] for x in results], "sides": results,
            "uncertainty": "heuristic_not_calibrated_not_physical_certification"}


def swap_artifact(artifact: dict) -> dict:
    result = dict(artifact)
    for key in ("coordinates", *FEATURE_KEYS):
        result[key] = artifact[key].flip(0)
    return result
