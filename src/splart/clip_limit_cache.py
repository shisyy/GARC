"""Fail-closed cache contract for counterfactual CLIP limit trajectories."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
from torch import Tensor

from splart.clip_limit import prompt_ensemble_sha256


CACHE_INDEX_SCHEMA = "splart-c-clip-ld-cache-index/v1"
CACHE_ARTIFACT_SCHEMA = "splart-c-clip-ld-cache-artifact/v1"
CACHE_ROW_KEYS = {"artifact", "artifact_sha256", "episode_id", "object_id", "render_metadata_sha256", "shape", "split"}
CACHE_ARTIFACT_KEYS = {
    "asset_bundle_sha256",
    "audit_thumbnail_sha256",
    "coordinates",
    "encoder_checkpoint_sha256",
    "episode_id",
    "object_id",
    "open_clip_version",
    "open_clip_wheel_sha256",
    "prompt_ensemble_sha256",
    "render_config_sha256",
    "render_metadata_sha256",
    "renderer_sha256",
    "schema",
    "signed_limit_evidence",
    "split",
    "view_count",
}
FORBIDDEN_MARKERS = ("sealed", "b_test", "full22", "box_e", "box_f")
CANDIDATE_SAMPLES = 129
CANDIDATE_MAX_OUTWARD_DISTANCE = 2.0


def fixed_candidate_grid(*, device: torch.device | None = None, dtype: torch.dtype = torch.float32) -> Tensor:
    """The preregistered, label-independent outward render grid."""

    return torch.linspace(0.0, CANDIDATE_MAX_OUTWARD_DISTANCE, CANDIDATE_SAMPLES, device=device, dtype=dtype)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _regular_artifact(index_path: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = index_path.parent / path
    if path.is_symlink():
        raise ValueError("semantic cache symlinks are forbidden")
    resolved = path.resolve()
    if any(marker in str(resolved).lower() for marker in FORBIDDEN_MARKERS):
        raise ValueError("protected path marker in semantic cache")
    if not resolved.is_file():
        raise ValueError("semantic cache artifact must be a regular file")
    return resolved


@dataclass(frozen=True)
class SemanticTrajectory:
    episode_id: str
    object_id: str
    split: str
    coordinates: Tensor
    evidence: Tensor
    view_count: int


def validate_cache_index(
    index_path: Path,
    *,
    source_index_sha256: str,
    expected_split_counts: Mapping[str, int],
    expected_identities: Iterable[tuple[str, str, str]] | None = None,
) -> list[tuple[dict[str, Any], Path]]:
    """Validate index, digests, identities and every tensor artifact."""

    if index_path.is_symlink():
        raise ValueError("semantic cache index must be a regular non-symlink file")
    index_path = index_path.resolve()
    if not index_path.is_file():
        raise ValueError("semantic cache index must be a regular non-symlink file")
    if any(marker in str(index_path).lower() for marker in FORBIDDEN_MARKERS):
        raise ValueError("protected path marker in semantic cache index")
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    if payload.get("schema") != CACHE_INDEX_SCHEMA:
        raise ValueError("unexpected semantic cache index schema")
    if payload.get("source_index_sha256") != source_index_sha256:
        raise ValueError("semantic cache is bound to a different geometry index")
    if payload.get("prompt_ensemble_sha256") != prompt_ensemble_sha256():
        raise ValueError("semantic cache prompt ensemble mismatch")
    checkpoint_sha256 = payload.get("encoder_checkpoint_sha256")
    if not isinstance(checkpoint_sha256, str) or len(checkpoint_sha256) != 64:
        raise ValueError("semantic cache has no encoder checkpoint binding")
    if payload.get("encoder") != "open_clip:ViT-B-32":
        raise ValueError("semantic cache encoder contract mismatch")
    open_clip_version = payload.get("open_clip_version")
    open_clip_wheel_sha256 = payload.get("open_clip_wheel_sha256")
    if not isinstance(open_clip_version, str) or not open_clip_version:
        raise ValueError("semantic cache has no OpenCLIP package version binding")
    if not isinstance(open_clip_wheel_sha256, str) or len(open_clip_wheel_sha256) != 64:
        raise ValueError("semantic cache has no OpenCLIP wheel binding")
    input_bindings = payload.get("input_bindings")
    if not isinstance(input_bindings, dict) or set(input_bindings) not in ({"selection"}, {"preregister"}):
        raise ValueError("semantic cache public mapping dependency binding mismatch")
    binding = next(iter(input_bindings.values()))
    if (
        not isinstance(binding, dict)
        or set(binding) != {"path", "sha256"}
        or not isinstance(binding["path"], str)
        or not isinstance(binding["sha256"], str)
        or len(binding["sha256"]) != 64
        or any(marker in binding["path"].lower() for marker in FORBIDDEN_MARKERS)
    ):
        raise ValueError("semantic cache public mapping dependency is invalid")
    renderer_sha256 = payload.get("renderer_sha256")
    render_config_sha256 = payload.get("render_config_sha256")
    if not all(isinstance(value, str) and len(value) == 64 for value in (renderer_sha256, render_config_sha256)):
        raise ValueError("semantic cache has no renderer/config binding")
    if payload.get("streaming_rgb_persisted") is not False:
        raise ValueError("semantic cache must not persist the full RGB trajectory")
    renderer_audit = payload.get("renderer_audit")
    if (
        not isinstance(renderer_audit, dict)
        or set(renderer_audit) != {
            "schema", "no_empty_views_checked", "repeat_bit_identical", "assets_repeat_audited", "views_checked",
            "render_config_sha256", "runtime"
        }
        or renderer_audit["schema"] != "splart-c-clip-ld-render-audit/v1"
        or renderer_audit["no_empty_views_checked"] is not True
        or renderer_audit["repeat_bit_identical"] is not True
        or not isinstance(renderer_audit["assets_repeat_audited"], int)
        or renderer_audit["assets_repeat_audited"] < 1
        or not isinstance(renderer_audit["views_checked"], int)
        or renderer_audit["views_checked"] < sum(expected_split_counts.values()) * 2 * CANDIDATE_SAMPLES * 6
        or renderer_audit["render_config_sha256"] != render_config_sha256
    ):
        raise ValueError("semantic cache has no valid renderer determinism/coverage audit")
    runtime = renderer_audit["runtime"]
    if (
        not isinstance(runtime, dict)
        or set(runtime) != {
            "device_type", "torch_version", "torch_cuda_version", "pytorch3d_version",
            "pytorch3d_distribution_sha256"
        }
        or runtime["device_type"] not in ("cpu", "cuda")
        or not isinstance(runtime["torch_version"], str)
        or (runtime["torch_cuda_version"] is not None and not isinstance(runtime["torch_cuda_version"], str))
    ):
        raise ValueError("semantic cache renderer runtime receipt is invalid")
    if runtime["device_type"] == "cuda" and (
        runtime["pytorch3d_version"] != "0.7.8"
        or not isinstance(runtime["pytorch3d_distribution_sha256"], str)
        or len(runtime["pytorch3d_distribution_sha256"]) != 64
        or not runtime["torch_cuda_version"]
    ):
        raise ValueError("semantic cache formal CUDA renderer is not bound to pytorch3d 0.7.8")
    if payload.get("protected_splits_read") != [] or payload.get("box_labels_read") not in (False, 0, []):
        raise ValueError("semantic cache records protected access")
    rows = payload.get("rows")
    if not isinstance(rows, list) or len(rows) != sum(expected_split_counts.values()):
        raise ValueError("semantic cache row count mismatch")

    counts: Counter[str] = Counter()
    identities: set[tuple[str, str, str]] = set()
    verified: list[tuple[dict[str, Any], Path]] = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != CACHE_ROW_KEYS:
            raise ValueError("unexpected semantic cache row contract")
        split, object_id, episode_id = row["split"], row["object_id"], row["episode_id"]
        render_metadata_sha256 = row["render_metadata_sha256"]
        identity = (split, object_id, episode_id)
        if split not in expected_split_counts or identity in identities:
            raise ValueError("duplicate or unauthorized semantic cache identity")
        if not all(isinstance(value, str) and value for value in identity):
            raise ValueError("semantic cache identities must be non-empty strings")
        if not isinstance(render_metadata_sha256, str) or len(render_metadata_sha256) != 64:
            raise ValueError("semantic cache row has no render metadata binding")
        if (
            not isinstance(row["shape"], list)
            or len(row["shape"]) != 3
            or row["shape"][:1] != [2]
            or row["shape"][-1:] != [1]
            or row["shape"][1] != CANDIDATE_SAMPLES
        ):
            raise ValueError("semantic trajectory shape must be [2,S,1], S>=2")
        path = _regular_artifact(index_path, row["artifact"])
        if sha256_file(path) != row["artifact_sha256"]:
            raise ValueError("semantic cache artifact digest mismatch")
        artifact = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(artifact, dict) or set(artifact) != CACHE_ARTIFACT_KEYS:
            raise ValueError("unexpected semantic cache artifact contract")
        if (
            artifact["schema"] != CACHE_ARTIFACT_SCHEMA
            or artifact["split"] != split
            or artifact["object_id"] != object_id
            or artifact["episode_id"] != episode_id
            or artifact["prompt_ensemble_sha256"] != prompt_ensemble_sha256()
            or artifact["encoder_checkpoint_sha256"] != checkpoint_sha256
            or artifact["renderer_sha256"] != renderer_sha256
            or artifact["render_config_sha256"] != render_config_sha256
            or artifact["render_metadata_sha256"] != render_metadata_sha256
            or artifact["open_clip_version"] != open_clip_version
            or artifact["open_clip_wheel_sha256"] != open_clip_wheel_sha256
        ):
            raise ValueError("semantic cache artifact identity/binding mismatch")
        coordinates = torch.as_tensor(artifact["coordinates"])
        evidence = torch.as_tensor(artifact["signed_limit_evidence"])
        if coordinates.shape != tuple(row["shape"][:-1]) or evidence.shape != tuple(row["shape"]):
            raise ValueError("semantic cache artifact shape mismatch")
        if not coordinates.is_floating_point() or not evidence.is_floating_point():
            raise ValueError("semantic cache tensors must be floating point")
        if not torch.isfinite(coordinates).all() or not torch.isfinite(evidence).all():
            raise ValueError("semantic cache tensors must be finite")
        if (coordinates < 0).any():
            raise ValueError("semantic coordinates must be non-negative")
        delta = coordinates[:, 1:] - coordinates[:, :-1]
        if not torch.all((delta > 0).all(dim=-1) | (delta < 0).all(dim=-1)):
            raise ValueError("semantic coordinates must be strictly monotone per side")
        expected_grid = fixed_candidate_grid(dtype=coordinates.dtype).expand(2, -1)
        if not torch.equal(coordinates.cpu(), expected_grid):
            raise ValueError("semantic coordinates do not match the fixed label-independent grid")
        if not isinstance(artifact["view_count"], int) or artifact["view_count"] < 2:
            raise ValueError("semantic cache must aggregate at least two views")
        if not isinstance(artifact["asset_bundle_sha256"], str) or len(artifact["asset_bundle_sha256"]) != 64:
            raise ValueError("semantic cache has no asset bundle binding")
        audit = artifact["audit_thumbnail_sha256"]
        expected_audit_keys = {f"side{side}:{index:03d}" for side in range(2) for index in (0, 64, 128)}
        if (
            not isinstance(audit, dict)
            or set(audit) != expected_audit_keys
            or not all(isinstance(value, str) and len(value) == 64 for value in audit.values())
        ):
            raise ValueError("semantic cache audit thumbnail hashes are incomplete")
        identities.add(identity)
        counts[split] += 1
        verified.append((row, path))
    if dict(counts) != dict(expected_split_counts):
        raise ValueError("semantic cache split counts mismatch")
    if renderer_audit["assets_repeat_audited"] != len({identity[1] for identity in identities}):
        raise ValueError("semantic cache renderer audit does not cover every object asset")
    if expected_identities is not None and identities != set(expected_identities):
        raise ValueError("semantic cache identities do not exactly match geometry")
    return verified


def load_semantic_cache(
    index_path: Path,
    *,
    source_index_sha256: str,
    expected_split_counts: Mapping[str, int],
    expected_identities: Iterable[tuple[str, str, str]] | None = None,
) -> dict[str, tuple[Tensor, Tensor, tuple[str, ...], tuple[str, ...]]]:
    groups: dict[str, list[SemanticTrajectory]] = {split: [] for split in expected_split_counts}
    for row, path in validate_cache_index(
        index_path,
        source_index_sha256=source_index_sha256,
        expected_split_counts=expected_split_counts,
        expected_identities=expected_identities,
    ):
        artifact = torch.load(path, map_location="cpu", weights_only=True)
        groups[row["split"]].append(
            SemanticTrajectory(
                episode_id=row["episode_id"],
                object_id=row["object_id"],
                split=row["split"],
                coordinates=torch.as_tensor(artifact["coordinates"], dtype=torch.float32),
                evidence=torch.as_tensor(artifact["signed_limit_evidence"], dtype=torch.float32),
                view_count=artifact["view_count"],
            )
        )
    result = {}
    for split, values in groups.items():
        values.sort(key=lambda value: value.episode_id)
        sample_counts = {value.coordinates.shape[1] for value in values}
        if len(sample_counts) != 1:
            raise ValueError("semantic trajectories must share a sample count")
        result[split] = (
            torch.stack([value.evidence for value in values]),
            torch.stack([value.coordinates for value in values]),
            tuple(value.object_id for value in values),
            tuple(value.episode_id for value in values),
        )
    return result


__all__ = [
    "CANDIDATE_MAX_OUTWARD_DISTANCE",
    "CANDIDATE_SAMPLES",
    "CACHE_ARTIFACT_KEYS",
    "CACHE_ARTIFACT_SCHEMA",
    "CACHE_INDEX_SCHEMA",
    "CACHE_ROW_KEYS",
    "FORBIDDEN_MARKERS",
    "SemanticTrajectory",
    "fixed_candidate_grid",
    "load_semantic_cache",
    "sha256_file",
    "validate_cache_index",
]
