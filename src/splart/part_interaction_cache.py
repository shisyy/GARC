"""Strict node9.3 cache schema and visible-deletion mask rules (no labels)."""
from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from build_relative_clip_cache import _hashes, _identity, _unit
from splart.clip_limit_cache import fixed_candidate_grid

INDEX_SCHEMA = "part-interaction-index/v1"
ARTIFACT_SCHEMA = "part-interaction/v1"
FEATURE_KEYS = ("static_embeddings", "mobile_embeddings", "partition_a_embeddings", "partition_b_embeddings")
HASH_KEYS = {
    "render_plan_sha256", "renderer_sha256", "render_config_sha256",
    "encoder_checkpoint_sha256", "open_clip_wheel_sha256", "builder_sha256",
    "mask_rule_sha256", "base_semantic_index_sha256",
}
STATIC_RGB = (148, 148, 148)
MOBILE_RGB = (230, 118, 34)


def validate_part_artifact(value: dict[str, Any]) -> None:
    keys = {"schema", "object_id", "split", "coordinates", "blank_embedding", "provenance", *FEATURE_KEYS}
    if not isinstance(value, dict) or set(value) != keys or value["schema"] != ARTIFACT_SCHEMA:
        raise ValueError("part interaction artifact schema mismatch")
    _identity(value["object_id"], value["split"])
    _hashes(value["provenance"], HASH_KEYS | {"asset_bundle_sha256"})
    grid = value["coordinates"]
    if not isinstance(grid, Tensor) or grid.dtype != torch.float32 or not torch.equal(grid, fixed_candidate_grid().expand(2, -1)):
        raise ValueError("part cache coordinates must be the exact fixed grid")
    for key in FEATURE_KEYS:
        _unit(value[key], (2, 129, 6, 512))
    _unit(value["blank_embedding"], (512,))


def validate_part_index(value: dict[str, Any]) -> None:
    if not isinstance(value, dict) or set(value) != {"schema", "rows", "provenance", "split_counts"} or value["schema"] != INDEX_SCHEMA:
        raise ValueError("part interaction index schema mismatch")
    _hashes(value["provenance"], HASH_KEYS)
    if not isinstance(value["rows"], list) or not value["rows"]:
        raise ValueError("part interaction index needs rows")
    counts, seen = {}, set()
    for row in value["rows"]:
        if not isinstance(row, dict) or set(row) != {"object_id", "split", "artifact", "artifact_sha256"}:
            raise ValueError("part index row has unapproved fields")
        _identity(row["object_id"], row["split"])
        if row["object_id"] in seen:
            raise ValueError("duplicate object")
        seen.add(row["object_id"])
        if row["artifact"] != f"artifacts/{row['object_id']}.pt":
            raise ValueError("artifact path must be canonical and relative")
        _hashes({"artifact_sha256": row["artifact_sha256"]}, {"artifact_sha256"})
        counts[row["split"]] = counts.get(row["split"], 0) + 1
    if counts != value["split_counts"]:
        raise ValueError("split count mismatch")


def visible_masks(images: Tensor) -> tuple[Tensor, Tensor]:
    """Recover masks from exact palette with scalar depth shade in [0.72,1].

    Permit only uint8 colors attainable by round(palette * shade); the tiny
    epsilon covers float32 renderer arithmetic at rounding boundaries.
    """
    if not isinstance(images, Tensor) or images.dtype != torch.uint8 or images.ndim != 4 or images.shape[1] != 3:
        raise ValueError("images must be uint8 [V,3,H,W]")
    pixels = images.permute(0, 2, 3, 1).to(torch.float64)
    masks = []
    for palette in (STATIC_RGB, MOBILE_RGB):
        colors = torch.tensor(palette, dtype=torch.float64, device=images.device)
        low = ((pixels - 0.5001) / colors).amax(dim=-1).clamp_min(0.72)
        high = ((pixels + 0.5001) / colors).amin(dim=-1).clamp_max(1.0)
        masks.append(low <= high)
    static, mobile = masks
    static &= (images[:, 0] == images[:, 1]) & (images[:, 1] == images[:, 2])
    background = (images == 255).all(dim=1)
    if (static & mobile).any() or not (static | mobile | background).all():
        raise ValueError("unknown nonbackground palette color")
    if not (static | mobile).flatten(1).any(dim=1).all():
        raise ValueError("empty foreground view")
    return static, mobile


def matched_partition(static: Tensor, mobile: Tensor) -> tuple[Tensor, Tensor]:
    """Exact area-preserving row-major interleave, independently per view."""
    if static.dtype != torch.bool or mobile.dtype != torch.bool or static.shape != mobile.shape or static.ndim != 3 or (static & mobile).any():
        raise ValueError("expected disjoint bool [V,H,W] masks")
    foreground = static | mobile
    partition = torch.zeros_like(foreground)
    for view in range(len(static)):
        positions = foreground[view].flatten().nonzero(as_tuple=False).flatten()
        n, k = positions.numel(), int(static[view].sum())
        if n == 0:
            raise ValueError("empty foreground view")
        rank = torch.arange(n, dtype=torch.int64, device=static.device)
        assignment = torch.div((rank + 1) * k, n, rounding_mode="floor") > torch.div(rank * k, n, rounding_mode="floor")
        partition[view].view(-1)[positions] = assignment
    return partition, foreground & ~partition


def deletion_images(images: Tensor):
    """Yield four six-view batches; never combine them into a 24-view batch."""
    static, mobile = visible_masks(images)
    a, b = matched_partition(static, mobile)
    for key, mask in zip(FEATURE_KEYS, (static, mobile, a, b)):
        yield key, torch.where(mask[:, None], images, torch.full_like(images, 255))
