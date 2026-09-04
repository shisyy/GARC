"""Create leakage-safe interior-state proxies from endpoint RGB-D observations.

This builder is an explicit fallback for the currently unavailable PartNet
Mobility URDF. It may consume endpoint geometry and ground-truth kinematics,
but writes only two interior states relabelled 0/1 to the model-facing root.
Hidden coordinates and endpoint targets stay in a separate sealed manifest.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import secrets
import shutil
from typing import Any, Iterable, Mapping, Sequence

import imageio.v3 as iio
import numpy as np
from scipy.spatial import cKDTree


PUBLIC_SCHEMA = "splart-middle-state-observations-v1"
QUERY_SCHEMA = "splart-endpoint-direction-queries-v1"
SEALED_SCHEMA = "splart-endpoint-extrapolation-sealed-v1"
RECEIPT_SCHEMA = "splart-middle-state-proxy-receipt-v1"
DEV3 = ("100247-Box", "102016-USB", "103042-Window")
STATE_VIEWS = 100
PROXY_QUALITY_THRESHOLDS = {
    "foreground_iou_min": 0.70,
    "static_iou_min": 0.70,
    "mobile_iou_min": 0.50,
    "psnr_min": 16.0,
    "depth_mae_m_max": 0.025,
}
ALIGNMENT_THRESHOLDS = {"median_m_max": 0.012, "p95_m_max": 0.050}
CLOSURE_THRESHOLDS = {"contact_fraction_margin_min": 0.01, "compactness_margin_min": 0.015}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def frame_lookup(meta: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {str(frame["file_path"]): frame for frame in meta["frames"]}


def validate_source_scene(scene_dir: Path, meta: Mapping[str, Any]) -> None:
    required = {"fl_x", "fl_y", "cx", "cy", "w", "h", "articulation", "frames"}
    if not required.issubset(meta):
        raise ValueError(f"missing source keys: {sorted(required - set(meta))}")
    lookup = frame_lookup(meta)
    expected = [f"color/train/{i:04d}.png" for i in range(2 * STATE_VIEWS)]
    if list(meta.get("train_filenames", [])) != expected:
        raise ValueError("source train order differs from fixed 200-view endpoint order")
    if [lookup[p]["state"] for p in expected] != [0] * STATE_VIEWS + [1] * STATE_VIEWS:
        raise ValueError("source must contain 100 state-0 then 100 state-1 views")
    for color_rel in expected:
        for modality in ("color", "depth"):
            path = scene_dir / color_rel.replace("color/", f"{modality}/", 1)
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"missing ordinary source asset: {path}")


def camera_points(
    depth_mm: np.ndarray, intrinsics: Sequence[float], stride: int
) -> tuple[np.ndarray, np.ndarray]:
    """Backproject positive depth in SplArt's saved OpenGL camera convention."""
    fx, fy, cx, cy = map(float, intrinsics)
    rows = np.arange(0, depth_mm.shape[0], stride)
    cols = np.arange(0, depth_mm.shape[1], stride)
    vv, uu = np.meshgrid(rows, cols, indexing="ij")
    depth = depth_mm[::stride, ::stride].astype(np.float32) / 1000.0
    valid = depth > 0
    return np.stack(
        (
            (uu[valid] - cx) * depth[valid] / fx,
            (cy - vv[valid]) * depth[valid] / fy,
            -depth[valid],
        ),
        axis=1,
    ), valid


def to_world(points_camera: np.ndarray, c2w: np.ndarray) -> np.ndarray:
    return points_camera @ c2w[:3, :3].T + c2w[:3, 3]


def transform_mobile(
    points: np.ndarray, articulation: Mapping[str, Any], delta_fraction: float
) -> np.ndarray:
    axis = np.asarray(articulation["axis"], dtype=np.float64)
    axis /= np.linalg.norm(axis)
    kind = int(articulation["type"])
    points64 = points.astype(np.float64, copy=False)
    if kind == 1:
        theta = float(articulation["angle"]) * delta_fraction
        pivot = np.asarray(articulation["pivot"], dtype=np.float64)
        skew = np.array(
            [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]]
        )
        rotation = np.eye(3) + math.sin(theta) * skew + (1 - math.cos(theta)) * (skew @ skew)
        points64 = (points64 - pivot) @ rotation.T + pivot
    elif kind == 2:
        points64 = points64 + axis * float(articulation["dist"]) * delta_fraction
    else:
        raise ValueError(f"only one-DOF revolute/prismatic proxy is supported, got type={kind}")
    return points64.astype(np.float32)


@dataclass
class Cloud:
    points: np.ndarray
    colors: np.ndarray
    labels: np.ndarray
    source_states: np.ndarray


def voxel_first(points: np.ndarray, voxel: float) -> np.ndarray:
    if voxel <= 0:
        return np.arange(len(points))
    keys = np.floor(points / voxel).astype(np.int64)
    _, first = np.unique(keys, axis=0, return_index=True)
    return np.sort(first)


def fuse_endpoint_cloud(scene_dir: Path, meta: Mapping[str, Any], stride: int, voxel: float) -> Cloud:
    lookup = frame_lookup(meta)
    intrinsics = (meta["fl_x"], meta["fl_y"], meta["cx"], meta["cy"])
    point_chunks: list[np.ndarray] = []
    color_chunks: list[np.ndarray] = []
    state_chunks: list[np.ndarray] = []
    for index in range(2 * STATE_VIEWS):
        relative = f"color/train/{index:04d}.png"
        frame = lookup[relative]
        color = iio.imread(scene_dir / relative)
        depth = iio.imread(scene_dir / relative.replace("color/", "depth/", 1))
        camera, valid = camera_points(depth, intrinsics, stride)
        sampled_color = color[::stride, ::stride][valid, :3]
        foreground = color[::stride, ::stride][valid, 3] > 0
        if not foreground.any():
            continue
        world = to_world(camera[foreground], np.asarray(frame["transform_matrix"], dtype=float))
        point_chunks.append(world.astype(np.float32))
        color_chunks.append(sampled_color[foreground].astype(np.uint8))
        state_chunks.append(np.full(foreground.sum(), int(frame["state"]), dtype=np.uint8))
    points = np.concatenate(point_chunks)
    colors = np.concatenate(color_chunks)
    states = np.concatenate(state_chunks)
    keep_chunks: list[np.ndarray] = []
    for source_id in (0, 1):
        chosen = np.flatnonzero(states == source_id)
        keep_chunks.append(chosen[voxel_first(points[chosen], voxel)])
    keep = np.sort(np.concatenate(keep_chunks))
    points, colors, states = points[keep], colors[keep], states[keep]
    labels = classify_static_mobile(points, states, meta["articulation"], voxel)
    # Static geometry is shared, so keep it from one endpoint only. Mobile
    # geometry stays represented by both endpoints for complementary coverage.
    keep = (labels == 1) | (states == 0)
    return Cloud(points[keep], colors[keep], labels[keep], states[keep])


def classify_static_mobile(
    points: np.ndarray,
    states: np.ndarray,
    articulation: Mapping[str, Any],
    voxel: float,
) -> np.ndarray:
    """Classify fused endpoint surface by competing rigid-motion hypotheses.

    Train endpoint part masks are absent from the released processed copy.  We
    therefore compare cross-state nearest-neighbour support under (a) identity
    and (b) the known one-DOF endpoint transform, which is available only to the
    builder. Test part masks are reserved for proxy-fidelity validation.
    """
    labels = np.zeros(len(points), dtype=np.uint8)
    p0, p1 = points[states == 0], points[states == 1]
    if not len(p0) or not len(p1):
        raise ValueError("both endpoint point clouds are required")
    tree0, tree1 = cKDTree(p0), cKDTree(p1)
    margin = max(voxel * 0.05, 1e-4)
    for source_state, source_points, target_tree in ((0, p0, tree1), (1, p1, tree0)):
        static_distance = target_tree.query(source_points, k=1, workers=-1)[0]
        moved = transform_mobile(source_points, articulation, 1.0 if source_state == 0 else -1.0)
        mobile_distance = target_tree.query(moved, k=1, workers=-1)[0]
        is_mobile = mobile_distance + margin < static_distance
        labels[np.flatnonzero(states == source_state)] = is_mobile.astype(np.uint8)
    if not np.any(labels == 1) or not np.any(labels == 0):
        raise ValueError("geometry-hypothesis classifier produced a degenerate partition")
    return labels


def cloud_at_fraction(cloud: Cloud, articulation: Mapping[str, Any], fraction: float) -> Cloud:
    points = cloud.points.copy()
    for source_state in (0, 1):
        chosen = (cloud.labels == 1) & (cloud.source_states == source_state)
        points[chosen] = transform_mobile(points[chosen], articulation, fraction - source_state)
    return Cloud(points, cloud.colors, cloud.labels, cloud.source_states)


def geometry_certificate(cloud: Cloud, articulation: Mapping[str, Any], voxel: float) -> dict[str, Any]:
    """Check endpoint alignment and infer the closed side without a label prior."""
    by = {
        (label, state): cloud.points[(cloud.labels == label) & (cloud.source_states == state)]
        for label in (0, 1)
        for state in (0, 1)
    }
    if any(not len(points) for points in by.values()):
        raise ValueError("geometry certificate requires both parts at both endpoints")

    def alignment(
        source: np.ndarray, target: np.ndarray, delta: float
    ) -> dict[str, float]:
        query = transform_mobile(source, articulation, delta) if delta else source
        distances = cKDTree(target).query(query, k=1, workers=-1)[0]
        return {"median_m": float(np.median(distances)), "p95_m": float(np.quantile(distances, 0.95))}

    directional = {
        "static_0_to_1": alignment(by[(0, 0)], by[(0, 1)], 0.0),
        "static_1_to_0": alignment(by[(0, 1)], by[(0, 0)], 0.0),
        "mobile_0_to_1": alignment(by[(1, 0)], by[(1, 1)], 1.0),
        "mobile_1_to_0": alignment(by[(1, 1)], by[(1, 0)], -1.0),
    }
    alignment_result = {
        "directional": directional,
        "worst_median_m": max(values["median_m"] for values in directional.values()),
        "worst_p95_m": max(values["p95_m"] for values in directional.values()),
    }
    alignment_pass = (
        alignment_result["worst_median_m"] <= ALIGNMENT_THRESHOLDS["median_m_max"]
        and alignment_result["worst_p95_m"] <= ALIGNMENT_THRESHOLDS["p95_m_max"]
    )

    static = by[(0, 0)]
    mobile0 = by[(1, 0)]
    static_tree = cKDTree(static)
    contact_epsilon = max(2.0 * voxel, 0.008)
    endpoint: dict[str, dict[str, float]] = {}
    for state in (0, 1):
        mobile = mobile0 if state == 0 else transform_mobile(mobile0, articulation, 1.0)
        nearest = static_tree.query(mobile, k=1, workers=-1)[0]
        combined = np.concatenate((static, mobile))
        diagonal = float(np.linalg.norm(combined.max(axis=0) - combined.min(axis=0)))
        endpoint[f"outside_state_{state}"] = {
            "contact_fraction": float(np.mean(nearest <= contact_epsilon)),
            "contact_gap_q05_m": float(np.quantile(nearest, 0.05)),
            "compactness_diagonal_m": diagonal,
        }
    q0, q1 = endpoint["outside_state_0"], endpoint["outside_state_1"]
    contact_delta = q0["contact_fraction"] - q1["contact_fraction"]
    compact_delta = (q1["compactness_diagonal_m"] - q0["compactness_diagonal_m"]) / max(
        q0["compactness_diagonal_m"], q1["compactness_diagonal_m"], 1e-9
    )
    contact_vote = 0 if contact_delta >= CLOSURE_THRESHOLDS["contact_fraction_margin_min"] else (
        1 if contact_delta <= -CLOSURE_THRESHOLDS["contact_fraction_margin_min"] else None
    )
    compact_vote = 0 if compact_delta >= CLOSURE_THRESHOLDS["compactness_margin_min"] else (
        1 if compact_delta <= -CLOSURE_THRESHOLDS["compactness_margin_min"] else None
    )
    closed_state = contact_vote if contact_vote is not None and compact_vote in (None, contact_vote) else None
    return {
        "alignment": alignment_result,
        "alignment_thresholds": ALIGNMENT_THRESHOLDS,
        "alignment_pass": alignment_pass,
        "closure": {
            "method": "contact_fraction_primary_with_compactness_nonconflict",
            "thresholds": CLOSURE_THRESHOLDS,
            "endpoint": endpoint,
            "contact_vote": contact_vote,
            "compactness_vote": compact_vote,
            "closed_query": f"outside_state_{closed_state}" if closed_state is not None else "closed_unknown",
            "identifiable": closed_state is not None,
            "proxy_semantics": "geometry_contact_compactness_without_raw_urdf",
        },
    }


def quality_gate(validation: Mapping[str, Any]) -> dict[str, Any]:
    mean = validation.get("mean", {})
    finite = all(np.isfinite(float(mean.get(key, np.nan))) for key in (
        "foreground_iou", "static_iou", "mobile_iou", "psnr", "depth_mae_m"
    ))
    count_exact = validation.get("view_count") == 6
    passed = finite and count_exact and (
        mean["foreground_iou"] >= PROXY_QUALITY_THRESHOLDS["foreground_iou_min"]
        and mean["static_iou"] >= PROXY_QUALITY_THRESHOLDS["static_iou_min"]
        and mean["mobile_iou"] >= PROXY_QUALITY_THRESHOLDS["mobile_iou_min"]
        and mean["psnr"] >= PROXY_QUALITY_THRESHOLDS["psnr_min"]
        and mean["depth_mae_m"] <= PROXY_QUALITY_THRESHOLDS["depth_mae_m_max"]
    )
    return {
        "pass": bool(passed),
        "finite": bool(finite),
        "validation_count_exact": bool(count_exact),
        "required_validation_count": 6,
        "thresholds": PROXY_QUALITY_THRESHOLDS,
    }


def render_cloud(
    cloud: Cloud,
    frame: Mapping[str, Any],
    intrinsics: Sequence[float],
    size: tuple[int, int],
    splat_radius: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fx, fy, cx, cy = map(float, intrinsics)
    width, height = size
    c2w = np.asarray(frame["transform_matrix"], dtype=float)
    camera = (cloud.points.astype(float) - c2w[:3, 3]) @ c2w[:3, :3]
    depth = -camera[:, 2]
    valid = depth > 1e-4
    u = np.rint(fx * camera[:, 0] / np.maximum(depth, 1e-4) + cx).astype(np.int32)
    v = np.rint(cy - fy * camera[:, 1] / np.maximum(depth, 1e-4)).astype(np.int32)
    valid &= (u >= 0) & (u < width) & (v >= 0) & (v < height)
    source = np.flatnonzero(valid)
    u, v, depth = u[valid], v[valid], depth[valid].astype(np.float32)
    pixel_chunks: list[np.ndarray] = []
    point_chunks: list[np.ndarray] = []
    depth_chunks: list[np.ndarray] = []
    for dv in range(-splat_radius, splat_radius + 1):
        for du in range(-splat_radius, splat_radius + 1):
            uu, vv = u + du, v + dv
            inside = (uu >= 0) & (uu < width) & (vv >= 0) & (vv < height)
            pixel_chunks.append((vv[inside] * width + uu[inside]).astype(np.int64))
            point_chunks.append(source[inside])
            depth_chunks.append(depth[inside])
    pixels = np.concatenate(pixel_chunks)
    point_ids = np.concatenate(point_chunks)
    depths = np.concatenate(depth_chunks)
    rgba = np.zeros((height * width, 4), dtype=np.uint8)
    depth_out = np.zeros(height * width, dtype=np.uint16)
    labels = np.full(height * width, 255, dtype=np.uint8)
    if len(pixels):
        order = np.lexsort((depths, pixels))
        sorted_pixels = pixels[order]
        first = np.r_[True, sorted_pixels[1:] != sorted_pixels[:-1]]
        chosen = order[first]
        target = pixels[chosen]
        selected = point_ids[chosen]
        rgba[target, :3] = cloud.colors[selected]
        rgba[target, 3] = 255
        depth_out[target] = np.clip(np.rint(depths[chosen] * 1000), 0, 65535).astype(np.uint16)
        labels[target] = cloud.labels[selected]
    return rgba.reshape(height, width, 4), depth_out.reshape(height, width), labels.reshape(height, width)


def source_hashes(scene_dir: Path, relatives: Iterable[str]) -> dict[str, str]:
    return {relative: sha256_file(scene_dir / relative) for relative in sorted(set(relatives))}


def plan_manifest(source_root: Path, output: Path, scenes: Sequence[str], mode: str) -> None:
    rng = secrets.SystemRandom()
    pairs = {
        scene: (0.25, 0.75) if mode == "smoke" else (round(rng.uniform(0.15, 0.35), 8), round(rng.uniform(0.65, 0.85), 8))
        for scene in scenes
    }
    episodes: dict[str, Any] = {}
    for scene in scenes:
        scene_dir = source_root / scene
        meta = load_json(scene_dir / "transforms.json")
        validate_source_scene(scene_dir, meta)
        lower, upper = pairs[scene]
        lookup = frame_lookup(meta)
        endpoint_views: dict[str, list[dict[str, Any]]] = {"outside_state_0": [], "outside_state_1": []}
        sealed_asset_root = output.parent / "evaluator-assets" / scene
        per_query_index = {"outside_state_0": 0, "outside_state_1": 0}
        for relative in meta.get("val_filenames", []):
            frame = lookup[relative]
            query = "outside_state_0" if int(frame["state"]) == 0 else "outside_state_1"
            assets = {
                "color": relative,
                "depth": relative.replace("color/", "depth/", 1),
                "part_seg": str(
                    sealed_asset_root / query / f"{per_query_index[query]:04d}.png"
                ),
            }
            per_query_index[query] += 1
            endpoint_views[query].append(
                {
                    "camera_to_world": frame["transform_matrix"],
                    "assets": assets,
                    "source_sha256": source_hashes(scene_dir, (assets["color"], assets["depth"])),
                    "part_seg_status": "builder_geometry_proxy_pending",
                }
            )
        episodes[scene] = {
            "scene_id": scene,
            "proxy_status": "rgbd_partseg_fusion_without_raw_urdf",
            "physical_coordinate": {
                "observed_state_0": lower,
                "observed_state_1": upper,
                "lower_limit": 0.0,
                "upper_limit": 1.0,
            },
            "target": {
                "local_scalars": {
                    "outside_state_0": -lower / (upper - lower),
                    "outside_state_1": (1.0 - lower) / (upper - lower),
                },
                "closed_query": "pending_geometry_certificate",
                "closed_semantics": "determined_postbuild_without_state_prior",
            },
            "camera": {
                "fl_x": meta["fl_x"], "fl_y": meta["fl_y"], "cx": meta["cx"], "cy": meta["cy"],
                "w": meta["w"], "h": meta["h"], "pose_convention": "nerfstudio_opengl_cam2raw",
                "depth_encoding": "uint16_millimetres",
                "color_encoding": "uint8_rgba_straight_alpha",
            },
            "articulation": dict(meta["articulation"]),
            "endpoint_metrics": {q: {"views": views} for q, views in endpoint_views.items()},
        }
    payload = {
        "schema": SEALED_SCHEMA,
        "mode": mode,
        "scene_order": list(scenes),
        "source_root": str(source_root.resolve()),
        "episodes": episodes,
        "leakage_boundary": "builder_and_evaluator_only_never_model_training",
    }
    write_json(output, payload)


def public_queries() -> dict[str, Any]:
    return {
        "schema": QUERY_SCHEMA,
        "coordinate": "observed_span_local_state_0_to_state_1",
        "queries": [
            {"query_id": "outside_state_0", "local_direction": -1},
            {"query_id": "outside_state_1", "local_direction": 1},
        ],
    }


def render_split(
    output_scene: Path,
    split: str,
    frames: Sequence[tuple[Mapping[str, Any], int]],
    clouds: Mapping[int, Cloud],
    intrinsics: Sequence[float],
    size: tuple[int, int],
    splat_radius: int,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for index, (frame, state) in enumerate(frames):
        color, depth, part = render_cloud(clouds[state], frame, intrinsics, size, splat_radius)
        relative = f"color/{split}/{index:04d}.png"
        for modality in ("color", "depth", "part-seg"):
            (output_scene / modality / split).mkdir(parents=True, exist_ok=True)
        iio.imwrite(output_scene / relative, color)
        iio.imwrite(output_scene / relative.replace("color/", "depth/", 1), depth)
        iio.imwrite(output_scene / relative.replace("color/", "part-seg/", 1), part)
        output.append({"file_path": relative, "transform_matrix": frame["transform_matrix"], "state": state})
    return output


def validate_proxy(scene_dir: Path, meta: Mapping[str, Any], cloud: Cloud, count: int, radius: int) -> dict[str, Any]:
    lookup = frame_lookup(meta)
    candidates = [lookup[p] for p in meta.get("test_filenames", []) if 0 <= float(lookup[p]["state"]) <= 1]
    if count <= 0 or not candidates:
        return {"view_count": 0, "scope": "proxy_fidelity_only"}
    chosen = np.linspace(0, len(candidates) - 1, min(count, len(candidates))).round().astype(int)
    intrinsics = (meta["fl_x"], meta["fl_y"], meta["cx"], meta["cy"])
    size = (int(meta["w"]), int(meta["h"]))
    rows: list[dict[str, float]] = []
    for index in chosen:
        frame = candidates[int(index)]
        predicted = render_cloud(
            cloud_at_fraction(cloud, meta["articulation"], float(frame["state"])), frame, intrinsics, size, radius
        )
        relative = str(frame["file_path"])
        gt_color = iio.imread(scene_dir / relative)[..., :3]
        gt_depth = iio.imread(scene_dir / relative.replace("color/", "depth/", 1))
        gt_part = iio.imread(scene_dir / relative.replace("color/", "part-seg/", 1))
        color, depth, part = predicted
        gt_fg, pred_fg = np.isin(gt_part, (0, 1)), np.isin(part, (0, 1))
        union, intersection = gt_fg | pred_fg, gt_fg & pred_fg
        mse = np.mean(((gt_color[union].astype(float) - color[union, :3].astype(float)) / 255) ** 2)
        depth_valid = intersection & (gt_depth > 0) & (depth > 0)
        row = {
            "foreground_iou": float(intersection.sum() / max(1, union.sum())),
            "psnr": float(-10 * math.log10(max(float(mse), 1e-12))),
            "depth_mae_m": float(np.mean(np.abs(gt_depth[depth_valid].astype(float) - depth[depth_valid])) / 1000)
            if depth_valid.any()
            else float("nan"),
        }
        for name, label_id in (("static_iou", 0), ("mobile_iou", 1)):
            a, b = gt_part == label_id, part == label_id
            row[name] = float((a & b).sum() / max(1, (a | b).sum()))
        rows.append(row)
    return {
        "view_count": len(rows),
        "mean": {key: float(np.nanmean([row[key] for row in rows])) for key in rows[0]},
        "scope": "proxy_fidelity_only_not_model_performance",
    }


def validation_asset_hashes(scene_dir: Path, meta: Mapping[str, Any], count: int) -> dict[str, str]:
    """Bind the exact hidden continuous-state assets selected by validate_proxy."""
    lookup = frame_lookup(meta)
    candidates = [lookup[p] for p in meta.get("test_filenames", []) if 0 <= float(lookup[p]["state"]) <= 1]
    chosen = np.linspace(0, len(candidates) - 1, min(count, len(candidates))).round().astype(int)
    relatives: list[str] = []
    for index in chosen:
        color = str(candidates[int(index)]["file_path"])
        relatives.extend(
            (color, color.replace("color/", "depth/", 1), color.replace("color/", "part-seg/", 1))
        )
    return source_hashes(scene_dir, relatives)


def build_scene(
    source_root: Path,
    output_root: Path,
    sealed: Mapping[str, Any],
    scene: str,
    stride: int,
    voxel: float,
    radius: int,
    validation_count: int,
    sealed_manifest_path: Path,
) -> dict[str, Any]:
    source_scene, output_scene = source_root / scene, output_root / scene
    if output_scene.exists():
        shutil.rmtree(output_scene)
    meta = load_json(source_scene / "transforms.json")
    validate_source_scene(source_scene, meta)
    physical = sealed["episodes"][scene]["physical_coordinate"]
    fractions = (float(physical["observed_state_0"]), float(physical["observed_state_1"]))
    cloud = fuse_endpoint_cloud(source_scene, meta, stride, voxel)
    clouds = {state: cloud_at_fraction(cloud, meta["articulation"], fraction) for state, fraction in enumerate(fractions)}
    lookup = frame_lookup(meta)
    train = [(lookup[f"color/train/{i:04d}.png"], 0 if i < 100 else 1) for i in range(200)]
    val_names = list(meta["val_filenames"])
    val = [(lookup[p], 0 if i < len(val_names) // 2 else 1) for i, p in enumerate(val_names)]
    intrinsics = (meta["fl_x"], meta["fl_y"], meta["cx"], meta["cy"])
    size = (int(meta["w"]), int(meta["h"]))
    train_frames = render_split(output_scene, "train", train, clouds, intrinsics, size, radius)
    val_frames = render_split(output_scene, "val", val, clouds, intrinsics, size, radius)
    endpoint_asset_hashes: dict[str, str] = {}
    for query, endpoint_state in (("outside_state_0", 0), ("outside_state_1", 1)):
        views = sealed["episodes"][scene]["endpoint_metrics"][query]["views"]
        for view_index, view in enumerate(views):
            frame = val[view_index + (0 if endpoint_state == 0 else len(val) // 2)][0]
            _, _, part = render_cloud(
                cloud_at_fraction(cloud, meta["articulation"], float(endpoint_state)),
                frame,
                intrinsics,
                size,
                radius,
            )
            target = Path(view["assets"]["part_seg"])
            target.parent.mkdir(parents=True, exist_ok=True)
            iio.imwrite(target, part)
            endpoint_asset_hashes[str(target)] = sha256_file(target)
    public_meta = {
        **{key: meta[key] for key in ("fl_x", "fl_y", "cx", "cy", "w", "h")},
        "articulation": {"type": int(meta["articulation"]["type"])},
        "benchmark": {"schema": PUBLIC_SCHEMA, "coordinate": "local_observed_span"},
        "frames": train_frames + val_frames,
        "train_filenames": [frame["file_path"] for frame in train_frames],
        "val_filenames": [frame["file_path"] for frame in val_frames],
    }
    write_json(output_scene / "transforms.json", public_meta)
    write_json(output_scene / "endpoint_queries.json", public_queries())
    validation = validate_proxy(scene_dir=source_scene, meta=meta, cloud=cloud, count=validation_count, radius=radius)
    quality = quality_gate(validation)
    geometry = geometry_certificate(cloud, meta["articulation"], voxel)
    public_files = sorted(path for path in output_scene.rglob("*") if path.is_file())
    payload_hashes = {path.relative_to(output_scene).as_posix(): sha256_file(path) for path in public_files}
    payload_tree_sha = canonical_sha256(payload_hashes)
    binding_id = hashlib.sha256(
        f"{scene}:{canonical_sha256(sealed)}:{payload_tree_sha}".encode()
    ).hexdigest()
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "scene_id": scene,
        "proxy_status": "rgbd_partseg_fusion_without_raw_urdf",
        "public_root": str(output_scene.resolve()),
        "sealed_manifest_sha256": canonical_sha256(sealed),
        "source_stride": stride,
        "voxel_size_m": voxel,
        "splat_radius_px": radius,
        "fused_point_count": int(len(cloud.points)),
        "public_payload_file_count": len(public_files),
        "public_payload_tree_sha256": payload_tree_sha,
        "hash_scope": "model_payload_only_excludes_this_receipt_and_complete",
        "binding_id": binding_id,
        "continuous_state_proxy_validation": validation,
        "quality_gate": quality,
        "geometry_alignment": {
            "result": geometry["alignment"],
            "thresholds": geometry["alignment_thresholds"],
            "pass": geometry["alignment_pass"],
        },
    }
    write_json(output_scene / "PROXY_RECEIPT.json", receipt)
    completion_pass = quality["pass"] and geometry["alignment_pass"] and geometry["closure"]["identifiable"]
    if completion_pass:
        write_json(
            output_scene / "COMPLETE.json",
            {
                "schema": "splart-middle-state-scene-complete-v1",
                "scene_id": scene,
                "proxy_receipt_sha256": sha256_file(output_scene / "PROXY_RECEIPT.json"),
                "sealed_manifest_sha256": canonical_sha256(sealed),
                "binding_id": binding_id,
                "all_predeclared_gates_pass": True,
                "closed_endpoint_certified": True,
            },
        )
    source_relatives = [
        f"{modality}/train/{index:04d}.png"
        for modality in ("color", "depth")
        for index in range(2 * STATE_VIEWS)
    ]
    public_all = sorted(path for path in output_scene.rglob("*") if path.is_file())
    episode_truth = copy.deepcopy(sealed["episodes"][scene])
    episode_truth["target"]["closed_query"] = geometry["closure"]["closed_query"]
    episode_truth["target"]["closed_semantics"] = geometry["closure"]["proxy_semantics"]
    for query in ("outside_state_0", "outside_state_1"):
        for view in episode_truth["endpoint_metrics"][query]["views"]:
            target = Path(view["assets"]["part_seg"])
            view["part_seg_status"] = "builder_geometry_proxy_complete"
            view["part_seg_sha256"] = sha256_file(target)
    postbuild = {
        "schema": "splart-endpoint-extrapolation-postbuild-seal-v1",
        "scene_id": scene,
        "binding_id": binding_id,
        "pass": completion_pass,
        "sealed_plan_file_sha256": sha256_file(sealed_manifest_path),
        "sealed_plan_content_sha256": canonical_sha256(sealed),
        "builder_sha256": sha256_file(Path(__file__)),
        "public_payload_tree_sha256": payload_tree_sha,
        "public_payload_sha256": payload_hashes,
        "public_complete_tree_sha256": canonical_sha256(
            {path.relative_to(output_scene).as_posix(): sha256_file(path) for path in public_all}
        ),
        "public_complete_sha256": {
            path.relative_to(output_scene).as_posix(): sha256_file(path) for path in public_all
        },
        "source_synthesis_sha256": source_hashes(source_scene, source_relatives),
        "source_transforms_sha256": sha256_file(source_scene / "transforms.json"),
        "continuous_validation_asset_sha256": validation_asset_hashes(source_scene, meta, validation_count),
        "sealed_endpoint_part_seg_sha256": endpoint_asset_hashes,
        "evaluator_truth": episode_truth,
        "quality_gate": quality,
        "geometry_certificate": geometry,
    }
    postbuild_path = sealed_manifest_path.parent / "postbuild" / f"{scene}.json"
    write_json(postbuild_path, postbuild)
    if not completion_pass:
        raise RuntimeError(f"predeclared proxy gates failed for {scene}; COMPLETE withheld")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan")
    plan.add_argument("--source-root", required=True, type=Path)
    plan.add_argument("--output", required=True, type=Path)
    plan.add_argument("--mode", required=True, choices=("smoke", "formal"))
    plan.add_argument("--scenes", nargs="+", default=list(DEV3))
    build = commands.add_parser("build")
    build.add_argument("--source-root", required=True, type=Path)
    build.add_argument("--output-root", required=True, type=Path)
    build.add_argument("--sealed-manifest", required=True, type=Path)
    build.add_argument("--scenes", nargs="+", default=list(DEV3))
    build.add_argument("--stride", type=int, default=8)
    build.add_argument("--voxel", type=float, default=0.004)
    build.add_argument("--splat-radius", type=int, default=1)
    build.add_argument("--validation-count", type=int, default=6)
    args = parser.parse_args()
    if args.command == "plan":
        plan_manifest(args.source_root, args.output, tuple(args.scenes), args.mode)
        print(json.dumps({"path": str(args.output), "sha256": sha256_file(args.output)}))
        return
    sealed = load_json(args.sealed_manifest)
    sealed_order = tuple(sealed.get("scene_order", ()))
    requested = tuple(args.scenes)
    if sealed.get("schema") != SEALED_SCHEMA:
        raise ValueError("sealed manifest schema mismatch")
    if requested != tuple(scene for scene in sealed_order if scene in set(requested)):
        raise ValueError("requested scenes are not an ordered subset of the sealed scene order")
    if args.sealed_manifest.resolve().is_relative_to(args.output_root.resolve()):
        raise ValueError("sealed manifest must live outside the public root")
    receipts = []
    for scene in args.scenes:
        receipts.append(
            build_scene(
                args.source_root, args.output_root, sealed, scene, args.stride, args.voxel,
                args.splat_radius, args.validation_count, args.sealed_manifest
            )
        )
        print(json.dumps({"scene": scene, "status": "complete"}))
    if all((args.output_root / scene / "COMPLETE.json").is_file() for scene in sealed_order):
        write_json(
            args.output_root / "COMPLETE.json",
            {
                "schema": "splart-middle-state-dev3-complete-v1",
                "scene_order": list(sealed_order),
                "sealed_manifest_sha256": canonical_sha256(sealed),
                "scene_receipt_sha256": {
                    scene: sha256_file(args.output_root / scene / "PROXY_RECEIPT.json") for scene in sealed_order
                },
            },
        )


if __name__ == "__main__":
    main()
