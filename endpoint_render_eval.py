"""Render and score candidate-dependent endpoint predictions.

This is the evaluator-only half of the benchmark.  It verifies the frozen
checkpoint/source/public tree, renders the held-out endpoint cameras at the
candidate's own predicted scalars, hashes those renders, and only then computes
the official image/depth/segmentation and articulation metrics.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Any, Mapping

from endpoint_eval import aggregate_scenes, evaluate_scene
from splart.endpoint_baselines import QUERY_IDS, content_sha256, sha256_file, validate_public_scene, write_json_atomic


def load_json(path: Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def scene_record(payload: Mapping[str, Any], scene_id: str) -> Mapping[str, Any]:
    episodes = payload.get("episodes")
    if not isinstance(episodes, Mapping) or scene_id not in episodes or not isinstance(episodes[scene_id], Mapping):
        raise ValueError(f"sealed manifest lacks episode {scene_id}")
    return episodes[scene_id]


def verify_hash(path: Path, expected: str, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be an ordinary file: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{label} sha256 mismatch: {actual} != {expected}")


def verify_candidate(prediction: Mapping[str, Any], public_scene: Path) -> dict[str, Path]:
    if prediction.get("baseline") != "splart-middle":
        raise ValueError("candidate renderer requires the splart-middle baseline")
    public = validate_public_scene(public_scene)
    public_input = prediction.get("public_input", {})
    if (
        not isinstance(public_input, Mapping)
        or public_input.get("root") != str(Path(public_scene).resolve())
        or public_input.get("tree_sha256") != public["public_tree_sha256"]
        or public_input.get("file_count") != len(public["public_tree"])
    ):
        raise ValueError("candidate public-input provenance mismatch")

    candidate = prediction.get("candidate")
    if not isinstance(candidate, Mapping):
        raise ValueError("prediction lacks candidate checkpoint provenance")
    paths: dict[str, Path] = {}
    for key in ("checkpoint", "config", "dataparser_transforms"):
        record = candidate.get(key)
        if not isinstance(record, Mapping):
            raise ValueError(f"candidate provenance lacks {key}")
        path = Path(record.get("path", "")).resolve()
        verify_hash(path, str(record.get("sha256", "")), f"candidate {key}")
        paths[key] = path
    if candidate["checkpoint"].get("step") != 24_999 or paths["checkpoint"].name != "step-000024999.ckpt":
        raise ValueError("candidate is not the fixed final training checkpoint")

    source = candidate.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("candidate provenance lacks source")
    source_dir = Path(source.get("path", "")).resolve()
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=source_dir, check=True, capture_output=True, text=True, timeout=30
    ).stdout.strip()
    tree = subprocess.run(
        ["git", "rev-parse", "HEAD^{tree}"], cwd=source_dir, check=True, capture_output=True, text=True, timeout=30
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=source_dir, check=True, capture_output=True, text=True, timeout=30
    ).stdout.strip()
    if head != source.get("commit") or tree != source.get("tree") or dirty:
        raise ValueError("candidate source provenance/cleanliness mismatch")
    paths["source"] = source_dir
    paths["checkpoint_dir"] = paths["checkpoint"].parent
    return paths


def resolve_asset(source_root: Path, scene_id: str, asset: str) -> Path:
    path = Path(asset)
    if not path.is_absolute():
        path = Path(source_root) / scene_id / path
    return path.resolve()


def expected_asset_hash(view: Mapping[str, Any], modality: str, asset: str) -> str:
    hashes = view.get("source_sha256", view.get("asset_sha256"))
    if not isinstance(hashes, Mapping):
        raise ValueError(f"sealed view lacks source_sha256 for {modality}")
    expected = hashes.get(asset, hashes.get(modality))
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError(f"sealed view lacks a valid sha256 for {modality}")
    return expected


def camera_from_view(episode: Mapping[str, Any], view: Mapping[str, Any], torch: Any, Cameras: Any) -> Any:
    camera = episode.get("camera")
    if not isinstance(camera, Mapping) or camera.get("pose_convention") != "nerfstudio_opengl_cam2raw":
        raise ValueError("sealed episode camera contract is absent or unsupported")
    c2w = torch.as_tensor(view.get("camera_to_world"), dtype=torch.float32)
    if c2w.shape == (4, 4):
        c2w = c2w[:3, :4]
    if c2w.shape != (3, 4) or not torch.isfinite(c2w).all():
        raise ValueError("held-out camera_to_world must be finite 3x4/4x4")

    def value(*names: str) -> float:
        for name in names:
            if name in camera:
                result = float(camera[name])
                if math.isfinite(result):
                    return result
        raise ValueError(f"camera contract lacks {names}")

    return Cameras(
        camera_to_worlds=c2w[None],
        fx=value("fx", "fl_x"),
        fy=value("fy", "fl_y"),
        cx=value("cx"),
        cy=value("cy"),
        width=int(value("w", "width")),
        height=int(value("h", "height")),
    )


def candidate_articulation_metrics(
    renderer: Any, episode: Mapping[str, Any], predicted_scalars: Mapping[str, float]
) -> dict:
    import torch
    from torch.nn.functional import normalize

    from splart.articulation_params import ArticulationParams, ArticulationType

    raw = episode.get("articulation")
    if not isinstance(raw, Mapping):
        raise ValueError("sealed episode lacks articulation ground truth")
    type_value = raw.get("type")
    if isinstance(type_value, str):
        type_value = {
            "revolute": ArticulationType.REVOLUTE,
            "prismatic": ArticulationType.PRISMATIC,
            "cylindrical": ArticulationType.CYLINDRICAL,
        }.get(type_value.lower())
    articulation_type = ArticulationType(int(type_value))
    device = renderer.model.device
    transform = renderer.ns_T_raw
    gt: dict[str, Any] = {"type": articulation_type}
    if "axis" in raw:
        raw_axis = torch.as_tensor(raw["axis"], dtype=torch.float32, device=device)
        gt["axis"] = normalize(transform[:3, :3] @ raw_axis, dim=0)
    if "pivot" in raw:
        raw_pivot = torch.as_tensor([*raw["pivot"], 1.0], dtype=torch.float32, device=device)
        gt["pivot"] = (transform @ raw_pivot)[:3]
    if "angle" in raw:
        gt["angle"] = float(raw["angle"])
    if "dist" in raw:
        gt["dist"] = float(raw["dist"]) * renderer.ns_s_raw
    gt_params = ArticulationParams(gt, trainable=False).to(device)

    learned = renderer.model.articulation_params
    valid = bool(learned.is_valid.item())
    type_correct = valid and int(learned.articulation_type.item()) == int(articulation_type)
    result: dict[str, Any] = {"valid": valid, "type_correct": type_correct}
    if not type_correct:
        return result
    span = float(predicted_scalars[QUERY_IDS[1]] - predicted_scalars[QUERY_IDS[0]])
    candidate_dict: dict[str, Any] = {"type": articulation_type, "axis": learned.axis.detach()}
    if articulation_type != ArticulationType.PRISMATIC:
        candidate_dict.update({"pivot": learned.pivot.detach(), "angle": learned.angle.detach() * span})
    if articulation_type != ArticulationType.REVOLUTE:
        candidate_dict["dist"] = learned.dist.detach() * span
    candidate_params = ArticulationParams(candidate_dict, trainable=False).to(device)
    metrics = renderer.model.evaluate_articulation(candidate_params, gt_params)
    for key in {"pivot", "t"}:
        if key in metrics:
            metrics[key] /= renderer.ns_s_raw
    result.update(metrics)
    return result


def gaussian_physical_certificate(model: Any, scalar: float) -> dict[str, Any]:
    """Deterministic candidate-derived contact/penetration proxy."""

    import torch
    from pytorch3d.transforms import axis_angle_to_matrix
    from torch.nn.functional import normalize

    from splart.articulation_params import ArticulationType

    with torch.inference_mode():
        states = model.states.squeeze(-1)
        means = model.means
        radii = model.scales.exp().amax(dim=-1)
        opacity = model.opacities.sigmoid().squeeze(-1)
        mobility = model.mobilities.sigmoid().squeeze(-1)
        target = ~model.mobilities.squeeze(-1).isnan()
        static_mask = target & (opacity >= 0.1) & (mobility < 0.5)
        mobile_mask = target & (opacity >= 0.1) & (mobility >= 0.5)
        if not static_mask.any() or not mobile_mask.any():
            return {"valid": False, "reason": "empty thresholded static/mobile support"}
        static_means = means[static_mask]
        static_radii = radii[static_mask]
        mobile_means = means[mobile_mask].clone()
        mobile_radii = radii[mobile_mask]
        mobile_states = states[mobile_mask]
        axis = normalize(model.articulation_params.axis, dim=0)
        factors = torch.where(mobile_states == 0, float(scalar), float(scalar) - 1.0)
        articulation_type = ArticulationType(int(model.articulation_params.articulation_type.item()))
        if articulation_type in {ArticulationType.REVOLUTE, ArticulationType.CYLINDRICAL}:
            pivot = model.articulation_params.pivot
            rotations = axis_angle_to_matrix(axis[None] * (model.articulation_params.angle * factors)[:, None])
            mobile_means = torch.bmm(rotations, (mobile_means - pivot)[:, :, None]).squeeze(-1) + pivot
        if articulation_type in {ArticulationType.PRISMATIC, ArticulationType.CYLINDRICAL}:
            mobile_means = mobile_means + axis * (model.articulation_params.dist * factors)[:, None]

        gaps = []
        for start in range(0, len(mobile_means), 256):
            distances = torch.cdist(mobile_means[start : start + 256], static_means)
            nearest_distance, nearest_index = distances.min(dim=1)
            gaps.append(nearest_distance - mobile_radii[start : start + 256] - static_radii[nearest_index])
        gap = torch.cat(gaps)
        min_abs_gap = gap.abs().min().item()
        max_penetration = torch.relu(-gap).max().item()
        terminal_contact_valid = min_abs_gap <= 0.01 and max_penetration <= 0.02
        return {
            "valid": True,
            "static_count": int(static_mask.sum().item()),
            "mobile_count": int(mobile_mask.sum().item()),
            "opacity_threshold": 0.1,
            "mobility_threshold": 0.5,
            "contact_tolerance_model_units": 0.01,
            "penetration_tolerance_model_units": 0.02,
            "minimum_absolute_surface_gap": min_abs_gap,
            "penetration_depth": max_penetration,
            "terminal_contact_valid": terminal_contact_valid,
        }


def evaluate_candidate(
    prediction_path: Path, sealed_path: Path, public_scene: Path, artifact_root: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    import numpy as np
    import torch
    from imageio.v3 import imread, imwrite
    from nerfstudio.cameras.cameras import Cameras
    from torch.nn.functional import one_hot
    from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    from vis_utils.utils import to_pt_depth

    from splart_renderer import SplartRenderer

    prediction = load_json(prediction_path)
    sealed = load_json(sealed_path)
    scene_id = prediction.get("scene_id")
    if not isinstance(scene_id, str):
        raise ValueError("prediction scene_id missing")
    episode = scene_record(sealed, scene_id)
    paths = verify_candidate(prediction, public_scene)
    if artifact_root.exists():
        raise FileExistsError(f"candidate render artifact root already exists: {artifact_root}")
    artifact_root.mkdir(parents=True)

    renderer = SplartRenderer(
        paths["checkpoint_dir"], load_step=24_999, data_dir=Path(public_scene), data_splits=("train",)
    )
    device = renderer.model.device
    psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips = LearnedPerceptualImagePatchSimilarity(normalize=True).to(device)
    source_root = Path(sealed.get("source_root", ""))
    if not source_root.is_absolute():
        raise ValueError("sealed source_root must be absolute")
    predicted_rows = {row["query_id"]: row for row in prediction["endpoint_predictions"]}
    predicted_scalars = {query_id: float(predicted_rows[query_id]["predicted_local_scalar"]) for query_id in QUERY_IDS}
    measurement: dict[str, Any] = {
        "schema_version": "splart-endpoint-candidate-measurement/v1",
        "scene_id": scene_id,
        "candidate_checkpoint_sha256": prediction["candidate"]["checkpoint"]["sha256"],
        "prediction_sha256": sha256_file(prediction_path),
        "sealed_manifest_sha256": sha256_file(sealed_path),
        "endpoint_metrics": {},
    }

    for query_id in QUERY_IDS:
        descriptor = episode.get("endpoint_metrics", {}).get(query_id)
        if not isinstance(descriptor, Mapping) or not isinstance(descriptor.get("views"), list):
            raise ValueError(f"sealed endpoint descriptor missing for {query_id}")
        query_dir = artifact_root / query_id
        query_dir.mkdir()
        view_metrics = []
        for index, view in enumerate(descriptor["views"]):
            if not isinstance(view, Mapping):
                raise ValueError("sealed endpoint view must be an object")
            assets = view.get("assets")
            if not isinstance(assets, Mapping) or set(assets) != {"color", "depth", "part_seg"}:
                raise ValueError("sealed endpoint view must bind color/depth/part_seg")
            resolved = {key: resolve_asset(source_root, scene_id, str(value)) for key, value in assets.items()}
            for modality, path in resolved.items():
                verify_hash(path, expected_asset_hash(view, modality, str(assets[modality])), f"GT {modality}")

            camera = camera_from_view(episode, view, torch, Cameras)
            outputs = renderer.render_view(
                camera,
                articulation_state=predicted_scalars[query_id],
                vis_articulation=False,
                pose_type="cam2raw",
                color_only=False,
                gen_part_seg=True,
            )
            gt_color_np = imread(resolved["color"])
            gt_color = renderer.model.get_gt_img(torch.from_numpy(np.asarray(gt_color_np)).to(device))
            gt_color = renderer.model.composite_with_background(gt_color, outputs["background"])
            pred_color = outputs["rgb"]
            pred_cf = pred_color.permute(2, 0, 1)[None]
            gt_cf = gt_color.permute(2, 0, 1)[None]

            gt_depth_np = imread(resolved["depth"])
            gt_depth = to_pt_depth(gt_depth_np).to(device).squeeze()
            pred_depth = outputs["depth"].squeeze()
            valid_depth = torch.as_tensor(gt_depth_np > 0, device=device)
            if not valid_depth.any():
                raise ValueError("held-out depth has no valid pixels")

            gt_seg_np = np.asarray(imread(resolved["part_seg"])).copy()
            gt_seg_np[gt_seg_np == 255] = 2
            gt_seg = one_hot(torch.from_numpy(gt_seg_np).to(device=device, dtype=torch.long), num_classes=3)
            pred_seg = outputs["part-seg"]
            pred_seg = torch.dstack((pred_seg[..., 2], pred_seg[..., 0], 1 - pred_seg.sum(dim=-1)))
            pred_seg = one_hot(torch.argmax(pred_seg, dim=-1), num_classes=3)
            ious = torch.nan_to_num(
                torch.min(gt_seg, pred_seg).sum(dim=(0, 1)) / torch.max(gt_seg, pred_seg).sum(dim=(0, 1)), nan=1.0
            )
            values = {
                "view_id": str(view.get("view_id", f"{index:04d}")),
                "psnr": float(psnr(pred_cf, gt_cf).item()),
                "ssim": float(ssim(pred_cf, gt_cf).item()),
                "lpips": float(lpips(pred_cf, gt_cf).item()),
                "depth_mae": float((gt_depth - pred_depth)[valid_depth].abs().mean().item()),
                "part_seg_ious": [float(value) for value in ious.tolist()],
            }

            color_path = query_dir / f"{index:04d}-color.png"
            depth_path = query_dir / f"{index:04d}-depth.npy"
            seg_path = query_dir / f"{index:04d}-part-seg.png"
            imwrite(color_path, (pred_color.clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8))
            np.save(depth_path, pred_depth.cpu().numpy(), allow_pickle=False)
            seg_labels = torch.argmax(pred_seg, dim=-1).cpu().numpy().astype(np.uint8)
            seg_labels[seg_labels == 2] = 255
            imwrite(seg_path, seg_labels)
            values["candidate_artifacts"] = {
                path.name: sha256_file(path) for path in (color_path, depth_path, seg_path)
            }
            view_metrics.append(values)
        measurement["endpoint_metrics"][query_id] = {"views": view_metrics}

    measurement["articulation"] = candidate_articulation_metrics(renderer, episode, predicted_scalars)
    physical_queries = {
        query_id: gaussian_physical_certificate(renderer.model, predicted_scalars[query_id]) for query_id in QUERY_IDS
    }
    valid_certificates = [value for value in physical_queries.values() if value.get("valid")]
    if len(valid_certificates) != len(QUERY_IDS):
        raise RuntimeError("candidate physical certificate lacks complete endpoint coverage")
    measurement["physics"] = {
        "method": "candidate-gaussian-radius-proxy-v1",
        "queries": physical_queries,
        "terminal_contact_valid": bool(valid_certificates)
        and all(value["terminal_contact_valid"] for value in valid_certificates),
        "penetration_depth": max((value["penetration_depth"] for value in valid_certificates), default=float("inf")),
    }
    measurement["content_sha256"] = content_sha256(measurement)
    scored = aggregate_scenes([evaluate_scene(prediction, sealed, measurement)])
    return measurement, scored


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--sealed-manifest", type=Path, required=True)
    parser.add_argument("--public-scene", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--measurement-output", type=Path, required=True)
    parser.add_argument("--score-output", type=Path, required=True)
    args = parser.parse_args()
    if args.measurement_output.exists() or args.score_output.exists():
        raise FileExistsError("measurement/score output must be fresh")
    measurement, scored = evaluate_candidate(
        args.prediction, args.sealed_manifest, args.public_scene, args.artifact_root
    )
    write_json_atomic(args.measurement_output, measurement)
    write_json_atomic(args.score_output, scored)
    print(args.score_output)


if __name__ == "__main__":
    main()
