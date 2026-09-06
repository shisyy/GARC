#!/usr/bin/env python3
"""Export frozen D2 trajectory profiles from public episodes, fail-closed."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from splart.gauge_energy_profile import PROFILE_CHANNELS, predictions_to_profile

SCHEMA = "splart-gauge-energy-public-episodes/v1"
FORBIDDEN_KEYS = {"joint_limits", "closed_side", "normalized_input_states", "target", "targets", "ground_truth"}
FORBIDDEN_PATH_MARKERS = ("sealed", "b_test", "full22")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        tensor = value.detach().cpu().contiguous()
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _reject_private(value: Any, *, path: str = "root") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).lower() in FORBIDDEN_KEYS:
                raise ValueError(f"forbidden model-facing key at {path}.{key}")
            _reject_private(child, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_private(child, path=f"{path}[{index}]")
    elif isinstance(value, str) and any(marker in value.lower() for marker in FORBIDDEN_PATH_MARKERS):
        raise ValueError(f"evaluator-only path marker at {path}")


def load_handoff(path: Path) -> dict[str, Any]:
    handoff = json.loads(path.read_text())
    _reject_private(handoff)
    if handoff.get("schema") != SCHEMA or handoff.get("training_handoff") != "UNLOCKED":
        raise ValueError("public profile handoff is not unlocked or has the wrong schema")
    episodes = handoff.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("public profile handoff has no episodes")
    ids = [row.get("object_id") for row in episodes]
    if any(not isinstance(value, str) or not value for value in ids) or len(ids) != len(set(ids)):
        raise ValueError("object IDs must be non-empty and object-disjoint")
    if any(row.get("split") not in {"train", "val"} for row in episodes):
        raise ValueError("only public train/val episodes are allowed")
    return handoff


def preflight_public_materialization(public_manifest_path: Path, receipt_path: Path) -> dict[str, Any]:
    """Verify every public episode payload without consuming supervision."""
    public = json.loads(public_manifest_path.read_text())
    receipt = json.loads(receipt_path.read_text())
    _reject_private(public)
    _reject_private(receipt)
    if public.get("schema") != "splart-endpoint-free-order-public/v1":
        raise ValueError("unexpected public episode manifest schema")
    if receipt.get("schema") != "splart-order-materialization-receipt/v1":
        raise ValueError("unexpected materialization receipt schema")
    episodes = public.get("episodes", [])
    objects = receipt.get("objects", [])
    if receipt.get("ready") != 12 or receipt.get("blocked") != 0 or len(episodes) != 12 or len(objects) != 12:
        raise ValueError("materialization is not exactly 12/12 complete")
    if receipt.get("public_manifest_sha256") != sha256_file(public_manifest_path):
        raise ValueError("receipt does not bind the public episode manifest")
    public_by_id = {row["object_id"]: row for row in episodes}
    receipt_by_id = {row["object_id"]: row for row in objects}
    if len(public_by_id) != 12 or set(public_by_id) != set(receipt_by_id):
        raise ValueError("public and materialized object sets differ")
    root = receipt_path.parent / "episodes"
    rows = []
    tree_digest = hashlib.sha256()
    for object_id in sorted(public_by_id):
        pub, rec = public_by_id[object_id], receipt_by_id[object_id]
        if pub["episode_id"] != rec["episode_id"] or rec.get("status") != "materialized":
            raise ValueError("episode identity/status mismatch")
        transform = root / pub["episode_id"] / "transforms.json"
        if transform.is_symlink() or not transform.is_file() or sha256_file(transform) != rec["transforms_sha256"]:
            raise ValueError(f"transforms provenance mismatch: {object_id}")
        payload = json.loads(transform.read_text())
        _reject_private(payload)
        frames = payload.get("frames", []) + payload.get("val_frames", [])
        if not frames or set(frame.get("state") for frame in frames) != {0, 1}:
            raise ValueError(f"episode states are not exactly {{0,1}}: {object_id}")
        files = [transform]
        for frame in frames:
            for key in ("file_path", "depth_file_path", "mask_path"):
                if key not in frame:
                    continue
                candidate = (transform.parent / frame[key]).resolve()
                if candidate.is_symlink() or not candidate.is_file() or transform.parent.resolve() not in candidate.parents:
                    raise ValueError(f"missing/unsafe public payload: {object_id}.{key}")
                files.append(candidate)
        for file in sorted(set(files), key=lambda item: str(item)):
            relative = file.relative_to(receipt_path.parent).as_posix()
            digest = sha256_file(file)
            tree_digest.update(relative.encode() + b"\0" + digest.encode() + b"\n")
        rows.append({"object_id": object_id, "episode_id": pub["episode_id"], "frames": len(frames), "files": len(set(files))})
    return {
        "schema": "splart-gauge-energy-materialization-preflight/v1",
        "public_manifest_sha256": sha256_file(public_manifest_path),
        "materialization_receipt_sha256": sha256_file(receipt_path),
        "materialized_tree_sha256": tree_digest.hexdigest(),
        "objects": rows,
    }


def build_handoff(public_manifest_path: Path, receipt_path: Path, training_path: Path) -> dict[str, Any]:
    preflight_public_materialization(public_manifest_path, receipt_path)
    public = json.loads(public_manifest_path.read_text())
    receipt = json.loads(receipt_path.read_text())
    training = json.loads(training_path.read_text())
    for value in (public, receipt, training):
        _reject_private(value)
    if public.get("schema") != "splart-endpoint-free-order-public/v1":
        raise ValueError("unexpected public episode manifest schema")
    if receipt.get("schema") != "splart-order-materialization-receipt/v1" or receipt.get("ready") != 12 or receipt.get("blocked") != 0:
        raise ValueError("materialization receipt is not complete")
    if receipt.get("public_manifest_sha256") != sha256_file(public_manifest_path):
        raise ValueError("receipt does not bind the public episode manifest")
    if training.get("schema") != "splart-frozen-d2-training-outputs/v1" or training.get("status") != "COMPLETE":
        raise ValueError("frozen D2 training outputs are incomplete")
    public_by_id = {row["object_id"]: row for row in public["episodes"]}
    receipt_by_id = {row["object_id"]: row for row in receipt["objects"]}
    trained_by_id = {row["object_id"]: row for row in training.get("objects", [])}
    if set(public_by_id) != set(receipt_by_id) or not trained_by_id or not set(trained_by_id).issubset(public_by_id):
        raise ValueError("trained objects must be a non-empty subset of the complete public materialization")
    materialized_root = receipt_path.parent
    episodes = []
    for object_id in sorted(trained_by_id):
        public_row, receipt_row, trained = public_by_id[object_id], receipt_by_id[object_id], trained_by_id[object_id]
        if public_row["episode_id"] != receipt_row["episode_id"] or receipt_row.get("status") != "materialized":
            raise ValueError("episode identity/status mismatch")
        transform = materialized_root / "episodes" / public_row["episode_id"] / "transforms.json"
        episodes.append({
            **trained,
            "episode_manifest": {"path": str(transform), "sha256": receipt_row["transforms_sha256"]},
        })
    return {"schema": SCHEMA, "training_handoff": "UNLOCKED", "episodes": episodes}


def _verified_file(path_text: str, expected_sha: str) -> Path:
    path = Path(path_text).resolve()
    if path.is_symlink() or not path.is_file() or sha256_file(path) != expected_sha:
        raise ValueError(f"file provenance mismatch: {path}")
    return path


def _git_provenance(path: Path, expected_commit: str) -> dict[str, str]:
    head = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(path), "status", "--porcelain"], text=True).strip()
    if head != expected_commit or dirty:
        raise ValueError("D2 source must be the exact clean frozen commit")
    tree = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD^{tree}"], text=True).strip()
    return {"commit": head, "tree": tree, "path": str(path)}


def export_one(row: dict[str, Any], d2_source: Path, stage: Path, device: str) -> dict[str, Any]:
    episode_manifest = _verified_file(row["episode_manifest"]["path"], row["episode_manifest"]["sha256"])
    episode = json.loads(episode_manifest.read_text())
    _reject_private(episode)
    if set(frame.get("state") for frame in episode.get("frames", [])) != {0, 1}:
        raise ValueError("episode manifest must expose only relabelled states 0 and 1")
    checkpoint = _verified_file(row["checkpoint"]["path"], row["checkpoint"]["sha256"])
    checkpoint_dir = checkpoint.parent
    config = _verified_file(row["config"]["path"], row["config"]["sha256"])
    transforms_file = _verified_file(row["dataparser_transforms"]["path"], row["dataparser_transforms"]["sha256"])

    sys.path[:0] = [str(d2_source), str(d2_source / "src")]
    from splart.contact_endpoint_field import EndpointFieldConfig
    from splart.endpoint_adapter import EndpointAdapterConfig, fit_endpoint_adapter, fixed_multiradius_counterfactuals
    from splart_renderer import SplartRenderer
    import posthoc_endpoint_adapter as d2_runner

    renderer = SplartRenderer(checkpoint_dir, load_step=int(row["checkpoint"]["step"]), data_dir=None, device=device)
    for parameter in renderer.model.parameters():
        parameter.requires_grad_(False)
    before = model_state_sha256(renderer.model)
    geometries, joint_kind, displacement = d2_runner.extract_dual_state_geometry(renderer.model)
    field_config = EndpointFieldConfig()
    radii = fixed_multiradius_counterfactuals(field_config)
    _, fitted = fit_endpoint_adapter(
        geometries, joint_kind=joint_kind, axis=renderer.model.articulation_params.axis.detach(),
        pivot=renderer.model.articulation_params.pivot.detach(), observed_displacement=displacement,
        field_config=field_config, adapter_config=EndpointAdapterConfig(), counterfactual_field_configs=radii,
    )
    after = model_state_sha256(renderer.model)
    if before != after or any(parameter.grad is not None for parameter in renderer.model.parameters()):
        raise RuntimeError("frozen base model state changed during profile export")
    features, scalars = predictions_to_profile(fitted.initial_fields, len(radii))
    if features.shape[-1] != len(PROFILE_CHANNELS) or not torch.isfinite(features).all():
        raise RuntimeError("invalid exported profile tensor")
    artifact = stage / f"{row['object_id']}.pt"
    torch.save({"features": features.cpu(), "scalars": scalars.cpu(), "object_id": row["object_id"], "split": row["split"]}, artifact)
    return {
        "object_id": row["object_id"], "split": row["split"], "artifact": artifact.name,
        "artifact_sha256": sha256_file(artifact), "shape": list(features.shape),
        "base_state_sha256_before": before, "base_state_sha256_after": after,
        "episode_manifest": {"path": str(episode_manifest), "sha256": sha256_file(episode_manifest)},
        "checkpoint": {"path": str(checkpoint), "sha256": sha256_file(checkpoint)},
        "config": {"path": str(config), "sha256": sha256_file(config)},
        "dataparser_transforms": {"path": str(transforms_file), "sha256": sha256_file(transforms_file)},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public-manifest", type=Path, required=True)
    parser.add_argument("--materialization-receipt", type=Path, required=True)
    parser.add_argument("--training-outputs", type=Path, required=True)
    parser.add_argument("--d2-source", type=Path, required=True)
    parser.add_argument("--d2-commit", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    launch = "\n".join(str(value).lower() for value in vars(args).values())
    if any(marker in launch for marker in FORBIDDEN_PATH_MARKERS):
        raise ValueError("model-facing launch contains evaluator-only path")
    handoff = build_handoff(args.public_manifest, args.materialization_receipt, args.training_outputs)
    provenance = _git_provenance(args.d2_source.resolve(), args.d2_commit)
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix="gauge-profile-stage-", dir=args.output_dir.parent.resolve()))
    try:
        rows = [export_one(row, args.d2_source.resolve(), stage, args.device) for row in handoff["episodes"]]
        index = {
            "schema": "splart-gauge-energy-profiles/v1",
            "public_manifest_sha256": sha256_file(args.public_manifest),
            "materialization_receipt_sha256": sha256_file(args.materialization_receipt),
            "training_outputs_sha256": sha256_file(args.training_outputs),
            "d2_source": provenance, "channels": list(PROFILE_CHANNELS), "rows": rows,
        }
        (stage / "index.json").write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
        os.replace(stage, args.output_dir)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    print(json.dumps({"output": str(args.output_dir), "objects": len(rows), "index_sha256": sha256_file(args.output_dir / "index.json")}))


if __name__ == "__main__":
    main()
