#!/usr/bin/env python3
"""Fit D2-CEA to a fixed terminal SplArt checkpoint without endpoint GT.

This model-facing process accepts only the public middle-state scene, the
terminal checkpoint, and clean source trees.  It never accepts evaluator or
sealed-manifest paths.  All SplArt tensors are frozen; only two bounded local
endpoint scalars are optimized on detached contact fields.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import torch

from splart.articulation_params import ArticulationType
from splart.contact_endpoint_field import EndpointFieldConfig
from splart.endpoint_adapter import EndpointAdapterConfig, EndpointGeometry, canonicalize_state1_mobile, fit_endpoint_adapter
from splart_renderer import SplartRenderer


FINAL_STEP = 24_999
BASE_COMMIT = "8709e841c41165f3e91a7d716b78d6891f6e91a5"
ADAPTER_COMMIT = "2b72d87"
QUERY_IDS = ("outside_state_0", "outside_state_1")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_provenance(path: Path, expected_prefix: str) -> dict[str, str]:
    path = path.resolve()
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=path, check=True, capture_output=True, text=True, timeout=30
    ).stdout.strip()
    tree = subprocess.run(
        ["git", "rev-parse", "HEAD^{tree}"], cwd=path, check=True, capture_output=True, text=True, timeout=30
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=path, check=True, capture_output=True, text=True, timeout=30
    ).stdout.strip()
    if not head.startswith(expected_prefix) or dirty:
        raise ValueError(f"source provenance mismatch or dirty tree: {path}")
    return {"path": str(path), "commit": head, "tree": tree}


def load_public_validator(base_source: Path) -> Any:
    module_path = base_source / "src" / "splart" / "endpoint_baselines.py"
    spec = importlib.util.spec_from_file_location("frozen_endpoint_baselines", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen public-input validator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def model_state_sha256(model: torch.nn.Module) -> str:
    """Hash model values one tensor at a time without retaining CPU copies."""

    digest = hashlib.sha256()
    with torch.no_grad():
        for name, value in sorted(model.state_dict().items()):
            tensor = value.detach().contiguous().cpu()
            digest.update(name.encode())
            digest.update(str(tensor.dtype).encode())
            digest.update(str(tuple(tensor.shape)).encode())
            digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def extract_dual_state_geometry(model: Any) -> tuple[tuple[EndpointGeometry, EndpointGeometry], str, torch.Tensor]:
    states = model.states.detach().squeeze(-1)
    target = ~model.mobilities.detach().squeeze(-1).isnan()
    opacity = model.opacities.detach().sigmoid().squeeze(-1)
    mobility = model.mobilities.detach().sigmoid().squeeze(-1)
    means = model.means.detach()
    scales = model.scales.detach().exp()
    axis = model.articulation_params.axis.detach()
    pivot = model.articulation_params.pivot.detach()
    articulation_type = ArticulationType(int(model.articulation_params.articulation_type.item()))
    if articulation_type == ArticulationType.REVOLUTE:
        joint_kind = "revolute"
        displacement = model.articulation_params.angle.detach()
    elif articulation_type == ArticulationType.PRISMATIC:
        joint_kind = "prismatic"
        displacement = model.articulation_params.dist.detach()
    else:
        raise ValueError(f"D2-CEA v1 supports revolute/prismatic joints, got {articulation_type}")

    geometries = []
    for state in (0, 1):
        mask = target & (states == state)
        if not mask.any():
            raise ValueError(f"checkpoint has no target Gaussians for state {state}")
        state_means = means[mask]
        state_scales = scales[mask]
        state_opacity = opacity[mask]
        state_mobility = mobility[mask]
        mobile_means = state_means
        if state == 1:
            mobile_means = canonicalize_state1_mobile(
                mobile_means,
                joint_kind=joint_kind,
                axis=axis,
                pivot=pivot,
                observed_displacement=displacement,
            )
        geometries.append(
            EndpointGeometry(
                static_means=state_means,
                static_scales=state_scales,
                mobile_means_state0=mobile_means,
                mobile_scales=state_scales,
                static_weights=state_opacity * (1.0 - state_mobility),
                mobile_weights=state_opacity * state_mobility,
            )
        )
    return (geometries[0], geometries[1]), joint_kind, displacement


def certificate_payload(certificate: Any) -> dict[str, Any]:
    def endpoint(value: Any) -> dict[str, Any]:
        return {
            "identifiable": bool(value.identifiable),
            "contact_gap": float(value.contact_gap.detach().cpu()),
            "contact_mass": float(value.contact_mass.detach().cpu()),
            "support_rise": float(value.support_rise.detach().cpu()),
            "inside_penetration_bands": float(value.inside_penetration_bands.detach().cpu()),
            "uncertainty": float(value.uncertainty.detach().cpu()),
            "boundary_selected": bool(value.boundary_selected),
        }

    return {
        "lower": endpoint(certificate.lower),
        "upper": endpoint(certificate.upper),
        "closed_end": certificate.closed_end,
        "closed_end_identifiable": bool(certificate.closed_end_identifiable),
        "contact_band": float(certificate.contact_band.detach().cpu()),
        "scene_scale": float(certificate.scene_scale.detach().cpu()),
    }


def write_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--public-scene", type=Path, required=True)
    parser.add_argument("--base-source", type=Path, required=True)
    parser.add_argument("--adapter-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "formal"), default="formal")
    args = parser.parse_args()
    launch_text = "\n".join(str(value) for value in vars(args).values()).lower()
    if any(marker in launch_text for marker in ("sealed", "evaluator_truth", "full22", "b_test")):
        raise ValueError("model-facing adapter launch contains evaluator-only path marker")

    base_source = args.base_source.resolve()
    adapter_source = args.adapter_source.resolve()
    base_provenance = git_provenance(base_source, BASE_COMMIT)
    adapter_provenance = git_provenance(adapter_source, ADAPTER_COMMIT)
    validator = load_public_validator(base_source)
    public = validator.validate_public_scene(args.public_scene)
    checkpoint_dir = args.checkpoint_dir.resolve()
    checkpoint = checkpoint_dir / f"step-{FINAL_STEP:09d}.ckpt"
    config_path = checkpoint_dir.parent / "config.yml"
    dataparser_path = checkpoint_dir.parent / "dataparser_transforms.json"
    for path in (checkpoint, config_path, dataparser_path):
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(path)

    renderer = SplartRenderer(checkpoint_dir, load_step=FINAL_STEP, data_dir=None, device="cuda")
    for parameter in renderer.model.parameters():
        parameter.requires_grad_(False)
    before_hash = model_state_sha256(renderer.model)
    geometries, joint_kind, displacement = extract_dual_state_geometry(renderer.model)
    if args.mode == "formal":
        field_config = EndpointFieldConfig()
        adapter_config = EndpointAdapterConfig()
    else:
        field_config = EndpointFieldConfig(
            samples_per_side=17,
            broad_phase_samples=5,
            max_broad_phase_pairs=2_048,
            pair_chunk_size=512,
            scalar_chunk_size=8,
        )
        adapter_config = EndpointAdapterConfig(iterations=2)
    adapter, fitted = fit_endpoint_adapter(
        geometries,
        joint_kind=joint_kind,
        axis=renderer.model.articulation_params.axis.detach(),
        pivot=renderer.model.articulation_params.pivot.detach(),
        observed_displacement=displacement,
        field_config=field_config,
        adapter_config=adapter_config,
    )
    after_hash = model_state_sha256(renderer.model)
    if before_hash != after_hash:
        raise RuntimeError("frozen SplArt state changed during endpoint fitting")
    if any(parameter.grad is not None for parameter in renderer.model.parameters()):
        raise RuntimeError("a frozen SplArt parameter received a gradient")

    lower = float(fitted.lower_scalar.cpu())
    upper = float(fitted.upper_scalar.cpu())
    closed_query = {
        "lower": QUERY_IDS[0],
        "upper": QUERY_IDS[1],
        "unknown": None,
    }[fitted.closed_end]
    source_record = dict(adapter_provenance)
    candidate = {
        "checkpoint": {"path": str(checkpoint), "sha256": sha256_file(checkpoint), "step": FINAL_STEP},
        "config": {"path": str(config_path), "sha256": sha256_file(config_path)},
        "dataparser_transforms": {"path": str(dataparser_path), "sha256": sha256_file(dataparser_path)},
        # Existing candidate-bound evaluator verifies this clean source record.
        "source": source_record,
    }
    module_paths = [adapter_source / "src" / "splart" / name for name in ("contact_endpoint_field.py", "endpoint_adapter.py")]
    prediction = {
        "schema_version": "splart-endpoint-prediction/v1",
        "scene_id": public["scene_id"],
        "baseline": "splart-middle",
        "source_commit": BASE_COMMIT,
        "training": {
            "from_scratch": True,
            "author_checkpoint": False,
            "endpoint_physics_enabled": True,
            "seed_or_checkpoint_selection": False,
            "base_parameters_frozen": True,
            "adapter_iterations": adapter_config.iterations,
        },
        "renderer": "original-splart-plus-d2-cea-scalars",
        "closed_prediction": {"status": "predicted" if closed_query else "unknown", "query_id": closed_query},
        "candidate": candidate,
        "public_input": {
            "root": public["scene_dir"],
            "tree_sha256": public["public_tree_sha256"],
            "file_count": len(public["public_tree"]),
        },
        "observed_states": [0, 1],
        "endpoint_predictions": [
            {
                "query_id": QUERY_IDS[0],
                "local_direction": -1,
                "predicted_local_scalar": lower,
                "predicted_closed": closed_query == QUERY_IDS[0],
            },
            {
                "query_id": QUERY_IDS[1],
                "local_direction": 1,
                "predicted_local_scalar": upper,
                "predicted_closed": closed_query == QUERY_IDS[1],
            },
        ],
        "adapter": {
            "name": "D2-CEA",
            "mode": args.mode,
            "base_checkpoint_sha256": candidate["checkpoint"]["sha256"],
            "base_source": base_provenance,
            "adapter_source": adapter_provenance,
            "module_sha256": {str(path.relative_to(adapter_source)): sha256_file(path) for path in module_paths},
            "runner_sha256": sha256_file(Path(__file__).resolve()),
            "model_state_sha256_before": before_hash,
            "model_state_sha256_after": after_hash,
            "joint_kind": joint_kind,
            "observed_displacement": float(displacement.detach().cpu()),
            "lower_identifiable": fitted.lower_identifiable,
            "upper_identifiable": fitted.upper_identifiable,
            "closed_end_identifiable": fitted.closed_end_identifiable,
            "final_loss": float(fitted.final_loss.cpu()),
            "state_certificates": [certificate_payload(value) for value in fitted.state_certificates],
            "field_config": vars(field_config),
            "adapter_config": vars(adapter_config),
        },
    }
    write_exclusive(args.output.resolve(), prediction)
    print(json.dumps({"output": str(args.output.resolve()), "lower": lower, "upper": upper, "closed": closed_query}))


if __name__ == "__main__":
    main()
