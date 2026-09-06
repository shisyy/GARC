#!/usr/bin/env python3
"""Write-once target-free evidence for frozen D2 and preregistered ablations.

This runner consumes only public checkpoint provenance.  It deliberately does
not accept labels, endpoint targets, split membership, or evaluator inputs.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

SCHEMA = "splart-d2-public-ablation-evidence/v2"
MODES = ("full", "single-radius", "no-contact", "no-penetration", "no-terminal-support")
FORBIDDEN = ("sealed", "b_test", "full22", "target", "ground_truth", "joint_limits", "split")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def profile_sufficiency() -> dict[str, Any]:
    """State which exact D2 outputs are identifiable from legacy mean profiles."""
    return {
        "legacy_contract": "[object,2,R,S,9], geometry-mean",
        "exactly_reconstructible": {"full": True, "single-radius": True},
        "exactly_reconstructible_reason": (
            "The frozen objective is linear across geometries and the full posterior/raw anchor "
            "is already averaged; selecting the registered middle radius preserves its field."
        ),
        "not_exactly_reconstructible": {
            "no-contact": "requires per-geometry softmax posterior after changing total energy",
            "no-penetration": "requires per-geometry softmax posterior after changing total energy",
            "no-terminal-support": "requires per-geometry softmax posterior after changing total energy",
        },
        "decision": "fresh_public_reexport_required",
    }


def mode_field_configs(base: Any, fixed: tuple[Any, ...], mode: str) -> tuple[Any, ...]:
    if mode == "full":
        return fixed
    if mode == "single-radius":
        return (fixed[1],)
    if mode == "no-contact":
        return tuple(replace(value, contact_weight=0.0) for value in fixed)
    if mode == "no-penetration":
        return tuple(replace(value, penetration_weight=0.0, inside_weight=0.0) for value in fixed)
    if mode == "no-terminal-support":
        return tuple(replace(value, support_weight=0.0) for value in fixed)
    raise ValueError(f"unknown mode: {mode}")


def prediction_payload(prediction: Any, configs: tuple[Any, ...], certificate_encoder: Callable[[Any], dict]) -> dict:
    return {
        "lower_scalar": float(prediction.lower_scalar.detach().cpu()),
        "upper_scalar": float(prediction.upper_scalar.detach().cpu()),
        "closed_end": prediction.closed_end,
        "closed_end_identifiable": bool(prediction.closed_end_identifiable),
        "lower_identifiable": bool(prediction.lower_identifiable),
        "upper_identifiable": bool(prediction.upper_identifiable),
        "final_loss": float(prediction.final_loss.detach().cpu()),
        "field_configs": [vars(value) for value in configs],
        "state_certificates": [certificate_encoder(value) for value in prediction.state_certificates],
    }


def _verified(record: dict[str, Any]) -> Path:
    path = Path(record["path"]).resolve()
    if path.is_symlink() or not path.is_file() or sha256_file(path) != record["sha256"]:
        raise ValueError(f"provenance mismatch: {path}")
    return path


def _state_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode() + str(tensor.dtype).encode() + str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def export_object(row: dict[str, Any], d2_source: Path, d2_provenance: dict[str, str], device: str) -> dict:
    checkpoint, config, transforms = (_verified(row[key]) for key in ("checkpoint", "config", "dataparser_transforms"))
    if int(row["checkpoint"]["step"]) != 24999:
        raise ValueError("only frozen step24999 checkpoints are admissible")
    sys.path[:0] = [str(d2_source), str(d2_source / "src")]
    from splart.contact_endpoint_field import EndpointFieldConfig
    from splart.endpoint_adapter import EndpointAdapterConfig, fit_endpoint_adapter, fixed_multiradius_counterfactuals
    from splart_renderer import SplartRenderer
    import posthoc_endpoint_adapter as runner

    renderer = SplartRenderer(checkpoint.parent, load_step=24999, data_dir=None, device=device)
    for parameter in renderer.model.parameters():
        parameter.requires_grad_(False)
    before = _state_hash(renderer.model)
    geometries, joint_kind, displacement = runner.extract_dual_state_geometry(renderer.model)
    base = EndpointFieldConfig()
    fixed = fixed_multiradius_counterfactuals(base)
    outputs = {}
    for mode in MODES:
        configs = mode_field_configs(base, fixed, mode)
        _, prediction = fit_endpoint_adapter(
            geometries, joint_kind=joint_kind, axis=renderer.model.articulation_params.axis.detach(),
            pivot=renderer.model.articulation_params.pivot.detach(), observed_displacement=displacement,
            field_config=base, adapter_config=EndpointAdapterConfig(), counterfactual_field_configs=configs,
        )
        outputs[mode] = prediction_payload(prediction, configs, runner.certificate_payload)
    after = _state_hash(renderer.model)
    if before != after or any(parameter.grad is not None for parameter in renderer.model.parameters()):
        raise RuntimeError("frozen base state changed")
    return {
        "schema": SCHEMA,
        "object_id": row["object_id"],
        "provenance": {"d2_commit": d2_provenance["commit"], "d2_tree": d2_provenance["tree"],
            "checkpoint": {"path": str(checkpoint), "sha256": sha256_file(checkpoint), "step": 24999},
            "config": {"path": str(config), "sha256": sha256_file(config)},
            "dataparser_transforms": {"path": str(transforms), "sha256": sha256_file(transforms)}},
        "base_state": {"before_sha256": before, "after_sha256": after, "unchanged": True},
        "modes": outputs,
    }


def _exclusive_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="public-only rows with checkpoint/config/dataparser hashes")
    parser.add_argument("--d2-source", type=Path, required=True)
    parser.add_argument("--d2-commit", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    launch = "\n".join(str(value).lower() for value in vars(args).values())
    if any(marker in launch for marker in FORBIDDEN):
        raise ValueError("protected/evaluator-only launch marker")
    source = args.d2_source.resolve()
    commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    tree = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD^{tree}"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"], text=True).strip()
    if commit != args.d2_commit or dirty:
        raise ValueError("D2 source is not the exact clean frozen commit")
    payload = json.loads(args.input.read_text())
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows or len({row.get("object_id") for row in rows}) != len(rows):
        raise ValueError("input must contain unique public object rows")
    args.output_dir.mkdir(mode=0o700, parents=True, exist_ok=False)
    evidence = []
    for row in rows:
        item = export_object(row, source, {"commit": commit, "tree": tree}, args.device)
        path = args.output_dir / f"{row['object_id']}.json"
        _exclusive_json(path, item)
        evidence.append({"object_id": row["object_id"], "artifact": path.name, "sha256": sha256_file(path)})
    index = {"schema": "splart-d2-public-ablation-index/v2", "sufficiency": profile_sufficiency(),
             "d2_source": {"commit": commit, "tree": tree}, "rows": evidence}
    _exclusive_json(args.output_dir / "index.json", index)
    os.chmod(args.output_dir, 0o700)
    print(json.dumps({"objects": len(evidence), "index_sha256": sha256_file(args.output_dir / "index.json")}))


if __name__ == "__main__":
    main()
