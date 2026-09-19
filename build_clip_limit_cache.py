#!/usr/bin/env python3
"""Stream counterfactual renders through frozen CLIP and retain evidence only.

The renderer is a hash-bound local plugin. It emits one candidate mini-batch
at a time; RGB tensors are encoded immediately and never written to the full
trajectory cache. Only six hashes of fixed 32x32 audit thumbnails are kept.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
from importlib import metadata
import json
from pathlib import Path
import sys
from types import ModuleType
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from prepare_clip_limit_render_plan import validate_plan
from run_glpdt_source_gate import FORBIDDEN_MARKERS, sha256_file
from splart.clip_limit import OPEN_CLIP_TORCH_VERSION, load_frozen_open_clip_vit_b32, prompt_ensemble_sha256
from splart.clip_limit_cache import CACHE_ARTIFACT_SCHEMA, CACHE_INDEX_SCHEMA, fixed_candidate_grid


AUDIT_INDICES = (0, 64, 128)
PLUGIN_METHODS = ("asset_bundle_sha256", "iter_candidate_batches", "audit_receipt")


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _load_plan(
    plan_path: Path, source_index: Path, domain: str, selection_path: Path | None, preregister_path: Path | None
) -> dict[str, Any]:
    if plan_path.is_symlink() or not plan_path.is_file():
        raise ValueError("render plan must be a regular non-symlink file")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    validate_plan(plan, source_index, domain, selection_path=selection_path, preregister_path=preregister_path)
    return plan


def _verify_open_clip_install(wheel: Path, wheel_sha256: str, version: str) -> None:
    if version != OPEN_CLIP_TORCH_VERSION:
        raise ValueError(f"OpenCLIP must be pinned to {OPEN_CLIP_TORCH_VERSION}")
    if wheel.is_symlink():
        raise ValueError("OpenCLIP wheel must be a regular non-symlink file")
    wheel = wheel.resolve()
    if not wheel.is_file() or sha256_file(wheel) != wheel_sha256:
        raise ValueError("OpenCLIP wheel digest mismatch")
    if metadata.version("open_clip_torch") != version:
        raise ValueError("installed OpenCLIP version differs from the staged wheel binding")


def _load_renderer_plugin(
    script: Path, script_sha256: str, asset_root: Path, domain: str, plan: dict[str, Any], device_name: str
) -> tuple[Any, str]:
    if script.is_symlink():
        raise ValueError("renderer plugin is missing, symlinked, or has the wrong digest")
    script = script.resolve()
    if not script.is_file() or sha256_file(script) != script_sha256:
        raise ValueError("renderer plugin is missing, symlinked, or has the wrong digest")
    if asset_root.is_symlink() or not asset_root.resolve().is_dir():
        raise ValueError("asset root must be a regular non-symlink directory")
    specification = importlib.util.spec_from_file_location("splart_c_clip_ld_renderer", script)
    if specification is None or specification.loader is None:
        raise ValueError("renderer plugin cannot be imported")
    module = importlib.util.module_from_spec(specification)
    assert isinstance(module, ModuleType)
    # Dynamic dataclass/type evaluation expects the module to be registered
    # while its body executes, just like a normal import.
    sys.modules[specification.name] = module
    try:
        specification.loader.exec_module(module)
    except Exception:
        sys.modules.pop(specification.name, None)
        raise
    factory = getattr(module, "create_renderer", None)
    if not callable(factory):
        raise ValueError("renderer plugin must define create_renderer")
    renderer = factory(asset_root.resolve(), domain, plan, device=device_name)
    if any(not callable(getattr(renderer, method, None)) for method in PLUGIN_METHODS):
        raise ValueError("renderer plugin does not satisfy the streaming contract")
    config_sha256 = getattr(renderer, "render_config_sha256", None)
    if not _is_sha256(config_sha256):
        raise ValueError("renderer plugin has no hash-bound render configuration")
    return renderer, config_sha256


def _audit_thumbnail_sha256(image: Tensor) -> str:
    if image.shape != (3, 224, 224) or image.dtype != torch.uint8:
        raise ValueError("audit image must be uint8 [3,224,224]")
    thumbnail = F.interpolate(image[None].float(), size=(32, 32), mode="area").round().to(torch.uint8)[0]
    return hashlib.sha256(thumbnail.contiguous().numpy().tobytes()).hexdigest()


def _encode_row(
    renderer: Any,
    discriminator: Any,
    row: dict[str, str],
    canonical_q: dict[str, list[float]],
    views: list[list[int]],
    batch_size: int,
) -> tuple[Tensor, dict[str, str]]:
    evidence = torch.empty(2, 129, 1, dtype=torch.float32)
    audits: dict[str, str] = {}
    for side in range(2):
        expected_start = 0
        batches = renderer.iter_candidate_batches(
            row=row,
            side=side,
            canonical_q=tuple(canonical_q[f"side{side}"]),
            views=tuple(tuple(value) for value in views),
            batch_size=batch_size,
        )
        for batch in batches:
            if not isinstance(batch, dict) or set(batch) != {"images", "start"}:
                raise ValueError("renderer batches must contain exactly start/images")
            start, images = batch["start"], batch["images"]
            if not isinstance(start, int) or start != expected_start:
                raise ValueError("renderer batches must be contiguous and ordered")
            if not isinstance(images, Tensor) or images.dtype != torch.uint8 or images.ndim != 5:
                raise ValueError("renderer images must be uint8 [N,V,3,224,224]")
            if images.shape[1:] != (len(views), 3, 224, 224) or not (1 <= images.shape[0] <= batch_size):
                raise ValueError("renderer image batch shape mismatch")
            end = start + images.shape[0]
            if end > 129:
                raise ValueError("renderer emitted too many candidates")
            evidence[side, start:end, 0] = discriminator(images).cpu()
            for audit_index in AUDIT_INDICES:
                if start <= audit_index < end:
                    key = f"side{side}:{audit_index:03d}"
                    audits[key] = _audit_thumbnail_sha256(images[audit_index - start, 0])
            expected_start = end
            del images
        if expected_start != 129:
            raise ValueError("renderer did not emit every fixed candidate")
    expected_audits = {f"side{side}:{index:03d}" for side in range(2) for index in AUDIT_INDICES}
    if set(audits) != expected_audits:
        raise ValueError("renderer did not cover the fixed audit candidates")
    return evidence, audits


def build_cache(
    plan_path: Path,
    source_index: Path,
    domain: str,
    selection_path: Path | None,
    preregister_path: Path | None,
    renderer_script: Path,
    renderer_sha256: str,
    asset_root: Path,
    output: Path,
    checkpoint: Path,
    checkpoint_sha256: str,
    open_clip_wheel: Path,
    open_clip_wheel_sha256: str,
    open_clip_version: str,
    device_name: str,
    batch_size: int,
) -> dict[str, Any]:
    if output.exists():
        raise ValueError("output directory already exists")
    if batch_size < 1 or batch_size > 32:
        raise ValueError("streaming render batch size must be in [1,32]")
    plan = _load_plan(plan_path, source_index, domain, selection_path, preregister_path)
    _verify_open_clip_install(open_clip_wheel, open_clip_wheel_sha256, open_clip_version)
    renderer, render_config_sha256 = _load_renderer_plugin(
        renderer_script, renderer_sha256, asset_root, domain, plan, device_name
    )
    discriminator = load_frozen_open_clip_vit_b32(checkpoint, checkpoint_sha256, torch.device(device_name))

    output.mkdir(parents=True, exist_ok=False)
    artifact_dir = output / "artifacts"
    artifact_dir.mkdir()
    coordinates = fixed_candidate_grid().expand(2, -1).clone()
    rows = []
    split_counts: dict[str, int] = {}
    for render_row in plan["rows"]:
        evidence, audits = _encode_row(
            renderer, discriminator, render_row, plan["canonical_q"], plan["views"], batch_size
        )
        asset_hash = renderer.asset_bundle_sha256(render_row)
        if not _is_sha256(asset_hash):
            raise ValueError("renderer returned an invalid asset bundle digest")
        identity = {key: render_row[key] for key in ("split", "object_id", "episode_id")}
        render_metadata_sha256 = _canonical_sha256(render_row)
        artifact = {
            "schema": CACHE_ARTIFACT_SCHEMA,
            **identity,
            "coordinates": coordinates,
            "signed_limit_evidence": evidence,
            "view_count": len(plan["views"]),
            "prompt_ensemble_sha256": prompt_ensemble_sha256(),
            "encoder_checkpoint_sha256": checkpoint_sha256,
            "open_clip_version": open_clip_version,
            "open_clip_wheel_sha256": open_clip_wheel_sha256,
            "renderer_sha256": renderer_sha256,
            "render_config_sha256": render_config_sha256,
            "asset_bundle_sha256": asset_hash,
            "audit_thumbnail_sha256": audits,
            "render_metadata_sha256": render_metadata_sha256,
        }
        target = artifact_dir / f"{identity['episode_id']}.pt"
        torch.save(artifact, target)
        rows.append(
            {
                "artifact": str(target.relative_to(output)),
                "artifact_sha256": sha256_file(target),
                **identity,
                "render_metadata_sha256": render_metadata_sha256,
                "shape": list(evidence.shape),
            }
        )
        split = identity["split"]
        split_counts[split] = split_counts.get(split, 0) + 1
    renderer_audit = renderer.audit_receipt()
    expected_objects = len({row["object_id"] for row in plan["rows"]})
    minimum_views = len(plan["rows"]) * 2 * 129 * len(plan["views"])
    if (
        not isinstance(renderer_audit, dict)
        or renderer_audit.get("no_empty_views_checked") is not True
        or renderer_audit.get("repeat_bit_identical") is not True
        or renderer_audit.get("assets_repeat_audited") != expected_objects
        or renderer_audit.get("render_config_sha256") != render_config_sha256
        or not isinstance(renderer_audit.get("views_checked"), int)
        or renderer_audit["views_checked"] < minimum_views
    ):
        raise RuntimeError("renderer did not satisfy the exact determinism/coverage audit")
    report = {
        "schema": CACHE_INDEX_SCHEMA,
        "encoder": "open_clip:ViT-B-32",
        "encoder_checkpoint_sha256": checkpoint_sha256,
        "open_clip_version": open_clip_version,
        "open_clip_wheel_sha256": open_clip_wheel_sha256,
        "prompt_ensemble_sha256": prompt_ensemble_sha256(),
        "renderer_sha256": renderer_sha256,
        "render_config_sha256": render_config_sha256,
        "renderer_audit": renderer_audit,
        "streaming_rgb_persisted": False,
        "audit_thumbnail_contract": "first_view_32x32_hash_at_side0/1_candidates_0/64/128",
        "source_index_sha256": plan["source_index_sha256"],
        "input_bindings": plan["input_bindings"],
        "render_plan_sha256": sha256_file(plan_path.resolve()),
        "split_counts": split_counts,
        "rows": rows,
        "protected_splits_read": [],
        "box_labels_read": False,
    }
    (output / "index.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--render-plan", type=Path, required=True)
    parser.add_argument("--source-index", type=Path, required=True)
    parser.add_argument("--domain", choices=("articraft", "njc"), required=True)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--preregister", type=Path)
    parser.add_argument("--renderer-script", type=Path, required=True)
    parser.add_argument("--renderer-sha256", required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--clip-checkpoint", type=Path, required=True)
    parser.add_argument("--clip-checkpoint-sha256", required=True)
    parser.add_argument("--open-clip-wheel", type=Path, required=True)
    parser.add_argument("--open-clip-wheel-sha256", required=True)
    parser.add_argument("--open-clip-version", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    launch = "\n".join(str(value).lower() for value in vars(args).values())
    if any(marker in launch for marker in FORBIDDEN_MARKERS):
        raise ValueError("cache build launch contains a protected path marker")
    print(
        json.dumps(
            build_cache(
                args.render_plan,
                args.source_index,
                args.domain,
                args.selection,
                args.preregister,
                args.renderer_script,
                args.renderer_sha256,
                args.asset_root,
                args.output,
                args.clip_checkpoint,
                args.clip_checkpoint_sha256,
                args.open_clip_wheel,
                args.open_clip_wheel_sha256,
                args.open_clip_version,
                args.device,
                args.batch_size,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
