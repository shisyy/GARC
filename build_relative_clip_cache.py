#!/usr/bin/env python3
"""Frozen per-view CLIP features for the node 9.1 label-free search diagnostic.

This is an ISOLATED offline ideal-mesh, stylized RGB diagnostic: the renderer
uses full assets and public observation fractions to generate poses, with fixed
part colors and depth shading, not reconstructed 3DGS or natural RGB. Neither
those pose fractions nor limits/assets/targets enter the model-facing cache.
Global images only; contact-region ROI features are not implemented.

The builder never opens a source index or endpoint labels. It accepts a pinned
existing render plan and exports only opaque identities, fixed gauge, frozen
embeddings, and allow-listed content hashes. Index publication is last; partial
builds cannot be mistaken for complete caches. Outputs are never overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from build_clip_limit_cache import _load_renderer_plugin, _verify_open_clip_install
from prepare_clip_limit_render_plan import (
    AZIMUTH_ELEVATION_DEGREES, PLAN_SCHEMA, _validate_row,
)
from splart.clip_limit import (
    FrozenCLIPLimitDiscriminator, OPEN_CLIP_TORCH_VERSION,
    load_frozen_open_clip_vit_b32,
)
from splart.clip_limit_cache import fixed_candidate_grid, sha256_file

INDEX_SCHEMA = "relative-clip-index/v1"
ARTIFACT_SCHEMA = "relative-clip/v1"
OPEN_PROMPTS = (
    "a photo of an articulated object in an open state",
    "a photo of an object with its movable part open",
    "a photo of an open articulated object",
)
CLOSED_PROMPTS = (
    "a photo of an articulated object in a closed state",
    "a photo of an object with its movable part closed",
    "a photo of a closed articulated object",
)
HASH_KEYS = {
    "render_plan_sha256", "renderer_sha256", "render_config_sha256",
    "encoder_checkpoint_sha256", "open_clip_wheel_sha256", "prompt_sha256",
    "builder_sha256",
}
ARTIFACT_KEYS = {
    "schema", "object_id", "split", "coordinates", "image_embeddings",
    "observed_embeddings", "text_direction", "provenance",
}


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def opaque_object_id(original_id: str) -> str:
    """Shared exporter contract; raw object names may contain category words."""
    return hashlib.sha256(("node91-object:" + original_id).encode("utf-8")).hexdigest()


def _identity(object_id: Any, split: Any) -> None:
    if not isinstance(object_id, str) or not re.fullmatch(r"[a-f0-9]{64}", object_id):
        raise ValueError("object identity must be an opaque SHA256 identifier")
    if split not in ("source_train", "source_validation"):
        raise ValueError("only the frozen source splits are permitted")


def _hashes(value: Any, keys: set[str]) -> None:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError("provenance contains missing or unapproved fields")
    if not all(isinstance(v, str) and re.fullmatch(r"[0-9a-f]{64}", v) for v in value.values()):
        raise ValueError("provenance must contain only SHA256 values")


def _unit(value: Any, shape: tuple[int, ...]) -> None:
    if not isinstance(value, Tensor) or value.dtype != torch.float32 or tuple(value.shape) != shape:
        raise ValueError(f"expected float32 embeddings of shape {shape}")
    if not torch.isfinite(value).all() or not torch.allclose(
        value.norm(dim=-1), torch.ones(shape[:-1]), atol=2e-5, rtol=2e-5
    ):
        raise ValueError("embeddings must be finite unit vectors")


def validate_relative_artifact(artifact: dict[str, Any]) -> None:
    """Strict allow-list: unrecognized metadata, including targets, is rejected."""
    if not isinstance(artifact, dict) or set(artifact) != ARTIFACT_KEYS or artifact["schema"] != ARTIFACT_SCHEMA:
        raise ValueError("relative cache artifact schema mismatch")
    _identity(artifact["object_id"], artifact["split"])
    _hashes(artifact["provenance"], HASH_KEYS | {"asset_bundle_sha256"})
    coordinates = artifact["coordinates"]
    expected = fixed_candidate_grid().expand(2, -1)
    if not isinstance(coordinates, Tensor) or coordinates.dtype != torch.float32 or not torch.equal(coordinates, expected):
        raise ValueError("cache coordinates must be the exact fixed two-side grid")
    _unit(artifact["image_embeddings"], (2, 129, 6, 512))
    _unit(artifact["observed_embeddings"], (2, 6, 512))
    _unit(artifact["text_direction"], (512,))
    if not torch.equal(artifact["observed_embeddings"], artifact["image_embeddings"][:, 0]):
        raise ValueError("observations must exactly equal each side's d=0 anchors")


def validate_relative_index(index: dict[str, Any]) -> None:
    if not isinstance(index, dict) or set(index) != {"schema", "rows", "provenance", "split_counts"}:
        raise ValueError("relative cache index has unapproved fields")
    if index["schema"] != INDEX_SCHEMA:
        raise ValueError("relative cache index schema mismatch")
    _hashes(index["provenance"], HASH_KEYS)
    if not isinstance(index["rows"], list) or not index["rows"]:
        raise ValueError("relative cache index must have rows")
    counts: dict[str, int] = {}
    seen: set[str] = set()
    for row in index["rows"]:
        if not isinstance(row, dict) or set(row) != {"object_id", "split", "artifact", "artifact_sha256"}:
            raise ValueError("relative index row has unapproved fields")
        _identity(row["object_id"], row["split"])
        if row["object_id"] in seen:
            raise ValueError("duplicate object")
        seen.add(row["object_id"])
        if row["artifact"] != f"artifacts/{row['object_id']}.pt":
            raise ValueError("artifact path must be canonical and relative")
        _hashes({"artifact_sha256": row["artifact_sha256"]}, {"artifact_sha256"})
        counts[row["split"]] = counts.get(row["split"], 0) + 1
    if counts != index["split_counts"]:
        raise ValueError("split count mismatch")


def load_render_plan(path: Path, expected_sha256: str, max_objects: int | None) -> dict[str, Any]:
    """Validate pose generation without calling the old source-index validator."""
    if path.is_symlink() or not path.is_file() or sha256_file(path) != expected_sha256:
        raise ValueError("render plan is missing or has wrong digest")
    plan = json.loads(path.read_text(encoding="utf-8"))
    if plan.get("schema") != PLAN_SCHEMA or plan.get("domain") != "articraft" or plan.get("target_labels_used") is not False:
        raise ValueError("requires the existing label-independent Articraft render plan")
    grid = fixed_candidate_grid().tolist()
    if plan.get("candidate_coordinates") != grid or plan.get("canonical_q") != {
        "side0": [-d for d in grid], "side1": [1.0 + d for d in grid]
    }:
        raise ValueError("original observed q=0/1 gauge and fixed outward grid required")
    if plan.get("views") != [list(v) for v in AZIMUTH_ELEVATION_DEGREES] or plan.get("required_render_shape") != [2, 129, 6, 3, 224, 224]:
        raise ValueError("requires all fixed six global views")
    rows = plan.get("rows")
    if not isinstance(rows, list) or len(rows) != 19:
        raise ValueError("requires the frozen Articraft19 plan")
    seen = set()
    for row in rows:
        _validate_row(row)
        # Raw offline identities are only needed for asset lookup; never export.
        if not re.fullmatch(r"[A-Za-z0-9_-][A-Za-z0-9_.-]*", row["object_id"]) or ".." in row["object_id"]:
            raise ValueError("invalid offline object identity")
        if row["episode_id"] != row["object_id"] or row["object_id"] in seen:
            raise ValueError("requires one original episode per distinct Articraft object")
        seen.add(row["object_id"])
    if sum(row["split"] == "source_train" for row in rows) != 13:
        raise ValueError("requires fixed 13/6 split")
    if max_objects is not None:
        if not 1 <= max_objects <= 13:
            raise ValueError("smoke max_objects must be 1..13 source-train objects")
        # Selection occurs on ORIGINAL source-train IDs before anonymization.
        rows = sorted((r for r in rows if r["split"] == "source_train"), key=lambda r: r["object_id"])[:max_objects]
    else:
        rows = sorted(rows, key=lambda r: (r["split"], r["object_id"]))
    # Only renderer-required metadata passes downstream. No source index is opened.
    return {key: plan[key] for key in ("schema", "domain", "target_labels_used", "canonical_q", "views", "required_render_shape")} | {"rows": rows}


@torch.no_grad()
def text_direction(model: Any, tokenizer: Any, device: torch.device) -> Tensor:
    tokens = tokenizer(OPEN_PROMPTS + CLOSED_PROMPTS).to(device)
    features = F.normalize(model.encode_text(tokens).float(), dim=-1)
    if features.shape != (6, 512) or not torch.isfinite(features).all():
        raise ValueError("unexpected text embeddings")
    opened = F.normalize(features[:3].mean(dim=0), dim=0)
    closed = F.normalize(features[3:].mean(dim=0), dim=0)
    difference = closed - opened
    if difference.norm() < 1e-8:
        raise ValueError("degenerate open-to-closed text direction")
    return F.normalize(difference, dim=0).cpu()


@torch.no_grad()
def encode_object(renderer: Any, model: Any, row: dict[str, Any], plan: dict[str, Any], batch_size: int, device: torch.device) -> Tensor:
    output = torch.empty(2, 129, 6, 512, dtype=torch.float32)
    for side in range(2):
        cursor = 0
        for batch in renderer.iter_candidate_batches(
            row=row, side=side, canonical_q=plan["canonical_q"][f"side{side}"],
            views=plan["views"], batch_size=batch_size,
        ):
            if not isinstance(batch, dict) or set(batch) != {"start", "images"} or batch["start"] != cursor:
                raise ValueError("renderer batches must be contiguous and ordered")
            images = batch["images"]
            if not isinstance(images, Tensor) or images.dtype != torch.uint8 or images.ndim != 5 or tuple(images.shape[1:]) != (6, 3, 224, 224):
                raise ValueError("renderer requires uint8 [N,6,3,224,224]")
            count = images.shape[0]
            if not 1 <= count <= batch_size or cursor + count > 129:
                raise ValueError("invalid renderer batch length")
            pixels = FrozenCLIPLimitDiscriminator.preprocess(images.reshape(-1, 3, 224, 224).to(device))
            features = F.normalize(model.encode_image(pixels).float(), dim=-1).cpu()
            _unit(features, (count * 6, 512))
            output[side, cursor:cursor + count] = features.reshape(count, 6, 512)
            cursor += count
        if cursor != 129:
            raise ValueError("renderer missed fixed candidates")
    return output


def build_cache(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise ValueError("output exists; cache artifacts are immutable and never overwritten")
    if not 1 <= args.batch_size <= 32:
        raise ValueError("batch size must be 1..32")
    plan = load_render_plan(args.render_plan, args.render_plan_sha256, args.max_objects)
    _verify_open_clip_install(args.open_clip_wheel, args.open_clip_wheel_sha256, OPEN_CLIP_TORCH_VERSION)
    renderer, config_hash = _load_renderer_plugin(
        args.renderer_script, args.renderer_sha256, args.asset_root, "articraft", plan, args.device
    )
    device = torch.device(args.device)
    wrapper = load_frozen_open_clip_vit_b32(args.clip_checkpoint, args.clip_checkpoint_sha256, device)
    model = wrapper.clip_model.eval()
    if any(p.requires_grad for p in model.parameters()):
        raise RuntimeError("CLIP must be frozen")
    import open_clip
    direction = text_direction(model, open_clip.get_tokenizer("ViT-B-32"), device)
    provenance = {
        "render_plan_sha256": args.render_plan_sha256, "renderer_sha256": args.renderer_sha256,
        "render_config_sha256": config_hash, "encoder_checkpoint_sha256": args.clip_checkpoint_sha256,
        "open_clip_wheel_sha256": args.open_clip_wheel_sha256,
        "prompt_sha256": _digest({"open": OPEN_PROMPTS, "closed": CLOSED_PROMPTS}),
        "builder_sha256": sha256_file(Path(__file__)),
    }
    _hashes(provenance, HASH_KEYS)
    (args.output / "artifacts").mkdir(parents=True, exist_ok=False)
    index: dict[str, Any] = {"schema": INDEX_SCHEMA, "rows": [], "provenance": provenance, "split_counts": {}}
    for number, row in enumerate(plan["rows"], 1):
        object_id = opaque_object_id(row["object_id"])
        print(json.dumps({"event": "object_start", "number": number, "total": len(plan["rows"]), "object_id": object_id}), flush=True)
        features = encode_object(renderer, model, row, plan, args.batch_size, device)
        artifact = {
            "schema": ARTIFACT_SCHEMA, "object_id": object_id, "split": row["split"],
            "coordinates": fixed_candidate_grid().expand(2, -1).clone(),
            "image_embeddings": features, "observed_embeddings": features[:, 0].clone(),
            "text_direction": direction.clone(),
            "provenance": provenance | {"asset_bundle_sha256": renderer.asset_bundle_sha256(row)},
        }
        validate_relative_artifact(artifact)
        relative = f"artifacts/{object_id}.pt"
        target = args.output / relative
        with target.open("xb") as stream:
            torch.save(artifact, stream)
        index["rows"].append({"object_id": object_id, "split": row["split"], "artifact": relative, "artifact_sha256": sha256_file(target)})
        index["split_counts"][row["split"]] = index["split_counts"].get(row["split"], 0) + 1
        print(json.dumps({"event": "object_done", "number": number, "total": len(plan["rows"]), "object_id": object_id}), flush=True)
    audit = renderer.audit_receipt()
    if audit.get("no_empty_views_checked") is not True or audit.get("repeat_bit_identical") is not True or audit.get("assets_repeat_audited") != len(plan["rows"]) or audit.get("views_checked", 0) < len(plan["rows"]) * 2 * 129 * 6 or audit.get("render_config_sha256") != config_hash:
        raise RuntimeError("renderer failed coverage or determinism audit")
    validate_relative_index(index)
    with (args.output / "index.json").open("x", encoding="utf-8") as stream:
        json.dump(index, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({"event": "complete", "objects": len(index["rows"]), "diagnostic": "ideal mesh stylized RGB; global views only; not 3DGS"}), flush=True)
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("render-plan", "renderer-script", "asset-root", "output", "clip-checkpoint", "open-clip-wheel"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in ("render-plan-sha256", "renderer-sha256", "clip-checkpoint-sha256", "open-clip-wheel-sha256"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-objects", type=int)
    args = parser.parse_args()
    for value in vars(args).values():
        if any(marker in str(value).lower() for marker in ("sealed", "b_test", "full22", "box_e", "box_f")):
            raise ValueError("protected path marker in cache launch")
    build_cache(args)


if __name__ == "__main__":
    main()
