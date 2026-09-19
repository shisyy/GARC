#!/usr/bin/env python3
"""Frozen visible-part deletion embeddings, bound to node9.1 baseline cache.

Stylized ideal-mesh diagnostic only. No source index or endpoint labels are
accepted. Batch one pose/six views, four deletions encoded separately. Publish
immutable content artifacts first and an atomic complete index last.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import build_relative_clip_cache as base
import splart.part_interaction_cache as rules
from splart.part_interaction_cache import FEATURE_KEYS, validate_part_artifact, validate_part_index


def load_base_index(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("base semantic index must be a regular file")
    value = json.loads(path.read_text(encoding="utf-8"))
    base.validate_relative_index(value)
    return value


def load_base_artifact(root: Path, row: dict, index: dict) -> dict:
    path = root / row["artifact"]
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root.resolve()) or base.sha256_file(path) != row["artifact_sha256"]:
        raise ValueError("base semantic artifact path or digest mismatch")
    artifact = torch.load(path, map_location="cpu", weights_only=True)
    base.validate_relative_artifact(artifact)
    if any(artifact[k] != row[k] for k in ("object_id", "split")) or any(artifact["provenance"][k] != v for k, v in index["provenance"].items()):
        raise ValueError("base semantic artifact binding mismatch")
    return artifact


@torch.no_grad()
def encode_views(model: Any, images: torch.Tensor, device: torch.device) -> torch.Tensor:
    if images.shape != (6, 3, 224, 224) or images.dtype != torch.uint8:
        raise ValueError("encoding requires exactly six uint8 views")
    pixels = base.FrozenCLIPLimitDiscriminator.preprocess(images.to(device))
    features = F.normalize(model.encode_image(pixels).float(), dim=-1).cpu()
    base._unit(features, (6, 512))
    return features


def reproduction_audit() -> dict:
    return {
        "schema": "part-render-reproduction-audit/v1", "status": "running",
        "fixed_max_render_attempts": 3, "atol": 1e-5, "rtol": 1e-5,
        "summary": {"candidates_checked": 0, "direct_matches": 0, "triggered_candidates": 0,
                    "additional_renders": 0, "failed_attempts": 0, "recovered_candidates": 0,
                    "unrecovered_candidates": 0},
        "events": [],
        "interpretation": "Bounded reproduction of pinned baseline content, not physical determinism. Rasterization can vary at a pixel; no candidate, seed, tolerance or semantic selection changes.",
    }


def _tensor_hash(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _rerender_candidate(renderer, row, plan, side, candidate):
    # The unchanged public iterator receives the exact full fixed coordinate list.
    # Earlier poses are traversed, never substituted for the requested candidate.
    for batch in renderer.iter_candidate_batches(row=row, side=side, canonical_q=plan["canonical_q"][f"side{side}"], views=plan["views"], batch_size=1):
        if batch["start"] == candidate:
            images = batch["images"]
            if not isinstance(images, torch.Tensor) or images.shape != (1, 6, 3, 224, 224) or images.dtype != torch.uint8:
                raise ValueError("retry renderer must emit one pose/six uint8 views")
            return images[0]
    raise ValueError("retry renderer missed fixed candidate")


@torch.no_grad()
def matched_joint_images(renderer, model, row, plan, baseline, device, side, candidate, images, audit):
    """At most three renders; accept only the original pinned feature tolerance."""
    expected = baseline["image_embeddings"][side, candidate]
    features = encode_views(model, images, device)
    summary = audit["summary"]
    summary["candidates_checked"] += 1
    if torch.allclose(features, expected, atol=1e-5, rtol=1e-5):
        summary["direct_matches"] += 1
        return images
    summary["triggered_candidates"] += 1
    event = {"object_id": base.opaque_object_id(row["object_id"]), "side": side,
             "candidate_index": candidate, "attempts": [], "accepted": False,
             "attempt_count": 0, "failed_attempts": 0}
    audit["events"].append(event)
    for attempt in range(1, 4):
        if attempt > 1:
            summary["additional_renders"] += 1
            images = _rerender_candidate(renderer, row, plan, side, candidate)
            features = encode_views(model, images, device)
        accepted = bool(torch.allclose(features, expected, atol=1e-5, rtol=1e-5))
        receipt = {"attempt": attempt, "max_abs_diff": float((features - expected).abs().max()),
                   "pixel_sha256": _tensor_hash(images), "feature_sha256": _tensor_hash(features),
                   "accepted": accepted}
        event["attempts"].append(receipt)
        event["attempt_count"] = attempt
        if accepted:
            event.update({"accepted": True, "accepted_pixel_sha256": receipt["pixel_sha256"],
                          "accepted_feature_sha256": receipt["feature_sha256"]})
            summary["recovered_candidates"] += 1
            print(json.dumps({"event": "render_reproduction_recovered", "object_id": event["object_id"], "side": side, "candidate_index": candidate, "attempt_count": attempt}), flush=True)
            return images
        event["failed_attempts"] += 1
        summary["failed_attempts"] += 1
    summary["unrecovered_candidates"] += 1
    raise ValueError(f"recomputed joint embedding differs from pinned baseline after 3 fixed render attempts: object_id={event['object_id']} side={side} candidate={candidate}")


def write_reproduction_audit(output: Path, audit: dict, status: str) -> None:
    audit["status"] = status
    with (output / "render_reproduction_audit.json").open("x", encoding="utf-8") as stream:
        json.dump(audit, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


@torch.no_grad()
def encode_object(renderer, model, row, plan, baseline, device, audit=None) -> dict[str, torch.Tensor]:
    if audit is None:
        audit = reproduction_audit()
    output = {key: torch.empty(2, 129, 6, 512, dtype=torch.float32) for key in FEATURE_KEYS}
    for side in range(2):
        cursor = 0
        for batch in renderer.iter_candidate_batches(row=row, side=side, canonical_q=plan["canonical_q"][f"side{side}"], views=plan["views"], batch_size=1):
            if not isinstance(batch, dict) or set(batch) != {"start", "images"} or batch["start"] != cursor or cursor >= 129:
                raise ValueError("renderer batches must be contiguous and ordered")
            images = batch["images"]
            if not isinstance(images, torch.Tensor) or images.shape != (1, 6, 3, 224, 224) or images.dtype != torch.uint8:
                raise ValueError("renderer must emit one pose/six uint8 views")
            images = images[0]
            images = matched_joint_images(renderer, model, row, plan, baseline, device, side, cursor, images, audit)
            for key, deleted in rules.deletion_images(images):
                output[key][side, cursor] = encode_views(model, deleted, device)
            cursor += 1
        if cursor != 129:
            raise ValueError("renderer missed fixed candidates")
    return output


def build_cache(args: argparse.Namespace) -> dict:
    if args.output.exists():
        raise ValueError("output exists; cache artifacts are immutable")
    if args.batch_size != 1:
        raise ValueError("node9.3 requires batch size exactly 1")
    plan = base.load_render_plan(args.render_plan, args.render_plan_sha256, args.max_objects)
    baseline_index = load_base_index(args.base_semantic_index)
    baseline_hash = base.sha256_file(args.base_semantic_index)
    baseline_rows = {row["object_id"]: row for row in baseline_index["rows"]}
    roster = {(base.opaque_object_id(row["object_id"]), row["split"]) for row in plan["rows"]}
    base_roster = {(row["object_id"], row["split"]) for row in baseline_index["rows"]}
    if not roster <= base_roster or (args.max_objects is None and roster != base_roster):
        raise ValueError("base semantic roster differs from render plan")
    device = torch.device(args.device)
    if device.type == "cuda":
        total = torch.cuda.get_device_properties(device).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, 8 * 1024 ** 3 / total), device)
    base._verify_open_clip_install(args.open_clip_wheel, args.open_clip_wheel_sha256, base.OPEN_CLIP_TORCH_VERSION)
    renderer, config_hash = base._load_renderer_plugin(args.renderer_script, args.renderer_sha256, args.asset_root, "articraft", plan, args.device)
    provenance = {
        "render_plan_sha256": args.render_plan_sha256, "renderer_sha256": args.renderer_sha256,
        "render_config_sha256": config_hash, "encoder_checkpoint_sha256": args.clip_checkpoint_sha256,
        "open_clip_wheel_sha256": args.open_clip_wheel_sha256,
        "builder_sha256": base.sha256_file(Path(__file__)), "mask_rule_sha256": base.sha256_file(Path(rules.__file__)),
        "base_semantic_index_sha256": baseline_hash,
    }
    for key in ("render_plan_sha256", "renderer_sha256", "render_config_sha256", "encoder_checkpoint_sha256", "open_clip_wheel_sha256"):
        if provenance[key] != baseline_index["provenance"][key]:
            raise ValueError(f"base semantic provenance mismatch: {key}")
    model = base.load_frozen_open_clip_vit_b32(args.clip_checkpoint, args.clip_checkpoint_sha256, device).clip_model.eval()
    if any(p.requires_grad for p in model.parameters()):
        raise RuntimeError("CLIP must be frozen")
    blank = encode_views(model, torch.full((6, 3, 224, 224), 255, dtype=torch.uint8), device)
    if not torch.allclose(blank, blank[0].expand_as(blank), atol=1e-5, rtol=1e-5):
        raise ValueError("blank embedding differs across identical views")
    (args.output / "artifacts").mkdir(parents=True, exist_ok=False)
    index = {"schema": rules.INDEX_SCHEMA, "rows": [], "provenance": provenance, "split_counts": {}}
    reproduction = reproduction_audit()
    for number, row in enumerate(plan["rows"], 1):
        object_id = base.opaque_object_id(row["object_id"])
        print(json.dumps({"event": "object_start", "number": number, "total": len(plan["rows"]), "object_id": object_id}), flush=True)
        baseline = load_base_artifact(args.base_semantic_index.parent, baseline_rows[object_id], baseline_index)
        asset_hash = renderer.asset_bundle_sha256(row)
        if asset_hash != baseline["provenance"]["asset_bundle_sha256"]:
            raise ValueError("asset bundle changed from baseline")
        try:
            auxiliary = encode_object(renderer, model, row, plan, baseline, device, reproduction)
        except Exception:
            write_reproduction_audit(args.output, reproduction, "failed")
            raise
        artifact = {
            "schema": rules.ARTIFACT_SCHEMA, "object_id": object_id, "split": row["split"],
            "coordinates": base.fixed_candidate_grid().expand(2, -1).clone(),
            **auxiliary,
            "blank_embedding": blank[0].clone(), "provenance": provenance | {"asset_bundle_sha256": asset_hash},
        }
        validate_part_artifact(artifact)
        relative = f"artifacts/{object_id}.pt"
        target = args.output / relative
        with target.open("xb") as stream:
            torch.save(artifact, stream)
        index["rows"].append({"object_id": object_id, "split": row["split"], "artifact": relative, "artifact_sha256": base.sha256_file(target)})
        index["split_counts"][row["split"]] = index["split_counts"].get(row["split"], 0) + 1
        print(json.dumps({"event": "object_done", "number": number, "total": len(plan["rows"]), "object_id": object_id}), flush=True)
    audit = renderer.audit_receipt()
    if audit.get("no_empty_views_checked") is not True or audit.get("repeat_bit_identical") is not True or audit.get("assets_repeat_audited") != len(plan["rows"]) or audit.get("views_checked", 0) < len(plan["rows"]) * 2 * 129 * 6 or audit.get("render_config_sha256") != config_hash:
        raise RuntimeError("renderer failed coverage or determinism audit")
    if base.sha256_file(args.base_semantic_index) != baseline_hash:
        raise ValueError("base semantic index changed during build")
    validate_part_index(index)
    write_reproduction_audit(args.output, reproduction, "complete")
    temporary = args.output / "index.json.partial"
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(index, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.rename(temporary, args.output / "index.json")
    print(json.dumps({"event": "complete", "objects": len(index["rows"]), "diagnostic": "ideal mesh visible deletion; not physical intervention or 3DGS"}), flush=True)
    return index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("render-plan", "renderer-script", "asset-root", "output", "clip-checkpoint", "open-clip-wheel", "base-semantic-index"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in ("render-plan-sha256", "renderer-sha256", "clip-checkpoint-sha256", "open-clip-wheel-sha256"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-objects", type=int)
    args = parser.parse_args()
    if any(marker in str(value).lower() for value in vars(args).values() for marker in ("sealed", "b_test", "full22", "box_e", "box_f")):
        raise ValueError("protected path marker in cache launch")
    build_cache(args)


if __name__ == "__main__":
    main()
