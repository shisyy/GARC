#!/usr/bin/env python3
"""Acquire raw and deletion CLIP arms from one saved RGB batch per candidate.

The immutable historical semantic cache supplies only frozen text directions,
identity and provenance. Historical image embeddings are never compared or used
to accept/reject acquired pixels. The unchanged renderer may execute its own
initial asset audit; acquisition never requests a feature-dependent rerender.

Output: relative/index.json, part/index.json, rgb/<opaque>/sideS_CCC.pt,
acquisition_audit.json (published last, binding both indices and all saved RGB).
No source index, endpoints or fitted parameters are read.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

import build_part_interaction_cache as previous

base = previous.base
rules = previous.rules


def publish_json(path: Path, value: dict) -> None:
    """Atomic publication within an exclusively created output directory."""
    if path.exists():
        raise ValueError("JSON artifact already exists; outputs are immutable")
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.rename(temporary, path)


def save_rgb_shard(root: Path, object_id: str, side: int, candidate: int, images: torch.Tensor) -> dict:
    if images.shape != (6, 3, 224, 224) or images.dtype != torch.uint8 or images.device.type != "cpu":
        raise ValueError("saved RGB must be original CPU uint8 six-view pixels")
    relative = f"rgb/{object_id}/side{side}_{candidate:03d}.pt"
    path = root / relative
    with path.open("xb") as stream:
        torch.save(images.contiguous().clone(), stream)
    return {"path": relative, "sha256": base.sha256_file(path), "side": side, "candidate_index": candidate}


@torch.no_grad()
def acquire_object(renderer, model, row, plan, device, root: Path):
    """One accepted RGB batch unconditionally supplies all five feature arrays."""
    object_id = base.opaque_object_id(row["object_id"])
    (root / "rgb" / object_id).mkdir(parents=True, exist_ok=False)
    joint = torch.empty(2, 129, 6, 512, dtype=torch.float32)
    auxiliary = {key: torch.empty_like(joint) for key in rules.FEATURE_KEYS}
    shards = []
    for side in range(2):
        cursor = 0
        batches = renderer.iter_candidate_batches(row=row, side=side, canonical_q=plan["canonical_q"][f"side{side}"], views=plan["views"], batch_size=1)
        for batch in batches:
            if not isinstance(batch, dict) or set(batch) != {"start", "images"} or batch["start"] != cursor or cursor >= 129:
                raise ValueError("renderer batches must be contiguous and ordered")
            images = batch["images"]
            if not isinstance(images, torch.Tensor) or images.shape != (1, 6, 3, 224, 224) or images.dtype != torch.uint8 or images.device.type != "cpu":
                raise ValueError("renderer must emit one pose/six original CPU uint8 views")
            images = images[0]
            # Save this exact acquisition before all encodings, never rerender it.
            shards.append(save_rgb_shard(root, object_id, side, cursor, images))
            joint[side, cursor] = previous.encode_views(model, images, device)
            for key, deleted in rules.deletion_images(images):
                auxiliary[key][side, cursor] = previous.encode_views(model, deleted, device)
            cursor += 1
        if cursor != 129:
            raise ValueError("renderer missed fixed candidates")
    return joint, auxiliary, shards


def save_artifact(root: Path, artifact: dict) -> dict:
    relative = f"artifacts/{artifact['object_id']}.pt"
    target = root / relative
    with target.open("xb") as stream:
        torch.save(artifact, stream)
    return {"object_id": artifact["object_id"], "split": artifact["split"], "artifact": relative, "artifact_sha256": base.sha256_file(target)}


def build_cache(args: argparse.Namespace) -> dict:
    if args.output.exists():
        raise ValueError("output exists; shared RGB acquisition is immutable")
    if args.batch_size != 1:
        raise ValueError("shared RGB acquisition requires batch size exactly 1")
    plan = base.load_render_plan(args.render_plan, args.render_plan_sha256, args.max_objects)
    historical = previous.load_base_index(args.base_semantic_index)
    historical_hash = base.sha256_file(args.base_semantic_index)
    historical_rows = {row["object_id"]: row for row in historical["rows"]}
    roster = {(base.opaque_object_id(row["object_id"]), row["split"]) for row in plan["rows"]}
    old_roster = {(row["object_id"], row["split"]) for row in historical["rows"]}
    if not roster <= old_roster or (args.max_objects is None and roster != old_roster):
        raise ValueError("historical semantic roster differs from render plan")
    if historical["provenance"]["prompt_sha256"] != base._digest({"open": base.OPEN_PROMPTS, "closed": base.CLOSED_PROMPTS}):
        raise ValueError("historical prompts differ from fixed prompts")
    device = torch.device(args.device)
    if device.type == "cuda":
        total = torch.cuda.get_device_properties(device).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, 8 * 1024 ** 3 / total), device)
    base._verify_open_clip_install(args.open_clip_wheel, args.open_clip_wheel_sha256, base.OPEN_CLIP_TORCH_VERSION)
    renderer, config_hash = base._load_renderer_plugin(args.renderer_script, args.renderer_sha256, args.asset_root, "articraft", plan, args.device)
    common = {
        "render_plan_sha256": args.render_plan_sha256, "renderer_sha256": args.renderer_sha256,
        "render_config_sha256": config_hash, "encoder_checkpoint_sha256": args.clip_checkpoint_sha256,
        "open_clip_wheel_sha256": args.open_clip_wheel_sha256,
    }
    if any(historical["provenance"][key] != value for key, value in common.items()):
        raise ValueError("historical semantic provenance mismatch")
    model = base.load_frozen_open_clip_vit_b32(args.clip_checkpoint, args.clip_checkpoint_sha256, device).clip_model.eval()
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("CLIP must be frozen")
    blank = previous.encode_views(model, torch.full((6, 3, 224, 224), 255, dtype=torch.uint8), device)
    if not torch.allclose(blank, blank[0].expand_as(blank), atol=1e-5, rtol=1e-5):
        raise ValueError("blank embedding differs across identical views")
    relative_root, part_root = args.output / "relative", args.output / "part"
    args.output.mkdir(parents=True, exist_ok=False)
    (relative_root / "artifacts").mkdir(parents=True)
    (part_root / "artifacts").mkdir(parents=True)
    builder_hash = base.sha256_file(Path(__file__))
    relative_provenance = common | {"prompt_sha256": historical["provenance"]["prompt_sha256"], "builder_sha256": builder_hash}
    relative_index = {"schema": base.INDEX_SCHEMA, "rows": [], "provenance": relative_provenance, "split_counts": {}}
    pending_auxiliary = []
    receipt_rows = []
    for number, row in enumerate(plan["rows"], 1):
        object_id = base.opaque_object_id(row["object_id"])
        print(json.dumps({"event": "object_start", "number": number, "total": len(plan["rows"]), "object_id": object_id}), flush=True)
        old = previous.load_base_artifact(args.base_semantic_index.parent, historical_rows[object_id], historical)
        asset_hash = renderer.asset_bundle_sha256(row)
        if asset_hash != old["provenance"]["asset_bundle_sha256"]:
            raise ValueError("asset bundle differs from historical cache")
        joint, auxiliary, shards = acquire_object(renderer, model, row, plan, device, args.output)
        raw = {
            "schema": base.ARTIFACT_SCHEMA, "object_id": object_id, "split": row["split"],
            "coordinates": base.fixed_candidate_grid().expand(2, -1).clone(),
            "image_embeddings": joint, "observed_embeddings": joint[:, 0].clone(),
            "text_direction": old["text_direction"].clone(),
            "provenance": relative_provenance | {"asset_bundle_sha256": asset_hash},
        }
        base.validate_relative_artifact(raw)
        relative_index["rows"].append(save_artifact(relative_root, raw))
        relative_index["split_counts"][row["split"]] = relative_index["split_counts"].get(row["split"], 0) + 1
        # At most19*4*2*129*6*512 floats (~242MB); never retain full RGB in memory.
        pending_auxiliary.append((object_id, row["split"], asset_hash, auxiliary))
        receipt_rows.append({"object_id": object_id, "split": row["split"], "historical_semantic_artifact_sha256": historical_rows[object_id]["artifact_sha256"], "rgb_shards": shards})
        print(json.dumps({"event": "object_done", "number": number, "total": len(plan["rows"]), "object_id": object_id}), flush=True)
    audit = renderer.audit_receipt()
    if audit.get("no_empty_views_checked") is not True or audit.get("repeat_bit_identical") is not True or audit.get("assets_repeat_audited") != len(plan["rows"]) or audit.get("views_checked", 0) < len(plan["rows"]) * 2 * 129 * 6 or audit.get("render_config_sha256") != config_hash:
        raise RuntimeError("unchanged renderer failed its coverage/initial-asset audit")
    if base.sha256_file(args.base_semantic_index) != historical_hash:
        raise ValueError("historical semantic index changed during acquisition")
    base.validate_relative_index(relative_index)
    publish_json(relative_root / "index.json", relative_index)
    relative_index_hash = base.sha256_file(relative_root / "index.json")
    part_provenance = common | {"builder_sha256": builder_hash, "mask_rule_sha256": base.sha256_file(Path(rules.__file__)), "base_semantic_index_sha256": relative_index_hash}
    part_index = {"schema": rules.INDEX_SCHEMA, "rows": [], "provenance": part_provenance, "split_counts": dict(relative_index["split_counts"])}
    for object_id, split, asset_hash, auxiliary in pending_auxiliary:
        artifact = {
            "schema": rules.ARTIFACT_SCHEMA, "object_id": object_id, "split": split,
            "coordinates": base.fixed_candidate_grid().expand(2, -1).clone(), **auxiliary,
            "blank_embedding": blank[0].clone(), "provenance": part_provenance | {"asset_bundle_sha256": asset_hash},
        }
        rules.validate_part_artifact(artifact)
        part_index["rows"].append(save_artifact(part_root, artifact))
    rules.validate_part_index(part_index)
    publish_json(part_root / "index.json", part_index)
    receipt = {
        "schema": "shared-rgb-acquisition/v1", "relative_index_sha256": relative_index_hash,
        "part_index_sha256": base.sha256_file(part_root / "index.json"),
        "original_base_semantic_index_sha256": historical_hash, "rows": receipt_rows,
        "acquisition": "single_render_shared_rgb",
    }
    publish_json(args.output / "acquisition_audit.json", receipt)
    print(json.dumps({"event": "complete", "objects": len(receipt_rows), "rgb_shards": sum(len(row["rgb_shards"]) for row in receipt_rows), "acquisition": "single_render_shared_rgb", "historical_joint_feature_selection": False}), flush=True)
    return receipt


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
        raise ValueError("protected path marker in shared RGB acquisition")
    build_cache(args)


if __name__ == "__main__":
    main()
