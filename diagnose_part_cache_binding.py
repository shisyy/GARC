#!/usr/bin/env python3
"""Bounded input-only diagnosis of node9.3 joint-feature baseline mismatch.

No tolerance changes, endpoint labels, seeds, cache writes or renderer edits.
The baseline has no stored pixels, so reproducible current rendering cannot
prove equality with historical pixels. This script reports that limitation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

import build_part_interaction_cache as cache


def tensor_sha256(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def difference(left, right):
    bit_identical = bool(torch.equal(left, right))
    within_pinned_tolerance = bool(torch.allclose(left, right, atol=1e-5, rtol=1e-5))
    left, right = left.double(), right.double()
    cosine = (left * right).sum(-1) / (left.norm(dim=-1) * right.norm(dim=-1))
    return {
        "bit_identical": bit_identical,
        "within_pinned_tolerance": within_pinned_tolerance,
        "max_abs": float((left - right).abs().max()),
        "per_view_max_abs": (left - right).abs().amax(-1).tolist(),
        "per_view_cosine_difference": (1 - cosine).tolist(),
    }


def candidate_batches(renderer, row, plan, side):
    return renderer.iter_candidate_batches(
        row=row, side=side, canonical_q=plan["canonical_q"][f"side{side}"],
        views=plan["views"], batch_size=1,
    )


def repeated_image(renderer, row, plan, side, candidate):
    for batch in candidate_batches(renderer, row, plan, side):
        if batch["start"] == candidate:
            return batch["images"][0]
    raise ValueError("repeat renderer missed candidate")


@torch.no_grad()
def scan(renderer, model, row, plan, baseline, device):
    # Mirror builder's blank and deletion call pattern. No thresholds changed.
    cache.encode_views(model, torch.full((6, 3, 224, 224), 255, dtype=torch.uint8), device)
    maximum = 0.0
    scanned = 0
    for side in range(2):
        cursor = 0
        for batch in candidate_batches(renderer, row, plan, side):
            if set(batch) != {"start", "images"} or batch["start"] != cursor or cursor >= 129:
                raise ValueError("renderer emitted invalid candidate order")
            images = batch["images"]
            if images.shape != (1, 6, 3, 224, 224) or images.dtype != torch.uint8:
                raise ValueError("renderer must emit exactly one six-view pose")
            images = images[0]
            joint = cache.encode_views(model, images, device)
            original = baseline["image_embeddings"][side, cursor]
            delta = difference(joint, original)
            maximum = max(maximum, delta["max_abs"])
            scanned += 1
            if not delta["within_pinned_tolerance"]:
                encoded_again = cache.encode_views(model, images, device)
                encoded_third = cache.encode_views(model, images, device)
                rerendered = repeated_image(renderer, row, plan, side, cursor)
                rerendered_embedding = cache.encode_views(model, rerendered, device)
                rendered_third = repeated_image(renderer, row, plan, side, cursor)
                third_embedding = cache.encode_views(model, rendered_third, device)
                return {
                    "status": "first_mismatch", "side": side, "candidate_index": cursor,
                    "candidates_scanned": scanned, "maximum_abs_seen": maximum,
                    "joint_vs_baseline": delta,
                    "same_pixels_repeat_vs_first": difference(encoded_again, joint),
                    "same_pixels_third_vs_first": difference(encoded_third, joint),
                    "same_pixels_repeat_vs_baseline": difference(encoded_again, original),
                    "rerender_vs_first": difference(rerendered_embedding, joint),
                    "rerender_vs_baseline": difference(rerendered_embedding, original),
                    "third_render_vs_first": difference(third_embedding, joint),
                    "third_render_vs_baseline": difference(third_embedding, original),
                    "pixel_sha256": [tensor_sha256(p) for p in (images, rerendered, rendered_third)],
                    "pixels_repeat_bit_identical": bool(torch.equal(images, rerendered) and torch.equal(images, rendered_third)),
                    "pixel_repeat_changed_values": [int((images != p).sum()) for p in (rerendered, rendered_third)],
                    "feature_sha256": [tensor_sha256(p) for p in (original, joint, encoded_again, encoded_third, rerendered_embedding, third_embedding)],
                    "baseline_pixels_available": False,
                    "interpretation_limit": "Current repeats separate same-pixel inference variation from current rerender variation; no saved baseline pixels exist to distinguish historical rasterization changes from historical inference differences.",
                }
            for _, deleted in cache.rules.deletion_images(images):
                cache.encode_views(model, deleted, device)
            cursor += 1
        if cursor != 129:
            raise ValueError("renderer missed candidates")
    return {"status": "no_mismatch", "candidates_scanned": scanned, "maximum_abs_seen": maximum,
            "interpretation_limit": "Fresh-process scan did not reproduce the earlier multi-object-process failure; allocator/kernel call history may differ. No guarantee of historical baseline pixel equality."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("render-plan", "renderer-script", "asset-root", "output", "clip-checkpoint", "open-clip-wheel", "base-semantic-index"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in ("render-plan-sha256", "renderer-sha256", "clip-checkpoint-sha256", "open-clip-wheel-sha256", "object-id"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if any(marker in str(value).lower() for value in vars(args).values() for marker in ("sealed", "b_test", "full22", "box_e", "box_f")):
        raise ValueError("protected path marker")
    if args.output.exists():
        raise ValueError("diagnostic output is immutable")
    cache.base._identity(args.object_id, "source_train")
    plan = cache.base.load_render_plan(args.render_plan, args.render_plan_sha256, None)
    rows = [row for row in plan["rows"] if cache.base.opaque_object_id(row["object_id"]) == args.object_id]
    if len(rows) != 1:
        raise ValueError("opaque object not uniquely present in pinned plan")
    row = rows[0]
    index = cache.load_base_index(args.base_semantic_index)
    base_rows = [item for item in index["rows"] if item["object_id"] == args.object_id]
    if len(base_rows) != 1 or base_rows[0]["split"] != row["split"]:
        raise ValueError("baseline roster mismatch")
    baseline = cache.load_base_artifact(args.base_semantic_index.parent, base_rows[0], index)
    device = torch.device(args.device)
    if device.type == "cuda":
        total = torch.cuda.get_device_properties(device).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, 8 * 1024 ** 3 / total), device)
    cache.base._verify_open_clip_install(args.open_clip_wheel, args.open_clip_wheel_sha256, cache.base.OPEN_CLIP_TORCH_VERSION)
    renderer, config_hash = cache.base._load_renderer_plugin(args.renderer_script, args.renderer_sha256, args.asset_root, "articraft", plan, args.device)
    expected = {
        "render_plan_sha256": args.render_plan_sha256, "renderer_sha256": args.renderer_sha256,
        "render_config_sha256": config_hash, "encoder_checkpoint_sha256": args.clip_checkpoint_sha256,
        "open_clip_wheel_sha256": args.open_clip_wheel_sha256,
        "asset_bundle_sha256": renderer.asset_bundle_sha256(row),
    }
    if any(baseline["provenance"][key] != value for key, value in expected.items()):
        raise ValueError("baseline provenance mismatch")
    model = cache.base.load_frozen_open_clip_vit_b32(args.clip_checkpoint, args.clip_checkpoint_sha256, device).clip_model.eval()
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("CLIP is not frozen")
    report = {
        "schema": "part-cache-binding-diagnostic/v1", "object_id": args.object_id,
        "provenance": expected | {"diagnostic_sha256": cache.base.sha256_file(Path(__file__)), "base_semantic_index_sha256": cache.base.sha256_file(args.base_semantic_index)},
        "runtime": {"torch": torch.__version__, "cuda": torch.version.cuda, "device": str(device),
                    "tf32_matmul": torch.backends.cuda.matmul.allow_tf32, "tf32_cudnn": torch.backends.cudnn.allow_tf32,
                    "cudnn_benchmark": torch.backends.cudnn.benchmark, "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()},
        **scan(renderer, model, row, plan, baseline, device),
    }
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
