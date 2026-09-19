#!/usr/bin/env python3
"""Trusted ideal-mesh exporter. Authored pose metadata never enters its output.

Only this isolated program may access the approved render plan and assets. It
exports two actual observed mobile poses, NOT the authored canonical-zero mesh.
No normalization is applied; canonical-zero frame_center/frame_scale are unused.
"""
import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from build_clip_limit_cache import _load_renderer_plugin
from build_relative_clip_cache import load_render_plan
from splart.relative_search import file_sha256, load_inputs, opaque_object_id, write_json_new
from splart.observed_surfaces import COUNTS, INDEX_SCHEMA, SCHEMA, load_surfaces, rotate_vertices, validate_surface


def export_object(renderer, row, base_value, provenance, base_artifact_sha256):
    """Trusted conversion discards canonical meshes and absolute pose metadata."""
    asset = renderer._load_asset(row)
    angles = renderer._angles(row, [0.0, 1.0], asset)
    axis = asset.axis.double()
    axis = axis / axis.norm()
    pivot = asset.pivot.double()
    mobile = torch.stack([rotate_vertices(asset.mobile_zero.vertices.double(), axis, pivot, angle) for angle in angles])
    artifact = {"schema": SCHEMA, "object_id": opaque_object_id(row["object_id"]), "split": row["split"],
                "static_vertices": asset.static.vertices.double().clone(), "static_faces": asset.static.faces.long().clone(),
                "mobile_vertices": mobile, "mobile_faces": asset.mobile_zero.faces.long().clone(),
                "axis": axis, "pivot": pivot.clone(), "angular_increment": float(angles[1] - angles[0]),
                "coordinates": base_value["coordinates"].clone(),
                "provenance": {**provenance, "asset_bundle_sha256": asset.bundle_sha256,
                               "base_input_artifact_sha256": base_artifact_sha256}}
    validate_surface(artifact)
    return artifact


def build(args):
    if args.output.exists():
        raise ValueError("output exists; export is immutable")
    if file_sha256(args.base_input_index) != args.base_input_index_sha256:
        raise ValueError("base input index hash mismatch")
    plan = load_render_plan(args.render_plan, args.render_plan_sha256, None)
    base_index, base_values = load_inputs(args.base_input_index)
    by_key = {(v["split"], v["object_id"]): v for v in base_values}
    base_rows = {(v["split"], v["object_id"]): v for v in base_index["rows"]}
    plan_keys = [(r["split"], opaque_object_id(r["object_id"])) for r in plan["rows"]]
    if (len(plan_keys) != 19 or len(set(plan_keys)) != 19 or set(plan_keys) != set(by_key)
            or {s: sum(v["split"] == s for v in base_values) for s in COUNTS} != COUNTS):
        raise ValueError("approved render plan and base input roster must match all 13/6 objects")
    renderer, _ = _load_renderer_plugin(args.renderer_script, args.renderer_sha256, args.asset_root, "articraft", plan, "cpu")
    provenance = {"render_plan_sha256": args.render_plan_sha256, "renderer_sha256": args.renderer_sha256,
                  "base_input_index_sha256": args.base_input_index_sha256, "exporter_sha256": file_sha256(Path(__file__)),
                  "consumer_sha256": file_sha256(ROOT / "src/splart/observed_surfaces.py")}
    (args.output / "artifacts").mkdir(parents=True, exist_ok=False)
    index = {"schema": INDEX_SCHEMA, "rows": [], "split_counts": COUNTS, "provenance": provenance}
    for number, row in enumerate(plan["rows"], 1):
        key = row["split"], opaque_object_id(row["object_id"])
        artifact = export_object(renderer, row, by_key[key], provenance, base_rows[key]["artifact_sha256"])
        relative = f"artifacts/{key[1]}.pt"
        target = args.output / relative
        with target.open("xb") as stream:
            torch.save(artifact, stream)
        index["rows"].append({"object_id": key[1], "split": key[0], "artifact": relative, "artifact_sha256": file_sha256(target)})
        print(json.dumps({"event": "object_exported", "number": number, "total": 19, "object_id": key[1],
                          "static_triangles": len(artifact["static_faces"]), "mobile_triangles": len(artifact["mobile_faces"])}), flush=True)
        renderer._assets.clear()
    # Stage and validate the complete join before publishing the final index.
    staged_index = args.output / "index.pending.json"
    write_json_new(staged_index, index)
    load_surfaces(staged_index, args.base_input_index)
    staged_index.rename(args.output / "index.json")
    print(json.dumps({"event": "complete", "objects": 19, "diagnostic": "full ideal observed meshes; not reconstructed",
                      "index_sha256": file_sha256(args.output / "index.json")}), flush=True)
    return index


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("render-plan", "renderer-script", "asset-root", "base-input-index", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in ("render-plan-sha256", "renderer-sha256", "base-input-index-sha256"):
        parser.add_argument(f"--{name}", required=True)
    args = parser.parse_args()
    build(args)
