#!/usr/bin/env python3
"""Publish immutable target-free surface measurements; no endpoint evaluation."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from splart.observed_surfaces import load_surfaces, swap_surface
from splart.relative_search import write_json_new
from splart.surface_contact import CONFIG, assert_exact_swap, backend_receipt, file_sha256, measure_surface


def run(surfaces: Path, base_inputs: Path, output_dir: Path, *, train_smoke2: bool = False) -> dict:
    if output_dir.exists():
        raise ValueError("measurement output directory must be new")
    started = time.monotonic()
    index, values = load_surfaces(surfaces, base_input_index=base_inputs)
    if train_smoke2:
        values = [value for value in values if value["split"] == "source_train"][:2]
        if len(values) != 2:
            raise ValueError("smoke requires two source training objects")
    output_dir.mkdir(parents=True)
    receipt = backend_receipt()
    write_json_new(output_dir / "backend_receipt.json", receipt)
    provenance = {"surface_index_sha256": file_sha256(surfaces), "base_input_index_sha256": file_sha256(base_inputs),
                  "engine_sha256": file_sha256(ROOT / "src/splart/surface_contact.py"),
                  "consumer_sha256": file_sha256(ROOT / "src/splart/observed_surfaces.py"),
                  "runner_sha256": file_sha256(Path(__file__)),
                  "backend_receipt_sha256": file_sha256(output_dir / "backend_receipt.json")}
    rows, diagnostics = [], []
    surface_rows = {(row["split"], row["object_id"]): row for row in index["rows"]}
    for value in values:
        object_started = time.monotonic()
        print(json.dumps({"stage": "measure_start", "object_id": value["object_id"], "split": value["split"]}), flush=True)
        artifact = measure_surface(value)
        swapped = measure_surface(swap_surface(value))
        assert_exact_swap(artifact, swapped)
        artifact["provenance"] = dict(provenance, surface_artifact_sha256=surface_rows[(value["split"], value["object_id"])]["artifact_sha256"])
        name = f"contact-{len(rows):03d}.pt"
        torch.save(artifact, output_dir / name)
        row = {"object_id": value["object_id"], "split": value["split"], "artifact": name,
               "artifact_sha256": file_sha256(output_dir / name)}
        rows.append(row)
        diagnostic = {"object_id": value["object_id"], "split": value["split"],
                      "observed_intersection_count": artifact["observed_intersection_count"],
                      "observed_anchor_intersection": artifact["observed_anchor_intersection"],
                      "observed_seen_pair_count": len(artifact["observed_seen_pairs"]),
                      "persistent_all_observed_pair_count": len(artifact["persistent_all_observed_pairs"]),
                      "side_intersection_poses": [int(side["surface_intersection"].sum()) for side in artifact["sides"]],
                      "side_novel_pair_poses": [int((side["novel_pair_count"] > 0).sum()) for side in artifact["sides"]],
                      "maximum_returned_contacts": max(int(side["contact_count"].max()) for side in [artifact["observed"], *artifact["sides"]]),
                      "exact_swap_passed": True, "enumeration_complete": True,
                      "boundary_confirmed": False, "elapsed_seconds": time.monotonic() - object_started}
        diagnostics.append(diagnostic)
        print(json.dumps(dict(stage="measure_complete", completed=len(rows), total=len(values), **diagnostic), sort_keys=True), flush=True)
    counts = {split: sum(row["split"] == split for row in rows) for split in ("source_train", "source_validation")}
    audit = {"schema": "surface-contact-audit/v1", "split_counts": counts, "rows": diagnostics,
             "config": asdict(CONFIG), "provenance": provenance, "exact_swap_passed": True,
             "all_contact_enumerations_complete": True, "endpoint_labels_read": False,
             "complete_fixed_source_roster": not train_smoke2, "endpoint_selection_authorized": False,
             "elapsed_seconds": time.monotonic() - started,
             "limitation": "discrete_surface_contact_not_solid_collision_or_continuous_motion_certificate"}
    write_json_new(output_dir / "audit.json", audit)
    write_json_new(output_dir / "index.json", {"schema": "surface-contact-index/v1", "rows": rows,
                   "provenance": provenance, "split_counts": counts, "audit_sha256": file_sha256(output_dir / "audit.json")})
    return {"index": str(output_dir / "index.json"), "objects": len(rows), "elapsed_seconds": time.monotonic() - started,
            "endpoint_selection_authorized": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--surfaces", type=Path, required=True)
    parser.add_argument("--base-inputs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-smoke2", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.surfaces, args.base_inputs, args.output_dir, train_smoke2=args.train_smoke2), sort_keys=True))
