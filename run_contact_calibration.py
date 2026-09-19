#!/usr/bin/env python3
"""Falsify a frozen observed-contact atlas on 64 interstitial poses, no labels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from splart.contact_calibration import (COUNTS, assert_midpoint_swap, load_calibration, load_contacts,
                                      measure_midpoints, swap_contact)
from splart.observed_surfaces import load_surfaces, swap_surface
from splart.relative_search import file_sha256, write_json_new
from splart.surface_contact import backend_receipt


def run(surfaces: Path, contacts: Path, base_inputs: Path, output_dir: Path) -> dict:
    if output_dir.exists():
        raise ValueError("calibration output directory must be new")
    started = time.monotonic()
    contact_index, contact_values = load_contacts(contacts, base_inputs)
    surface_index, surface_values = load_surfaces(surfaces, base_input_index=base_inputs)
    if contact_index["provenance"]["surface_index_sha256"] != file_sha256(surfaces):
        raise ValueError("surface/contact index binding mismatch")
    original_backend = json.loads((contacts.parent / "backend_receipt.json").read_text(encoding="utf-8"))
    if backend_receipt() != original_backend:
        raise ValueError("calibration backend differs from frozen contact acquisition")
    contact_rows = {(row["split"], row["object_id"]): row for row in contact_index["rows"]}
    surface_rows = {(row["split"], row["object_id"]): row for row in surface_index["rows"]}
    provenance = {"contact_index_sha256": file_sha256(contacts), "base_input_index_sha256": file_sha256(base_inputs),
                  "surface_index_sha256": file_sha256(surfaces), "calibrator_sha256": file_sha256(ROOT / "src/splart/contact_calibration.py"),
                  "runner_sha256": file_sha256(Path(__file__)),
                  "backend_receipt_sha256": contact_index["provenance"]["backend_receipt_sha256"]}
    output_dir.mkdir(parents=True)
    rows = []
    for surface in surface_values:
        identity = surface["split"], surface["object_id"]
        contact = contact_values[identity]
        if contact["provenance"]["surface_artifact_sha256"] != surface_rows[identity]["artifact_sha256"]:
            raise ValueError("contact uses another observed surface artifact")
        print(json.dumps({"stage": "midpoint_start", "object_id": surface["object_id"], "split": surface["split"]}), flush=True)
        artifact = measure_midpoints(surface, contact)
        swapped = measure_midpoints(swap_surface(surface), swap_contact(contact))
        assert_midpoint_swap(artifact, swapped)
        artifact["exact_swap_passed"] = True
        artifact["provenance"] = dict(provenance, contact_artifact_sha256=contact_rows[identity]["artifact_sha256"])
        name = f"midpoint-{len(rows):03d}.json"
        write_json_new(output_dir / name, artifact)
        row = {key: artifact[key] for key in ("object_id", "split", "calibration_passed", "midpoint_new_event_count", "enumeration_complete", "exact_swap_passed")}
        row.update(contact_artifact_sha256=contact_rows[identity]["artifact_sha256"],
                   midpoint_artifact=name, midpoint_artifact_sha256=file_sha256(output_dir / name))
        rows.append(row)
        print(json.dumps(dict(stage="midpoint_complete", completed=len(rows), total=len(surface_values), **row)), flush=True)
    write_json_new(output_dir / "audit.json", {"schema": "contact-calibration-audit/v1", "rows": rows,
                   "provenance": provenance, "split_counts": COUNTS, "endpoint_labels_read": False, "atlas_updated": False,
                   "all_enumerations_complete": True, "exact_swap_passed": True, "elapsed_seconds": time.monotonic() - started})
    write_json_new(output_dir / "index.json", {"schema": "contact-calibration-index/v1", "rows": rows,
                   "provenance": provenance, "split_counts": COUNTS, "audit_sha256": file_sha256(output_dir / "audit.json")})
    # Consumer-side verification must succeed before this job reports success.
    load_calibration(output_dir / "index.json", contacts, base_inputs)
    return {"index": str(output_dir / "index.json"), "objects": len(rows),
            "calibration_passed": sum(row["calibration_passed"] for row in rows),
            "elapsed_seconds": time.monotonic() - started}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--surfaces", type=Path, required=True)
    parser.add_argument("--contacts", type=Path, required=True)
    parser.add_argument("--base-inputs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.surfaces, args.contacts, args.base_inputs, args.output_dir), sort_keys=True))
