#!/usr/bin/env python3
"""Audit exact coverage and non-duplication of the node7.2 delta queue."""

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    public_manifest_path = args.root / "public/profile_manifest.json"
    original_handoff_path = args.root / "redistribution-v1/queue_handoff_receipt.json"
    addendum_path = args.root / "scripts/node72_gpu6_handoff_addendum.json"
    finalize_paths = {
        gpu: args.root / f"redistribution-v1/finalize-gpu{gpu}.json" for gpu in (2, 3)
    }
    receipt_paths = {
        gpu: args.root / f"redistribution-v1/receipts/worker-gpu{gpu}.json" for gpu in (2, 3, 5, 6)
    }
    manifest = json.loads(public_manifest_path.read_text())
    handoff = json.loads(original_handoff_path.read_text())
    addendum = json.loads(addendum_path.read_text())
    receipts = {gpu: json.loads(path.read_text()) for gpu, path in receipt_paths.items()}
    finalizers = {gpu: json.loads(path.read_text()) for gpu, path in finalize_paths.items()}

    original_pending = {
        episode for values in handoff["assignment"].values() for episode in values
    }
    assigned_rows = [
        (row["episode_id"], gpu, row["object_id"], row["status"])
        for gpu, receipt in receipts.items() for row in receipt["objects"]
    ]
    assigned_ids = [row[0] for row in assigned_rows]
    public_ids = {
        row["episode_id"] for row in manifest["profiles"] if not row["existing_node71_profile"]
    }
    canonical_old = [
        json.loads((args.root / f"scratch-25k-v1/receipts/worker-gpu{gpu}.json").read_text())
        for gpu in (2, 3)
    ]
    completed_ids = {
        row["episode_id"] for receipt in canonical_old for row in receipt["objects"]
        if row["status"] == "complete"
    }
    handed_off_ids = {
        row["episode_id"] for receipt in canonical_old for row in receipt["objects"]
        if row["status"] == "handed_off"
    }
    claim_paths = sorted((args.root / "redistribution-v1/claims").glob("*.json"))
    claims = [json.loads(path.read_text()) for path in claim_paths]
    claim_ids = [claim["episode_id"] for claim in claims]
    expected_assignment = {episode: int(gpu) for episode, gpu in addendum["assignment"].items()}

    checks = {
        "original_handoff_pass": handoff["status"] == "PASS",
        "finalizers_pass": all(value["status"] == "PASS" for value in finalizers.values()),
        "eight_delta_rows": len(assigned_rows) == 8,
        "delta_rows_unique": len(assigned_ids) == len(set(assigned_ids)) == 8,
        "delta_union_exact_original_pending": set(assigned_ids) == original_pending,
        "addendum_assignment_exact": all(expected_assignment[eid] == gpu for eid, gpu, _, _ in assigned_rows),
        "receipts_fresh_25k": all(
            receipt["initialization"] == "fresh random; no checkpoint/load/resume"
            and receipt["max_num_iterations"] == 25000
            and receipt["score_files_read"] == []
            and receipt["target_files_read"] == []
            and receipt["sealed_files_read"] == []
            for receipt in receipts.values()
        ),
        "valid_queue_statuses": all(status in {"pending", "running", "complete"} for _, _, _, status in assigned_rows),
        "claims_unique_subset": len(claim_ids) == len(set(claim_ids)) and set(claim_ids).issubset(set(assigned_ids)),
        "claims_match_assigned_gpu": all(expected_assignment[claim["episode_id"]] == claim["gpu"] for claim in claims),
        "canonical_old_has_16_complete": len(completed_ids) == 16,
        "canonical_old_handed_off_exactly_8": handed_off_ids == original_pending,
        "complete_plus_delta_cover_public24": completed_ids | set(assigned_ids) == public_ids,
        "no_complete_delta_overlap": not (completed_ids & set(assigned_ids)),
    }
    checks["all_checks_pass"] = all(checks.values())
    input_paths = {
        "public_manifest": public_manifest_path,
        "original_handoff": original_handoff_path,
        "gpu6_addendum": addendum_path,
        **{f"finalize_gpu{gpu}": path for gpu, path in finalize_paths.items()},
        **{f"delta_receipt_gpu{gpu}": path for gpu, path in receipt_paths.items()},
        **{f"claim_{path.stem}": path for path in claim_paths},
    }
    output = {
        "schema": "splart-node7.2-redistributed-queue-audit/v1",
        "captured_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "PASS" if checks["all_checks_pass"] else "FAIL",
        "checks": checks,
        "assignment": {str(gpu): [row[0] for row in assigned_rows if row[1] == gpu] for gpu in (2, 3, 5, 6)},
        "queue_status": {row[0]: row[3] for row in assigned_rows},
        "receipt_hash_ledger": {name: sha256(path) for name, path in input_paths.items()},
        "score_files_read": [],
        "target_files_read": [],
        "sealed_files_read": [],
        "protected_files_read": [],
    }
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": output["status"], "receipt_sha256": sha256(args.output)}, sort_keys=True))
    if output["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
