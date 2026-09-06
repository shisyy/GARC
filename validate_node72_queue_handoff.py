#!/usr/bin/env python3
"""Freeze and validate the public-only node7.2 GPU5 queue handoff."""

import argparse
import hashlib
import json
import os
from pathlib import Path


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--receipt-gpu2", required=True, type=Path)
    parser.add_argument("--receipt-gpu3", required=True, type=Path)
    parser.add_argument("--delta-gpu2", required=True, type=Path)
    parser.add_argument("--delta-gpu3", required=True, type=Path)
    parser.add_argument("--delta-gpu5", required=True, type=Path)
    parser.add_argument("--runner", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest = json.loads(args.manifest.read_text())
    public_ids = {
        row["episode_id"] for row in manifest["profiles"] if not row["existing_node71_profile"]
    }
    receipts = [json.loads(args.receipt_gpu2.read_text()), json.loads(args.receipt_gpu3.read_text())]
    status_by_id = {row["episode_id"]: row["status"] for receipt in receipts for row in receipt["objects"]}
    completed = {key for key, status in status_by_id.items() if status == "complete"}
    running = {key for key, status in status_by_id.items() if status == "running"}
    pending = {key for key, status in status_by_id.items() if status == "pending"}
    deltas = {}
    for gpu in (2, 3, 5):
        payload = json.loads(getattr(args, f"delta_gpu{gpu}").read_text())
        assert all(payload.get(key) == [] for key in (
            "score_files_read", "target_files_read", "sealed_files_read", "protected_files_read"
        ))
        deltas[gpu] = [row["episode_id"] for row in payload["episodes"]]
    flat = [episode_id for gpu in (2, 3, 5) for episode_id in deltas[gpu]]
    checks = {
        "public_roster_24": len(public_ids) == 24,
        "completed_14": len(completed) == 14,
        "running_2": len(running) == 2,
        "pending_8": len(pending) == 8,
        "delta_union_exact_pending": set(flat) == pending,
        "delta_no_duplicates": len(flat) == len(set(flat)) == 8,
        "no_completed_or_running_in_delta": not (set(flat) & (completed | running)),
        "roster_partition_complete": completed | running | pending == public_ids,
        "runner_parents_stopped": all(
            Path(f"/proc/{pid}/stat").read_text().split()[2] == "T" for pid in (3478290, 3478300)
        ),
    }
    # Exercise O_EXCL on the actual destination filesystem without claiming an episode.
    probe = args.output.parent / ".atomic-claim-probe"
    descriptor = os.open(probe, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(descriptor)
    try:
        try:
            second = os.open(probe, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            os.close(second)
            checks["o_excl_atomic_claim"] = False
        except FileExistsError:
            checks["o_excl_atomic_claim"] = True
    finally:
        probe.unlink()
    checks["all_checks_pass"] = all(checks.values())
    output = {
        "schema": "splart-node7.2-queue-handoff/v1",
        "status": "PASS" if checks["all_checks_pass"] else "FAIL",
        "checks": checks,
        "completed_episode_count": len(completed),
        "running_episode_count": len(running),
        "pending_episode_count": len(pending),
        "assignment": {f"gpu{gpu}": deltas[gpu] for gpu in (2, 3, 5)},
        "input_hashes": {
            "manifest": sha256(args.manifest),
            "receipt_gpu2": sha256(args.receipt_gpu2),
            "receipt_gpu3": sha256(args.receipt_gpu3),
            "delta_gpu2": sha256(args.delta_gpu2),
            "delta_gpu3": sha256(args.delta_gpu3),
            "delta_gpu5": sha256(args.delta_gpu5),
            "runner": sha256(args.runner),
        },
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
