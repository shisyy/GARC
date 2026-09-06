#!/usr/bin/env python3
"""Audit the final eight-object node7.2 queue after GPU4/7 diversion."""

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    scripts = args.root / "scripts"
    handoff_path = args.root / "redistribution-v1/queue_handoff_receipt.json"
    gpu6_path = scripts / "node72_gpu6_handoff_addendum.json"
    gpu47_path = scripts / "node72_gpu47_handoff_addendum.json"
    guardian_path = args.root / "redistribution-v1/gpu5_handoff_guardian_v3_receipt.json"
    paths = {
        "handoff": handoff_path,
        "gpu6_addendum": gpu6_path,
        "gpu47_addendum": gpu47_path,
        "guardian": guardian_path,
    }
    handoff = load(handoff_path)
    gpu6 = load(gpu6_path)
    gpu47 = load(gpu47_path)
    guardian = load(guardian_path)

    original_pending = {
        episode for values in handoff["assignment"].values() for episode in values
    }
    expected = {episode: int(gpu) for episode, gpu in gpu6["assignment"].items()}
    expected.update({episode: int(gpu) for episode, gpu in gpu47["assignment"].items()})
    final_lists = {}
    for gpu in (2, 3, 4, 6, 7):
        path = scripts / f"node72_delta_gpu{gpu}.json"
        paths[f"delta_gpu{gpu}"] = path
        final_lists[gpu] = [row["episode_id"] for row in load(path)["episodes"]]
    gpu5_source_path = scripts / "node72_delta_gpu5.json"
    paths["delta_gpu5_source"] = gpu5_source_path
    gpu5_source = [row["episode_id"] for row in load(gpu5_source_path)["episodes"]]
    final_lists[5] = [episode for episode in gpu5_source if expected[episode] == 5]
    actual = {
        episode: gpu for gpu, episodes in final_lists.items() for episode in episodes
    }
    flattened = [episode for episodes in final_lists.values() for episode in episodes]

    claim_paths = sorted((args.root / "redistribution-v1/claims").glob("*.json"))
    claims = [load(path) for path in claim_paths]
    for path in claim_paths:
        paths[f"claim_{path.stem}"] = path
    checks = {
        "original_handoff_pass": handoff["status"] == "PASS",
        "guardian_pass": guardian["status"] == "PASS",
        "eight_objects": len(flattened) == 8,
        "eight_unique_objects": len(flattened) == len(set(flattened)) == 8,
        "exact_original_pending_coverage": set(flattened) == original_pending,
        "final_assignment_exact": actual == expected,
        "gpu4_one_object": len(final_lists[4]) == 1,
        "gpu7_one_object": len(final_lists[7]) == 1,
        "gpu5_one_training_object": len(final_lists[5]) == 1,
        "claims_unique": len(claims) == len({row["episode_id"] for row in claims}),
        "claims_subset_of_queue": {row["episode_id"] for row in claims}.issubset(original_pending),
        "claims_match_final_assignment": all(expected[row["episode_id"]] == row["gpu"] for row in claims),
        "gpu5_verified_external_claim_skip": guardian["verified_external_claims"] == gpu47["assignment"],
        "old_gpu5_parent_removed": guardian["old_stopped_parent_exists"] is False,
        "no_protected_reads": all(
            value.get(field) == []
            for value in (handoff, gpu6, gpu47, guardian)
            for field in ("score_files_read", "target_files_read", "sealed_files_read", "protected_files_read")
        ),
    }
    checks["all_checks_pass"] = all(checks.values())
    output = {
        "schema": "splart-node7.2-gpu4-gpu7-queue-audit/v1",
        "captured_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "PASS" if checks["all_checks_pass"] else "FAIL",
        "checks": checks,
        "final_assignment": {str(gpu): final_lists[gpu] for gpu in sorted(final_lists)},
        "receipt_hash_ledger": {name: sha256(path) for name, path in paths.items()},
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
