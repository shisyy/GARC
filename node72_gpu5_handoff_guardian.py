#!/usr/bin/env python3
"""Finalize stopped GPU5 parent and verify existing-claim skip semantics."""

import datetime as dt
import hashlib
import json
import subprocess
import time
from pathlib import Path


ROOT = Path("/data1/public/yptang/splart-endpoint-node72-data")
CHILD_PID = 3631201
CURRENT = "ep72-5861eb1e5d15e433"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def state(pid):
    path = Path(f"/proc/{pid}/stat")
    return path.read_text().split()[2] if path.exists() else None


def main():
    guardian_receipt = ROOT / "redistribution-v1/gpu5_handoff_guardian_v3_receipt.json"
    if guardian_receipt.exists():
        raise FileExistsError(guardian_receipt)
    checkpoint = ROOT / f"scratch-25k-v1/model_ckpts/{CURRENT}/splart/node72-scratch25k-v1/nerfstudio_models/step-000024999.ckpt"
    deadline = time.monotonic() + 3600
    while state(CHILD_PID) not in {None, "Z"} or not checkpoint.is_file():
        if time.monotonic() > deadline:
            raise TimeoutError("GPU5 in-flight child did not complete")
        time.sleep(15)
    finalize = ROOT / "redistribution-v1/finalize-gpu5.json"
    finalize_result = subprocess.run([
        "python3", str(ROOT / "scripts/finalize_node72_stopped_worker.py"),
        "--root", str(ROOT), "--gpu", "5", "--parent-pid", "3631198",
        "--child-pid", str(CHILD_PID), "--session", "node72-scratch25k-v1-gpu5-delta",
        "--current", CURRENT, "--assignment", str(ROOT / "scripts/node72_gpu47_handoff_addendum.json"),
        "--log-path", str(ROOT / f"redistribution-v1/logs/{CURRENT}.log"),
        "--receipt-path", str(ROOT / "redistribution-v1/receipts/worker-gpu5.json"),
        "--output", str(finalize),
    ], check=True, text=True, capture_output=True)

    skip_receipt = ROOT / "redistribution-v1/receipts/worker-gpu5-skip-verification.json"
    skip_log = ROOT / "redistribution-v1/logs/worker-gpu5-skip-verification.log"
    command = [
        "/data/yptang/workspace/20260720_SplArt/.conda/splart/bin/python",
        str(ROOT / "scripts/run_order_scratch_baselines.py"),
        "--public-manifest", str(ROOT / "public/profile_manifest.json"),
        "--episode-list", str(ROOT / "scripts/node72_delta_gpu5_remaining.json"),
        "--claim-root", str(ROOT / "redistribution-v1/claims"),
        "--dataset-root", str(ROOT / "public/materialized-new24-v1"),
        "--output-root", str(ROOT / "scratch-25k-v1/model_ckpts"),
        "--log-root", str(ROOT / "redistribution-v1/logs"),
        "--receipt", str(skip_receipt), "--gpu", "5", "--shard-index", "5",
        "--num-shards", "6", "--minimum-free-mib", "22000",
        "--minimum-output-free-mib", "131072", "--max-iterations", "25000",
        "--timestamp-tag", "node72-scratch25k-v1",
    ]
    with skip_log.open("w") as handle:
        subprocess.run(command, check=True, stdout=handle, stderr=subprocess.STDOUT)
    if state(3631198) is not None:
        raise RuntimeError("stopped GPU5 parent still exists after finalization")
    skip = json.loads(skip_receipt.read_text())
    expected = {
        "ep72-cc94a9639dbc9e14": 4,
        "ep72-9033adcfed953617": 7,
    }
    if not all(
        row["status"] == "skipped_existing_claim"
        and row["existing_claim_gpu"] == expected[row["episode_id"]]
        for row in skip["objects"]
    ):
        raise RuntimeError("GPU5 did not verify both external claims")
    payload = {
        "schema": "splart-node7.2-gpu5-handoff-guardian/v1",
        "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "PASS",
        "finalize_receipt_sha256": sha256(finalize),
        "skip_verification_receipt_sha256": sha256(skip_receipt),
        "skip_verification_log_sha256": sha256(skip_log),
        "old_stopped_parent_exists": state(3631198) is not None,
        "verified_external_claims": expected,
        "score_files_read": [],
        "target_files_read": [],
        "sealed_files_read": [],
        "protected_files_read": [],
    }
    guardian_receipt.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "PASS", "receipt_sha256": sha256(guardian_receipt)}, sort_keys=True))


if __name__ == "__main__":
    main()
