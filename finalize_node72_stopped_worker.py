#!/usr/bin/env python3
"""Finalize one stopped node7.2 runner after its in-flight child completes."""

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
from pathlib import Path


FRESH = "No Nerfstudio checkpoint to load, so training from scratch."
ERRORS = re.compile(r"\b(traceback|error|exception|failed|nan|inf)\b", re.I)


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def state(pid):
    path = Path(f"/proc/{pid}/stat")
    return path.read_text().split()[2] if path.exists() else None


def atomic_json(path, payload):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--gpu", required=True, type=int)
    parser.add_argument("--parent-pid", required=True, type=int)
    parser.add_argument("--child-pid", required=True, type=int)
    parser.add_argument("--session", required=True)
    parser.add_argument("--current", required=True)
    parser.add_argument("--assignment", required=True, type=Path)
    parser.add_argument("--log-path", type=Path)
    parser.add_argument("--receipt-path", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if state(args.child_pid) not in {None, "Z"}:
        raise RuntimeError("in-flight child has not completed")
    if state(args.parent_pid) != "T":
        raise RuntimeError("old runner parent is not stopped")
    checkpoint = args.root / "scratch-25k-v1/model_ckpts" / args.current / "splart/node72-scratch25k-v1/nerfstudio_models/step-000024999.ckpt"
    log = args.log_path or args.root / "scratch-25k-v1/logs" / f"{args.current}.log"
    log_text = log.read_text(errors="replace")
    if not checkpoint.is_file() or FRESH not in log_text or ERRORS.search(log_text):
        raise RuntimeError("completed child evidence failed")
    receipt_path = args.receipt_path or args.root / f"scratch-25k-v1/receipts/worker-gpu{args.gpu}.json"
    evidence = args.root / "redistribution-v1/evidence/pre-handoff-receipts"
    evidence.mkdir(parents=True, exist_ok=True)
    preserved = evidence / receipt_path.name
    if preserved.exists():
        raise FileExistsError(preserved)
    shutil.copy2(receipt_path, preserved)
    assignment = json.loads(args.assignment.read_text())["assignment"]
    receipt = json.loads(receipt_path.read_text())
    handed_off = {}
    for row in receipt["objects"]:
        if row["episode_id"] == args.current:
            row.update({"status": "complete", "returncode": 0, "finished_at": now(), "completion_verified_by_handoff": True})
        elif row["status"] == "pending":
            destination = assignment[row["episode_id"]]
            row.update({"status": "handed_off", "handed_off_to_gpu": destination, "handed_off_at": now()})
            handed_off[row["episode_id"]] = destination
    os.kill(args.parent_pid, signal.SIGKILL)
    subprocess.run(["tmux", "kill-session", "-t", args.session], check=False)
    receipt["queue_handoff"] = {
        "schema": "splart-node7.2-stopped-parent-supersession/v1",
        "preserved_original_sha256": sha256(preserved),
        "handoff_evidence": str(args.output),
    }
    receipt["finished_at"] = now()
    atomic_json(receipt_path, receipt)
    output = {
        "schema": "splart-node7.2-stopped-worker-finalization/v1",
        "status": "PASS",
        "gpu": args.gpu,
        "completed_episode_id": args.current,
        "checkpoint_sha256": sha256(checkpoint),
        "log_sha256": sha256(log),
        "preserved_receipt_sha256": sha256(preserved),
        "updated_receipt_sha256": sha256(receipt_path),
        "old_parent_terminated": state(args.parent_pid) is None,
        "handed_off": handed_off,
        "score_files_read": [],
        "target_files_read": [],
        "sealed_files_read": [],
        "protected_files_read": [],
    }
    atomic_json(args.output, output)
    print(json.dumps({"status": "PASS", "receipt_sha256": sha256(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
