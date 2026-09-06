#!/usr/bin/env python3
"""Run independent from-scratch SplArt baselines for one public object shard."""

import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
from pathlib import Path


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def write_json_atomic(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def free_memory_mib(gpu):
    output = subprocess.check_output([
        "nvidia-smi", "-i", str(gpu), "--query-gpu=memory.free",
        "--format=csv,noheader,nounits",
    ], text=True)
    return int(output.strip())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--public-manifest", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--log-root", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--gpu", required=True, type=int)
    parser.add_argument("--shard-index", required=True, type=int)
    parser.add_argument("--num-shards", default=2, type=int)
    parser.add_argument("--minimum-free-mib", default=22000, type=int)
    parser.add_argument("--minimum-output-free-mib", default=8192, type=int)
    parser.add_argument("--max-iterations", default=25000, type=int)
    parser.add_argument("--timestamp-tag", default="node71-scratch-v2")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--episode-list", type=Path)
    parser.add_argument("--claim-root", type=Path)
    args = parser.parse_args()

    manifest = json.loads(args.public_manifest.read_text())
    if manifest.get("score_files_read") != []:
        raise ValueError("public manifest must attest score_files_read=[]")
    if manifest.get("target_files_read", []) != [] or manifest.get("sealed_files_read", []) != []:
        raise ValueError("public manifest must attest target_files_read=[] and sealed_files_read=[]")
    roster = manifest.get("profiles", manifest.get("episodes", []))
    roster = [row for row in roster if not row.get("existing_node71_profile", False)]
    episodes = [row for index, row in enumerate(roster) if index % args.num_shards == args.shard_index]
    if args.episode_list is not None:
        explicit = json.loads(args.episode_list.read_text())
        if any(explicit.get(key, []) != [] for key in (
            "score_files_read", "target_files_read", "sealed_files_read", "protected_files_read"
        )):
            raise ValueError("explicit list is not public-only")
        roster_by_id = {row["episode_id"]: row for row in roster}
        requested = explicit["episodes"]
        if len(requested) != len({row["episode_id"] for row in requested}):
            raise ValueError("duplicate episode in explicit list")
        episodes = []
        for requested_row in requested:
            row = roster_by_id.get(requested_row["episode_id"])
            if row is None or row["object_id"] != requested_row["object_id"]:
                raise ValueError("explicit list row does not match frozen public manifest")
            episodes.append(row)
    if args.limit is not None:
        episodes = episodes[:args.limit]
    receipt = {
        "schema": "splart-order-scratch-worker/v1",
        "score_files_read": [],
        "target_files_read": [],
        "sealed_files_read": [],
        "protected_files_read": [],
        "sealed_mapping_read": False,
        "gpu": args.gpu,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "initialization": "fresh random; no checkpoint/load/resume",
        "max_num_iterations": args.max_iterations,
        "objects": [{"episode_id": row["episode_id"], "object_id": row["object_id"], "status": "pending"} for row in episodes],
        "started_at": now(),
    }
    write_json_atomic(args.receipt, receipt)
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    implementation_root = Path("/data/yptang/workspace/20260720_SplArt/src/splart_official")
    ns_train = "/data/yptang/workspace/20260720_SplArt/.conda/splart/bin/ns-train"
    for row, status in zip(episodes, receipt["objects"]):
        free_mib = free_memory_mib(args.gpu)
        status["launch_free_memory_mib"] = free_mib
        if free_mib < args.minimum_free_mib:
            status.update({"status": "blocked_memory_gate", "required_free_memory_mib": args.minimum_free_mib, "finished_at": now()})
            write_json_atomic(args.receipt, receipt)
            raise SystemExit(f"GPU{args.gpu} has only {free_mib} MiB free")
        output_free_mib = shutil.disk_usage(args.output_root.parent).free // (1024 * 1024)
        status["launch_output_free_mib"] = output_free_mib
        if output_free_mib < args.minimum_output_free_mib:
            status.update({"status": "blocked_disk_gate", "required_output_free_mib": args.minimum_output_free_mib, "finished_at": now()})
            write_json_atomic(args.receipt, receipt)
            raise SystemExit(f"output filesystem has only {output_free_mib} MiB free")
        episode_id = row["episode_id"]
        claim_path = None
        if args.claim_root is not None:
            args.claim_root.mkdir(parents=True, exist_ok=True)
            claim_path = args.claim_root / f"{episode_id}.json"
            claim_payload = json.dumps({
                "schema": "splart-node7.2-atomic-claim/v1",
                "episode_id": episode_id,
                "object_id": row["object_id"],
                "gpu": args.gpu,
                "runner_pid": os.getpid(),
                "claimed_at": now(),
                "status": "claimed",
                "score_files_read": [],
                "target_files_read": [],
                "sealed_files_read": [],
                "protected_files_read": [],
            }, indent=2, sort_keys=True) + "\n"
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            try:
                descriptor = os.open(claim_path, flags, 0o600)
            except FileExistsError:
                existing_claim = json.loads(claim_path.read_text())
                if (
                    existing_claim.get("episode_id") != episode_id
                    or existing_claim.get("object_id") != row["object_id"]
                    or existing_claim.get("status") not in {"claimed", "running", "complete"}
                ):
                    status.update({"status": "blocked_invalid_existing_claim", "finished_at": now()})
                    write_json_atomic(args.receipt, receipt)
                    raise SystemExit(f"invalid existing claim: {claim_path}")
                status.update({
                    "status": "skipped_existing_claim",
                    "existing_claim_gpu": existing_claim.get("gpu"),
                    "existing_claim_path": str(claim_path),
                    "finished_at": now(),
                })
                write_json_atomic(args.receipt, receipt)
                continue
            with os.fdopen(descriptor, "w") as handle:
                handle.write(claim_payload)
                handle.flush()
                os.fsync(handle.fileno())
        dataset = args.dataset_root / "episodes" / episode_id
        if not (dataset / "transforms.json").is_file():
            status.update({"status": "blocked_missing_dataset", "finished_at": now()})
            write_json_atomic(args.receipt, receipt)
            raise SystemExit(f"missing dataset for {episode_id}")
        experiment_root = args.output_root / episode_id
        if experiment_root.exists():
            status.update({"status": "blocked_nonfresh_output", "finished_at": now()})
            write_json_atomic(args.receipt, receipt)
            raise SystemExit(f"refusing existing output: {experiment_root}")
        args.log_root.mkdir(parents=True, exist_ok=True)
        log_path = args.log_root / f"{episode_id}.log"
        command = [
            ns_train, "splart",
            "--output-dir", str(args.output_root),
            "--experiment-name", episode_id,
            "--timestamp", args.timestamp_tag,
            "--vis", "tensorboard",
            "--max-num-iterations", str(args.max_iterations),
            "--pipeline.model.num-random", "999999",
            "--pipeline.model.random-scale", "1.3",
            "--data", str(dataset),
        ]
        status.update({"status": "running", "started_at": now(), "log_path": str(log_path), "command_has_checkpoint_argument": False})
        write_json_atomic(args.receipt, receipt)
        with log_path.open("w") as log:
            process = subprocess.Popen(command, cwd=implementation_root, env=environment, stdout=log, stderr=subprocess.STDOUT)
            status["pid"] = process.pid
            write_json_atomic(args.receipt, receipt)
            returncode = process.wait()
        status.update({"status": "complete" if returncode == 0 else "failed", "returncode": returncode, "finished_at": now()})
        if claim_path is not None:
            claim = json.loads(claim_path.read_text())
            claim.update({"status": status["status"], "returncode": returncode, "finished_at": status["finished_at"]})
            write_json_atomic(claim_path, claim)
        write_json_atomic(args.receipt, receipt)
        if returncode != 0:
            raise SystemExit(f"training failed for {episode_id}: {returncode}")
    receipt["finished_at"] = now()
    write_json_atomic(args.receipt, receipt)


if __name__ == "__main__":
    main()
