#!/usr/bin/env python3
"""Build a target-free manifest for completed scratch-25k checkpoints."""

import argparse
import json
import subprocess
from pathlib import Path

from export_gauge_profiles import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public-manifest", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, action="append", required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--run-name", default="node71-scratch-v4")
    parser.add_argument("--object-id", action="append", default=[], help="Exact public object ID to include; repeatable")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--d2-source", type=Path)
    parser.add_argument("--d2-commit")
    parser.add_argument("--dry-run-receipt", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    public = json.loads(args.public_manifest.read_text())
    public_rows = public.get("episodes", public.get("profiles"))
    if not isinstance(public_rows, list) or not public_rows:
        raise ValueError("public manifest has neither episodes nor profiles")
    for key in ("protected_files_read", "score_files_read", "sealed_files_read", "target_files_read"):
        if key in public and public[key] != []:
            raise ValueError(f"public manifest is not target-free: {key}")
    public_ids = {row["object_id"]: row for row in public_rows}
    complete = {}
    for receipt_path in args.receipt:
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("sealed_mapping_read") is not False or any(receipt.get(key, []) != [] for key in
                ("protected_files_read", "score_files_read", "sealed_files_read", "target_files_read")):
            raise ValueError("worker receipt is not target-free")
        for row in receipt.get("objects", []):
            if row.get("status") == "complete" and row.get("returncode") == 0:
                complete[row["object_id"]] = row
    objects = []
    for object_id in sorted(complete):
        if object_id not in public_ids:
            raise ValueError("completed object is absent from public manifest")
        episode_id = complete[object_id]["episode_id"]
        if not (episode_id.startswith("ep-") or episode_id.startswith("ep72-")):
            raise ValueError(f"unsupported episode ID: {episode_id}")
        run = args.checkpoint_root / episode_id / "splart" / args.run_name
        checkpoint = run / "nerfstudio_models" / "step-000024999.ckpt"
        config = run / "config.yml"
        transforms = run / "dataparser_transforms.json"
        if not all(path.is_file() and not path.is_symlink() for path in (checkpoint, config, transforms)):
            raise ValueError(f"complete receipt lacks immutable 25k outputs: {object_id}")
        objects.append({
            "object_id": object_id, "split": "unassigned",
            "checkpoint": {"path": str(checkpoint), "sha256": sha256_file(checkpoint), "step": 24999},
            "config": {"path": str(config), "sha256": sha256_file(config)},
            "dataparser_transforms": {"path": str(transforms), "sha256": sha256_file(transforms)},
        })
    if not objects:
        raise ValueError("no completed scratch-25k checkpoints")
    result = {
        "schema": "splart-frozen-d2-training-outputs/v1", "status": "COMPLETE",
        "selection": "all receipt-complete scratch-25k checkpoints; no checkpoint selection",
        "split_status": "unassigned until target-free public split handoff", "objects": objects,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if args.dry_run_receipt:
        if not args.d2_source or not args.d2_commit or args.dry_run_receipt.exists():
            raise ValueError("dry-run receipt requires a fresh path and frozen D2 source/commit")
        head = subprocess.check_output(["git", "-C", str(args.d2_source), "rev-parse", "HEAD"], text=True).strip()
        tree = subprocess.check_output(["git", "-C", str(args.d2_source), "rev-parse", "HEAD^{tree}"], text=True).strip()
        dirty = subprocess.check_output(["git", "-C", str(args.d2_source), "status", "--porcelain"], text=True).strip()
        if head != args.d2_commit or dirty:
            raise ValueError("D2 source is not the exact clean frozen commit")
        receipt = {
            "schema": "splart-gauge-energy-export-dry-run/v1", "status": "READY_FOR_IDLE_GPU",
            "gpu_started": False, "head_training_started": False,
            "target_files_read": [], "sealed_files_read": [], "score_files_read": [],
            "input_manifest": {"path": str(args.output), "sha256": sha256_file(args.output)},
            "public_manifest_sha256": sha256_file(args.public_manifest),
            "d2_source": {"path": str(args.d2_source), "commit": head, "tree": tree, "clean": True},
            "base_state_guard": "required sha256_before == sha256_after; fail closed during export",
            "objects": [{"object_id": row["object_id"], "checkpoint_sha256": row["checkpoint"]["sha256"],
                         "config_sha256": row["config"]["sha256"],
                         "dataparser_transforms_sha256": row["dataparser_transforms"]["sha256"],
                         "step": row["checkpoint"]["step"]} for row in objects],
        }
        args.dry_run_receipt.parent.mkdir(parents=True, exist_ok=True)
        args.dry_run_receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"objects": len(objects), "output": str(args.output), "sha256": sha256_file(args.output)}))


if __name__ == "__main__":
    main()
