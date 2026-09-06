#!/usr/bin/env python3
"""Build a target-free manifest for completed scratch-25k checkpoints."""

import argparse
import json
from pathlib import Path

from export_gauge_profiles import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public-manifest", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, action="append", required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    public = json.loads(args.public_manifest.read_text())
    public_ids = {row["object_id"]: row for row in public["episodes"]}
    complete = {}
    for receipt_path in args.receipt:
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("sealed_mapping_read") is not False or receipt.get("score_files_read") != []:
            raise ValueError("worker receipt is not target-free")
        for row in receipt.get("objects", []):
            if row.get("status") == "complete" and row.get("returncode") == 0:
                complete[row["object_id"]] = row
    objects = []
    for object_id in sorted(complete):
        if object_id not in public_ids:
            raise ValueError("completed object is absent from public manifest")
        episode_id = complete[object_id]["episode_id"]
        run = args.checkpoint_root / episode_id / "splart" / "node71-scratch-v4"
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
    print(json.dumps({"objects": len(objects), "output": str(args.output), "sha256": sha256_file(args.output)}))


if __name__ == "__main__":
    main()
