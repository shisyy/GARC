#!/usr/bin/env python3
"""Create an immutable reference-only index after rehashing profile batches."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

EXPECTED_D2 = "dd78dcc5355fd0f41fb00541d6a18dc36d87ef91"
EXPECTED_SHAPE = [2, 3, 257, 9]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-index", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-objects", type=int, default=12)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    seen, rows, batches = set(), [], []
    for index_path in args.batch_index:
        index_path = index_path.resolve()
        index = json.loads(index_path.read_text())
        if index.get("schema") != "splart-gauge-energy-profiles/v1" or index.get("d2_source", {}).get("commit") != EXPECTED_D2:
            raise ValueError("batch schema/frozen D2 provenance mismatch")
        d2_path = Path(index["d2_source"]["path"])
        if subprocess.check_output(["git", "-C", str(d2_path), "status", "--porcelain"], text=True).strip():
            raise ValueError("frozen D2 source is dirty")
        head = subprocess.check_output(["git", "-C", str(d2_path), "rev-parse", "HEAD"], text=True).strip()
        tree = subprocess.check_output(["git", "-C", str(d2_path), "rev-parse", "HEAD^{tree}"], text=True).strip()
        if head != EXPECTED_D2 or tree != index["d2_source"]["tree"]:
            raise ValueError("frozen D2 commit/tree changed")
        for row in index["rows"]:
            object_id = row["object_id"]
            if object_id in seen:
                raise ValueError(f"duplicate object: {object_id}")
            seen.add(object_id)
            artifact = index_path.parent / row["artifact"]
            if row["shape"] != EXPECTED_SHAPE or row["base_state_sha256_before"] != row["base_state_sha256_after"]:
                raise ValueError(f"shape/frozen-state audit failed: {object_id}")
            if Path(row["checkpoint"]["path"]).name != "step-000024999.ckpt":
                raise ValueError(f"checkpoint is not fixed step24999: {object_id}")
            if sha256_file(artifact) != row["artifact_sha256"]:
                raise ValueError(f"artifact rehash failed: {object_id}")
            rows.append({**row, "artifact": str(artifact), "source_index_sha256": sha256_file(index_path)})
        batches.append({"path": str(index_path), "sha256": sha256_file(index_path), "objects": len(index["rows"])})
    if len(rows) != args.expected_objects or len(seen) != args.expected_objects:
        raise ValueError(f"expected {args.expected_objects} unique objects, got {len(seen)}")
    merged = {"schema": "splart-gauge-energy-profiles-merged/v1", "d2_commit": EXPECTED_D2,
              "artifact_policy": "reference-only; source artifacts unchanged", "batches": batches,
              "objects": sorted(rows, key=lambda row: row["object_id"])}
    args.output_dir.mkdir(mode=0o700, parents=True)
    output = args.output_dir / "index.json"
    output.write_text(json.dumps(merged, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"objects": len(rows), "output": str(output), "sha256": sha256_file(output)}))


if __name__ == "__main__":
    main()
