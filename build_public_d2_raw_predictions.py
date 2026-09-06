#!/usr/bin/env python3
"""Build exact36 target-free raw endpoint-distance predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any

SCHEMA = "splart-public-raw-distance-predictions/v2"
METHODS = ("scratch", "symmetric_linear", "full_d2", "single_radius", "no_contact",
           "no_penetration", "no_terminal_support")
D2_MODE = {"full_d2": "full", "single_radius": "single-radius", "no_contact": "no-contact",
           "no_penetration": "no-penetration", "no_terminal_support": "no-terminal-support"}
FORBIDDEN_KEYS = {"target", "targets", "split", "membership", "ground_truth", "score", "scores", "aggregate"}
OPAQUE_ID = re.compile(r"^(?:ep|ep72)-[A-Za-z0-9]+$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def reject_private(value: Any, path: str = "root") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).lower() in FORBIDDEN_KEYS:
                raise ValueError(f"forbidden key at {path}.{key}")
            reject_private(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            reject_private(child, f"{path}[{index}]")
    elif isinstance(value, str) and any(marker in value.lower() for marker in ("sealed", "b_test", "full22")):
        raise ValueError(f"protected path at {path}")


def scalar_to_distance(lower: float, upper: float) -> list[float]:
    distance = [-float(lower), float(upper) - 1.0]
    if any(not math.isfinite(value) or value < 0.0 for value in distance):
        raise ValueError("D2 endpoint scalars produced invalid outward distance")
    return distance


def build_method(method: str, index: dict[str, Any], index_path: Path) -> dict[str, Any]:
    if method not in METHODS:
        raise ValueError("unknown method")
    rows = index.get("rows")
    if index.get("objects") != 36 or not isinstance(rows, list) or len(rows) != 36:
        raise ValueError("source index must contain exact36 objects")
    output = []
    for source_row in rows:
        object_id = source_row.get("object_id")
        if not isinstance(object_id, str) or not OPAQUE_ID.fullmatch(object_id):
            raise ValueError("non-opaque or unsafe object ID")
        artifact = Path(source_row["artifact"])
        if artifact.is_symlink() or not artifact.is_file() or sha256_file(artifact) != source_row["artifact_sha256"]:
            raise ValueError(f"artifact provenance mismatch: {object_id}")
        evidence = json.loads(artifact.read_text())
        reject_private(evidence)
        if method == "scratch":
            distance, config = [0.0, 0.0], {"constant_observation_distance": [0.0, 0.0]}
        elif method == "symmetric_linear":
            distance, config = [0.5, 0.5], {"constant_observation_distance": [0.5, 0.5]}
        else:
            mode = D2_MODE[method]
            prediction = evidence["modes"][mode]
            distance = scalar_to_distance(prediction["lower_scalar"], prediction["upper_scalar"])
            config = {"mode": mode, "field_configs": prediction["field_configs"]}
        provenance = evidence["provenance"]
        if (provenance["d2_commit"] != index["d2_source"]["commit"] or
                provenance["d2_tree"] != index["d2_source"]["tree"]):
            raise ValueError("D2 provenance mismatch")
        output.append({"object_id": object_id, "observation_distance": distance,
                       "source_artifact_sha256": source_row["artifact_sha256"],
                       "d2_commit": provenance["d2_commit"], "d2_tree": provenance["d2_tree"],
                       "mode_config_sha256": canonical_sha(config)})
    if len({row["object_id"] for row in output}) != 36:
        raise ValueError("methods require 36 unique opaque IDs")
    payload = {"schema": SCHEMA, "method": method, "objects": 36,
               "source_index_sha256": sha256_file(index_path), "rows": sorted(output, key=lambda row: row["object_id"])}
    reject_private(payload)
    return payload


def write_exclusive(path: Path, value: dict[str, Any], mode: int = 0o400) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-index", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    launch = f"{args.evidence_index}\n{args.output_root}".lower()
    if any(marker in launch for marker in ("sealed", "b_test", "full22")):
        raise ValueError("protected launch path")
    index = json.loads(args.evidence_index.read_text())
    reject_private(index)
    args.output_root.mkdir(mode=0o700, parents=True, exist_ok=False)
    files = []
    for method in METHODS:
        path = args.output_root / f"{method}.json"
        write_exclusive(path, build_method(method, index, args.evidence_index))
        files.append({"method": method, "file": path.name, "sha256": sha256_file(path)})
    receipt = {"schema": "splart-public-raw-distance-receipt/v2", "methods": len(files),
               "objects_per_method": 36, "files": files}
    write_exclusive(args.output_root / "receipt.json", receipt)
    os.chmod(args.output_root, 0o700)
    print(json.dumps({"root": str(args.output_root), "receipt_sha256": sha256_file(args.output_root / "receipt.json")}))


if __name__ == "__main__":
    main()
