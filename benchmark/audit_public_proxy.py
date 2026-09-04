"""Fail-closed leakage and geometry audit for public middle-state proxies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any

import imageio.v3 as iio
import numpy as np

from benchmark.endpoint_proxy import DEV3, PUBLIC_SCHEMA, QUERY_SCHEMA, canonical_sha256, load_json, sha256_file


FORBIDDEN_KEY = re.compile(
    r"physical.*fraction|fraction.*physical|true.*limit|limit.*true|full.*range|range.*full|"
    r"local.*scalar|scalar.*local|closed.*query|query.*closed|endpoint.*scalar|urdf|pivot|axis|angle|dist",
    re.IGNORECASE,
)
FORBIDDEN_NAME = re.compile(r"closed|limit|fraction|urdf", re.IGNORECASE)
FORBIDDEN_VALUE = re.compile(
    r"physical_coordinate|local_scalars|closed_query|lower_limit|upper_limit|"
    r"arbor-sealed|evaluator-assets",
    re.IGNORECASE,
)


def walk_keys(value: Any, prefix: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if FORBIDDEN_KEY.fullmatch(str(key)):
                found.append(path)
            found.extend(walk_keys(child, path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(walk_keys(child, f"{prefix}[{index}]"))
    return found


def walk_values(value: Any, prefix: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            found.extend(walk_values(child, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(walk_values(child, f"{prefix}[{index}]"))
    elif isinstance(value, str) and FORBIDDEN_VALUE.search(value):
        found.append(prefix)
    return found


def audit_scene(scene_root: Path) -> dict[str, Any]:
    meta_path = scene_root / "transforms.json"
    query_path = scene_root / "endpoint_queries.json"
    meta, queries = load_json(meta_path), load_json(query_path)
    failures: list[str] = []
    if meta.get("benchmark", {}).get("schema") != PUBLIC_SCHEMA:
        failures.append("public schema mismatch")
    if queries.get("schema") != QUERY_SCHEMA:
        failures.append("query schema mismatch")
    if set(meta.get("articulation", {})) != {"type"}:
        failures.append("public articulation must contain joint type only")
    states = [frame.get("state") for frame in meta.get("frames", [])]
    if states[:200] != [0] * 100 + [1] * 100 or any(state not in (0, 1) for state in states):
        failures.append("states are not exactly relabelled 0/1")
    if len(meta.get("train_filenames", [])) != 200 or len(meta.get("val_filenames", [])) != 20:
        failures.append("expected 200 train and 20 validation observations")
    allowed_json = {"transforms.json", "endpoint_queries.json", "PROXY_RECEIPT.json", "COMPLETE.json"}
    json_paths = {path.name for path in scene_root.glob("*.json")}
    if json_paths != allowed_json:
        failures.append(f"root JSON allowlist mismatch: {sorted(json_paths)}")
    for path in scene_root.rglob("*"):
        if path.is_symlink():
            failures.append(f"symlink forbidden: {path.relative_to(scene_root)}")
    json_payloads: list[tuple[Path, Any]] = []
    for path in sorted(scene_root.rglob("*.json")):
        try:
            json_payloads.append((path, json.loads(path.read_text(encoding="utf-8"))))
        except (OSError, json.JSONDecodeError) as error:
            failures.append(f"invalid JSON {path.relative_to(scene_root)}: {error}")
    forbidden = []
    forbidden_values = []
    for path, payload in json_payloads:
        forbidden.extend(f"{path.name}:{item}" for item in walk_keys(payload))
        forbidden_values.extend(f"{path.name}:{item}" for item in walk_values(payload))
    if forbidden:
        failures.append(f"forbidden metadata keys: {forbidden}")
    if forbidden_values:
        failures.append(f"forbidden metadata values: {forbidden_values}")
    expected_queries = [
        {"query_id": "outside_state_0", "local_direction": -1},
        {"query_id": "outside_state_1", "local_direction": 1},
    ]
    if queries.get("queries") != expected_queries:
        failures.append("query payload contains more than direction identifiers")
    for path in scene_root.rglob("*"):
        if path.is_file() and FORBIDDEN_NAME.search(path.name):
            failures.append(f"forbidden public filename: {path.relative_to(scene_root)}")
    expected_assets = {
        f"{modality}/{split}/{index:04d}.png"
        for modality in ("color", "depth", "part-seg")
        for split, count in (("train", 200), ("val", 20))
        for index in range(count)
    }
    actual_assets = {path.relative_to(scene_root).as_posix() for path in scene_root.rglob("*.png")}
    if actual_assets != expected_assets:
        failures.append(
            f"PNG allowlist mismatch: missing={sorted(expected_assets-actual_assets)[:8]}, "
            f"extra={sorted(actual_assets-expected_assets)[:8]}"
        )
    allowed_files = expected_assets | allowed_json
    actual_files = {path.relative_to(scene_root).as_posix() for path in scene_root.rglob("*") if path.is_file()}
    if actual_files != allowed_files:
        failures.append(
            f"file allowlist mismatch: missing={sorted(allowed_files-actual_files)[:8]}, "
            f"extra={sorted(actual_files-allowed_files)[:8]}"
        )
    frame_lookup = {frame["file_path"]: frame for frame in meta.get("frames", [])}
    for split, count in (("train", 200), ("val", 20)):
        for index in range(count):
            relative = f"color/{split}/{index:04d}.png"
            if relative not in frame_lookup:
                failures.append(f"missing frame metadata: {relative}")
                continue
            assets = [
                scene_root / relative,
                scene_root / relative.replace("color/", "depth/", 1),
                scene_root / relative.replace("color/", "part-seg/", 1),
            ]
            if not all(path.is_file() and not path.is_symlink() for path in assets):
                failures.append(f"missing/non-ordinary assets: {relative}")
                continue
            if index in (0, count - 1):
                color, depth, part = map(iio.imread, assets)
                if color.ndim != 3 or color.shape[2] != 4 or depth.shape != color.shape[:2] or part.shape != depth.shape:
                    failures.append(f"asset geometry mismatch: {relative}")
                if not set(np.unique(part)).issubset({0, 1, 255}):
                    failures.append(f"unexpected part labels: {relative}")
                if not np.any(depth > 0) or not np.any(color[..., 3] > 0):
                    failures.append(f"empty render: {relative}")
    return {
        "scene_id": scene_root.name,
        "pass": not failures,
        "failures": failures,
        "transforms_sha256": sha256_file(meta_path),
        "queries_sha256": sha256_file(query_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-root", required=True, type=Path)
    parser.add_argument("--scenes", nargs="+", default=list(DEV3))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    results = [audit_scene(args.public_root / scene) for scene in args.scenes]
    payload = {
        "schema": "splart-middle-state-public-audit-v1",
        "public_root": str(args.public_root.resolve()),
        "scene_order": list(args.scenes),
        "pass": all(item["pass"] for item in results),
        "results": results,
    }
    payload["content_sha256"] = canonical_sha256(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload))
    if not payload["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
