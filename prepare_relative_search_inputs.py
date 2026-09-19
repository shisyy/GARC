#!/usr/bin/env python3
"""Offline whitelist exporter. Raw source artifacts contain labels; never export them.

This executable is separate from both predictor and evaluator. It copies only
fixed geometry channels and verifies provenance; it never indexes true_endpoints.
"""
import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from run_glpdt_source_gate import ARTIFACT_KEYS, validate_index
from splart.gauge_energy_profile import PROFILE_CHANNELS
from splart.relative_search import file_sha256, opaque_object_id, validate_input, write_json_new


def sample_geometry(raw: dict, coordinates: torch.Tensor) -> torch.Tensor:
    """Original observed gauge only: side0 q=-d, side1 q=1+d."""
    if set(raw) != ARTIFACT_KEYS or tuple(raw["channels"]) != PROFILE_CHANNELS:
        raise ValueError("unexpected raw source geometry schema")
    q = torch.as_tensor(raw["q"], dtype=torch.float32)
    fields = torch.as_tensor(raw["fields"], dtype=torch.float32)
    if q.ndim != 1 or not torch.all(q[1:] > q[:-1]):
        raise ValueError("invalid source q grid")
    if fields.ndim != 5 or fields.shape[1] != 2 or fields.shape[3:] != (len(q), 9):
        raise ValueError("invalid source field shape")
    out = []
    for side in range(2):
        query = -coordinates[side] if side == 0 else 1.0 + coordinates[side]
        if query.min() < q[0] or query.max() > q[-1]:
            raise ValueError("fixed query outside geometry cache; do not clamp silently")
        upper = torch.searchsorted(q, query).clamp(1, len(q) - 1)
        lower = upper - 1
        weight = ((query - q[lower]) / (q[upper] - q[lower]))[None, None, :, None]
        left, right = fields[:, side, :, lower, :8], fields[:, side, :, upper, :8]
        out.append(left + weight * (right - left))
    return torch.stack(out, dim=1)


def run(source_index: Path, semantic_index: Path, output_dir: Path) -> dict:
    if output_dir.exists():
        raise ValueError("export directory must be new")
    verified = {(r["split"], opaque_object_id(r["object_id"])): (r, p) for r, p in validate_index(source_index.resolve())}
    semantic = json.loads(semantic_index.read_text(encoding="utf-8"))
    if set(semantic) != {"schema", "rows", "provenance", "split_counts"} or semantic["schema"] != "relative-clip-index/v1":
        raise ValueError("unexpected relative CLIP index")
    source_hash = file_sha256(source_index)
    output_dir.mkdir(parents=True)
    rows, identities = [], set()
    for row in sorted(semantic["rows"], key=lambda r: (r["split"], r["object_id"])):
        if set(row) != {"object_id", "split", "artifact", "artifact_sha256"}:
            raise ValueError("unexpected semantic row")
        identity = (row["split"], row["object_id"])
        if identity not in verified or identity in identities:
            raise ValueError("unexpected or duplicate semantic identity")
        identities.add(identity)
        relative = Path(row["artifact"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("semantic path must be within cache")
        spath = semantic_index.parent / relative
        if spath.is_symlink() or not spath.resolve().is_relative_to(semantic_index.parent.resolve()):
            raise ValueError("semantic artifact escapes cache")
        if file_sha256(spath) != row["artifact_sha256"]:
            raise ValueError("semantic digest mismatch")
        sem = torch.load(spath, map_location="cpu", weights_only=True)
        expected = {"schema", "object_id", "split", "coordinates", "image_embeddings", "observed_embeddings", "text_direction", "provenance"}
        if set(sem) != expected or sem["schema"] != "relative-clip/v1" or (sem["split"], sem["object_id"]) != identity:
            raise ValueError("semantic artifact contract mismatch")
        raw_row, raw_path = verified[identity]
        raw = torch.load(raw_path, map_location="cpu", weights_only=True)
        if (raw["split"], opaque_object_id(raw["object_id"])) != identity:
            raise ValueError("geometry identity mismatch")
        # Copy an explicit whitelist; do not propagate raw dictionaries/metadata.
        value = {k: sem[k] for k in ("object_id", "split", "coordinates", "image_embeddings", "observed_embeddings", "text_direction")}
        value.update(schema="relative-search-input/v1", geometry=sample_geometry(raw, sem["coordinates"]),
                     provenance={"geometry_artifact_sha256": raw_row["artifact_sha256"],
                                 "semantic_artifact_sha256": row["artifact_sha256"], "source_index_sha256": source_hash})
        validate_input(value)
        artifact = f"input-{len(rows):03d}.pt"
        torch.save(value, output_dir / artifact)
        rows.append({"object_id": identity[1], "split": identity[0], "artifact": artifact, "artifact_sha256": file_sha256(output_dir / artifact)})
    payload = {"schema": "relative-search-input-index/v1", "rows": rows, "source_index_sha256": source_hash,
               "diagnostic": "Articraft source-only full-mesh surrogate geometry; stylized renders; not reconstructed-scene validation"}
    write_json_new(output_dir / "index.json", payload)
    return {"index": str(output_dir / "index.json"), "objects": len(rows)}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source-index", type=Path, required=True)
    p.add_argument("--semantic-index", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(run(a.source_index, a.semantic_index, a.output_dir), sort_keys=True))
