#!/usr/bin/env python3
"""Rebase target-free geometry inputs onto one shared-RGB semantic acquisition."""
import argparse
import json
from pathlib import Path
import re
import sys

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from build_relative_clip_cache import validate_relative_artifact, validate_relative_index
from splart.part_interaction_cache import validate_part_index
from splart.relative_search import file_sha256, load_inputs, validate_input, write_json_new


def _digest(value):
    if not isinstance(value, str) or not re.fullmatch(r"[a-f0-9]{64}", value):
        raise ValueError("expected SHA256 digest")


def _safe_file(root: Path, relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("artifact path must remain within acquisition root")
    path = root / path
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError("missing or unsafe acquisition artifact")
    return path


def verify_shared_acquisition(acquisition_root: Path) -> tuple[dict, dict, dict]:
    """Verify common index bindings and every saved RGB shard's bytes."""
    audit = json.loads((acquisition_root / "acquisition_audit.json").read_text(encoding="utf-8"))
    expected = {"schema", "relative_index_sha256", "part_index_sha256", "original_base_semantic_index_sha256", "rows", "acquisition"}
    if set(audit) != expected or audit["schema"] != "shared-rgb-acquisition/v1" or audit["acquisition"] != "single_render_shared_rgb":
        raise ValueError("unexpected shared-RGB acquisition audit")
    for key in ("relative_index_sha256", "part_index_sha256", "original_base_semantic_index_sha256"):
        _digest(audit[key])
    relative_path = _safe_file(acquisition_root, "relative/index.json")
    part_path = _safe_file(acquisition_root, "part/index.json")
    if file_sha256(relative_path) != audit["relative_index_sha256"] or file_sha256(part_path) != audit["part_index_sha256"]:
        raise ValueError("shared-RGB cache index binding mismatch")
    relative = json.loads(relative_path.read_text(encoding="utf-8"))
    part = json.loads(part_path.read_text(encoding="utf-8"))
    validate_relative_index(relative)
    validate_part_index(part)
    if part["provenance"]["base_semantic_index_sha256"] != audit["relative_index_sha256"]:
        raise ValueError("part cache is not bound to this newly acquired baseline")
    for key in ("render_plan_sha256", "renderer_sha256", "render_config_sha256", "encoder_checkpoint_sha256", "open_clip_wheel_sha256", "builder_sha256"):
        if relative["provenance"][key] != part["provenance"][key]:
            raise ValueError("baseline/part acquisition implementation mismatch")
    roster = {(r["split"], r["object_id"]) for r in relative["rows"]}
    if roster != {(r["split"], r["object_id"]) for r in part["rows"]}:
        raise ValueError("shared cache rosters differ")
    # Index signatures alone cannot catch a truncated/deleted cache or RGB bank.
    for directory, index in (("relative", relative), ("part", part)):
        for row in index["rows"]:
            path = _safe_file(acquisition_root / directory, row["artifact"])
            if file_sha256(path) != row["artifact_sha256"]:
                raise ValueError("shared cache artifact digest mismatch")
    seen = set()
    for row in audit["rows"]:
        if set(row) != {"object_id", "split", "historical_semantic_artifact_sha256", "rgb_shards"}:
            raise ValueError("unexpected RGB audit object row")
        identity = (row["split"], row["object_id"])
        if identity in seen or identity not in roster:
            raise ValueError("unexpected or duplicate RGB object identity")
        seen.add(identity)
        _digest(row["historical_semantic_artifact_sha256"])
        positions = set()
        for shard in row["rgb_shards"]:
            if set(shard) != {"path", "sha256", "side", "candidate_index"}:
                raise ValueError("unexpected RGB shard row")
            position = (shard["side"], shard["candidate_index"])
            if position in positions or position[0] not in (0, 1) or position[1] not in range(129):
                raise ValueError("missing/duplicate/out-of-grid RGB shard")
            positions.add(position)
            if shard["path"] != f"rgb/{row['object_id']}/side{position[0]}_{position[1]:03d}.pt":
                raise ValueError("noncanonical RGB shard path")
            _digest(shard["sha256"])
            path = _safe_file(acquisition_root, shard["path"])
            if file_sha256(path) != shard["sha256"]:
                raise ValueError("saved RGB shard digest mismatch")
        if len(positions) != 258:
            raise ValueError("complete two-side129 RGB bank required")
    if seen != roster:
        raise ValueError("RGB audit roster differs from both caches")
    return audit, relative, part


def rebase_value(old: dict, semantic: dict, new_semantic_sha256: str) -> dict:
    validate_input(old)
    validate_relative_artifact(semantic)
    if (semantic["split"], semantic["object_id"]) != (old["split"], old["object_id"]):
        raise ValueError("fresh semantic identity mismatch")
    if not torch.equal(old["coordinates"], semantic["coordinates"]):
        raise ValueError("fresh semantic observed-gauge grid changed")
    if not torch.equal(old["text_direction"], semantic["text_direction"]):
        raise ValueError("fresh acquisition changed the frozen text direction")
    _digest(new_semantic_sha256)
    value = dict(old)
    value["image_embeddings"] = semantic["image_embeddings"]
    value["observed_embeddings"] = semantic["observed_embeddings"]
    value["provenance"] = {**old["provenance"], "semantic_artifact_sha256": new_semantic_sha256}
    validate_input(value)
    return value


def run(old_inputs: Path, acquisition_root: Path, output_dir: Path, allow_train_smoke2: bool = False) -> dict:
    if output_dir.exists():
        raise ValueError("new input export directory must not exist")
    torch.set_num_threads(1)
    old_index, values = load_inputs(old_inputs)
    audit, relative, _ = verify_shared_acquisition(acquisition_root)
    required_counts = {"source_train": 2} if allow_train_smoke2 else {"source_train": 13, "source_validation": 6}
    if relative["split_counts"] != required_counts:
        raise ValueError("requires fixed full13/6 acquisition or explicit two-training-object smoke")
    old_values = {(v["split"], v["object_id"]): v for v in values}
    new_identities = {(r["split"], r["object_id"]) for r in relative["rows"]}
    if (not new_identities <= set(old_values)) or (not allow_train_smoke2 and new_identities != set(old_values)):
        raise ValueError("old label-free inputs and fresh acquisition rosters differ")
    old_hashes = {(r["split"], r["object_id"]): r["historical_semantic_artifact_sha256"] for r in audit["rows"]}
    for identity in new_identities:
        if old_hashes[identity] != old_values[identity]["provenance"]["semantic_artifact_sha256"]:
            raise ValueError("acquisition historical semantic source differs from old inputs")
    output_dir.mkdir(parents=True)
    rows = []
    for row in sorted(relative["rows"], key=lambda r: (r["split"], r["object_id"])):
        path = _safe_file(acquisition_root / "relative", row["artifact"])
        semantic = torch.load(path, map_location="cpu", weights_only=True)
        if any(semantic["provenance"][k] != v for k, v in relative["provenance"].items()):
            raise ValueError("fresh artifact/index provenance mismatch")
        value = rebase_value(old_values[(row["split"], row["object_id"])], semantic, row["artifact_sha256"])
        target = output_dir / f"input-{len(rows):03d}.pt"
        with target.open("xb") as stream:
            torch.save(value, stream)
        rows.append({"object_id": value["object_id"], "split": value["split"], "artifact": target.name, "artifact_sha256": file_sha256(target)})
    index = {"schema": "relative-search-input-index/v1", "rows": rows, "source_index_sha256": old_index["source_index_sha256"],
             "diagnostic": old_index["diagnostic"] + "; fresh same-RGB baseline/visible-deletion acquisition"}
    write_json_new(output_dir / "index.json", index)
    receipt = {"schema": "shared-rgb-rebased-inputs/v1", "old_input_index_sha256": file_sha256(old_inputs),
               "new_input_index_sha256": file_sha256(output_dir / "index.json"),
               "acquisition_audit_sha256": file_sha256(acquisition_root / "acquisition_audit.json"),
               "relative_index_sha256": audit["relative_index_sha256"], "part_index_sha256": audit["part_index_sha256"],
               "source_index_sha256": old_index["source_index_sha256"], "endpoint_labels_read": False,
               "geometry_coordinates_text_preserved": True, "objects": len(rows)}
    write_json_new(output_dir / "rebase_audit.json", receipt)
    return receipt


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--old-inputs", type=Path, required=True)
    p.add_argument("--acquisition-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--allow-train-smoke2", action="store_true")
    a = p.parse_args()
    print(json.dumps(run(a.old_inputs, a.acquisition_root, a.output_dir, a.allow_train_smoke2), sort_keys=True))
