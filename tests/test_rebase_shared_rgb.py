import json
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from build_relative_clip_cache import HASH_KEYS
from splart.part_interaction import PROVENANCE_KEYS, interaction_predict
from splart.relative_search import METHODS, file_sha256, load_inputs, opaque_object_id, predict, write_json_new
from rebase_shared_rgb_inputs import rebase_value, run, verify_shared_acquisition


def acquisition(tmp_path):
    torch.set_num_threads(1)
    shared = tmp_path / "shared"
    old_dir = tmp_path / "old"
    old_dir.mkdir()
    (shared / "relative/artifacts").mkdir(parents=True)
    (shared / "part/artifacts").mkdir(parents=True)
    relative_prov = {key: "a" * 64 for key in HASH_KEYS}
    relative_rows, part_rows, audit_rows, old_rows = [], [], [], []
    for i in range(2):
        oid = opaque_object_id(f"synthetic-{i}")
        d = torch.linspace(0, 2, 129).repeat(2, 1)
        old_e = torch.zeros(2, 129, 6, 512)
        old_e[..., 0] = 1
        new_e = torch.zeros_like(old_e)
        new_e[..., 0], new_e[..., 1] = 0.6, 0.8
        t = torch.zeros(512)
        t[0] = 1
        value = {"schema": "relative-search-input/v1", "object_id": oid, "split": "source_train", "coordinates": d,
                 "geometry": torch.zeros(1, 2, 1, 129, 8), "image_embeddings": old_e,
                 "observed_embeddings": old_e[:, 0].clone(), "text_direction": t,
                 "provenance": {"geometry_artifact_sha256": "b" * 64, "semantic_artifact_sha256": "c" * 64, "source_index_sha256": "d" * 64}}
        old_path = old_dir / f"{i}.pt"
        torch.save(value, old_path)
        old_rows.append({"object_id": oid, "split": "source_train", "artifact": old_path.name, "artifact_sha256": file_sha256(old_path)})
        sem = {k: value[k] for k in ("object_id", "split", "coordinates", "text_direction")}
        sem.update(schema="relative-clip/v1", image_embeddings=new_e, observed_embeddings=new_e[:, 0].clone(),
                   provenance={**relative_prov, "asset_bundle_sha256": "e" * 64})
        rel_path = shared / f"relative/artifacts/{oid}.pt"
        torch.save(sem, rel_path)
        relative_rows.append({"object_id": oid, "split": "source_train", "artifact": f"artifacts/{oid}.pt", "artifact_sha256": file_sha256(rel_path)})
        part_path = shared / f"part/artifacts/{oid}.pt"
        part_path.write_bytes(b"synthetic cache hash fixture; not an embedding")
        part_rows.append({"object_id": oid, "split": "source_train", "artifact": f"artifacts/{oid}.pt", "artifact_sha256": file_sha256(part_path)})
        shards = []
        (shared / f"rgb/{oid}").mkdir(parents=True)
        for side in range(2):
            for candidate in range(129):
                path = f"rgb/{oid}/side{side}_{candidate:03d}.pt"
                (shared / path).write_bytes(b"synthetic RGB byte-hash fixture")
                shards.append({"path": path, "sha256": file_sha256(shared / path), "side": side, "candidate_index": candidate})
        audit_rows.append({"object_id": oid, "split": "source_train", "historical_semantic_artifact_sha256": "c" * 64, "rgb_shards": shards})
    write_json_new(old_dir / "index.json", {"schema": "relative-search-input-index/v1", "rows": old_rows, "source_index_sha256": "d" * 64, "diagnostic": "synthetic source-only"})
    write_json_new(shared / "relative/index.json", {"schema": "relative-clip-index/v1", "rows": relative_rows, "provenance": relative_prov, "split_counts": {"source_train": 2}})
    part_prov = {k: "a" * 64 for k in PROVENANCE_KEYS}
    part_prov["base_semantic_index_sha256"] = file_sha256(shared / "relative/index.json")
    write_json_new(shared / "part/index.json", {"schema": "part-interaction-index/v1", "rows": part_rows, "provenance": part_prov, "split_counts": {"source_train": 2}})
    write_json_new(shared / "acquisition_audit.json", {"schema": "shared-rgb-acquisition/v1", "relative_index_sha256": file_sha256(shared / "relative/index.json"),
                    "part_index_sha256": file_sha256(shared / "part/index.json"), "original_base_semantic_index_sha256": "f" * 64,
                    "rows": audit_rows, "acquisition": "single_render_shared_rgb"})
    return old_dir / "index.json", shared


def test_fresh_rebase_preserves_geometry_gauge_text_and_raw_recovery(tmp_path):
    old_index, shared = acquisition(tmp_path)
    receipt = run(old_index, shared, tmp_path / "new", allow_train_smoke2=True)
    old_payload, old_values = load_inputs(old_index)
    new_payload, new_values = load_inputs(tmp_path / "new/index.json")
    assert receipt["endpoint_labels_read"] is False
    assert old_payload["source_index_sha256"] == new_payload["source_index_sha256"]
    for old, new in zip(old_values, new_values):
        for field in ("geometry", "coordinates", "text_direction"):
            assert torch.equal(old[field], new[field])
        assert old["provenance"]["geometry_artifact_sha256"] == new["provenance"]["geometry_artifact_sha256"]
        assert not torch.equal(old["image_embeddings"], new["image_embeddings"])
        assert torch.equal(new["observed_embeddings"], new["image_embeddings"][:, 0])
    for method in METHODS:
        old_raw_predictor = predict(new_values[0], method, new_values[1])
        new_raw_mode = interaction_predict(new_values[0], method, "raw", {}, new_values[1])
        assert old_raw_predictor == new_raw_mode


def test_changed_saved_rgb_fails_verification(tmp_path):
    _, shared = acquisition(tmp_path)
    audit, _, _ = verify_shared_acquisition(shared)
    path = shared / audit["rows"][0]["rgb_shards"][0]["path"]
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="RGB shard digest mismatch"):
        verify_shared_acquisition(shared)


def test_changed_text_and_historical_semantic_source_rejected(tmp_path):
    old_index, shared = acquisition(tmp_path)
    _, values = load_inputs(old_index)
    old = values[0]
    sem = torch.load(shared / f"relative/artifacts/{old['object_id']}.pt", weights_only=True)
    sem["text_direction"] = -sem["text_direction"]
    with pytest.raises(ValueError, match="text direction"):
        rebase_value(old, sem, "a" * 64)
    audit_path = shared / "acquisition_audit.json"
    audit = json.loads(audit_path.read_text())
    audit["rows"][0]["historical_semantic_artifact_sha256"] = "0" * 64
    audit_path.write_text(json.dumps(audit))
    with pytest.raises(ValueError, match="historical semantic source"):
        run(old_index, shared, tmp_path / "new", allow_train_smoke2=True)


def test_full_export_refuses_smoke_roster(tmp_path):
    old_index, shared = acquisition(tmp_path)
    with pytest.raises(ValueError, match="full13/6"):
        run(old_index, shared, tmp_path / "new")
