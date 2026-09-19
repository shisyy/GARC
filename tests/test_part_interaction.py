import json
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from test_relative_search import fixture
from splart.relative_search import METHODS, file_sha256, predict, swap_input, write_json_new
from splart.part_interaction import FEATURE_KEYS, MODES, PROVENANCE_KEYS, interaction_evidence, interaction_predict, load_interactions, residual, swap_artifact
from run_part_interaction import run, verify_all_modes


def artifact(value, zero=False):
    e = value["image_embeddings"]
    blank = torch.tensor([1., 0.])
    return {"schema": "part-interaction/v1", "object_id": value["object_id"], "split": value["split"],
            "coordinates": value["coordinates"], "static_embeddings": e.clone() if zero else e.flip(-1),
            "mobile_embeddings": blank.expand_as(e).clone(), "partition_a_embeddings": -e,
            "partition_b_embeddings": blank.expand_as(e).clone(), "blank_embedding": blank,
            "provenance": {k: "a" * 64 for k in PROVENANCE_KEYS | {"asset_bundle_sha256"}}}


@pytest.mark.parametrize("method", METHODS)
def test_raw_exact_recovery(method):
    v, donor = fixture(), fixture("b", 0.2)
    owner = donor if method == "object_shuffle" else v
    assert interaction_predict(v, method, "raw", artifact(owner), owner) == predict(v, method, donor)


@pytest.mark.parametrize("mode", MODES)
def test_exact_swap_all_fields_and_repeated_outputs(mode):
    v, owner = fixture(), fixture("b", 0.3)
    a = artifact(owner)
    for method in METHODS:
        p = interaction_predict(v, method, mode, a, owner)
        assert p == interaction_predict(v, method, mode, a, owner)
        q = interaction_predict(swap_input(v), method, mode, swap_artifact(a), swap_input(owner))
        assert p["sides"] == list(reversed(q["sides"]))
        if method in ("coordinate_only", "geometry_only"):
            assert p == predict(v, method)


def test_zero_residual_is_not_normalized_and_abstains():
    v = fixture()
    a = artifact(v, zero=True)
    r = residual(v, a, "part_interaction")
    assert torch.equal(r, torch.zeros_like(r))
    p = interaction_predict(v, "joint", "part_interaction", a)
    assert all(s["abstain"] and "zero_interaction_signal" in s["reasons"] for s in p["sides"])
    assert all(torch.isfinite(torch.tensor(s["cost"])).all() for s in p["sides"])


def test_partition_changes_representation_and_retains_nonunit_residual():
    v, a = fixture(), artifact(fixture())
    r = residual(v, a, "matched_partition")
    assert torch.allclose(r, 2 * v["image_embeddings"], atol=1e-7)
    assert torch.allclose(r.norm(dim=-1), torch.full_like(r[..., 0], 2.0), atol=1e-6)
    assert not torch.equal(interaction_evidence(v, a, "part_interaction")[0], interaction_evidence(v, a, "matched_partition")[0])


def make_bound_cache(tmp_path, values):
    base_rows, cache_rows = [], []
    shared = {k: "a" * 64 for k in PROVENANCE_KEYS}
    for value in values:
        base_rows.append({"object_id": value["object_id"], "split": value["split"], "artifact": "unused.pt",
                          "artifact_sha256": value["provenance"]["semantic_artifact_sha256"]})
    counts = {s: sum(v["split"] == s for v in values) for s in ("source_train", "source_validation")}
    base = tmp_path / "base.json"
    write_json_new(base, {"schema": "relative-clip-index/v1", "rows": base_rows, "provenance": shared, "split_counts": counts})
    shared["base_semantic_index_sha256"] = file_sha256(base)
    for i, value in enumerate(values):
        a = artifact(value)
        a["provenance"] = {**shared, "asset_bundle_sha256": "b" * 64}
        path = tmp_path / f"cache-{i}.pt"
        torch.save(a, path)
        cache_rows.append({"object_id": value["object_id"], "split": value["split"], "artifact": path.name, "artifact_sha256": file_sha256(path)})
    index = tmp_path / "cache.json"
    write_json_new(index, {"schema": "part-interaction-index/v1", "rows": cache_rows, "provenance": shared, "split_counts": counts})
    return index, base


def test_strict_cache_binding_rejects_semantic_digest_mismatch(tmp_path):
    values = [fixture(), fixture("b")]
    index, base = make_bound_cache(tmp_path, values)
    assert len(load_interactions(index, base, values, allow_train_smoke2=True)) == 2
    values[0]["provenance"]["semantic_artifact_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="semantic artifact hash mismatch"):
        load_interactions(index, base, values)


def test_all_modes_seal_and_two_step_donor(tmp_path):
    values, rows = [], []
    for i in range(19):
        v = fixture(str(i), 0.01 * i)
        v["split"] = "source_train" if i < 13 else "source_validation"
        values.append(v)
        path = tmp_path / f"input-{i}.pt"
        torch.save(v, path)
        rows.append({"object_id": v["object_id"], "split": v["split"], "artifact": path.name, "artifact_sha256": file_sha256(path)})
    cache, base = make_bound_cache(tmp_path, values)
    inputs = tmp_path / "inputs.json"
    write_json_new(inputs, {"schema": "relative-search-input-index/v1", "rows": rows, "source_index_sha256": "0" * 64, "diagnostic": "synthetic"})
    result = run(inputs, cache, base, tmp_path / "out")
    assert result["rows_per_mode"] == 95
    assert verify_all_modes(tmp_path / "out")["verified_modes"] == list(MODES)
    payload = json.loads((tmp_path / "out/donor_interaction/predictions.json").read_text())
    train = sorted(v["object_id"] for v in values if v["split"] == "source_train")
    diag = next(x for x in payload["interaction_diagnostics"] if x["object_id"] == train[0] and x["method"] == "object_shuffle")
    assert diag["effective_semantic_owner"] == train[2]
    changed = tmp_path / "out/raw/predictions.json"
    changed.write_text(changed.read_text() + " ")
    with pytest.raises(ValueError, match="seal mismatch"):
        verify_all_modes(tmp_path / "out")
