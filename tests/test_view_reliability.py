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
from splart.view_reliability import MODES, view_weights, weighted_predict
from run_view_reliability import run


@pytest.mark.parametrize("mode", MODES)
def test_shared_weights_exact_swap_and_finite(mode):
    v = fixture()
    weights, diag = view_weights(v, mode)
    swapped_weights, swapped_diag = view_weights(swap_input(v), mode)
    assert torch.equal(weights, swapped_weights)
    assert diag == swapped_diag
    assert torch.isfinite(weights).all() and (weights >= 0).all()
    assert abs(float(weights.sum()) - 1) < 1e-14
    assert 1 - 1e-12 <= diag["effective_view_count"] <= 3 + 1e-12


@pytest.mark.parametrize("method", METHODS)
def test_uniform_exactly_recovers_old_output(method):
    v, donor = fixture(), fixture("b", 0.2)
    assert weighted_predict(v, method, "uniform", donor) == predict(v, method, donor)


@pytest.mark.parametrize("mode", MODES)
def test_all_output_fields_exact_swap_and_repeat(mode):
    v, donor, weight_donor = fixture(), fixture("b", 0.2), fixture("c", 0.4)
    for method in METHODS:
        p = weighted_predict(v, method, mode, donor, weight_donor)
        assert p == weighted_predict(v, method, mode, donor, weight_donor)
        q = weighted_predict(swap_input(v), method, mode, swap_input(donor), swap_input(weight_donor))
        assert p["sides"] == list(reversed(q["sides"]))
        if method in ("coordinate_only", "geometry_only"):
            assert p == predict(v, method, donor)


def test_zero_observed_projection_fallback_and_weakness_preserved():
    v = fixture()
    v["image_embeddings"][1, 0] = v["image_embeddings"][0, 0]
    v["observed_embeddings"] = v["image_embeddings"][:, 0].clone()
    w, diag = view_weights(v, "reliability")
    assert torch.equal(w, torch.ones_like(w) / len(w))
    assert diag["all_unobservable_uniform_fallback"]
    p = weighted_predict(v, "joint", "reliability")
    assert all("weak_observed_semantic_direction" in s["reasons"] for s in p["sides"])


def test_formula_uses_observed_signal_not_noise_only_and_zero_signal_view_zero():
    v = fixture()
    w, diag = view_weights(v, "reliability")
    n, _ = view_weights(v, "inverse_noise_only")
    assert not torch.allclose(w, n)
    assert sum(diag["projected_second_difference_noise"]) > 0
    v["image_embeddings"][1, 0, 0] = v["image_embeddings"][0, 0, 0]
    v["observed_embeddings"] = v["image_embeddings"][:, 0].clone()
    w, _ = view_weights(v, "reliability")
    assert w[0] == 0


def test_nonuniform_modes_do_not_remove_view_or_opening_abstention():
    v, donor = fixture(), fixture("b", 0.4)
    protected = {"view_or_geometry_disagreement", "opening_limit_not_identifiable_from_closure_semantics", "weak_observed_semantic_direction"}
    original = predict(v, "joint")
    for mode in MODES:
        p = weighted_predict(v, "joint", mode, donor, donor)
        for old_side, new_side in zip(original["sides"], p["sides"]):
            assert protected.intersection(old_side["reasons"]) == protected.intersection(new_side["reasons"])


def test_mismatched_and_self_donors_rejected():
    v, donor = fixture(), fixture("b", 0.4)
    with pytest.raises(ValueError, match="distinct"):
        weighted_predict(v, "object_shuffle", "reliability", v)
    with pytest.raises(ValueError, match="distinct"):
        weighted_predict(v, "joint", "shuffled_weights", donor, v)
    donor["split"] = "source_validation"
    with pytest.raises(ValueError, match="split and grid"):
        weighted_predict(v, "object_shuffle", "reliability", donor)
    with pytest.raises(ValueError, match="split and grid"):
        weighted_predict(v, "joint", "shuffled_weights", donor, donor)


def test_all_four_modes_sealed_without_labels(tmp_path):
    rows = []
    for i in range(19):
        v = fixture(str(i), 0.01 * i)
        v["split"] = "source_train" if i < 13 else "source_validation"
        path = tmp_path / f"{i}.pt"
        torch.save(v, path)
        rows.append({"object_id": v["object_id"], "split": v["split"], "artifact": path.name, "artifact_sha256": file_sha256(path)})
    index = tmp_path / "index.json"
    write_json_new(index, {"schema": "relative-search-input-index/v1", "rows": rows, "source_index_sha256": "0" * 64, "diagnostic": "synthetic"})
    result = run(index, tmp_path / "out")
    assert result["rows_per_mode"] == 95
    manifest = json.loads((tmp_path / "out/all_modes.seal.json").read_text())
    assert manifest["modes"] == list(MODES)
    for mode in MODES:
        output = tmp_path / "out" / manifest["predictions"][mode]["path"]
        assert manifest["predictions"][mode]["sha256"] == file_sha256(output)
        seal = json.loads(output.with_suffix(".seal.json").read_text())
        assert seal["prediction_sha256"] == file_sha256(output)
        payload = json.loads(output.read_text())
        assert payload["exact_swap_error"] == 0 and payload["endpoint_labels_used"] is False
        assert len(payload["weight_diagnostics"]) == 19
