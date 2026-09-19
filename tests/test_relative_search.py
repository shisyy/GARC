from dataclasses import asdict
import json
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from splart.relative_search import CONFIG, METHODS, evidence, file_sha256, opaque_object_id, predict, swap_input, validate_input, write_json_new
from prepare_relative_search_inputs import sample_geometry
from run_relative_search import run
from evaluate_relative_search import run as evaluate
from splart.gauge_energy_profile import PROFILE_CHANNELS


def fixture(object_id="a", offset=0.0):
    d = torch.linspace(0, 2, 17).repeat(2, 1)
    q = torch.stack((-d[0], 1 + d[1]))
    angle = (q + offset)[..., None] * torch.tensor([0.3, 0.4, 0.5])[None, None]
    e = torch.stack((angle.cos(), angle.sin()), -1)
    g = torch.zeros(2, 2, 3, len(d[0]), 8)
    g[..., 7] = (d[:, None] - 0.75 - offset).square()
    return {"schema": "relative-search-input/v1", "object_id": opaque_object_id(object_id), "split": "source_train",
            "coordinates": d, "geometry": g, "image_embeddings": e, "observed_embeddings": e[:, 0].clone(),
            "text_direction": torch.tensor([0., 1.]),
            "provenance": {"source_index_sha256": "0" * 64, "semantic_artifact_sha256": "1" * 64, "geometry_artifact_sha256": "2" * 64}}


@pytest.mark.parametrize("method", METHODS)
def test_exact_side_swap_and_repeat(method):
    value, donor = fixture(), fixture("b", 0.3)
    original = predict(value, method, donor)
    assert original == predict(value, method, donor)
    swapped = predict(swap_input(value), method, swap_input(donor))
    assert original["sides"] == list(reversed(swapped["sides"]))


def test_strict_whitelist_rejects_labels_and_anchor_mismatch():
    value = fixture()
    validate_input(value)
    contaminated = dict(value, true_endpoints=torch.zeros(2))
    with pytest.raises(ValueError, match="whitelist"):
        validate_input(contaminated)
    value["observed_embeddings"] = value["observed_embeddings"].flip(0)
    with pytest.raises(ValueError, match="exactly equal"):
        validate_input(value)


def test_controls_are_active_and_modality_isolated():
    value = fixture()
    changed = fixture()
    changed["geometry"][..., 7] = (changed["coordinates"][:, None] - 1.5).square()
    assert predict(value, "coordinate_only") == predict(changed, "coordinate_only")
    assert predict(value, "semantic_only") == predict(changed, "semantic_only")
    assert predict(value, "geometry_only")["distance"] != predict(changed, "geometry_only")["distance"]
    donor = fixture("b", 0.3)
    assert predict(value, "joint")["sides"] != predict(value, "object_shuffle", donor)["sides"]


def test_prompt_sign_flip_preserves_search_but_reverses_closure_interpretation():
    value = fixture()
    flipped = dict(value, text_direction=-value["text_direction"])
    assert torch.equal(evidence(value)[1], evidence(flipped)[1])
    assert predict(value, "joint")["distance"] == predict(flipped, "joint")["distance"]


def test_original_observed_gauge_and_target_noninterference():
    q = torch.linspace(-2, 3, 513)
    fields = q.view(1, 1, 1, -1, 1).expand(2, 2, 3, -1, 9).clone()
    raw = {"schema": "fixture", "object_id": "a", "split": "source_train", "q": q,
           "channels": PROFILE_CHANNELS, "fields": fields, "true_endpoints": object()}
    d = fixture()["coordinates"]
    got = sample_geometry(raw, d)
    assert got.shape == (2, 2, 3, 17, 8)
    assert torch.allclose(got[:, 0, :, :, 0], -d[0].expand(2, 3, -1), atol=1e-6)
    assert torch.allclose(got[:, 1, :, :, 0], (1 + d[1]).expand(2, 3, -1), atol=1e-6)
    raw["true_endpoints"] = "poisoned label unused"
    assert torch.equal(got, sample_geometry(raw, d))


def test_boundary_and_flat_evidence_abstain_but_keep_fallback():
    value = fixture()
    value["geometry"].zero_()
    out = predict(value, "geometry_only")
    assert out["distance"] == [0., 0.]
    assert all(x["abstain"] and x["evidence_interval"] == [0., 2.] for x in out["sides"])


def test_fixed_config_and_sealed_prediction_flow(tmp_path):
    assert json.loads((ROOT / "configs/relative_search_v1.json").read_text()) == asdict(CONFIG)
    rows = []
    for i, value in enumerate((fixture(), fixture("b", 0.3))):
        path = tmp_path / f"{i}.pt"
        torch.save(value, path)
        rows.append({"object_id": value["object_id"], "split": value["split"], "artifact": path.name, "artifact_sha256": file_sha256(path)})
    index = tmp_path / "index.json"
    write_json_new(index, {"schema": "relative-search-input-index/v1", "rows": rows, "source_index_sha256": "0" * 64, "diagnostic": "synthetic plumbing only"})
    output = tmp_path / "predictions.json"
    result = run(index, output)
    assert result["rows"] == 10 and result["exact_swap_error"] == 0
    seal = json.loads(output.with_suffix(".seal.json").read_text())
    assert seal["prediction_sha256"] == file_sha256(output)
    # Mutating predictions must fail before the evaluator even tries to open
    # an endpoint-bearing source index (this path deliberately does not exist).
    output.write_text(output.read_text() + " ")
    with pytest.raises(ValueError, match="seal mismatch"):
        evaluate(output, tmp_path / "missing-label-index.json", tmp_path / "report.json")


def test_full_gate_scores_all_abstentions_and_rejects_missing_object(tmp_path):
    source_rows, input_rows = [], []
    for i in range(19):
        original = f"obj-{i:02d}"
        split = "source_train" if i < 13 else "source_validation"
        raw_path = tmp_path / f"raw-{i}.pt"
        torch.save({"true_endpoints": torch.tensor([-0.5, 1.5])}, raw_path)
        source_rows.append({"artifact": raw_path.name, "artifact_sha256": file_sha256(raw_path), "object_id": original, "split": split, "shape": []})
    source_index = tmp_path / "source-index.json"
    write_json_new(source_index, {"schema": "splart-source-full-trajectory-index/v1", "protected_splits_read": [], "box_labels_read": False, "rows": source_rows})
    for i, row in enumerate(source_rows):
        value = fixture(row["object_id"])
        value["split"] = row["split"]
        value["provenance"]["source_index_sha256"] = file_sha256(source_index)
        path = tmp_path / f"input-{i}.pt"
        torch.save(value, path)
        input_rows.append({"artifact": path.name, "artifact_sha256": file_sha256(path), "object_id": value["object_id"], "split": value["split"]})
    input_index = tmp_path / "inputs.json"
    write_json_new(input_index, {"schema": "relative-search-input-index/v1", "rows": input_rows, "source_index_sha256": file_sha256(source_index), "diagnostic": "synthetic"})
    output = tmp_path / "predictions.json"
    run(input_index, output)
    report = evaluate(output, source_index, tmp_path / "report.json")
    coordinate = report["metrics"]["source_validation"]["coordinate_only"]
    assert report["complete_fixed_source_gate"] is True
    assert coordinate["objects"] == 6 and coordinate["mean_side_nmae"] == 0.5
    assert coordinate["abstention_fraction"] == 1 and coordinate["accepted_side_nmae_secondary"] is None
    pred = json.loads(output.read_text())
    first_id = pred["rows"][0]["object_id"]
    pred["rows"] = [r for r in pred["rows"] if r["object_id"] != first_id]
    dropped = tmp_path / "dropped.json"
    write_json_new(dropped, pred)
    seal = json.loads(output.with_suffix(".seal.json").read_text())
    seal["prediction_sha256"] = file_sha256(dropped)
    write_json_new(dropped.with_suffix(".seal.json"), seal)
    with pytest.raises(ValueError, match="requires all 13/6"):
        evaluate(dropped, source_index, tmp_path / "missing-report.json")
