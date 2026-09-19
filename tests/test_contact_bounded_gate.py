"""Label-free four-mode gate sealing integration, with synthetic loader stubs."""
import json
import pytest
import torch
from test_contact_bounded_semantics import fixture, gate, MODES, file_sha256, write_json_new


def test_full_four_mode_runner_seals_and_controls(tmp_path, monkeypatch):
    values, contacts, calibrations, input_rows = [], {}, {}, []
    for i in range(19):
        value, contact, calibration = fixture(str(i), 4 + i % 3, "source_train" if i < 13 else "source_validation")
        values.append(value)
        identity = value["split"], value["object_id"]
        contacts[identity], calibrations[identity] = contact, calibration
        artifact = tmp_path / f"input-{i}.pt"
        torch.save(value, artifact)
        input_rows.append({"object_id": value["object_id"], "split": value["split"], "artifact": artifact.name,
                           "artifact_sha256": file_sha256(artifact)})
    inputs = tmp_path / "inputs.json"
    write_json_new(inputs, {"schema": "relative-search-input-index/v1", "rows": input_rows,
                           "source_index_sha256": "0" * 64, "diagnostic": "synthetic runner plumbing"})
    contact_index, calibration_index = tmp_path / "contacts.json", tmp_path / "calibration.json"
    write_json_new(contact_index, {"synthetic_loader_stub": True})
    write_json_new(calibration_index, {"synthetic_loader_stub": True})
    # Strict disk loader negative tests belong to test_contact_calibration.py;
    # here all-method prediction, donor routing and output seals are integrated.
    monkeypatch.setattr(gate, "load_contacts", lambda *a: ({}, contacts))
    monkeypatch.setattr(gate, "load_calibration", lambda *a: ({}, calibrations))
    out = tmp_path / "results"
    assert gate.run(inputs, contact_index, calibration_index, out)["rows_per_mode"] == 95
    assert gate.verify_all_modes(out)["verified_modes"] == list(MODES)
    raw = json.loads((out / "raw/predictions.json").read_text())
    fixed = [r for r in raw["rows"] if r["method"] in ("coordinate_only", "semantic_only")]
    for mode in MODES:
        result = json.loads((out / mode / "predictions.json").read_text())
        assert [r for r in result["rows"] if r["method"] in ("coordinate_only", "semantic_only")] == fixed
    diag = out / "bounded/routing_diagnostics.json"
    diag.write_text(diag.read_text() + " ")
    with pytest.raises(ValueError, match="seal mismatch"):
        gate.verify_all_modes(out)



