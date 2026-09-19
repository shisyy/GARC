"""Synthetic, label-free tests of the complete four-mode sealing boundary."""
import json
from pathlib import Path
import shutil
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
import run_geometry_first_gate as gate
from run_relative_search import run as original_run
from splart.relative_search import METHODS, file_sha256, opaque_object_id, write_json_new


def value_fixture(identity, split, offset=0.):
    d = torch.linspace(0, 2, 17).repeat(2, 1)
    q = torch.stack((-d[0], 1 + d[1]))
    angle = (q + offset)[..., None] * torch.tensor([0.3, 0.4, 0.5])[None, None]
    e = torch.stack((angle.cos(), angle.sin()), -1)
    g = torch.zeros(2, 2, 3, 17, 8)
    g[..., 7] = (d[:, None] - 0.75 - offset).square()
    for radius in range(3):
        g[:, :, radius, :, 2] = 0.05 * d + torch.relu(d - (0.625 + radius * 0.125))
    return {"schema": "relative-search-input/v1", "object_id": opaque_object_id(identity), "split": split,
            "coordinates": d, "geometry": g, "image_embeddings": e, "observed_embeddings": e[:, 0].clone(),
            "text_direction": torch.tensor([0., 1.]),
            "provenance": {"source_index_sha256": "0" * 64, "semantic_artifact_sha256": "1" * 64,
                           "geometry_artifact_sha256": "2" * 64}}


def make_inputs(path, count=19):
    rows, values = [], []
    for i in reversed(range(count)):
        value = value_fixture(str(i), "source_train" if i < 13 else "source_validation", 0.01 * i)
        artifact = path / f"input-{i}.pt"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        torch.save(value, artifact)
        values.append(value)
        rows.append({"object_id": value["object_id"], "split": value["split"], "artifact": artifact.name,
                     "artifact_sha256": file_sha256(artifact)})
    index = path / "index.json"
    write_json_new(index, {"schema": "relative-search-input-index/v1", "rows": rows,
                           "source_index_sha256": "0" * 64, "diagnostic": "synthetic label-free fixture"})
    return index, values


@pytest.fixture(scope="module")
def completed_gate(tmp_path_factory):
    base = tmp_path_factory.mktemp("geometry-first")
    inputs, values = make_inputs(base / "inputs")
    output = base / "results"
    result = gate.run(inputs, output)
    return inputs, values, output, result


def test_all_modes_sealed_with_exact_raw_and_original_controls(completed_gate):
    inputs, _, output, result = completed_gate
    assert result["rows_per_mode"] == 95
    assert gate.verify_all_modes(output)["verified_modes"] == list(gate.MODES)
    original = output.parent / "original.json"
    original_run(inputs, original)
    raw = json.loads((output / "raw/predictions.json").read_text())
    assert raw["rows"] == json.loads(original.read_text())["rows"]
    controls = [r for r in raw["rows"] if r["method"] in ("coordinate_only", "semantic_only")]
    for mode in gate.MODES:
        report = json.loads((output / mode / "predictions.json").read_text())
        assert report["exact_swap_error"] == 0
        assert [r for r in report["rows"] if r["method"] in ("coordinate_only", "semantic_only")] == controls
        assert report["endpoint_labels_used"] is False
        assert report["trained_parameters"] == 0


def test_sorted_same_split_first_and_second_donor_routes(completed_gate):
    _, values, output, _ = completed_gate
    donors = gate.donor_map(values)
    for mode in gate.MODES:
        diagnostics = json.loads((output / mode / "routing_diagnostics.json").read_text())
        for row in diagnostics["rows"]:
            first = donors[row["object_id"]]
            second = donors[first["object_id"]]
            assert first["split"] == second["split"] == row["split"]
            assert row["first_donor"] == first["object_id"]
            if mode == "donor_semantics" and row["method"] == "object_shuffle":
                assert row["semantic_owner"] == row["routing_owner"] == second["object_id"]
            if mode == "shuffled_verification" and row["method"] == "object_shuffle":
                assert row["semantic_owner"] == first["object_id"]
                assert row["routing_owner"] == second["object_id"]
            if mode == "shuffled_verification" and row["method"] == "joint":
                assert row["semantic_owner"] == row["object_id"]
                assert row["routing_owner"] == first["object_id"]
        assert diagnostics["summary"] == gate.diagnostic_summary(diagnostics["rows"])


@pytest.mark.parametrize("artifact", ["geometry_first/predictions.json", "donor_semantics/routing_diagnostics.json"])
def test_tampering_rejected_before_independent_evaluator(completed_gate, tmp_path, artifact):
    _, _, output, _ = completed_gate
    copied = tmp_path / "results"
    shutil.copytree(output, copied)
    changed = copied / artifact
    changed.write_text(changed.read_text() + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="seal mismatch"):
        gate.verify_all_modes(copied)


def test_missing_mode_rejected(completed_gate, tmp_path):
    _, _, output, _ = completed_gate
    copied = tmp_path / "results"
    shutil.copytree(output, copied)
    seal = copied / "all_modes.seal.json"
    payload = json.loads(seal.read_text())
    payload["predictions"].pop("donor_semantics")
    seal.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="four preregistered"):
        gate.verify_all_modes(copied)


def test_existing_output_and_incomplete_roster_rejected(completed_gate, tmp_path):
    inputs, _, output, _ = completed_gate
    with pytest.raises(ValueError, match="must be new"):
        gate.run(inputs, output)
    incomplete, _ = make_inputs(tmp_path / "inputs", count=18)
    with pytest.raises(ValueError, match="full fixed 13/6"):
        gate.run(incomplete, tmp_path / "results")


def test_runner_has_no_endpoint_loader_or_source_argument():
    import ast
    tree = ast.parse((ROOT / "run_geometry_first_gate.py").read_text(encoding="utf-8"))
    modules = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    assert not any(m and (m.startswith("evaluate_") or m.startswith("prepare_") or m.startswith("run_glpdt")) for m in modules)
    strings = [node.value for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, str)]
    assert "--source-index" not in strings
    assert "true_endpoints" not in strings
