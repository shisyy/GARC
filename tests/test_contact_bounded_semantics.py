"""Synthetic geometry barriers and closure-only localization, without labels."""
import ast
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from splart.relative_search import METHODS, opaque_object_id, predict, swap_input, file_sha256, write_json_new
from splart.contact_bounded_semantics import MODES, bounded_predict, swap_contact
from splart.contact_calibration import first_event, unseen_face_pairs
import run_contact_bounded_gate as gate


def fixture(identity="a", peak=4, split="source_train"):
    d = torch.linspace(0, 2, 129).repeat(2, 1)
    scores = torch.full((2, 129, 6), .3)
    scores[0, 0] = .2
    scores[1, 0] = -.2
    scores[0, peak] = .8
    embeddings = torch.stack(((1 - scores.square()).sqrt(), scores), -1)
    geometry = torch.zeros(1, 2, 1, 129, 8)
    geometry[..., 7] = (d[:, None] - .75).square()
    value = {"schema": "relative-search-input/v1", "object_id": opaque_object_id(identity), "split": split,
             "coordinates": d, "geometry": geometry, "image_embeddings": embeddings,
             "observed_embeddings": embeddings[:, 0].clone(), "text_direction": torch.tensor([0., 1.]),
             "provenance": {"geometry_artifact_sha256": "1" * 64, "semantic_artifact_sha256": "2" * 64,
                            "source_index_sha256": "0" * 64}}
    sides = []
    for event in (8, 12):
        pairs = [torch.empty(0, 2, dtype=torch.int64) for _ in range(129)]
        pairs[event] = torch.tensor([[1, 1]])
        sides.append({"triangle_pairs": pairs})
    contact = {"object_id": value["object_id"], "split": split, "coordinates": d,
               "observed_seen_pairs": torch.tensor([[0, 0]]), "sides": sides}
    calibration = {"object_id": value["object_id"], "split": split, "calibration_passed": True,
                   "enumeration_complete": True, "exact_swap_passed": True}
    return value, contact, calibration


def test_geometry_last_safe_prefix_and_closure_semantic_selection():
    value, contact, calibration = fixture()
    geometry = bounded_predict(value, "geometry_only", contact, calibration)
    joint = bounded_predict(value, "joint", contact, calibration)
    assert geometry["distance"] == [float(value["coordinates"][0, 7]), float(value["coordinates"][1, 11])]
    closed, opened = [s["contact_bounded"] for s in joint["sides"]]
    assert closed["prefix"] == [1, 7] and closed["selected_index"] == 4
    assert closed["semantic_activation"] and closed["selection_changed"] and closed["semantic_verified"]
    assert not opened["semantic_activation"] and opened["fallback_reason"] == "opening_role_not_localized"
    assert opened["selected_index"] == 11


@pytest.mark.parametrize("failure", ["nohit", "calibration", "immediate"])
def test_unsupported_geometry_forbids_semantics_and_keeps_raw_geometry(failure):
    value, contact, calibration = fixture()
    if failure == "nohit":
        for side in contact["sides"]:
            side["triangle_pairs"] = [torch.empty(0, 2, dtype=torch.int64) for _ in range(129)]
    elif failure == "calibration":
        calibration["calibration_passed"] = False
    else:
        for side in contact["sides"]:
            side["triangle_pairs"][1] = torch.tensor([[2, 2]])
    result = bounded_predict(value, "joint", contact, calibration)
    assert result["distance"] == predict(value, "geometry_only")["distance"]
    assert all(s["abstain"] and not s["contact_bounded"]["semantic_activation"] for s in result["sides"])


def test_unknown_role_and_failed_semantic_verification_keep_geometry():
    value, contact, calibration = fixture()
    value["image_embeddings"][0, 0, :3] = value["image_embeddings"][1, 0, :3]
    value["image_embeddings"][1, 0, :3] = torch.tensor([(1 - .2 ** 2) ** .5, .2])
    value["observed_embeddings"] = value["image_embeddings"][:, 0].clone()
    result = bounded_predict(value, "joint", contact, calibration)
    assert all(not s["contact_bounded"]["semantic_activation"] for s in result["sides"])
    value, contact, calibration = fixture()
    value["image_embeddings"][0, 1:8] = value["image_embeddings"][0, :1]
    result = bounded_predict(value, "joint", contact, calibration)
    assert result["sides"][0]["contact_bounded"]["fallback_reason"] == "semantic_verification_failed"
    assert result["sides"][0]["contact_bounded"]["selected_index"] == 7


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("method", METHODS)
def test_exact_swap_all_predictions_diagnostics_and_raw_controls(mode, method):
    value, contact, calibration = fixture()
    donor, _, _ = fixture("b", 6)
    route, _, _ = fixture("c", 5)
    original = bounded_predict(value, method, contact, calibration, mode, donor, route)
    swapped = bounded_predict(swap_input(value), method, swap_contact(contact), calibration, mode,
                              swap_input(donor), swap_input(route))
    assert original == dict(swapped, sides=list(reversed(swapped["sides"])), distance=list(reversed(swapped["distance"])))
    if mode == "raw" or method in ("coordinate_only", "semantic_only"):
        assert original == predict(value, method, donor)


def test_donor_changes_localization_and_never_uses_donor_geometry():
    value, contact, calibration = fixture()
    donor, _, _ = fixture("b", 6)
    donor["geometry"].fill_(100)
    own = bounded_predict(value, "joint", contact, calibration)
    replaced = bounded_predict(value, "joint", contact, calibration, "donor_semantics", donor)
    assert own["sides"][0]["contact_bounded"]["selected_index"] == 4
    assert replaced["sides"][0]["contact_bounded"]["selected_index"] == 6


def test_face_subdivision_exclusion_counterexample_is_explicit_limitation():
    # Face IDs 1/1 can be children of already contacted triangles 0/0. Without
    # parent-face topology, both-unseen classification changes under subdivision
    # even if the geometric contact region is unchanged. This is NOT invariance.
    observed = torch.tensor([[0, 0]])
    assert len(unseen_face_pairs(torch.tensor([[0, 0]]), observed)) == 0
    assert len(unseen_face_pairs(torch.tensor([[1, 1]]), observed)) == 1
    assert len(unseen_face_pairs(torch.tensor([[0, 1], [1, 0]]), observed)) == 0




def test_no_label_bearing_imports_or_argument():
    for path in (ROOT / "run_contact_bounded_gate.py", ROOT / "src/splart/contact_bounded_semantics.py"):
        tree = ast.parse(path.read_text())
        modules = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
        assert not any(m and (m.startswith("evaluate_") or m.startswith("prepare_") or m.startswith("run_glpdt")) for m in modules)
        strings = [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)]
        assert "--source-index" not in strings and "true_endpoints" not in strings
