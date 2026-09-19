from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from splart.contact_calibration import (CONTACT_HASH_KEYS, MIDPOINTS, assert_midpoint_swap, first_event,
                                       load_calibration, load_contacts, measure_midpoints, swap_contact,
                                       unseen_face_pairs, validate_contact, validate_midpoints)
from splart.relative_search import file_sha256, write_json_new
from splart.surface_contact import CONFIG, _pack, annotate_pairs


def tensor(pairs):
    return torch.tensor(pairs, dtype=torch.int64).reshape(-1, 2)


def query(pairs):
    return {"surface_intersection": bool(pairs), "surface_clearance": 0. if pairs else 1.,
            "backend_unsigned_distance": None if pairs else 1., "triangle_pairs": pairs,
            "contact_count": len(pairs), "pair_count": len(pairs), "enumeration_complete": True,
            "enumeration_capacity": 64, "enumeration_attempts": 1}


def contact(value, provenance):
    reference = {"seen_pairs": {(0, 0)}, "persistent_pairs": {(0, 0)}}
    observed = _pack([annotate_pairs(query([(0, 0)]), reference) for _ in range(65)])
    sides = [_pack([annotate_pairs(query([(0, 0)] if i < first else [(0, 0), (1, 1)]), reference)
                    for i in range(129)]) for first in (10, 20)]
    return {"schema": "surface-contact-measurement/v1", "object_id": value["object_id"], "split": value["split"],
            "coordinates": value["coordinates"], "observed_coordinates": torch.linspace(0, 1, 65, dtype=torch.float64),
            "observed": observed, "sides": sides, "observed_seen_pairs": tensor([(0, 0)]),
            "persistent_all_observed_pairs": tensor([(0, 0)]), "observed_intersection_count": 65,
            "observed_anchor_intersection": [True, True], "observed_overlapping_or_touching": True,
            "physical_validity": "unresolved_visual_surface_intersections_and_no_solid_containment_test",
            "novelty_caution": "new_triangle_pairs_can_be_tessellation_or_sliding_not_new_constraints",
            "boundary_confirmed": False, "endpoint_selection_authorized": False, "config": asdict(CONFIG),
            "provenance": dict(provenance, surface_artifact_sha256="f" * 64)}


def value(index=0):
    e = torch.zeros(2, 129, 6, 2)
    e[..., 0] = 1
    return {"schema": "relative-search-input/v1", "object_id": f"{index:064x}",
            "split": "source_train" if index < 13 else "source_validation", "coordinates": torch.linspace(0, 2, 129).repeat(2, 1),
            "geometry": torch.zeros(1, 2, 1, 129, 8), "image_embeddings": e, "observed_embeddings": e[:, 0].clone(),
            "text_direction": torch.tensor([1., 0.]), "provenance": {"geometry_artifact_sha256": "a" * 64,
            "semantic_artifact_sha256": "b" * 64, "source_index_sha256": "c" * 64}}


def test_both_unseen_faces_not_pair_recombination_and_first_event():
    seen = tensor([(0, 0), (1, 1)])
    assert unseen_face_pairs(tensor([(0, 1), (0, 4), (4, 0), (4, 4)]), seen).tolist() == [[4, 4]]
    artifact = contact(value(), {key: "0" * 64 for key in CONTACT_HASH_KEYS})
    assert first_event(artifact, 0) == 10 and first_event(artifact, 1) == 20
    assert first_event(swap_contact(artifact), 0) == 20
    artifact["observed_seen_pairs"] = tensor([(0, 0), (1, 1)])
    assert first_event(artifact, 0) is None


def test_face_subdivision_is_structural_limitation_not_invariance_claim():
    # Same coarse surface patch pair can be seen at a hinge and later distally.
    # Refinement gives the distal regions unseen child IDs without moving a
    # surface: this changes the exclusion decision. This is a combinatorial
    # counterexample, not a backend subdivision accuracy assertion.
    assert len(unseen_face_pairs(tensor([(0, 0)]), tensor([(0, 0)]))) == 0
    assert len(unseen_face_pairs(tensor([(1, 1)]), tensor([(0, 0)]))) == 1


def test_validate_contact_rejects_extra_fields_changed_atlas_and_incomplete():
    v = value()
    provenance = {key: "0" * 64 for key in CONTACT_HASH_KEYS}
    original = contact(v, provenance)
    validate_contact(original, v, provenance)
    for changed in (dict(original, endpoint=0), dict(original, observed_seen_pairs=tensor([]))):
        with pytest.raises(ValueError):
            validate_contact(changed, v, provenance)
    changed = deepcopy(original)
    changed["sides"][0]["enumeration_complete"] = False
    with pytest.raises(ValueError, match="complete"):
        validate_contact(changed, v, provenance)


def test_calibration_module_has_no_endpoint_renderer_or_exporter_imports():
    import ast
    module = ROOT / "src/splart/contact_calibration.py"
    imports = [node.module or "" for node in ast.walk(ast.parse(module.read_text())) if isinstance(node, ast.ImportFrom)]
    assert not any(any(word in name for word in ("renderer", "evaluate", "export")) for name in imports)
    assert len(MIDPOINTS) == 64 and all(q * 64 % 1 == 0.5 for q in MIDPOINTS)


def midpoint_artifact(v, provenance, fail=False):
    queries = [dict(query([]), both_unseen_pairs=[], both_unseen_pair_count=0) for _ in MIDPOINTS]
    if fail:
        queries[4] = dict(query([[3, 3]]), both_unseen_pairs=[[3, 3]], both_unseen_pair_count=1)
    return {"schema": "contact-midpoint-measurement/v1", "object_id": v["object_id"], "split": v["split"],
            "midpoints": list(MIDPOINTS), "queries": queries, "midpoint_new_event_count": int(fail),
            "calibration_passed": not fail, "enumeration_complete": True, "atlas_updated": False,
            "exact_swap_passed": True, "provenance": dict(provenance, contact_artifact_sha256="d" * 64)}


@pytest.mark.parametrize("fail", [False, True])
def test_midpoint_falsification_and_malformed_rejection(fail):
    v, provenance = value(), {"test": "0" * 64}
    artifact = midpoint_artifact(v, provenance, fail)
    row = {key: artifact[key] for key in ("object_id", "split", "midpoint_new_event_count", "calibration_passed", "enumeration_complete", "exact_swap_passed")}
    row["contact_artifact_sha256"] = "d" * 64
    cache = contact(v, {key: "0" * 64 for key in CONTACT_HASH_KEYS})
    validate_midpoints(artifact, row, cache, provenance)
    changed = deepcopy(artifact)
    changed["queries"][0]["backend_unsigned_distance"] = float("nan")
    with pytest.raises(ValueError, match="clearance"):
        validate_midpoints(changed, row, cache, provenance)
    changed = deepcopy(artifact)
    changed["queries"][0]["triangle_pairs"] = [[0.1, 0]]
    with pytest.raises(ValueError, match="integer"):
        validate_midpoints(changed, row, cache, provenance)
    changed = dict(artifact, atlas_updated=True)
    with pytest.raises(ValueError, match="mutated atlas"):
        validate_midpoints(changed, row, cache, provenance)


def build_cache(root):
    base_dir, contact_dir = root / "base", root / "contacts"
    base_dir.mkdir()
    contact_dir.mkdir()
    values, base_rows = [value(i) for i in range(19)], []
    for i, v in enumerate(values):
        path = base_dir / f"{i}.pt"
        torch.save(v, path)
        base_rows.append({"object_id": v["object_id"], "split": v["split"], "artifact": path.name, "artifact_sha256": file_sha256(path)})
    base_index = base_dir / "index.json"
    write_json_new(base_index, {"schema": "relative-search-input-index/v1", "rows": base_rows,
                               "source_index_sha256": "c" * 64, "diagnostic": "synthetic"})
    write_json_new(contact_dir / "backend_receipt.json", {"synthetic": True})
    provenance = {"surface_index_sha256": "e" * 64, "base_input_index_sha256": file_sha256(base_index),
                  "engine_sha256": file_sha256(ROOT / "src/splart/surface_contact.py"),
                  "consumer_sha256": file_sha256(ROOT / "src/splart/observed_surfaces.py"),
                  "runner_sha256": file_sha256(ROOT / "run_surface_contact.py"),
                  "backend_receipt_sha256": file_sha256(contact_dir / "backend_receipt.json")}
    rows = []
    for i, v in enumerate(values):
        path = contact_dir / f"contact-{i:03d}.pt"
        torch.save(contact(v, provenance), path)
        rows.append({"object_id": v["object_id"], "split": v["split"], "artifact": path.name, "artifact_sha256": file_sha256(path)})
    counts = {"source_train": 13, "source_validation": 6}
    write_json_new(contact_dir / "audit.json", {"provenance": provenance, "config": asdict(CONFIG), "split_counts": counts,
                   "exact_swap_passed": True, "all_contact_enumerations_complete": True, "complete_fixed_source_roster": True,
                   "endpoint_labels_read": False})
    write_json_new(contact_dir / "index.json", {"schema": "surface-contact-index/v1", "rows": rows, "provenance": provenance,
                   "split_counts": counts, "audit_sha256": file_sha256(contact_dir / "audit.json")})
    return base_index, contact_dir / "index.json"


def test_strict_full_contact_loader_hash_and_roster(tmp_path):
    base, contacts = build_cache(tmp_path)
    index, loaded = load_contacts(contacts, base)
    assert len(loaded) == 19
    first = contacts.parent / index["rows"][0]["artifact"]
    with first.open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(ValueError, match="hash"):
        load_contacts(contacts, base)


def test_strict_calibration_loader_and_tampered_rows(tmp_path):
    base, contacts = build_cache(tmp_path)
    contact_index, loaded = load_contacts(contacts, base)
    output = tmp_path / "calibration"
    output.mkdir()
    provenance = {"contact_index_sha256": file_sha256(contacts), "base_input_index_sha256": file_sha256(base),
                  "surface_index_sha256": contact_index["provenance"]["surface_index_sha256"],
                  "backend_receipt_sha256": contact_index["provenance"]["backend_receipt_sha256"],
                  "calibrator_sha256": file_sha256(ROOT / "src/splart/contact_calibration.py"),
                  "runner_sha256": file_sha256(ROOT / "run_contact_calibration.py")}
    rows = []
    for i, contact_row in enumerate(contact_index["rows"]):
        artifact = midpoint_artifact(value(i), provenance, fail=i == 0)
        artifact["provenance"]["contact_artifact_sha256"] = contact_row["artifact_sha256"]
        path = output / f"midpoint-{i:03d}.json"
        write_json_new(path, artifact)
        row = {key: artifact[key] for key in ("object_id", "split", "midpoint_new_event_count", "calibration_passed", "enumeration_complete", "exact_swap_passed")}
        row.update(contact_artifact_sha256=contact_row["artifact_sha256"], midpoint_artifact=path.name, midpoint_artifact_sha256=file_sha256(path))
        rows.append(row)
    write_json_new(output / "audit.json", {"provenance": provenance, "rows": rows, "endpoint_labels_read": False, "atlas_updated": False})
    write_json_new(output / "index.json", {"schema": "contact-calibration-index/v1", "rows": rows, "provenance": provenance,
                   "split_counts": {"source_train": 13, "source_validation": 6}, "audit_sha256": file_sha256(output / "audit.json")})
    _, result = load_calibration(output / "index.json", contacts, base)
    assert len(result) == 19 and sum(row["calibration_passed"] for row in result.values()) == 18
    with (output / rows[0]["midpoint_artifact"]).open("a") as stream:
        stream.write(" ")
    with pytest.raises(ValueError, match="hash"):
        load_calibration(output / "index.json", contacts, base)


def test_backend_midpoint_measurement_keeps_atlas_and_exact_swap():
    from test_surface_contact import fixture
    from splart.surface_contact import measure_surface
    from splart.observed_surfaces import swap_surface
    surface = fixture()
    cache = measure_surface(surface)
    atlas = cache["observed_seen_pairs"].clone()
    measured = measure_midpoints(surface, cache)
    assert measured["calibration_passed"]
    assert torch.equal(atlas, cache["observed_seen_pairs"])
    assert_midpoint_swap(measured, measure_midpoints(swap_surface(surface), swap_contact(cache)))
    assert len(measured["queries"]) == 64 and measured["atlas_updated"] is False
