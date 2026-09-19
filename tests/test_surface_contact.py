"""Backend tests are mandatory on the remote environment, never auto-skipped."""
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from splart.observed_surfaces import rotate_vertices, swap_surface
from splart.surface_contact import (CONFIG, ContactEnumerationOverflow, TrianglePairQuery,
                                   annotate_pairs, assert_exact_swap, measure_surface,
                                   observed_pair_reference, rotation_transform)


def cube(center=(0., 0., 0.)):
    vertices = np.array([(x, y, z) for x in (-1., 1.) for y in (-1., 1.) for z in (-1., 1.)]) + center
    faces = np.array([(0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5),
                      (0, 4, 5), (0, 5, 1), (2, 3, 7), (2, 7, 6),
                      (0, 2, 6), (0, 6, 4), (1, 5, 7), (1, 7, 3)], dtype=np.int32)
    return vertices, faces


def thin_pair():
    return (np.array([[-2., -2., 0.], [2., -2., 0.], [0., 2., 0.]]), np.array([[0, 1, 2]]),
            np.array([[0., -.25, -1.], [0., .25, -1.], [0., 0., 1.]]), np.array([[0, 1, 2]]))


def fixture(offset_z=0.):
    from splart.observed_surfaces import ARTIFACT_HASH_KEYS
    static, sf, mobile, mf = thin_pair()
    axis, pivot = torch.tensor([0., 0., 1.], dtype=torch.float64), torch.zeros(3, dtype=torch.float64)
    first = torch.tensor(mobile + [0., 0., offset_z], dtype=torch.float64)
    second = rotate_vertices(first, axis, pivot, 0.2)
    return {"schema": "observed-surfaces/v1", "object_id": "a" * 64, "split": "source_train",
            "static_vertices": torch.tensor(static), "static_faces": torch.tensor(sf, dtype=torch.int64),
            "mobile_vertices": torch.stack((first, second)), "mobile_faces": torch.tensor(mf, dtype=torch.int64),
            "axis": axis, "pivot": pivot, "angular_increment": 0.2,
            "coordinates": torch.linspace(0, 2, 129).repeat(2, 1),
            "provenance": {key: "0" * 64 for key in ARTIFACT_HASH_KEYS}}


def test_rotation_transform_replays_rigid_motion_without_backend():
    axis = np.array([0., 0., 1.])
    pivot = np.array([1., 2., 3.])
    vertices = np.array([[2., 2., 3.], [1., 4., 3.]])
    rotation, translation = rotation_transform(axis, pivot, np.pi / 2)
    assert np.allclose(vertices @ rotation.T + translation, [[1., 3., 3.], [-1., 2., 3.]], atol=1e-14)
    assert np.array_equal(rotation_transform(axis, pivot, -0.)[0], np.eye(3))


def test_pair_allowance_and_sliding_recombination_are_not_boundaries():
    def row(pairs):
        return {"triangle_pairs": pairs, "enumeration_complete": True}
    reference = observed_pair_reference([row([(0, 0), (1, 1)]), row([(0, 0), (1, 1)])])
    held = annotate_pairs(row([(0, 0)]), reference)
    assert held["novel_pair_count"] == 0 and held["persistent_observed_pair_count"] == 1
    # A sliding patch may meet a new face pair between individually observed
    # faces. Exact-pair novelty must NOT become a physical boundary claim.
    sliding = annotate_pairs(row([(0, 1)]), reference)
    assert sliding["novel_pair_count"] == 1
    assert sliding["observed_region_recombination_count"] == 1
    assert sliding["boundary_confirmed"] is False
    unseen_patch = annotate_pairs(row([(3, 4)]), reference)
    assert unseen_patch["novel_pair_count"] == 1 and not unseen_patch["boundary_confirmed"]


def test_incomplete_reference_fails_closed():
    with pytest.raises(ContactEnumerationOverflow):
        observed_pair_reference([{"triangle_pairs": [], "enumeration_complete": False}])


def test_no_endpoint_or_renderer_imports_and_frozen_sampling():
    import ast
    source = ROOT / "src/splart/surface_contact.py"
    tree = ast.parse(source.read_text())
    imports = [node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert not any("renderer" in module or "evaluate" in module or "export" in module for module in imports)
    assert CONFIG.observed_samples == 65
    assert CONFIG.endpoint_selection_authorized is False
    assert "torch.load" not in source.read_text()


def test_backend_known_separated_to_contact_onset():
    a, af = cube()
    b, bf = cube((3., 0., 0.))
    query = TrianglePairQuery(a, af, b, bf)
    rows = [query.query(translation=[-step, 0., 0.]) for step in np.linspace(0, 2, 9)]
    assert rows[0]["surface_clearance"] == pytest.approx(1.)
    assert all(not row["surface_intersection"] for row in rows[:4])
    first = next(i for i, row in enumerate(rows) if row["surface_intersection"])
    assert first in (4, 5)  # exact touching or first inward discrete step
    assert rows[first]["triangle_pairs"] and rows[first]["enumeration_complete"]


def test_backend_thin_face_crossing_despite_separated_vertices():
    static, sf, mobile, mf = thin_pair()
    assert np.linalg.norm(static[:, None] - mobile[None], axis=-1).min() > 1.
    query = TrianglePairQuery(static, sf, mobile, mf)
    crossing = query.query()
    assert crossing["surface_intersection"] and crossing["triangle_pairs"] == [(0, 0)]
    assert crossing["surface_clearance"] == 0
    separated = query.query(translation=[0., 0., 3.])
    assert not separated["surface_intersection"] and separated["surface_clearance"] > 0


def test_backend_sliding_contact_new_tessellation_pairs_not_new_boundary():
    static = np.array([[-2., -2., 0.], [0., -2., 0.], [2., -2., 0.],
                       [-2., 2., 0.], [0., 2., 0.], [2., 2., 0.]])
    faces = np.array([[0, 1, 3], [1, 4, 3], [1, 2, 4], [2, 5, 4]])
    _, _, mobile, mf = thin_pair()
    query = TrianglePairQuery(static, faces, mobile, mf)
    observed = [query.query(translation=[x, 0, 0]) for x in (-0.75, -0.5)]
    reference = observed_pair_reference(observed)
    rows = [annotate_pairs(query.query(translation=[x, 0, 0]), reference)
            for x in np.linspace(-0.5, 0.75, 11)]
    assert all(row["surface_intersection"] for row in rows)
    assert rows[-1]["novel_pair_count"] > 0
    assert all(not row["boundary_confirmed"] for row in rows)


def test_backend_enumeration_overflow_rejects_not_truncates():
    with pytest.raises(ContactEnumerationOverflow, match="completeness unproven"):
        TrianglePairQuery(*thin_pair(), maximum_contacts=1).query()


def test_backend_vertex_face_reordering_preserves_geometry():
    static, sf = cube()
    mobile, mf = cube((1.5, 0., 0.))
    original = TrianglePairQuery(static, sf, mobile, mf).query()
    permutation = np.array([7, 3, 2, 5, 4, 6, 1, 0])
    inverse = np.argsort(permutation)
    face_permutation = np.arange(len(sf))[::-1]
    reordered = TrianglePairQuery(static[permutation], inverse[sf[face_permutation]],
                                  mobile[permutation], inverse[mf[face_permutation]]).query()
    assert reordered["surface_intersection"] == original["surface_intersection"]
    assert reordered["surface_clearance"] == original["surface_clearance"]
    restored_pairs = {(int(face_permutation[a]), int(face_permutation[b])) for a, b in reordered["triangle_pairs"]}
    assert restored_pairs == set(original["triangle_pairs"])


def subdivide(vertices, faces):
    result, output = vertices.tolist(), []
    for a, b, c in faces:
        ia, ib, ic = len(result), len(result) + 1, len(result) + 2
        result.extend(((vertices[a] + vertices[b]) / 2, (vertices[b] + vertices[c]) / 2,
                       (vertices[c] + vertices[a]) / 2))
        output.extend(((a, ia, ic), (ia, b, ib), (ic, ib, c), (ia, ib, ic)))
    return np.array(result), np.array(output)


def test_backend_subdivision_consistency_within_grid_resolution():
    static, sf = cube()
    mobile, mf = cube((3., 0., 0.))
    fine_static, fine_sf = subdivide(static, sf)
    fine_mobile, fine_mf = subdivide(mobile, mf)
    original = TrianglePairQuery(static, sf, mobile, mf)
    refined = TrianglePairQuery(fine_static, fine_sf, fine_mobile, fine_mf)
    coarse_hits, refined_hits = [], []
    for index, step in enumerate(np.linspace(0, 2, 17)):
        first, second = original.query(translation=[-step, 0, 0]), refined.query(translation=[-step, 0, 0])
        assert first["surface_clearance"] == pytest.approx(second["surface_clearance"], abs=1e-12)
        if first["surface_intersection"]:
            coarse_hits.append(index)
        if second["surface_intersection"]:
            refined_hits.append(index)
    assert abs(coarse_hits[0] - refined_hits[0]) <= 1


def test_backend_persistent_hinge_contacts_never_new_boundary_and_exact_swap():
    value = fixture()
    original = measure_surface(value)
    swapped = measure_surface(swap_surface(value))
    assert_exact_swap(original, swapped)
    assert original["observed_intersection_count"] == 65
    assert len(original["persistent_all_observed_pairs"]) == 1
    assert all(int(side["novel_pair_count"].sum()) == 0 for side in original["sides"])
    assert original["observed_overlapping_or_touching"]
    assert original["boundary_confirmed"] is False


def test_backend_no_contact_abstains_and_exact_swap():
    value = fixture(offset_z=3.)
    original = measure_surface(value)
    assert_exact_swap(original, measure_surface(swap_surface(value)))
    assert original["observed_intersection_count"] == 0
    assert all(not side["surface_intersection"].any() for side in original["sides"])
    assert original["boundary_confirmed"] is False
    assert original["endpoint_selection_authorized"] is False
