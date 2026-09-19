import json
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from splart.geometry_first import MODES, fit_positive_hinge, geometry_first_predict, geometry_proposals, semantic_role
from splart.relative_search import METHODS, opaque_object_id, predict, swap_input


def fixture(name="a", knots=(14, 16, 18)):
    d = torch.linspace(0, 2, 41).repeat(2, 1)
    scores = torch.stack((0.1 + d[0] * 0.3, -0.1 - d[1] * 0.3))[:, :, None].expand(2, 41, 6)
    e = torch.stack((scores, (1 - scores.square()).sqrt()), -1)
    g = torch.zeros(2, 2, len(knots), 41, 8)
    for radius, knot in enumerate(knots):
        g[:, :, radius, :, 2] = 0.04 * radius + (0.3 + radius) * (d - d[:, knot:knot + 1]).clamp_min(0)
    g[..., 7] = (d[:, None] - 0.4).square()
    return {"schema": "relative-search-input/v1", "object_id": opaque_object_id(name), "split": "source_train",
            "coordinates": d, "geometry": g, "image_embeddings": e, "observed_embeddings": e[:, 0].clone(),
            "text_direction": torch.tensor([1., 0.]),
            "provenance": {"source_index_sha256": "0" * 64, "semantic_artifact_sha256": "1" * 64,
                           "geometry_artifact_sha256": "2" * 64}}


def replace_scores(value, scores):
    value["image_embeddings"] = torch.stack((scores, (1 - scores.square()).sqrt()), -1)
    value["observed_embeddings"] = value["image_embeddings"][:, 0].clone()


@pytest.mark.parametrize("curve", [lambda x: torch.zeros_like(x), lambda x: 0.3 + 0.2 * x,
                                  lambda x: 0.3 - 0.2 * x, lambda x: -(x - 0.8).clamp_min(0)])
def test_flat_affine_and_negative_hinges_reject(curve):
    d = torch.linspace(0, 2, 41, dtype=torch.float64)
    assert not fit_positive_hinge(d, curve(d))["accepted"]


def test_known_positive_hinge_and_scale_normalization():
    d = torch.linspace(0, 2, 41, dtype=torch.float64)
    for scale in (0.3, 1.0, 100.0):
        result = fit_positive_hinge(d, scale * (0.5 + 0.1 * d + (d - d[16]).clamp_min(0)))
        assert result["accepted"] and result["knot_index"] == 16
        assert result["hinge_bic"] < result["affine_bic"]


def test_deduplication_never_double_votes_exact_origins_or_curves():
    value = fixture()
    original = geometry_proposals(value)
    assert original[0]["unique_origin_count"] == 1
    assert original[0]["unique_curve_count"] == 3
    value["geometry"] = value["geometry"].repeat(3, 1, 1, 1, 1)
    value["geometry"] = torch.cat((value["geometry"], value["geometry"][:, :, :1]), dim=2)
    assert original == geometry_proposals(value)


def test_strict_majority_and_compactness_required():
    value = fixture()
    value["geometry"][:, :, 1, :, 2] = 0.1
    value["geometry"][:, :, 2, :, 2] = 0.2
    proposal = geometry_proposals(value)[0]
    assert proposal["supporting_curve_count"] == 1
    assert proposal["unique_curve_count"] == 3
    assert not proposal["geometric_confirmed"]
    value = fixture(knots=(4, 16, 36))
    proposal = geometry_proposals(value)[0]
    assert proposal["strict_majority"] and not proposal["compact_spread"]
    assert not proposal["geometric_confirmed"]


def test_geometry_medoid_then_semantics_changes_only_within_proposals():
    value = fixture()
    geometric = geometry_first_predict(value, "geometry_only")
    joint = geometry_first_predict(value, "joint")
    for side in range(2):
        assert geometric["sides"][side]["geometry_first"]["selected_index"] == 16
        diagnostic = joint["sides"][side]["geometry_first"]
        assert diagnostic["selected_index"] == 18
        assert diagnostic["selected_index"] in diagnostic["candidate_indices"]
        assert diagnostic["semantic_verified"] and diagnostic["selection_changed"]
        # Unconstrained semantic progress continues beyond all proposals.
        assert joint["distance"][side] < float(value["coordinates"][side, -1])
    assert not joint["sides"][0]["abstain"]
    assert joint["sides"][1]["abstain"]
    assert "opening_limit_not_identifiable_from_closure_semantics" in joint["sides"][1]["reasons"]


def test_no_geometry_retains_only_unconfirmed_geometry_fallback():
    value = fixture()
    value["geometry"][..., 2] = 0
    expected = predict(value, "geometry_only")
    actual = geometry_first_predict(value, "joint")
    assert actual["distance"] == expected["distance"]
    for side in actual["sides"]:
        assert side["abstain"]
        assert "no_stable_geometric_boundary" in side["reasons"]
        assert side["evidence_interval"] == [0., 2.]
        assert side["geometry_first"]["role"] == "not_evaluated"
        assert not side["geometry_first"]["semantic_activation"]


@pytest.mark.parametrize("votes", [[1, 1, 1, -1, -1, -1], [1, 1, -1, -1, -1, 0]])
def test_tied_or_zero_vote_ambiguous_role_rejects(votes):
    value = fixture()
    scores = value["image_embeddings"][..., 0] * torch.tensor(votes)
    replace_scores(value, scores)
    assert semantic_role(value, 0)["role"] == "unknown"
    out = geometry_first_predict(value, "joint")
    geometric = geometry_first_predict(value, "geometry_only")
    assert out["distance"] == geometric["distance"]
    assert all(s["abstain"] and not s["geometry_first"]["semantic_activation"] for s in out["sides"])


def test_weak_direction_and_failed_candidate_verification_explicit():
    value = fixture()
    scores = value["image_embeddings"][..., 0].clone()
    scores[:, 0] *= 0.001
    replace_scores(value, scores)
    out = geometry_first_predict(value, "joint")
    assert all("weak_observed_semantic_direction" in s["reasons"] for s in out["sides"])
    value = fixture()
    scores = value["image_embeddings"][..., 0].clone()
    scores[:, 1:] = 0
    replace_scores(value, scores)
    out = geometry_first_predict(value, "joint")
    assert all("semantic_verification_failed" in s["reasons"] for s in out["sides"])
    assert out["distance"] == geometry_first_predict(value, "geometry_only")["distance"]


def test_zero_is_not_a_negative_vote_and_exact_margin_ties_use_medoid():
    value = fixture()
    scores = value["image_embeddings"][..., 0].clone() * torch.tensor([1., -1., -1., -1., -1., 0.])
    replace_scores(value, scores)
    role = semantic_role(value, 0)
    assert role["negative_votes"] == 4 and role["zero_votes"] == 1
    assert role["role"] == "opening_like"
    value = fixture()
    scores = value["image_embeddings"][..., 0].clone()
    scores[0, 1:] = 0.5
    scores[1, 1:] = -0.5
    replace_scores(value, scores)
    out = geometry_first_predict(value, "joint")
    assert all(s["geometry_first"]["selected_index"] == 16 for s in out["sides"])
    assert all(s["geometry_first"]["semantic_verified"] for s in out["sides"])
    assert all(not s["geometry_first"]["selection_changed"] for s in out["sides"])


def test_candidates_must_progress_beyond_both_not_just_own_anchor():
    value = fixture()
    scores = value["image_embeddings"][..., 0].clone()
    # Route closure from another object onto this opening side. A score of zero
    # is beyond its own -0.1 anchor, but not beyond the opposite +0.1 anchor.
    scores[1, 1:] = 0
    replace_scores(value, scores)
    donor = swap_input(fixture("b"))
    out = geometry_first_predict(value, "joint", "shuffled_verification", route_donor=donor)
    assert out["sides"][1]["geometry_first"]["role"] == "closure_like"
    assert not out["sides"][1]["geometry_first"]["semantic_verified"]


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("method", METHODS)
def test_exact_side_swap_repeat_and_finite_serialization(mode, method):
    value, donor, route = fixture(), fixture("b"), fixture("c")
    original = geometry_first_predict(value, method, mode, donor, route)
    assert original == geometry_first_predict(value, method, mode, donor, route)
    swapped = geometry_first_predict(swap_input(value), method, mode, swap_input(donor), swap_input(route))
    assert original["sides"] == list(reversed(swapped["sides"]))
    assert original["distance"] == list(reversed(swapped["distance"]))
    json.dumps(original, allow_nan=False)
    if mode == "raw":
        assert original == predict(value, method, donor)
    if method in ("coordinate_only", "semantic_only"):
        assert original == predict(value, method)


def test_donor_geometry_never_creates_proposals_and_controls_are_active():
    value, donor = fixture(), fixture("b", knots=(4, 5, 6))
    scores = donor["image_embeddings"][..., 0].clone()
    scores[:, 1:] = 0
    replace_scores(donor, scores)
    own = geometry_first_predict(value, "joint")
    control = geometry_first_predict(value, "joint", "donor_semantics", semantic_donor=donor)
    assert own["distance"] != control["distance"]
    for side in range(2):
        assert control["sides"][side]["geometry_first"]["candidate_indices"] == [14, 16, 18]
    donor["geometry"].zero_()
    assert control == geometry_first_predict(value, "joint", "donor_semantics", semantic_donor=donor)


def test_fail_closed_labels_donor_and_grid():
    value, donor = fixture(), fixture("b")
    with pytest.raises(ValueError, match="whitelist"):
        geometry_first_predict(dict(value, endpoint="forbidden"), "joint")
    with pytest.raises(ValueError, match="distinct"):
        geometry_first_predict(value, "joint", "donor_semantics", semantic_donor=value)
    donor["coordinates"] *= 2
    with pytest.raises(ValueError, match="grid"):
        geometry_first_predict(value, "joint", "donor_semantics", semantic_donor=donor)
