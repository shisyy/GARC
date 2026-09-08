import copy
import inspect
import json
from pathlib import Path

import pytest
import torch

from scripts.run_rza_acwt_label_free import (FROZEN_CONFIG_SHA256, canonical_sha256,
                                               run_cell, verify_provenance_binding)
from splart.rza_acwt import (_canonical_sha, _fit_model, _sha_tensor,
                              canonical_object_scalar, empirical_rank,
                              feature_crossfit_rza_acwt, global_row_identity,
                              preserve_recipient_displacement, receipt_passes,
                              rza_acwt_ablation)


def contract():
    return {
        "rank_relative_tolerance": 1e-12, "condition_max": 1e6,
        "eigengap_multiplier": 100., "crossfit_folds": 5,
        "crossfit_absolute_spearman_at_most": 1.,
        "crossfit_distance_correlation_at_most": 1.,
        "normalized_residual_mean_max": 1e-10,
        "normalized_residual_raw_z_correlation_max": 1e-10,
        "recipient_u_max_error": 1e-12, "reconstruction_max_error": 1e-10,
        "orthogonality_max": 1e-10, "held_boundary_clipped_fraction_max": 1.,
        "residual_energy_ratio_at_least": 0.,
        "shuffle_rms_over_original_sd_at_least": 0.,
        "shuffled_norm_p99_ratio_at_most": 100., "train_coverage": 1.,
        "held_coverage": 1., "train_self_rate": 0.,
        "held_effective_donors_per_object": 1.,
    }


def _analytic_vector(object_index, gauge, width, offset):
    return torch.tensor([
        torch.sin(torch.tensor((object_index + 1) * (column + 1) * .173
                               + gauge * .019 + offset)).item()
        + .17 * torch.cos(torch.tensor((object_index + 3) * (column + 2) * .071)).item()
        for column in range(width)], dtype=torch.float64)


def fixture():
    rows, train, held, train_field, held_field, train_z, held_z = [], [], [], [], [], {}, {}
    for domain_index, domain in enumerate(("articraft", "njc")):
        offset = domain_index * .37
        for split, count, start in (("train", 20, 0), ("held", 5, 30)):
            for local in range(count):
                object_index = start + local
                obj = f"{domain}-{split}-{local:02d}"
                z = .31 + .041 * object_index + offset
                (train_z if split == "train" else held_z)[obj] = z
                # Strong one-direction conditional covariance plus nonzero bulk.
                x = -1 + 2 * local / max(count - 1, 1)
                location = torch.tensor([1 + z + .2*x*x, 2 - .5*z + .1*x*x,
                                         -.3 + .7*z, .4 - .2*z + .15*x*x,
                                         .8 + .1*z, -.6 + .2*z], dtype=torch.float64)
                axis = torch.tensor([1., .2, -.1, .05, .03, -.02], dtype=torch.float64)
                axis = axis / axis.norm()
                coefficient = (1 + .9*x + .8*x*x) * (1 if local % 2 else -1)
                bulk = _analytic_vector(object_index, 0, 6, offset)
                bulk = bulk - (bulk @ axis) * axis
                mean = location + coefficient * axis + .19 * bulk
                for gauge in range(3):
                    semantic = _analytic_vector(object_index, gauge, 24, offset)
                    mechanical = torch.tensor([
                        (object_index + 1) ** ((column % 3) + 1) * 1e-3
                        + (column + 1) * gauge * .013 + offset
                        for column in range(8)] + [z], dtype=torch.float64)
                    index = len(rows)
                    destination = train if split == "train" else held
                    destination.append(index)
                    # Deliberately collide local gauge IDs across domains.
                    rows.append({"domain": domain, "object_group_id": obj, "joint_id": obj,
                                 "gauge_id": f"{split}-{local:02d}-g{gauge}",
                                 "semantic": semantic, "mechanical": mechanical,
                                 "observed_displacement": z + (gauge - 1) * 1e-15})
                    field = mean + torch.tensor([gauge, -gauge, .5*gauge, -.25*gauge,
                                                 .1*gauge, -.05*gauge], dtype=torch.float64) * .01
                    (train_field if split == "train" else held_field).append(field)
    return (rows, train, held, torch.stack(train_field), torch.stack(held_field),
            train_z, held_z)


def run_fixture(args=None, context="semantic_rza_acwt/final-source-features"):
    args = fixture() if args is None else args
    crossfit = feature_crossfit_rza_acwt(args[0], args[1], "semantic", contract())
    return rza_acwt_ablation(*args, contract(), context, crossfit)


def test_location_raw_z_anchor_axis_scales_and_exact_reconstruction():
    z = torch.linspace(-1.3, 1.7, 25, dtype=torch.float64)
    x, _, _ = empirical_rank(z)
    axis = torch.tensor([1., .2, -.1, .03], dtype=torch.float64)
    axis /= axis.norm()
    rows = []
    for index, (zi, xi) in enumerate(zip(z, x)):
        location = torch.tensor([1 + zi + xi*xi, 2 - .2*zi, -.4 + .3*zi, .1 - .7*zi])
        bulk = torch.tensor([0., torch.sin(zi*3), torch.cos(zi*2), torch.sin(zi*5)])
        bulk -= (bulk @ axis) * axis
        rows.append(location + (1 + xi + xi*xi) * (-1 if index % 2 else 1) * axis + .2*bulk)
    model = _fit_model(z, x, torch.stack(rows), contract())
    assert model["reconstruction_max_error"] < 1e-10
    assert model["normalized_residual_mean_max"] < 1e-10
    assert model["normalized_residual_raw_z_correlation_max"] < 1e-10
    assert model["q_constant_orthogonality"] < 1e-12
    assert model["q_raw_z_orthogonality"] < 1e-12
    assert model["eigengap"] > model["eigengap_threshold"]
    assert abs(float(model["axis"].norm()) - 1) < 1e-12
    assert model["axis"][model["axis_pivot"]] >= 0
    assert model["radial_scale_min"] > 0 and model["parallel_scale_min"] > 0
    assert model["perpendicular_scale_min"] > 0


def test_global_row_identity_accepts_cross_domain_local_collision_and_rejects_global_collision():
    rows, train, held, train_field, held_field, train_z, held_z = fixture()
    assert len({rows[i]["gauge_id"] for i in train}) < len(train)
    assert len({global_row_identity(rows[i]) for i in train}) == len(train)
    bad = copy.deepcopy(rows)
    bad[train[1]]["gauge_id"] = bad[train[0]]["gauge_id"]
    bad[train[1]]["domain"] = bad[train[0]]["domain"]
    with pytest.raises(ValueError, match="global row identity"):
        rza_acwt_ablation(bad, train, held, train_field, held_field, train_z, held_z,
                          contract(), "semantic_rza_acwt/final-source-features",
                          feature_crossfit_rza_acwt(rows, train, "semantic", contract()))


def test_near_tie_reductions_reverse_order_exact_hashes_stats_predictions_and_receipt():
    args = fixture()
    rows, train, held, train_field, held_field, *_ = args
    forward_z = canonical_object_scalar(rows, train, "semantic")
    reverse_z = canonical_object_scalar(rows, list(reversed(train)), "semantic")
    assert forward_z == reverse_z
    first = run_fixture(args)
    reversed_args = (rows, list(reversed(train)), list(reversed(held)), train_field.flip(0),
                     held_field.flip(0), args[5], args[6])
    second = run_fixture(reversed_args)
    first_train = {global_row_identity(rows[i]): first[0][k] for k, i in enumerate(train)}
    second_train = {global_row_identity(rows[i]): second[0][k]
                    for k, i in enumerate(reversed(train))}
    first_held = {global_row_identity(rows[i]): first[1][k] for k, i in enumerate(held)}
    second_held = {global_row_identity(rows[i]): second[1][k]
                   for k, i in enumerate(reversed(held))}
    assert all(torch.equal(first_train[key], second_train[key]) for key in first_train)
    assert all(torch.equal(first_held[key], second_held[key]) for key in first_held)
    assert first[2] == second[2]
    assert _canonical_sha(first[2]) == _canonical_sha(second[2])


def test_crossfit_is_fold_local_complete_and_semantic_returns_to_raw_coordinates():
    rows, train, *_ = fixture()
    first = feature_crossfit_rza_acwt(rows, train, "semantic", contract())
    second = feature_crossfit_rza_acwt(rows, list(reversed(train)), "semantic", contract())
    assert first == second
    for domain, value in first.items():
        assert torch.tensor(value["audit_payload"]["standardized"]).shape == (20, 24)
        assert len(value["folds"]) == 5
        assert all(fold["model_provenance"]["eigengap"] >
                   fold["model_provenance"]["eigengap_threshold"] for fold in value["folds"])
        assert all(fold["joint_domain_fit_object_hash"] == first["articraft"]["folds"][i]["joint_domain_fit_object_hash"]
                   for i, fold in enumerate(value["folds"]))


def test_mechanical_crossfit_zero_fills_fold_dropped_columns_and_records_joint_domain_fit():
    rows, train, *_ = fixture()
    first = feature_crossfit_rza_acwt(rows, train, "mechanical", contract())
    assignments = {domain: value["audit_payload"]["fold_assignment"]
                   for domain, value in first.items()}
    changed = copy.deepcopy(rows)
    for row in changed:
        if row["object_group_id"] not in assignments[row["domain"]]:
            continue
        fold = assignments[row["domain"]][row["object_group_id"]]
        row["mechanical"][0] = (float(int(row["object_group_id"].split("-")[-1]) + 1)
                                if fold == 0 else 0.)
        row["mechanical"][1] = 1.
    result = feature_crossfit_rza_acwt(changed, train, "mechanical", contract())
    for domain, value in result.items():
        standardized = torch.tensor(value["audit_payload"]["standardized"], dtype=torch.float64)
        assert standardized.shape == (20, 8) and torch.isfinite(standardized).all()
        masks = [fold["mechanical_keep_mask"] for fold in value["folds"]]
        assert masks[0][0] is False and any(mask[0] for mask in masks[1:])
        assert all(mask[1] is False and mask[-1] is True for mask in masks)
        for index, obj in enumerate(value["audit_payload"]["object_ids"]):
            if value["audit_payload"]["fold_assignment"][obj] == 0:
                assert standardized[index, 0] == 0
        assert all(fold["joint_domain_fit_object_hash"] ==
                   result["articraft"]["folds"][index]["joint_domain_fit_object_hash"]
                   for index, fold in enumerate(value["folds"]))


def test_double_recompute_receipt_mutation_and_alias_are_rejected():
    receipt = run_fixture()[2]
    expected = run_fixture()[2]
    context = "semantic_rza_acwt/final-source-features"
    assert receipt_passes(receipt, expected, contract(), context, {"articraft", "njc"})
    assert not receipt_passes(receipt, receipt, contract(), context, {"articraft", "njc"})
    bad = copy.deepcopy(receipt)
    bad["domains"]["articraft"]["model_provenance"]["axis_sha256"] = "0" * 64
    assert not receipt_passes(bad, expected, contract(), context, {"articraft", "njc"})
    bad = copy.deepcopy(receipt)
    bad["domains"]["njc"]["crossfit"]["folds"][0]["preprocessor_sha256"] = "f" * 64
    assert not receipt_passes(bad, expected, contract(), context, {"articraft", "njc"})


def test_run_cell_separate_and_gate_recompute_are_bitwise_and_mechanical_d_is_exact():
    args = fixture()
    rows, train, held, train_field, held_field, train_z, held_z = args
    cell = run_cell(rows, train, held, train_field, held_field, train_z, held_z,
                    contract(), "semantic_rza_acwt/final-source-features", "semantic")
    assert cell[4]["repeat_bit_identical"] and cell[4]["row_order_invariant"]
    assert cell[4]["candidate_prediction_sha256"] == cell[4]["gate_recompute_prediction_sha256"]
    geometry = torch.arange(12, dtype=torch.float64).reshape(3, 4)
    mechanical = torch.randn(3, 5, dtype=torch.float64)
    result = preserve_recipient_displacement(geometry, mechanical)
    assert torch.equal(result[:, -1], mechanical[:, -1])


def test_score_free_config_runner_and_provenance_contract():
    config = json.loads(Path("configs/rza_acwt_v1.json").read_text())
    assert config["model"]["axis_rank"] == 1 and config["model"]["candidate_search"] is False
    assert not config["source_labels_opened"] and not config["source_labels_hashed"]
    assert not config["source_scores_computed"] and not config["training_started"]
    source = inspect.getsource(__import__("scripts.run_rza_acwt_label_free", fromlist=["main"]))
    assert "labels.pt" not in source and "--labels" not in source and "--score" not in source
    assert "run_rqlsot_feasibility.py" in source and "rqlsot.py" in source
    result = {"schema": "splart-rza-acwt-feasibility/v1", "config_sha256": FROZEN_CONFIG_SHA256,
              "feature_provenance": {"source": "opaque"}, "code_files_sha256": {"runner": "a"},
              "source_labels_opened": False, "source_labels_hashed": False,
              "source_scores_computed": False, "training_started": False,
              "remote_execution_started": False, "box_labels_read": [], "protected_splits_read": []}
    result["feature_provenance_sha256"] = canonical_sha256(result["feature_provenance"])
    result["code_sha256"] = canonical_sha256(result["code_files_sha256"])
    receipt = {"schema": "splart-rza-acwt-feasibility-receipt/v1",
               "config_sha256": FROZEN_CONFIG_SHA256, "feasibility_sha256": "0" * 64,
               "decision": "PRUNE_LABEL_FREE", "code_sha256": result["code_sha256"],
               "feature_provenance_sha256": result["feature_provenance_sha256"],
               "source_labels_opened": False, "source_labels_hashed": False,
               "source_scores_computed": False, "training_started": False,
               "remote_execution_started": False, "box_labels_read": [], "protected_splits_read": []}
    assert verify_provenance_binding(result, receipt)


def test_config_hash_is_frozen():
    data = Path("configs/rza_acwt_v1.json").read_bytes().replace(b"\r\n", b"\n")
    assert __import__("hashlib").sha256(data).hexdigest() == FROZEN_CONFIG_SHA256
