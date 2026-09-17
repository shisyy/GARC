from dataclasses import replace

import pytest
import torch

import run_se_mdc_njc_gate as njc_gate
import run_se_mdc_source_gate as source_gate
from run_glpdt_source_gate import CONFIG
from splart.se_mdc import SEMDCHead


class Float32BatchShapeDriftHead(SEMDCHead):
    """Mimic batch-shape-only float32 kernel drift, not structural coupling."""

    def forward(self, features, coordinates, anchor_distance):
        output = super().forward(features, coordinates, anchor_distance)
        if features.dtype == torch.float32:
            drift = features.new_tensor(features.shape[0] * 2.0e-5)
            output = replace(output, distance=output.distance + drift)
        return output


class TrueCrossBatchCouplingHead(SEMDCHead):
    """Deliberately leak the current batch mean into every prediction."""

    def forward(self, features, coordinates, anchor_distance):
        output = super().forward(features, coordinates, anchor_distance)
        leaked = 0.1 * features.mean()
        return replace(output, distance=output.distance + leaked)


def fixture(batch: int = 8):
    generator = torch.Generator().manual_seed(817)
    features = torch.randn(batch, 2, 3, 9, 9, generator=generator)
    coordinates = torch.linspace(0.05, 1.8, 9).view(1, 1, 1, -1).expand(
        batch, 2, 3, -1
    )
    anchor = torch.rand(batch, 2, generator=generator) + 0.2
    target = torch.rand(batch, 2, generator=generator) + 0.2
    objects = tuple("a" if index < batch // 2 else "b" for index in range(batch))
    return features, coordinates, anchor, target, objects


def test_frozen_training_contract_and_control_matrix_model_counts():
    assert (CONFIG.seed, CONFIG.steps) == (2202, 4000)
    coupled = source_gate.make_model("se_mdc")
    independent = source_gate.make_model("se_mdc_independent")
    assert sum(p.numel() for p in coupled.parameters()) == 98563
    assert sum(p.numel() for p in independent.parameters()) == 98563
    assert source_gate.ARCHITECTURES == ("cr_fpl", "se_mdc", "se_mdc_independent")
    assert source_gate.LOSSES == ("original", "minimax")


def test_tail_metrics_include_legacy_sided_log_object_macro_and_signed_views():
    _, _, _, target, objects = fixture()
    prediction = target * torch.tensor([[0.5, 2.0]])
    metrics = source_gate.tail_metrics(prediction, target, objects, "candidate")
    required = {
        "candidate_target_relative_p99_error",
        "candidate_target_relative_p99_per_side",
        "candidate_target_relative_p99_max_side",
        "candidate_absolute_log_ratio_q99_per_side",
        "candidate_absolute_log_ratio_max",
        "candidate_object_macro_target_relative_p99_per_side",
        "candidate_object_macro_absolute_log_ratio_q99_per_side",
        "candidate_signed_relative_q01_per_side",
        "candidate_signed_relative_q99_per_side",
    }
    assert required <= set(metrics)
    assert metrics["candidate_target_relative_p99_per_side"] == [0.5, 1.0]


def test_reports_strict_invariance_dual_risk_and_object_macro_metrics():
    model = SEMDCHead().eval()
    features, coordinates, anchor, target, objects = fixture()
    with torch.no_grad():
        output = model(features, coordinates, anchor)
        invariance = source_gate.invariance_diagnostics(
            model, features, coordinates, anchor
        )
        metrics = source_gate.summarize_metrics(
            output, anchor, target, objects
        )
        njc_gate.add_object_macro_metrics(metrics, output, anchor, target, objects)
    assert max(invariance.values()) <= 1e-6
    assert "heldout_harder_side_allocation_argmax_hit_rate" in metrics
    assert "heldout_allocation_error_pearson" in metrics
    assert "candidate_object_macro_endpoint_nmae" in metrics
    assert len(metrics["candidate_object_macro_side_nmae"]) == 2


def test_paired_object_wins_are_aggregated_not_episode_counted():
    _, _, _, target, objects = fixture()
    candidate = target.clone()
    reference = target + 1.0
    wins = source_gate.paired_object_wins(candidate, reference, target, objects)
    assert wins == {"candidate": 2, "reference": 0, "ties": 0}


def test_frozen_node816_report_and_njc_checkpoint_hashes_are_bound():
    assert source_gate.SOURCE_CR_FPL_REPORT_SHA256 == (
        "647f846262e9a46d0882c21f91a259bb94e5afec98370cd224f90355c5092fc7"
    )
    assert source_gate.SOURCE_CR_FPL_CHECKPOINT_SHA256 == (
        "7d81987c6aad190922d95d183ab823811db76afb2670adc945fd9dfba84fb3d7"
    )
    assert njc_gate.NJC_CR_FPL_REPORT_SHA256 == (
        "2e07f767c6deba1da83c766c5d7aeb6ae09f69fb4fc91779188a15239f7f9c55"
    )
    assert njc_gate.NJC_CR_FPL_CHECKPOINT_SHA256 == (
        "22d38b52aa77766d775d36c773c664f7690c5807945d2aa3c4ecf2326bd0b4c5"
    )


def test_float32_batch_shape_drift_does_not_false_fail_double_structural_audit():
    model = Float32BatchShapeDriftHead().eval()
    features, coordinates, anchor, _, _ = fixture(4)
    with torch.no_grad():
        batched = model(features, coordinates, anchor).distance
        singles = torch.cat(
            [
                model(
                    features[index : index + 1],
                    coordinates[index : index + 1],
                    anchor[index : index + 1],
                ).distance
                for index in range(4)
            ],
            dim=0,
        )
        assert float((batched - singles).abs().max()) > 1e-6
        diagnostics = source_gate.invariance_diagnostics(
            model, features, coordinates, anchor
        )
    assert max(diagnostics.values()) <= 1e-6


def test_true_cross_batch_coupling_still_fails_double_structural_audit():
    model = TrueCrossBatchCouplingHead().eval()
    features, coordinates, anchor, _, _ = fixture(4)
    with pytest.raises(ValueError, match="strict invariance check failed"):
        source_gate.invariance_diagnostics(model, features, coordinates, anchor)
