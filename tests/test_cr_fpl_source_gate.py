import torch

from run_cr_fpl_source_gate import CONFIG, normalize_source_for_requery, summarize_metrics
from splart.cr_fpl import CRFPLHead


def test_requery_normalization_preserves_physical_coordinates():
    assert (CONFIG.seed, CONFIG.steps) == (2202, 4000)
    features = torch.randn(2, 2, 3, 7, 9)
    coordinates = torch.rand(2, 2, 3, 7)
    anchor = torch.rand(2, 2) + 0.1
    target = torch.rand(2, 2) + 0.1
    data = {
        "source_train": (features, coordinates, anchor, target, ("a", "b")),
        "source_validation": (features + 1, coordinates, anchor, target, ("c", "d")),
    }
    normalized = normalize_source_for_requery(data)
    assert torch.equal(normalized["source_train"][1], coordinates)
    assert not torch.equal(normalized["source_train"][0], features)


def test_source_metrics_cover_one_loop_sides_tail_and_loop_diagnostics():
    generator = torch.Generator().manual_seed(816)
    features = torch.randn(6, 2, 3, 7, 9, generator=generator)
    coordinates = torch.linspace(0.05, 2.0, 7).view(1, 1, 1, 7).expand(6, 2, 3, -1)
    anchor = torch.rand(6, 2, generator=generator) + 0.1
    target = torch.rand(6, 2, generator=generator) + 0.1
    output = CRFPLHead()(features, coordinates, anchor)
    metrics = summarize_metrics(output, anchor, target)
    assert len(metrics["cr_fpl_side_nmae"]) == 2
    assert metrics["cr_fpl_worst_side_nmae"] >= metrics["cr_fpl_mean_side_nmae"]
    assert metrics["one_loop_same_parameters_nmae"] >= 0.0
    assert metrics["cr_fpl_target_relative_p99_error"] >= 0.0
    assert len(metrics["loop_fixed_point_residual_mean"]) == 4
    assert len(metrics["loop_query_residual_l2_mean"]) == 4
