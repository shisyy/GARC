import torch

from run_ald_pdl_source_gate import summarize_metrics
from splart.ald_pdl import ALDPDLHead


def test_source_metrics_include_same_parameter_one_loop_sides_tail_and_representability():
    generator = torch.Generator().manual_seed(815)
    features = torch.randn(6, 2, 3, 7, 9, generator=generator)
    coordinates = torch.randn(6, 2, 3, 7, generator=generator)
    anchor = torch.rand(6, 2, generator=generator) + 0.1
    target = torch.rand(6, 2, generator=generator) + 0.1
    output = ALDPDLHead()(features, coordinates, anchor)
    metrics = summarize_metrics(output, anchor, target)
    assert metrics["endpoint_count"] == 12
    assert metrics["representable_endpoint_count"] == 12
    assert len(metrics["ald_pdl_side_nmae"]) == 2
    assert metrics["ald_pdl_worst_side_nmae"] >= metrics["ald_pdl_mean_side_nmae"]
    assert metrics["one_loop_same_parameters_nmae"] >= 0.0
    assert metrics["ald_pdl_target_relative_p99_error"] >= 0.0
