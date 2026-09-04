from types import SimpleNamespace

import torch

from splart.pcfg_certificate import deterministic_sample_indices, pcfg_certificate


def _model(*, symmetric: bool = False) -> SimpleNamespace:
    ys = torch.tensor([-0.6, -0.2, 0.2, 0.6], dtype=torch.float64)
    lower_static = torch.stack((torch.zeros(4, dtype=torch.float64), ys, torch.zeros(4, dtype=torch.float64)), -1)
    upper_ys = ys if symmetric else ys[:1]
    upper_static = torch.stack((torch.full_like(upper_ys, 3.0), upper_ys, torch.zeros_like(upper_ys)), -1)
    mobile = torch.stack((torch.ones(4, dtype=torch.float64), ys, torch.zeros(4, dtype=torch.float64)), -1)
    means = torch.cat((lower_static, upper_static, mobile))
    static_count = len(lower_static) + len(upper_static)
    mobility_logits = torch.cat((torch.full((static_count, 1), -10.0), torch.full((4, 1), 10.0)))
    return SimpleNamespace(
        states=torch.zeros((len(means), 1), dtype=torch.long),
        means=means,
        scales=torch.full((len(means), 3), torch.log(torch.tensor(0.1, dtype=torch.float64))),
        opacities=torch.full((len(means), 1), 10.0, dtype=torch.float64),
        mobilities=mobility_logits,
        articulation_params=SimpleNamespace(
            articulation_type=torch.tensor(2),
            axis=torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64),
            dist=torch.tensor(1.0, dtype=torch.float64),
        ),
    )


def test_pcfg_recovers_unique_broad_closed_side_and_terminal_stops() -> None:
    prediction = pcfg_certificate(_model(), -0.8, 1.8, 1.0)
    assert prediction.lower.terminal_valid
    assert prediction.upper.terminal_valid
    assert prediction.closed_identifiable
    assert prediction.closed_end == "lower"
    assert prediction.closed_mean_gain_lower > prediction.closed_mean_gain_upper
    central = prediction.lower.per_radius[1]
    assert central["outside_penetration_q99_m"] > central["inside_penetration_q99_m"]


def test_pcfg_abstains_for_symmetric_contact_area() -> None:
    prediction = pcfg_certificate(_model(symmetric=True), -0.8, 1.8, 1.0)
    assert prediction.closed_end == "unknown"
    assert not prediction.closed_identifiable


def test_pcfg_does_not_label_the_less_negative_contact_gain_as_closed() -> None:
    prediction = pcfg_certificate(_model(), -0.3, 1.3, 1.0)
    assert prediction.closed_mean_gain_lower <= 0.0
    assert prediction.closed_mean_gain_upper <= 0.0
    assert prediction.closed_end == "unknown"


def test_even_stride_sampling_matches_v4_rule() -> None:
    assert deterministic_sample_indices(3, cap=5).tolist() == [0, 1, 2]
    assert deterministic_sample_indices(10, cap=4).tolist() == [0, 2, 5, 7]
