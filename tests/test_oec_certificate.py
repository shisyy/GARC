from types import SimpleNamespace

import torch

from splart.oec_certificate import _surface_gaps, oec_certificate


def _model(kind=2, state1=False, anisotropic=False):
    dtype = torch.float64
    static = torch.tensor([[-.9, -.3, 0.], [-.9, .3, 0.], [2.9, 0., 0.]], dtype=dtype)
    mobile = torch.tensor([[0., -.3, 0.], [0., .3, 0.]], dtype=dtype)
    means = torch.cat((static, mobile))
    states = torch.zeros((5, 1), dtype=torch.long)
    if state1:
        means[3:, 0] += 1
        states[3:] = 1
    scales = torch.full((5, 3), .1, dtype=dtype)
    if anisotropic:
        scales[:, 0] = .45
        scales[:, 1:] = .03
    quats = torch.tensor([[1., 0., 0., 0.]], dtype=dtype).repeat(5, 1)
    return SimpleNamespace(
        states=states, means=means, scales=scales.log(), quats=quats,
        opacities=torch.full((5, 1), 10., dtype=dtype),
        mobilities=torch.cat((torch.full((3, 1), -10.), torch.full((2, 1), 10.))).to(dtype),
        articulation_params=SimpleNamespace(articulation_type=torch.tensor(kind), axis=torch.tensor([1., 0., 0.], dtype=dtype),
            dist=torch.tensor(1., dtype=dtype), angle=torch.tensor(1., dtype=dtype), pivot=torch.zeros(3, dtype=dtype)))


def test_prismatic_oec_runs_and_exposes_fixed_three_radii():
    result = oec_certificate(_model(), -.8, 1.8, 1.)
    assert len(result.lower.per_radius) == 3
    assert result.static_count == 3 and result.mobile_count == 2


def test_revolute_oec_runs():
    model = _model(kind=1)
    model.articulation_params.axis = torch.tensor([0., 0., 1.], dtype=torch.float64)
    result = oec_certificate(model, -.8, 1.8, 1.)
    assert result.lower.endpoint_scalar == -.8


def test_state_swap_preserves_physical_evidence():
    a = oec_certificate(_model(state1=False), -.8, 1.8, 1.)
    b = oec_certificate(_model(state1=True), -.8, 1.8, 1.)
    assert abs(a.closed_mean_gain_lower-b.closed_mean_gain_lower) < 1e-8
    assert abs(a.closed_mean_gain_upper-b.closed_mean_gain_upper) < 1e-8


def test_free_space_abstains():
    model = _model()
    model.means[:3] += torch.tensor([0., 20., 0.], dtype=torch.float64)
    result = oec_certificate(model, -.8, 1.8, 1.)
    assert result.closed_end == "unknown"
    assert not result.lower.terminal_valid and not result.upper.terminal_valid


def test_anisotropic_support_differs_from_amax_sphere():
    dtype = torch.float64
    means_m = torch.tensor([[0., 0., 0.]], dtype=dtype)
    means_s = torch.tensor([[0., 1., 0.]], dtype=dtype)
    rot = torch.eye(3, dtype=dtype)[None]
    scale = torch.tensor([[.5, .05, .05]], dtype=dtype)
    ellipsoid_gap = _surface_gaps(means_m, rot, scale, means_s, rot, scale, 1.)
    sphere_gap = torch.tensor([1.-1.], dtype=dtype)  # two amax=.5 spheres
    assert ellipsoid_gap.item() > .89
    assert sphere_gap.item() == 0.

