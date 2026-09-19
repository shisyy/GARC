import torch

from splart.clip_limit import CSTRHead, CSTR_CHANNELS, counterfactual_semantic_tangent_residual, exact_cstr_swap_error
from splart.cr_fpl import CRFPLHead, CRFPLOutput


def raw_trajectory(batch: int = 3):
    coordinates = torch.linspace(0.0, 2.0, 129).view(1, 1, -1).expand(batch, 2, -1).clone()
    offset = torch.arange(batch * 2, dtype=torch.float32).reshape(batch, 2, 1)
    evidence = (offset + 0.3 * coordinates + coordinates.square()).unsqueeze(-1)
    return evidence, coordinates


def model_inputs(batch: int = 3):
    generator = torch.Generator().manual_seed(819)
    features = torch.randn(batch, 2, 3, 11, 9, generator=generator)
    coordinates = torch.linspace(0.05, 2.05, 11).view(1, 1, 1, 11).expand(batch, 2, 3, -1).clone()
    anchor = torch.rand(batch, 2, generator=generator) + 0.2
    raw, semantic_coordinates = raw_trajectory(batch)
    cstr = counterfactual_semantic_tangent_residual(raw, semantic_coordinates)
    return features, coordinates, anchor, cstr, semantic_coordinates


def test_cstr_annihilates_object_internal_affine_trajectories():
    coordinates = torch.linspace(0.0, 2.0, 129).view(1, 1, -1).expand(4, 2, -1).clone()
    slope = torch.tensor([0.1, -0.3, 1.2, -2.0]).view(4, 1, 1)
    intercept = torch.arange(8, dtype=torch.float32).reshape(4, 2, 1)
    evidence = (intercept + slope * coordinates).unsqueeze(-1)
    cstr = counterfactual_semantic_tangent_residual(evidence, coordinates)
    assert cstr.shape == (4, 2, 129, CSTR_CHANNELS)
    assert float(cstr.max()) < 2e-5


def test_cstr_is_prompt_sign_invariant_and_activates_on_nonlinearity():
    evidence, coordinates = raw_trajectory()
    positive = counterfactual_semantic_tangent_residual(evidence, coordinates)
    negative = counterfactual_semantic_tangent_residual(-evidence, coordinates)
    assert torch.allclose(positive, negative, atol=0.0, rtol=0.0)
    assert torch.all(positive[..., 80:, 0].mean(dim=-1) > 0.1)
    assert torch.all(positive[..., 80:, 1].mean(dim=-1) > 0.1)
    assert torch.all(positive[..., 80:, 2].mean(dim=-1) > 0.1)


def test_cstr_zero_channels_exactly_recover_same_weight_cr_fpl():
    torch.manual_seed(819)
    geometry = CRFPLHead()
    model = CSTRHead()
    incompatible = model.load_state_dict(geometry.state_dict(), strict=False)
    assert incompatible.unexpected_keys == []
    assert set(incompatible.missing_keys) == {
        "shared_side.semantic_residual.value.weight",
        "shared_side.semantic_residual.gate.weight",
    }
    features, coordinates, anchor, cstr, semantic_coordinates = model_inputs()
    baseline = geometry(features, coordinates, anchor)
    candidate = model(features, coordinates, anchor, torch.zeros_like(cstr), semantic_coordinates)
    for field in CRFPLOutput.__dataclass_fields__:
        assert torch.equal(getattr(candidate, field), getattr(baseline, field))
    assert torch.equal(candidate.loop_semantic_residual_l2, torch.zeros_like(candidate.loop_semantic_residual_l2))


def test_cstr_is_requeried_every_loop_and_has_exact_side_swap_equivariance():
    model = CSTRHead()
    with torch.no_grad():
        model.shared_side.fixed_point_readout.bias.fill_(0.8)
    inputs = model_inputs()
    output = model(*inputs)
    assert output.loop_semantic_evidence.shape == (3, 2, CSTR_CHANNELS, 4)
    assert torch.any(output.loop_semantic_evidence[..., 1:] != output.loop_semantic_evidence[..., :1])
    assert exact_cstr_swap_error(model.eval(), *inputs) == 0.0
