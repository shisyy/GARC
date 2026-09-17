import pytest
import torch

from splart.cr_fpl import (
    CR_FPL_LOOPS,
    CR_FPL_RESIDUAL_SCALE,
    CR_FPL_WIDTH,
    CRFPLHead,
    cr_fpl_deep_supervision_loss,
    differentiable_profile_query,
    exact_cr_fpl_swap_error,
)


def profiles(batch: int = 3):
    generator = torch.Generator().manual_seed(816)
    features = torch.randn(batch, 2, 3, 11, 9, generator=generator)
    base = torch.linspace(0.05, 2.05, 11)
    coordinates = base.view(1, 1, 1, 11).expand(batch, 2, 3, -1).clone()
    coordinates[:, 0] = coordinates[:, 0].flip(-1)
    anchor = torch.rand(batch, 2, generator=generator) * 1.5 + 0.1
    return features, coordinates, anchor


def test_linear_profile_query_is_exact_for_both_storage_orders_and_differentiable():
    coordinates = torch.tensor([[[0.0, 1.0, 2.0], [2.0, 1.0, 0.0]]])
    features = torch.stack((2.0 * coordinates + 1.0, -coordinates + 4.0), dim=-1)
    query = torch.tensor([0.25], requires_grad=True)
    sampled = differentiable_profile_query(features, coordinates, query)
    assert torch.allclose(sampled[0, :, 0], torch.full((2,), 1.5))
    assert torch.allclose(sampled[0, :, 1], torch.full((2,), 3.75))
    sampled.sum().backward()
    assert query.grad is not None and torch.isfinite(query.grad).all()
    assert query.grad.abs().item() > 0.0


def test_four_tied_loops_width_positive_output_and_convex_updates():
    model = CRFPLHead()
    assert CR_FPL_LOOPS == 4
    assert CR_FPL_WIDTH == 64
    assert CR_FPL_RESIDUAL_SCALE == 0.25
    assert len([name for name, _ in model.named_modules() if name.endswith("loop_block")]) == 1
    calls = []
    hook = model.shared_side.loop_block.register_forward_hook(lambda *_: calls.append(1))
    output = model(*profiles())
    hook.remove()
    assert len(calls) == 2 * CR_FPL_LOOPS
    assert output.loop_distances.shape == (3, 2, 4)
    assert (output.loop_distances > 0).all()
    assert torch.all((output.loop_convex_weights > 0) & (output.loop_convex_weights < 1))
    previous = torch.cat(
        (torch.zeros_like(output.loop_log_distance_ratios[..., :1]), output.loop_log_distance_ratios[..., :-1]),
        dim=-1,
    )
    lower = torch.minimum(previous, output.loop_fixed_point_proposals)
    upper = torch.maximum(previous, output.loop_fixed_point_proposals)
    assert torch.all(output.loop_log_distance_ratios >= lower)
    assert torch.all(output.loop_log_distance_ratios <= upper)


def test_state_dependent_requery_activates_after_first_update():
    model = CRFPLHead()
    with torch.no_grad():
        model.shared_side.fixed_point_readout.bias.fill_(0.8)
        model.shared_side.log_energy_readout.weight.zero_()
        model.shared_side.log_energy_readout.bias.zero_()
    output = model(*profiles())
    assert torch.equal(output.loop_query_residual_l2[..., 0], torch.zeros_like(output.loop_query_residual_l2[..., 0]))
    assert torch.all(output.loop_query_residual_l2[..., 1] > 0)
    assert torch.all(output.loop_prediction_residuals[..., 0] > 0)


def test_exact_swap_and_deep_supervision_gradients():
    torch.manual_seed(816)
    model = CRFPLHead()
    inputs = profiles()
    assert exact_cr_fpl_swap_error(model.eval(), *inputs) == 0.0
    model.train()
    output = model(*inputs)
    target = torch.full_like(output.distance, 0.75)
    violation = torch.full_like(output.terminal_violation, 0.25)
    loss = cr_fpl_deep_supervision_loss(output, target, violation)
    loss.backward()
    assert torch.isfinite(loss)
    for parameter in (
        model.shared_side.loop_block.attention.in_proj_weight,
        model.shared_side.feedback_tokenizer[0].weight,
        model.shared_side.fixed_point_readout.weight,
        model.shared_side.convex_weight_readout.weight,
    ):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()


def test_fail_closed_inputs_and_nonpositive_target():
    model = CRFPLHead()
    features, coordinates, anchor = profiles()
    with pytest.raises(ValueError, match="non-negative outward"):
        model(features, -coordinates, anchor)
    output = model(features, coordinates, anchor)
    with pytest.raises(ValueError, match="positive"):
        cr_fpl_deep_supervision_loss(output, torch.zeros_like(anchor))
