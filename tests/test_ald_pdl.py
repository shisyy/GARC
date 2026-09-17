import pytest
import torch

from splart.ald_pdl import (
    ALD_PDL_LOOPS,
    ALD_PDL_RESIDUAL_SCALE,
    ALD_PDL_WIDTH,
    ALDPDLHead,
    ald_pdl_deep_supervision_loss,
    dual_penalty_update,
    exact_ald_pdl_swap_error,
)


def profiles(batch: int = 3):
    generator = torch.Generator().manual_seed(815)
    features = torch.randn(batch, 2, 3, 11, 9, generator=generator)
    coordinates = torch.randn(batch, 2, 3, 11, generator=generator)
    anchor = torch.rand(batch, 2, generator=generator) * 2.0 + 0.05
    return features, coordinates, anchor


def test_four_tied_loops_width_and_same_parameter_early_exit():
    model = ALDPDLHead()
    assert ALD_PDL_LOOPS == 4
    assert ALD_PDL_WIDTH == 64
    assert ALD_PDL_RESIDUAL_SCALE == 0.25
    assert len([name for name, _ in model.named_modules() if name.endswith("loop_block")]) == 1
    calls = []
    hook = model.shared_side.loop_block.register_forward_hook(lambda *_: calls.append(1))
    output = model(*profiles())
    hook.remove()
    assert len(calls) == 2 * ALD_PDL_LOOPS
    assert output.loop_distances.shape == (3, 2, 4)
    assert torch.equal(output.loop_distances[..., 0], output.loop_distances[..., :1].squeeze(-1))


def test_log_ratio_is_positive_and_not_bounded_by_glpdt_additive_cap():
    model = ALDPDLHead()
    with torch.no_grad():
        model.shared_side.primal_readout.bias.fill_(2.0)
        model.shared_side.log_energy_readout.weight.zero_()
        model.shared_side.log_energy_readout.bias.zero_()
    features, coordinates, anchor = profiles()
    output = model(features, coordinates, anchor)
    assert (output.loop_distances > 0).all()
    assert torch.all(output.loop_distances[..., 0] > anchor * 3.0)
    assert torch.all(output.loop_distances[..., -1] > output.loop_distances[..., 0])


def test_dual_penalty_changes_only_for_energy_increases():
    penalty = torch.zeros(3)
    previous = torch.tensor([1.0, 1.0, 1.0])
    current = torch.tensor([0.5, 1.0, 1.5])
    updated = dual_penalty_update(previous, current, penalty)
    assert torch.equal(updated, torch.tensor([0.0, 0.0, 0.5]))
    assert torch.equal(dual_penalty_update(None, current, penalty), penalty)


def test_exact_swap_and_deep_supervision_gradients():
    torch.manual_seed(815)
    model = ALDPDLHead()
    inputs = profiles()
    assert exact_ald_pdl_swap_error(model.eval(), *inputs) == 0.0
    model.train()
    output = model(*inputs)
    target = torch.full_like(output.distance, 0.75)
    violation = torch.full_like(output.terminal_violation, 0.25)
    loss = ald_pdl_deep_supervision_loss(output, target, violation)
    loss.backward()
    assert torch.isfinite(loss)
    for parameter in (
        model.shared_side.loop_block.attention.in_proj_weight,
        model.shared_side.primal_readout.weight,
        model.shared_side.log_energy_readout.weight,
    ):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()


def test_fail_closed_inputs_and_nonpositive_target():
    model = ALDPDLHead()
    features, coordinates, anchor = profiles()
    with pytest.raises(ValueError, match="anchor_distance"):
        model(features, coordinates, -anchor)
    output = model(features, coordinates, anchor)
    with pytest.raises(ValueError, match="positive"):
        ald_pdl_deep_supervision_loss(output, torch.zeros_like(anchor))
