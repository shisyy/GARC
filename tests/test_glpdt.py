import pytest
import torch

from splart.glpdt import (
    GLPDT_CORRECTION_FRACTION,
    GLPDT_LOOPS,
    GLPDT_RESIDUAL_SCALE,
    GLPDTHead,
    exact_glpdt_swap_error,
    glpdt_deep_supervision_loss,
)


def profiles(batch: int = 3):
    generator = torch.Generator().manual_seed(814)
    features = torch.randn(batch, 2, 3, 11, 9, generator=generator)
    coordinates = torch.randn(batch, 2, 3, 11, generator=generator)
    anchor = torch.rand(batch, 2, generator=generator) * 2.0 + 0.05
    return features, coordinates, anchor


def test_four_tied_prenorm_loops_with_fixed_input_recall():
    model = GLPDTHead()
    assert GLPDT_LOOPS == 4
    assert GLPDT_RESIDUAL_SCALE == 0.25
    assert len([name for name, _ in model.named_modules() if name.endswith("loop_block")]) == 1
    calls = []
    hook = model.shared_side.loop_block.register_forward_hook(lambda *_: calls.append(1))
    output = model(*profiles())
    hook.remove()
    assert len(calls) == 2 * GLPDT_LOOPS
    assert output.loop_distances.shape == (3, 2, 4)
    assert output.loop_state_residual_l2.shape == (3, 2, 4)
    assert (output.loop_state_residual_l2 > 0).all()


def test_positive_bounded_anchor_correction_and_dual_state():
    model = GLPDTHead()
    features, coordinates, anchor = profiles()
    output = model(features, coordinates, anchor)
    lower = anchor * (1.0 - GLPDT_CORRECTION_FRACTION)
    upper = anchor + GLPDT_CORRECTION_FRACTION * (anchor + 1.0)
    assert (output.loop_distances > 0).all()
    assert torch.all(output.loop_distances >= lower.unsqueeze(-1) - 1e-7)
    assert torch.all(output.loop_distances <= upper.unsqueeze(-1) + 1e-7)
    assert torch.all((output.loop_terminal_violations > 0) & (output.loop_terminal_violations < 1))


def test_exact_swap_equivariance_includes_every_loop_diagnostic():
    torch.manual_seed(814)
    model = GLPDTHead().eval()
    features, coordinates, anchor = profiles()
    assert exact_glpdt_swap_error(model, features, coordinates, anchor) == 0.0


def test_deep_supervision_reaches_tied_block_and_both_readouts():
    torch.manual_seed(814)
    model = GLPDTHead()
    output = model(*profiles())
    distance_target = torch.full_like(output.distance, 0.75)
    violation_target = torch.zeros_like(output.terminal_violation)
    loss = glpdt_deep_supervision_loss(output, distance_target, violation_target)
    loss.backward()
    assert torch.isfinite(loss)
    for parameter in (
        model.shared_side.loop_block.attention.in_proj_weight,
        model.shared_side.correction_readout.weight,
        model.shared_side.violation_readout.weight,
    ):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()


def test_fail_closed_shapes_anchor_and_nonfinite_profiles():
    model = GLPDTHead()
    features, coordinates, anchor = profiles()
    with pytest.raises(ValueError, match="anchor_distance"):
        model(features, coordinates, -anchor)
    with pytest.raises(ValueError, match="coordinates"):
        model(features, coordinates[..., :-1], anchor)
    features[0, 0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        model(features, coordinates, anchor)
