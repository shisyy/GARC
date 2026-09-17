import pytest
import torch

from splart.cr_fpl import CRFPLHead
from splart.se_mdc import (
    SE_MDC_LOOPS,
    SE_MDC_RESIDUAL_SCALE,
    SE_MDC_WIDTH,
    SEMDCHead,
    fixed_point_loss_terms,
    exact_se_mdc_swap_error,
)


def profiles(batch: int = 6):
    generator = torch.Generator().manual_seed(817)
    features = torch.randn(batch, 2, 3, 11, 9, generator=generator)
    coordinates = torch.linspace(0.05, 2.05, 11).view(1, 1, 1, 11).expand(
        batch, 2, 3, -1
    )
    anchor = torch.rand(batch, 2, generator=generator) * 1.5 + 0.1
    return features, coordinates, anchor


def test_four_tied_width64_loops_and_coupled_convex_allocation():
    model = SEMDCHead()
    assert (SE_MDC_LOOPS, SE_MDC_WIDTH, SE_MDC_RESIDUAL_SCALE) == (4, 64, 0.25)
    assert len([name for name, _ in model.named_modules() if name.endswith("loop_block")]) == 1
    assert len([name for name, _ in model.named_modules() if name.endswith("dual_coupler")]) == 1
    calls = []
    hook = model.loop_block.register_forward_hook(lambda *_: calls.append(1))
    output = model(*profiles())
    hook.remove()
    assert len(calls) == 4
    assert output.loop_distances.shape == (6, 2, 4)
    assert (output.loop_distances > 0).all()
    assert torch.all((output.loop_convex_weights > 0) & (output.loop_convex_weights < 1))
    assert torch.allclose(
        output.loop_dual_allocation.sum(dim=1),
        torch.ones_like(output.loop_dual_allocation[:, 0]),
    )


def test_exact_side_swap_batch_permutation_chunking_and_single_batch_invariance():
    torch.manual_seed(817)
    model = SEMDCHead().eval()
    features, coordinates, anchor = profiles(4)
    reference = model(features, coordinates, anchor)
    assert exact_se_mdc_swap_error(model, features, coordinates, anchor) <= 1e-6
    permutation = torch.tensor([2, 0, 3, 1])
    permuted = model(features[permutation], coordinates[permutation], anchor[permutation])
    for field in type(reference).__dataclass_fields__:
        assert torch.allclose(
            getattr(permuted, field), getattr(reference, field)[permutation], atol=1e-6, rtol=0
        )
        chunked = torch.cat(
            [
                getattr(model(features[:1], coordinates[:1], anchor[:1]), field),
                getattr(model(features[1:], coordinates[1:], anchor[1:]), field),
            ],
            dim=0,
        )
        singles = torch.cat(
            [
                getattr(
                    model(
                        features[index : index + 1],
                        coordinates[index : index + 1],
                        anchor[index : index + 1],
                    ),
                    field,
                )
                for index in range(4)
            ],
            dim=0,
        )
        assert torch.allclose(chunked, getattr(reference, field), atol=1e-6, rtol=0)
        assert torch.allclose(singles, getattr(reference, field), atol=1e-6, rtol=0)


def test_independent_control_is_parameter_matched_and_has_no_cross_side_effect():
    torch.manual_seed(817)
    coupled = SEMDCHead(coupling_mode="coupled")
    independent = SEMDCHead(coupling_mode="independent")
    independent.load_state_dict(coupled.state_dict())
    assert sum(p.numel() for p in coupled.parameters()) == sum(
        p.numel() for p in independent.parameters()
    )
    features, coordinates, anchor = profiles(3)
    with torch.no_grad():
        independent.fixed_point_readout.bias.fill_(0.5)
    before = independent(features, coordinates, anchor).distance[:, 0]
    changed = features.clone()
    changed[:, 1] = changed[:, 1] * 50.0 + 100.0
    after = independent(changed, coordinates, anchor).distance[:, 0]
    assert torch.equal(before, after)


def test_object_minimax_aggregates_episodes_before_smooth_max_and_backpropagates():
    torch.manual_seed(817)
    model = SEMDCHead()
    features, coordinates, anchor = profiles(6)
    output = model(features, coordinates, anchor)
    target = torch.rand_like(anchor) + 0.2
    object_ids = ("a", "a", "a", "b", "b", "b")
    terms = fixed_point_loss_terms(
        output,
        target,
        torch.full_like(anchor, 0.2),
        object_ids,
        minimax=True,
    )
    assert set(terms) == {
        "log_distance",
        "fixed_point_proposal",
        "log_energy",
        "residual_monotonicity",
        "object_smooth_worst_side",
        "dual_alignment",
        "terminal_violation",
    }
    torch.stack(tuple(terms.values())).sum().backward()
    for parameter in (
        model.loop_block.attention.in_proj_weight,
        model.dual_coupler.attention.in_proj_weight,
        model.dual_coupler.risk_readout.weight,
        model.fixed_point_readout.weight,
    ):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()

    # CR-FPL/minimax is a valid 2x2 control and omits only architectural dual alignment.
    cr_output = CRFPLHead()(features, coordinates, anchor)
    cr_terms = fixed_point_loss_terms(cr_output, target, object_ids=object_ids, minimax=True)
    assert "object_smooth_worst_side" in cr_terms
    assert "dual_alignment" not in cr_terms


def test_fail_closed_inputs_and_nonpositive_target():
    model = SEMDCHead()
    features, coordinates, anchor = profiles()
    with pytest.raises(ValueError, match="non-negative outward"):
        model(features, -coordinates, anchor)
    output = model(features, coordinates, anchor)
    with pytest.raises(ValueError, match="positive"):
        fixed_point_loss_terms(output, torch.zeros_like(anchor))
