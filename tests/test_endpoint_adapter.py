from __future__ import annotations

import torch

from splart.contact_endpoint_field import EndpointFieldConfig
from splart.endpoint_adapter import (
    EndpointAdapterConfig,
    EndpointGeometry,
    canonicalize_state1_mobile,
    fit_endpoint_adapter,
    swap_endpoint_scalars,
)


DTYPE = torch.float64


def _scales(count: int) -> torch.Tensor:
    return torch.full((count, 3), 0.05, dtype=DTYPE)


def _field_config() -> EndpointFieldConfig:
    return EndpointFieldConfig(
        lower_scan=-2.0,
        upper_scan=3.0,
        samples_per_side=121,
        support_step=0.025,
        softargmin_temperature=0.025,
        closed_mass_margin=0.08,
        pair_chunk_size=2,
        scalar_chunk_size=11,
    )


def _adapter_config(iterations: int = 80) -> EndpointAdapterConfig:
    return EndpointAdapterConfig(iterations=iterations, learning_rate=0.04, anchor_weight=0.1)


def _two_state_prismatic_geometry(*, requires_grad: bool = False) -> tuple[EndpointGeometry, EndpointGeometry]:
    static0 = torch.tensor(
        [[0.0, -0.08, 0.0], [0.0, 0.08, 0.0], [2.9, -0.08, 0.0]],
        dtype=DTYPE,
        requires_grad=requires_grad,
    )
    mobile0 = torch.tensor(
        [[1.0, -0.08, 0.0], [1.0, 0.08, 0.0]], dtype=DTYPE, requires_grad=requires_grad
    )
    # A second independent reconstruction at state 1, canonicalized before it
    # enters D2-CEA.  Small opposing noise prevents the test from degenerating
    # into two identical field objects.
    mobile1_observed = torch.tensor([[2.0, -0.078, 0.0], [2.0, 0.078, 0.0]], dtype=DTYPE)
    mobile1_canonical = canonicalize_state1_mobile(
        mobile1_observed,
        joint_kind="prismatic",
        axis=torch.tensor([1.0, 0.0, 0.0], dtype=DTYPE),
        observed_displacement=1.0,
    )
    static1 = static0.detach().clone()
    geometry0 = EndpointGeometry(static0, _scales(3), mobile0, _scales(2))
    geometry1 = EndpointGeometry(static1, _scales(3), mobile1_canonical, _scales(2))
    return geometry0, geometry1


def test_dual_state_adapter_fits_both_physical_stops_and_closed_side() -> None:
    geometries = _two_state_prismatic_geometry()
    adapter, prediction = fit_endpoint_adapter(
        geometries,
        joint_kind="prismatic",
        axis=torch.tensor([1.0, 0.0, 0.0], dtype=DTYPE),
        observed_displacement=1.0,
        field_config=_field_config(),
        adapter_config=_adapter_config(),
    )
    lower, upper = adapter.endpoint_scalars()
    assert abs(lower.item() - (-0.9)) < 0.08
    assert abs(upper.item() - 1.8) < 0.08
    assert prediction.lower_identifiable and prediction.upper_identifiable
    assert prediction.closed_end == "lower"
    assert prediction.closed_end_identifiable


def test_adapter_detaches_reconstruction_and_articulation_tensors() -> None:
    geometries = _two_state_prismatic_geometry(requires_grad=True)
    axis = torch.tensor([1.0, 0.02, 0.0], dtype=DTYPE, requires_grad=True)
    displacement = torch.tensor(1.0, dtype=DTYPE, requires_grad=True)
    adapter, prediction = fit_endpoint_adapter(
        geometries,
        joint_kind="prismatic",
        axis=axis,
        observed_displacement=displacement,
        field_config=_field_config(),
        adapter_config=_adapter_config(iterations=5),
    )
    assert geometries[0].static_means.grad is None
    assert geometries[0].mobile_means_state0.grad is None
    assert axis.grad is None and displacement.grad is None
    assert adapter.lower_logit.grad is not None and torch.isfinite(adapter.lower_logit.grad)
    assert adapter.upper_logit.grad is not None and torch.isfinite(adapter.upper_logit.grad)
    assert torch.isfinite(prediction.final_loss)


def test_state_swap_equivariance_for_endpoint_scalars_and_closed_label() -> None:
    geometry = _two_state_prismatic_geometry()[0]
    _, forward = fit_endpoint_adapter(
        [geometry],
        joint_kind="prismatic",
        axis=torch.tensor([1.0, 0.0, 0.0], dtype=DTYPE),
        observed_displacement=1.0,
        field_config=_field_config(),
        adapter_config=_adapter_config(),
    )
    swapped_mobile_state0 = geometry.mobile_means_state0 + torch.tensor([1.0, 0.0, 0.0], dtype=DTYPE)
    swapped_geometry = EndpointGeometry(
        geometry.static_means,
        geometry.static_scales,
        swapped_mobile_state0,
        geometry.mobile_scales,
    )
    _, swapped = fit_endpoint_adapter(
        [swapped_geometry],
        joint_kind="prismatic",
        axis=torch.tensor([1.0, 0.0, 0.0], dtype=DTYPE),
        observed_displacement=-1.0,
        field_config=_field_config(),
        adapter_config=_adapter_config(),
    )
    expected_lower, expected_upper = swap_endpoint_scalars(forward.lower_scalar, forward.upper_scalar)
    assert torch.allclose(swapped.lower_scalar, expected_lower, atol=0.08)
    assert torch.allclose(swapped.upper_scalar, expected_upper, atol=0.08)
    assert forward.closed_end == "lower"
    assert swapped.closed_end == "upper"


def test_canonicalize_revolute_state1_is_inverse_motion() -> None:
    point0 = torch.tensor([[1.0, 0.0, 0.0]], dtype=DTYPE)
    point1 = torch.tensor([[0.0, 1.0, 0.0]], dtype=DTYPE)
    recovered = canonicalize_state1_mobile(
        point1,
        joint_kind="revolute",
        axis=torch.tensor([0.0, 0.0, 1.0], dtype=DTYPE),
        pivot=torch.zeros(3, dtype=DTYPE),
        observed_displacement=torch.pi / 2,
    )
    assert torch.allclose(recovered, point0, atol=1.0e-8)
