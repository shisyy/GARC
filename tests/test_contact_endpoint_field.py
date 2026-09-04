from __future__ import annotations

import math

import torch

from splart.contact_endpoint_field import (
    EndpointFieldConfig,
    certify_contact_feasible_endpoints,
    infer_contact_feasible_endpoints,
    trajectory_voxel_pair_index,
)


DTYPE = torch.float64
SCALE = 0.05


def _scales(n: int) -> torch.Tensor:
    return torch.full((n, 3), SCALE, dtype=DTYPE)


def _config(**overrides) -> EndpointFieldConfig:
    values = {
        "lower_scan": -2.0,
        "upper_scan": 3.0,
        "samples_per_side": 401,
        "support_step": 0.025,
        "softargmin_temperature": 0.02,
        "closed_mass_margin": 0.08,
    }
    values.update(overrides)
    return EndpointFieldConfig(**values)


def test_prismatic_two_physical_stops_and_broad_closed_contact() -> None:
    # State 0 -> state 1 translates +1 along x.  Surface contact occurs near
    # scalars -0.8 and +1.8 because each pair has combined radius 0.1.
    mobile = torch.tensor([[1.0, -0.08, 0.0], [1.0, 0.08, 0.0]], dtype=DTYPE)
    static = torch.tensor(
        [[0.0, -0.08, 0.0], [0.0, 0.08, 0.0], [2.9, -0.08, 0.0]],  # broad closure face  # localized opposite stop
        dtype=DTYPE,
    )
    prediction = infer_contact_feasible_endpoints(
        static,
        _scales(len(static)),
        mobile,
        _scales(len(mobile)),
        joint_kind="prismatic",
        axis=torch.tensor([1.0, 0.0, 0.0], dtype=DTYPE),
        observed_displacement=torch.tensor(1.0, dtype=DTYPE),
        config=_config(),
    )

    assert prediction.lower.identifiable
    assert prediction.upper.identifiable
    assert abs(prediction.lower.value.item() - (-0.9)) < 0.035
    assert abs(prediction.upper.value.item() - 1.8) < 0.035
    assert prediction.closed_end == "lower"
    assert prediction.identifiable


def test_revolute_two_stops_are_recovered_in_observed_span_coordinates() -> None:
    observed_angle = math.pi / 4.0
    lower_scalar, upper_scalar = -0.75, 2.0

    def point(angle: float, radial_offset: float = 0.0) -> list[float]:
        radius = 1.0 + radial_offset
        return [radius * math.cos(angle), radius * math.sin(angle), 0.0]

    mobile = torch.tensor([point(0.0), point(0.0, 0.12)], dtype=DTYPE)
    lower_angle = lower_scalar * observed_angle
    upper_angle = upper_scalar * observed_angle
    static = torch.tensor(
        [
            point(lower_angle),
            point(lower_angle, 0.12),  # broad closed contact
            point(upper_angle),  # localized open stop
        ],
        dtype=DTYPE,
    )
    prediction = infer_contact_feasible_endpoints(
        static,
        _scales(len(static)),
        mobile,
        _scales(len(mobile)),
        joint_kind="revolute",
        axis=torch.tensor([0.0, 0.0, 1.0], dtype=DTYPE),
        pivot=torch.zeros(3, dtype=DTYPE),
        observed_displacement=torch.tensor(observed_angle, dtype=DTYPE),
        config=_config(support_step=0.035),
    )

    assert prediction.lower.identifiable
    assert prediction.upper.identifiable
    # Spherical envelopes touch slightly before centre coincidence.
    assert abs(prediction.lower.value.item() - lower_scalar) < 0.18
    assert abs(prediction.upper.value.item() - upper_scalar) < 0.18
    assert prediction.closed_end == "lower"


def test_no_contact_fails_closed_instead_of_hallucinating_limits() -> None:
    static = torch.tensor([[20.0, 10.0, 0.0]], dtype=DTYPE)
    mobile = torch.tensor([[0.0, 0.0, 0.0]], dtype=DTYPE)
    prediction = infer_contact_feasible_endpoints(
        static,
        _scales(1),
        mobile,
        _scales(1),
        joint_kind="prismatic",
        axis=torch.tensor([1.0, 0.0, 0.0], dtype=DTYPE),
        observed_displacement=1.0,
        config=_config(),
    )

    assert not prediction.lower.identifiable
    assert not prediction.upper.identifiable
    assert not prediction.identifiable
    assert prediction.closed_end == "unknown"
    assert torch.isnan(prediction.lower.value)
    assert torch.isnan(prediction.upper.value)
    assert torch.isfinite(prediction.lower.raw_value)
    assert torch.isfinite(prediction.upper.raw_value)


def test_symmetric_terminal_contacts_leave_closed_label_unidentified() -> None:
    mobile = torch.tensor([[1.0, 0.0, 0.0]], dtype=DTYPE)
    static = torch.tensor([[0.0, 0.0, 0.0], [2.9, 0.0, 0.0]], dtype=DTYPE)
    prediction = infer_contact_feasible_endpoints(
        static,
        _scales(2),
        mobile,
        _scales(1),
        joint_kind="prismatic",
        axis=torch.tensor([1.0, 0.0, 0.0], dtype=DTYPE),
        observed_displacement=1.0,
        config=_config(),
    )

    assert prediction.lower.identifiable and prediction.upper.identifiable
    assert prediction.closed_end == "unknown"
    assert not prediction.closed_end_identifiable
    assert not prediction.identifiable


def test_energy_fields_backpropagate_to_geometry_and_screw_motion() -> None:
    static = torch.tensor([[0.0, 0.0, 0.0], [2.9, 0.0, 0.0]], dtype=DTYPE, requires_grad=True)
    mobile = torch.tensor([[1.0, 0.0, 0.0]], dtype=DTYPE, requires_grad=True)
    displacement = torch.tensor(1.0, dtype=DTYPE, requires_grad=True)
    prediction = infer_contact_feasible_endpoints(
        static,
        _scales(2),
        mobile,
        _scales(1),
        joint_kind="prismatic",
        axis=torch.tensor([1.0, 0.03, 0.0], dtype=DTYPE, requires_grad=True),
        observed_displacement=displacement,
        config=_config(samples_per_side=101),
    )
    loss = (prediction.lower.field.posterior * prediction.lower.field.total_energy).sum() + (
        prediction.upper.field.posterior * prediction.upper.field.total_energy
    ).sum()
    loss.backward()

    assert static.grad is not None and torch.isfinite(static.grad).all()
    assert mobile.grad is not None and torch.isfinite(mobile.grad).all()
    assert displacement.grad is not None and torch.isfinite(displacement.grad)
    assert displacement.grad.abs() > 0


def test_revolute_field_backpropagates_to_axis_pivot_angle_and_scales() -> None:
    static = torch.tensor([[0.0, -1.0, 0.0], [0.0, 1.0, 0.0]], dtype=DTYPE, requires_grad=True)
    mobile = torch.tensor([[1.0, 0.0, 0.0]], dtype=DTYPE, requires_grad=True)
    static_scales = _scales(2).requires_grad_()
    mobile_scales = _scales(1).requires_grad_()
    axis = torch.tensor([0.02, -0.01, 1.0], dtype=DTYPE, requires_grad=True)
    pivot = torch.tensor([0.01, 0.02, 0.0], dtype=DTYPE, requires_grad=True)
    angle = torch.tensor(math.pi / 4.0, dtype=DTYPE, requires_grad=True)
    prediction = infer_contact_feasible_endpoints(
        static,
        static_scales,
        mobile,
        mobile_scales,
        joint_kind="revolute",
        axis=axis,
        pivot=pivot,
        observed_displacement=angle,
        config=_config(samples_per_side=101),
    )
    objective = prediction.lower.field.total_energy.mean() + prediction.upper.field.total_energy.mean()
    objective.backward()

    for tensor in (static, mobile, static_scales, mobile_scales, axis, pivot, angle):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()


def test_large_dense_pair_request_uses_capped_trajectory_broad_phase() -> None:
    static = torch.stack((torch.linspace(-1.0, 4.0, 200), torch.zeros(200), torch.zeros(200)), dim=-1).to(DTYPE)
    mobile = torch.stack((torch.linspace(0.0, 1.0, 160), torch.full((160,), 0.02), torch.zeros(160)), dim=-1).to(DTYPE)
    config = _config(
        max_dense_pairs=100,
        max_broad_phase_pairs=256,
        broad_phase_samples=9,
        pair_chunk_size=31,
        scalar_chunk_size=7,
        samples_per_side=19,
    )
    prediction = infer_contact_feasible_endpoints(
        static,
        _scales(len(static)),
        mobile,
        _scales(len(mobile)),
        joint_kind="prismatic",
        axis=torch.tensor([1.0, 0.0, 0.0], dtype=DTYPE),
        observed_displacement=1.0,
        config=config,
    )
    assert prediction.lower.field.scalars.numel() == config.samples_per_side
    assert torch.isfinite(prediction.lower.field.total_energy).all()


def test_low_opacity_part_weight_cannot_create_a_false_stop() -> None:
    mobile = torch.tensor([[1.0, 0.0, 0.0]], dtype=DTYPE)
    static = torch.tensor([[0.0, 0.0, 0.0], [2.9, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=DTYPE)
    prediction = infer_contact_feasible_endpoints(
        static,
        _scales(3),
        mobile,
        _scales(1),
        static_weights=torch.tensor([1.0, 1.0, 1.0e-4], dtype=DTYPE),
        mobile_weights=torch.ones(1, dtype=DTYPE),
        joint_kind="prismatic",
        axis=torch.tensor([1.0, 0.0, 0.0], dtype=DTYPE),
        observed_displacement=1.0,
        config=_config(min_geometry_weight=1.0e-2),
    )
    assert abs(prediction.lower.raw_value.item() - (-0.9)) < 0.04


def test_published_scalar_has_a_recomputed_matching_certificate() -> None:
    mobile = torch.tensor([[1.0, 0.0, 0.0]], dtype=DTYPE)
    static = torch.tensor([[0.0, 0.0, 0.0], [2.9, 0.0, 0.0]], dtype=DTYPE)
    config = _config(samples_per_side=83)
    prediction = infer_contact_feasible_endpoints(
        static,
        _scales(2),
        mobile,
        _scales(1),
        joint_kind="prismatic",
        axis=torch.tensor([1.0, 0.0, 0.0], dtype=DTYPE),
        observed_displacement=1.0,
        config=config,
    )
    certified = certify_contact_feasible_endpoints(
        static,
        _scales(2),
        mobile,
        _scales(1),
        lower_scalar=prediction.lower.raw_value,
        upper_scalar=prediction.upper.raw_value,
        joint_kind="prismatic",
        axis=torch.tensor([1.0, 0.0, 0.0], dtype=DTYPE),
        observed_displacement=1.0,
        config=config,
    )
    assert torch.allclose(prediction.lower.contact_gap, certified.lower.contact_gap)
    assert torch.allclose(prediction.upper.contact_gap, certified.upper.contact_gap)
    assert torch.allclose(prediction.lower.contact_mass, certified.lower.contact_mass)


def test_trajectory_voxel_pairs_are_deterministic_and_capped() -> None:
    static = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=DTYPE)
    mobile = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.04, 0.0]], dtype=DTYPE)
    scalars = torch.linspace(-1.0, 2.0, 13, dtype=DTYPE)
    kwargs = dict(
        joint_kind="prismatic",
        axis=torch.tensor([1.0, 0.0, 0.0], dtype=DTYPE),
        pivot=torch.zeros(3, dtype=DTYPE),
        observed_displacement=torch.tensor(1.0, dtype=DTYPE),
        voxel_size=0.25,
        neighbor_radius=1,
        max_pairs=4,
    )
    first = trajectory_voxel_pair_index(static, mobile, scalars, **kwargs)
    second = trajectory_voxel_pair_index(static, mobile, scalars, **kwargs)
    assert first.shape[1] <= 4
    assert torch.equal(first, second)
