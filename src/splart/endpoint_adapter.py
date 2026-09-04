"""Detached dual-state contact endpoint adapter (D2-CEA).

The adapter learns only two scalars in the relabelled observation coordinate
system.  Scene geometry and screw parameters are always detached, so fitting
the adapter cannot change SplArt reconstruction, mobility, or articulation.
No endpoint image, absolute interior fraction, or URDF limit is accepted by
this API.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, Sequence

import torch
from torch import Tensor, nn

from splart.contact_endpoint_field import (
    CertifiedEndpointPair,
    ContactEndpointPrediction,
    EndpointEnergyField,
    EndpointFieldConfig,
    JointKind,
    certify_contact_feasible_endpoints,
    infer_contact_feasible_endpoints,
)


@dataclass(frozen=True)
class EndpointAdapterConfig:
    iterations: int = 300
    learning_rate: float = 3.0e-2
    anchor_weight: float = 1.0e-1
    min_extension: float = 2.0e-2
    interpolation_width_steps: float = 0.35

    def validate(self, field: EndpointFieldConfig) -> None:
        if self.iterations < 1 or self.learning_rate <= 0.0:
            raise ValueError("adapter iterations and learning rate must be positive")
        if self.anchor_weight < 0.0 or self.interpolation_width_steps <= 0.0:
            raise ValueError("adapter weights must be non-negative and interpolation width positive")
        if not 0.0 < self.min_extension < min(-field.lower_scan, field.upper_scan - 1.0):
            raise ValueError("min_extension must fit strictly inside both scan intervals")


@dataclass(frozen=True)
class EndpointGeometry:
    """One independently reconstructed state, canonicalized to state 0."""

    static_means: Tensor
    static_scales: Tensor
    mobile_means_state0: Tensor
    mobile_scales: Tensor
    static_weights: Optional[Tensor] = None
    mobile_weights: Optional[Tensor] = None
    pair_index: Optional[Tensor] = None

    def detached(self) -> "EndpointGeometry":
        def maybe_detach(value: Optional[Tensor]) -> Optional[Tensor]:
            return None if value is None else value.detach()

        return EndpointGeometry(
            static_means=self.static_means.detach(),
            static_scales=self.static_scales.detach(),
            mobile_means_state0=self.mobile_means_state0.detach(),
            mobile_scales=self.mobile_scales.detach(),
            static_weights=maybe_detach(self.static_weights),
            mobile_weights=maybe_detach(self.mobile_weights),
            pair_index=maybe_detach(self.pair_index),
        )


@dataclass(frozen=True)
class EndpointAdapterPrediction:
    lower_scalar: Tensor
    upper_scalar: Tensor
    closed_end: str
    closed_end_identifiable: bool
    lower_identifiable: bool
    upper_identifiable: bool
    state_certificates: tuple[CertifiedEndpointPair, ...]
    initial_fields: tuple[ContactEndpointPrediction, ...]
    final_loss: Tensor


def canonicalize_state1_mobile(
    means_state1: Tensor,
    *,
    joint_kind: JointKind,
    axis: Tensor,
    observed_displacement: Tensor | float,
    pivot: Optional[Tensor] = None,
) -> Tensor:
    """Map state-1 mobile centres back to the state-0 coordinate system."""

    axis = torch.as_tensor(axis, dtype=means_state1.dtype, device=means_state1.device)
    axis = axis / torch.linalg.vector_norm(axis).clamp_min(torch.finfo(means_state1.dtype).eps)
    displacement = torch.as_tensor(observed_displacement, dtype=means_state1.dtype, device=means_state1.device)
    if joint_kind == "prismatic":
        return means_state1 - axis * displacement
    if joint_kind != "revolute":
        raise ValueError("joint_kind must be 'revolute' or 'prismatic'")
    if pivot is None:
        pivot = torch.zeros(3, dtype=means_state1.dtype, device=means_state1.device)
    else:
        pivot = torch.as_tensor(pivot, dtype=means_state1.dtype, device=means_state1.device)
    theta = -displacement
    x = means_state1 - pivot
    cross = torch.linalg.cross(axis.expand_as(x), x, dim=-1)
    projection = (x * axis).sum(dim=-1, keepdim=True) * axis
    rotated = x * torch.cos(theta) + cross * torch.sin(theta) + projection * (1.0 - torch.cos(theta))
    return rotated + pivot


def swap_endpoint_scalars(lower: Tensor | float, upper: Tensor | float) -> tuple[Tensor, Tensor]:
    """Convert endpoint scalars after exchanging observed state 0 and state 1."""

    lower_t = torch.as_tensor(lower)
    upper_t = torch.as_tensor(upper, dtype=lower_t.dtype, device=lower_t.device)
    return 1.0 - upper_t, 1.0 - lower_t


class TrainableEndpointAdapter(nn.Module):
    """Two bounded scene parameters optimized against detached CEF fields."""

    def __init__(
        self,
        field_config: EndpointFieldConfig = EndpointFieldConfig(),
        adapter_config: EndpointAdapterConfig = EndpointAdapterConfig(),
        *,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        field_config.validate()
        adapter_config.validate(field_config)
        self.field_config = field_config
        self.adapter_config = adapter_config
        lower_fraction = (-0.5 - field_config.lower_scan) / (
            -adapter_config.min_extension - field_config.lower_scan
        )
        upper_fraction = (1.5 - (1.0 + adapter_config.min_extension)) / (
            field_config.upper_scan - (1.0 + adapter_config.min_extension)
        )
        lower_fraction = min(max(lower_fraction, 1.0e-4), 1.0 - 1.0e-4)
        upper_fraction = min(max(upper_fraction, 1.0e-4), 1.0 - 1.0e-4)
        lower_logit = torch.logit(torch.tensor(lower_fraction, dtype=dtype, device=device))
        upper_logit = torch.logit(torch.tensor(upper_fraction, dtype=dtype, device=device))
        self.lower_logit = nn.Parameter(lower_logit)
        self.upper_logit = nn.Parameter(upper_logit)

    def endpoint_scalars(self) -> tuple[Tensor, Tensor]:
        cfg, eps = self.field_config, self.adapter_config.min_extension
        lower = cfg.lower_scan + (-eps - cfg.lower_scan) * torch.sigmoid(self.lower_logit)
        upper = 1.0 + eps + (cfg.upper_scan - (1.0 + eps)) * torch.sigmoid(self.upper_logit)
        return lower, upper

    def _interpolated_energy(self, field: EndpointEnergyField, scalar: Tensor) -> Tensor:
        if field.scalars.numel() < 2:
            raise ValueError("adapter requires a sampled endpoint field")
        # Piecewise-linear interpolation gives a non-vanishing gradient between
        # samples.  The earlier narrow RBF could strand a scalar between rugged
        # contact minima while still publishing a deeply penetrating point.
        grid = field.scalars.detach()
        index = torch.searchsorted(grid, scalar.detach()).clamp(1, grid.numel() - 1)
        lo, hi = index - 1, index
        fraction = (scalar - grid[lo]) / (grid[hi] - grid[lo]).clamp_min(torch.finfo(grid.dtype).eps)
        energy = field.total_energy.detach()
        return energy[lo] * (1.0 - fraction) + energy[hi] * fraction

    def loss_from_predictions(self, predictions: Sequence[ContactEndpointPrediction]) -> Tensor:
        if not predictions:
            raise ValueError("at least one independently reconstructed state is required")
        lower, upper = self.endpoint_scalars()
        physical_terms = []
        anchor_lower = []
        anchor_upper = []
        for prediction in predictions:
            physical_terms.extend(
                (
                    self._interpolated_energy(prediction.lower.field, lower),
                    self._interpolated_energy(prediction.upper.field, upper),
                )
            )
            anchor_lower.append(prediction.lower.raw_value.detach())
            anchor_upper.append(prediction.upper.raw_value.detach())
        physical = torch.stack(physical_terms).mean()
        lower_target = torch.stack(anchor_lower).mean()
        upper_target = torch.stack(anchor_upper).mean()
        scan_span = self.field_config.upper_scan - self.field_config.lower_scan
        anchor = ((lower - lower_target) / scan_span).square() + ((upper - upper_target) / scan_span).square()
        return physical + self.adapter_config.anchor_weight * anchor


def _geometry_kwargs(geometry: EndpointGeometry) -> dict:
    return {
        "static_weights": geometry.static_weights,
        "mobile_weights": geometry.mobile_weights,
        "pair_index": geometry.pair_index,
    }


def fit_endpoint_adapter(
    geometries: Sequence[EndpointGeometry],
    *,
    joint_kind: JointKind,
    axis: Tensor,
    observed_displacement: Tensor | float,
    pivot: Optional[Tensor] = None,
    field_config: EndpointFieldConfig = EndpointFieldConfig(),
    adapter_config: EndpointAdapterConfig = EndpointAdapterConfig(),
    counterfactual_field_configs: Optional[Sequence[EndpointFieldConfig]] = None,
) -> tuple[TrainableEndpointAdapter, EndpointAdapterPrediction]:
    """Fit D2-CEA with a fixed schedule and return recomputed certificates."""

    if not geometries:
        raise ValueError("at least one endpoint geometry is required")
    detached = tuple(geometry.detached() for geometry in geometries)
    axis_detached = torch.as_tensor(axis).detach()
    displacement_detached = torch.as_tensor(observed_displacement).detach()
    pivot_detached = None if pivot is None else torch.as_tensor(pivot).detach()
    field_configs = tuple(counterfactual_field_configs or (field_config,))
    if not field_configs:
        raise ValueError("at least one field configuration is required")
    for candidate_config in field_configs:
        if (candidate_config.lower_scan, candidate_config.upper_scan, candidate_config.samples_per_side) != (
            field_config.lower_scan,
            field_config.upper_scan,
            field_config.samples_per_side,
        ):
            raise ValueError("counterfactual fields must share the adapter scan grid")
    initial = tuple(
        infer_contact_feasible_endpoints(
            geometry.static_means,
            geometry.static_scales,
            geometry.mobile_means_state0,
            geometry.mobile_scales,
            joint_kind=joint_kind,
            axis=axis_detached,
            observed_displacement=displacement_detached,
            pivot=pivot_detached,
            config=candidate_config,
            **_geometry_kwargs(geometry),
        )
        for geometry in detached
        for candidate_config in field_configs
    )
    dtype, device = detached[0].static_means.dtype, detached[0].static_means.device
    adapter = TrainableEndpointAdapter(field_config, adapter_config, dtype=dtype, device=device)
    optimizer = torch.optim.Adam(adapter.parameters(), lr=adapter_config.learning_rate)
    final_loss = torch.zeros((), dtype=dtype, device=device)
    for _ in range(adapter_config.iterations):
        optimizer.zero_grad(set_to_none=True)
        final_loss = adapter.loss_from_predictions(initial)
        final_loss.backward()
        optimizer.step()

    lower, upper = adapter.endpoint_scalars()
    certificates = tuple(
        certify_contact_feasible_endpoints(
            geometry.static_means,
            geometry.static_scales,
            geometry.mobile_means_state0,
            geometry.mobile_scales,
            lower_scalar=lower,
            upper_scalar=upper,
            joint_kind=joint_kind,
            axis=axis_detached,
            observed_displacement=displacement_detached,
            pivot=pivot_detached,
            config=candidate_config,
            **_geometry_kwargs(geometry),
        )
        for geometry in detached
        for candidate_config in field_configs
    )
    closed_labels = [certificate.closed_end for certificate in certificates]
    lower_votes = closed_labels.count("lower")
    upper_votes = closed_labels.count("upper")
    required_votes = max(1, (2 * len(closed_labels) + 2) // 3)
    closed_identifiable = max(lower_votes, upper_votes) >= required_votes and lower_votes != upper_votes
    closed_end = "lower" if closed_identifiable and lower_votes > upper_votes else (
        "upper" if closed_identifiable else "unknown"
    )
    prediction = EndpointAdapterPrediction(
        lower_scalar=lower.detach(),
        upper_scalar=upper.detach(),
        closed_end=closed_end,
        closed_end_identifiable=closed_identifiable,
        lower_identifiable=all(certificate.lower.identifiable for certificate in certificates),
        upper_identifiable=all(certificate.upper.identifiable for certificate in certificates),
        state_certificates=certificates,
        initial_fields=initial,
        final_loss=final_loss.detach(),
    )
    return adapter, prediction


def fixed_multiradius_counterfactuals(base: EndpointFieldConfig) -> tuple[EndpointFieldConfig, ...]:
    """Predeclared radius/delta probes with a hard penetration barrier."""

    return tuple(
        replace(
            base,
            radius_scale=radius_scale,
            support_step=support_step,
            penetration_weight=64.0,
            inside_weight=64.0,
        )
        for radius_scale, support_step in ((0.75, 0.02), (1.0, 0.04), (1.25, 0.08))
    )


__all__ = [
    "EndpointAdapterConfig",
    "EndpointAdapterPrediction",
    "EndpointGeometry",
    "TrainableEndpointAdapter",
    "canonicalize_state1_mobile",
    "fit_endpoint_adapter",
    "fixed_multiradius_counterfactuals",
    "swap_endpoint_scalars",
]
