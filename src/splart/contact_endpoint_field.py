"""Contact-feasible endpoint inference for articulated Gaussian geometry.

This module deliberately does not consume endpoint labels, absolute interior
fractions, endpoint images, or URDF limits.  It treats the two observed states
as coordinates 0 and 1 and asks where the learned screw trajectory first
encounters a physically supported contact on either side of that interval.

The prototype uses a conservative spherical envelope for each Gaussian.  The
energy field is differentiable with respect to Gaussian centres/scales and the
predicted screw parameters (apart from the standard subgradient at a nearest
pair switch).  Large models should pass a deterministic, geometry-only
``pair_index`` produced by a broad-phase collision detector; dense pairing is
intentionally bounded to prevent accidental quadratic allocations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch
from torch import Tensor
from torch.nn import functional as F


JointKind = Literal["revolute", "prismatic"]
ClosedEnd = Literal["lower", "upper", "unknown"]


@dataclass(frozen=True)
class EndpointFieldConfig:
    """Scene-independent settings expressed in observed-span coordinates."""

    lower_scan: float = -2.0
    upper_scan: float = 3.0
    samples_per_side: int = 257
    support_step: float = 0.04
    radius_scale: float = 1.0
    contact_band_radius: float = 0.35
    contact_band_scene: float = 2.0e-3
    support_margin_bands: float = 0.15
    max_contact_gap_bands: float = 0.75
    max_inside_penetration_bands: float = 0.35
    max_normalized_entropy: float = 0.82
    closed_mass_margin: float = 0.12
    softargmin_temperature: float = 0.035
    penetration_weight: float = 6.0
    support_weight: float = 2.0
    inside_weight: float = 5.0
    max_dense_pairs: int = 2_000_000

    def validate(self) -> None:
        if not self.lower_scan < 0.0:
            raise ValueError("lower_scan must be strictly below observed state 0")
        if not self.upper_scan > 1.0:
            raise ValueError("upper_scan must be strictly above observed state 1")
        if self.samples_per_side < 5:
            raise ValueError("samples_per_side must be at least 5")
        if self.support_step <= 0.0:
            raise ValueError("support_step must be positive")
        if self.radius_scale <= 0.0:
            raise ValueError("radius_scale must be positive")
        if self.contact_band_radius <= 0.0 or self.contact_band_scene <= 0.0:
            raise ValueError("contact bandwidths must be positive")
        if self.softargmin_temperature <= 0.0:
            raise ValueError("softargmin_temperature must be positive")
        if self.max_dense_pairs < 1:
            raise ValueError("max_dense_pairs must be positive")


@dataclass(frozen=True)
class EndpointEnergyField:
    """Differentiable evidence evaluated along one extrapolation direction."""

    scalars: Tensor
    signed_gap: Tensor
    contact_energy: Tensor
    penetration_energy: Tensor
    support_energy: Tensor
    inside_penetration_energy: Tensor
    contact_mass: Tensor
    support_rise: Tensor
    total_energy: Tensor
    posterior: Tensor


@dataclass(frozen=True)
class EndpointEstimate:
    """One endpoint estimate and its physical evidence certificate."""

    value: Tensor
    raw_value: Tensor
    interval: tuple[Tensor, Tensor]
    identifiable: bool
    uncertainty: Tensor
    contact_gap: Tensor
    contact_mass: Tensor
    support_rise: Tensor
    normalized_entropy: Tensor
    boundary_selected: bool
    field: EndpointEnergyField


@dataclass(frozen=True)
class ContactEndpointPrediction:
    """Fail-closed endpoint prediction.

    ``lower.value`` or ``upper.value`` is NaN when that stop is not physically
    identifiable.  The raw soft estimate remains available for diagnostics but
    must not be treated as a predicted endpoint.
    """

    lower: EndpointEstimate
    upper: EndpointEstimate
    closed_end: ClosedEnd
    closed_end_identifiable: bool
    identifiable: bool
    closed_end_uncertainty: Tensor
    scene_scale: Tensor
    contact_band: Tensor


def dense_pair_index(num_mobile: int, num_static: int, *, device: torch.device) -> Tensor:
    """Return all mobile/static candidate pairs for small analytic problems."""

    mobile_ids = torch.arange(num_mobile, device=device).repeat_interleave(num_static)
    static_ids = torch.arange(num_static, device=device).repeat(num_mobile)
    return torch.stack((mobile_ids, static_ids), dim=0)


def _validate_geometry(name: str, means: Tensor, scales: Tensor) -> None:
    if means.ndim != 2 or means.shape[-1] != 3:
        raise ValueError(f"{name}_means must have shape [N, 3]")
    if scales.shape != means.shape:
        raise ValueError(f"{name}_scales must match {name}_means")
    if means.shape[0] == 0:
        raise ValueError(f"{name} geometry must be non-empty")
    if not means.is_floating_point() or not scales.is_floating_point():
        raise TypeError(f"{name} geometry must use a floating dtype")
    if means.device != scales.device:
        raise ValueError(f"{name} means/scales must share a device")
    if not torch.isfinite(means).all() or not torch.isfinite(scales).all():
        raise ValueError(f"{name} geometry must be finite")
    if not (scales > 0).all():
        raise ValueError(f"{name}_scales must be strictly positive")


def _normalize_axis(axis: Tensor, *, dtype: torch.dtype, device: torch.device) -> Tensor:
    axis = torch.as_tensor(axis, dtype=dtype, device=device)
    if axis.shape != (3,) or not torch.isfinite(axis).all():
        raise ValueError("axis must be a finite tensor with shape [3]")
    norm = torch.linalg.vector_norm(axis)
    if float(norm.detach().cpu()) <= torch.finfo(dtype).eps:
        raise ValueError("axis must be non-zero")
    return axis / norm


def _transform_mobile(
    mobile_means: Tensor,
    scalars: Tensor,
    *,
    joint_kind: JointKind,
    axis: Tensor,
    pivot: Tensor,
    observed_displacement: Tensor,
) -> Tensor:
    """Transform state-0 mobile centres to arbitrary observed-span scalars."""

    if joint_kind == "prismatic":
        offset = scalars[:, None, None] * observed_displacement * axis[None, None, :]
        return mobile_means[None, :, :] + offset

    theta = scalars * observed_displacement
    x = mobile_means - pivot
    # Rodrigues' formula, vectorized over scan scalars and mobile Gaussians.
    cross = torch.linalg.cross(axis.expand_as(x), x, dim=-1)
    projection = (x * axis).sum(dim=-1, keepdim=True) * axis
    cos = torch.cos(theta)[:, None, None]
    sin = torch.sin(theta)[:, None, None]
    rotated = x[None, :, :] * cos + cross[None, :, :] * sin + projection[None, :, :] * (1.0 - cos)
    return rotated + pivot


def _pair_gaps(
    transformed_mobile: Tensor, static_means: Tensor, mobile_radii: Tensor, static_radii: Tensor, pair_index: Tensor
) -> Tensor:
    mobile_ids, static_ids = pair_index
    delta = transformed_mobile[:, mobile_ids] - static_means[static_ids][None, :, :]
    centre_distance = torch.linalg.vector_norm(delta, dim=-1)
    return centre_distance - (mobile_radii[mobile_ids] + static_radii[static_ids])[None, :]


def _weighted_contact_mass(pair_gaps: Tensor, band: Tensor, pair_weights: Tensor) -> Tensor:
    kernel = torch.exp(-0.5 * (pair_gaps / band).square())
    return (kernel * pair_weights[None, :]).sum(dim=-1) / pair_weights.sum().clamp_min(torch.finfo(pair_gaps.dtype).eps)


def _field_for_side(
    scalars: Tensor,
    direction: float,
    *,
    mobile_means: Tensor,
    static_means: Tensor,
    mobile_radii: Tensor,
    static_radii: Tensor,
    pair_index: Tensor,
    pair_weights: Tensor,
    joint_kind: JointKind,
    axis: Tensor,
    pivot: Tensor,
    observed_displacement: Tensor,
    band: Tensor,
    config: EndpointFieldConfig,
) -> EndpointEnergyField:
    def gaps_at(q: Tensor) -> Tensor:
        transformed = _transform_mobile(
            mobile_means, q, joint_kind=joint_kind, axis=axis, pivot=pivot, observed_displacement=observed_displacement
        )
        return _pair_gaps(transformed, static_means, mobile_radii, static_radii, pair_index)

    pair_gaps = gaps_at(scalars)
    outside_pair_gaps = gaps_at(scalars + direction * config.support_step)
    inside_pair_gaps = gaps_at(scalars - direction * config.support_step)

    signed_gap = pair_gaps.min(dim=-1).values
    outside_gap = outside_pair_gaps.min(dim=-1).values
    inside_gap = inside_pair_gaps.min(dim=-1).values
    normalized_gap = signed_gap / band

    # Surface contact, no penetration at the endpoint, a rising collision
    # barrier immediately beyond it, and a collision-free point just inside.
    contact_energy = normalized_gap.square()
    penetration_energy = F.relu(-normalized_gap).square()
    support_rise = F.relu((signed_gap - outside_gap) / band)
    support_energy = F.relu(config.support_margin_bands - support_rise).square()
    inside_penetration_energy = F.relu(-inside_gap / band).square()
    contact_mass = _weighted_contact_mass(pair_gaps, band, pair_weights)

    total = (
        contact_energy
        + config.penetration_weight * penetration_energy
        + config.support_weight * support_energy
        + config.inside_weight * inside_penetration_energy
    )
    logits = -(total - total.detach().min()) / config.softargmin_temperature
    posterior = torch.softmax(logits, dim=0)
    return EndpointEnergyField(
        scalars=scalars,
        signed_gap=signed_gap,
        contact_energy=contact_energy,
        penetration_energy=penetration_energy,
        support_energy=support_energy,
        inside_penetration_energy=inside_penetration_energy,
        contact_mass=contact_mass,
        support_rise=support_rise,
        total_energy=total,
        posterior=posterior,
    )


def _posterior_interval(scalars: Tensor, posterior: Tensor, mass: float = 0.9) -> tuple[Tensor, Tensor]:
    cdf = posterior.cumsum(dim=0)
    tail = (1.0 - mass) / 2.0
    lo_idx = torch.searchsorted(cdf.detach(), torch.as_tensor(tail, dtype=cdf.dtype, device=cdf.device))
    hi_idx = torch.searchsorted(cdf.detach(), torch.as_tensor(1.0 - tail, dtype=cdf.dtype, device=cdf.device))
    lo_idx = lo_idx.clamp_max(scalars.numel() - 1)
    hi_idx = hi_idx.clamp_max(scalars.numel() - 1)
    return scalars[lo_idx], scalars[hi_idx]


def _summarize_endpoint(field: EndpointEnergyField, config: EndpointFieldConfig) -> EndpointEstimate:
    index = int(field.total_energy.detach().argmin().cpu())
    raw_value = (field.scalars * field.posterior).sum()
    eps = torch.finfo(field.posterior.dtype).eps
    entropy = -(field.posterior * field.posterior.clamp_min(eps).log()).sum()
    entropy = entropy / torch.log(torch.as_tensor(field.posterior.numel(), dtype=entropy.dtype, device=entropy.device))
    gap_bands = field.contact_energy[index].clamp_min(0.0).sqrt()
    inside_penetration_bands = field.inside_penetration_energy[index].clamp_min(0.0).sqrt()
    boundary_selected = index in {0, field.scalars.numel() - 1}
    identifiable = bool(
        not boundary_selected
        and float(gap_bands.detach().cpu()) <= config.max_contact_gap_bands
        and float(field.support_rise[index].detach().cpu()) >= config.support_margin_bands
        and float(inside_penetration_bands.detach().cpu()) <= config.max_inside_penetration_bands
        and float(entropy.detach().cpu()) <= config.max_normalized_entropy
    )

    contact_confidence = torch.exp(-0.5 * gap_bands.square())
    support_confidence = torch.sigmoid(8.0 * (field.support_rise[index] - config.support_margin_bands))
    concentration = (1.0 - entropy).clamp(0.0, 1.0)
    confidence = contact_confidence * support_confidence * concentration
    if boundary_selected:
        confidence = confidence * 0.0
    uncertainty = (1.0 - confidence).clamp(0.0, 1.0)
    value = raw_value if identifiable else torch.full_like(raw_value, torch.nan)
    return EndpointEstimate(
        value=value,
        raw_value=raw_value,
        interval=_posterior_interval(field.scalars, field.posterior),
        identifiable=identifiable,
        uncertainty=uncertainty,
        contact_gap=field.signed_gap[index],
        contact_mass=field.contact_mass[index],
        support_rise=field.support_rise[index],
        normalized_entropy=entropy,
        boundary_selected=boundary_selected,
        field=field,
    )


def infer_contact_feasible_endpoints(
    static_means: Tensor,
    static_scales: Tensor,
    mobile_means_state0: Tensor,
    mobile_scales: Tensor,
    *,
    joint_kind: JointKind,
    axis: Tensor,
    observed_displacement: Tensor | float,
    pivot: Optional[Tensor] = None,
    static_weights: Optional[Tensor] = None,
    mobile_weights: Optional[Tensor] = None,
    pair_index: Optional[Tensor] = None,
    config: EndpointFieldConfig = EndpointFieldConfig(),
) -> ContactEndpointPrediction:
    """Infer two physical stops from geometry and a learned screw trajectory.

    State 0 is the reference geometry, state 1 lies at scalar 1, and all
    returned scalars use that same observed-span coordinate system.  The method
    has no parameter through which absolute interior fractions or true joint
    limits can enter.
    """

    config.validate()
    _validate_geometry("static", static_means, static_scales)
    _validate_geometry("mobile", mobile_means_state0, mobile_scales)
    if static_means.device != mobile_means_state0.device or static_means.dtype != mobile_means_state0.dtype:
        raise ValueError("static and mobile geometry must share dtype and device")
    if joint_kind not in {"revolute", "prismatic"}:
        raise ValueError("joint_kind must be 'revolute' or 'prismatic'")

    dtype, device = static_means.dtype, static_means.device
    axis = _normalize_axis(axis, dtype=dtype, device=device)
    displacement = torch.as_tensor(observed_displacement, dtype=dtype, device=device)
    if displacement.ndim != 0 or not torch.isfinite(displacement):
        raise ValueError("observed_displacement must be a finite scalar")
    if float(displacement.detach().abs().cpu()) <= torch.finfo(dtype).eps:
        raise ValueError("observed_displacement must be non-zero")
    if pivot is None:
        pivot = torch.zeros(3, dtype=dtype, device=device)
    else:
        pivot = torch.as_tensor(pivot, dtype=dtype, device=device)
    if pivot.shape != (3,) or not torch.isfinite(pivot).all():
        raise ValueError("pivot must be a finite tensor with shape [3]")

    num_mobile, num_static = mobile_means_state0.shape[0], static_means.shape[0]
    if pair_index is None:
        if num_mobile * num_static > config.max_dense_pairs:
            raise ValueError(
                "dense contact pairing exceeds max_dense_pairs; pass a deterministic geometry-only broad-phase pair_index"
            )
        pair_index = dense_pair_index(num_mobile, num_static, device=device)
    else:
        pair_index = torch.as_tensor(pair_index, dtype=torch.long, device=device)
    if pair_index.ndim != 2 or pair_index.shape[0] != 2 or pair_index.shape[1] == 0:
        raise ValueError("pair_index must have shape [2, P] with P > 0")
    if pair_index[0].min() < 0 or pair_index[0].max() >= num_mobile:
        raise ValueError("pair_index contains an invalid mobile index")
    if pair_index[1].min() < 0 or pair_index[1].max() >= num_static:
        raise ValueError("pair_index contains an invalid static index")

    def weights_or_ones(value: Optional[Tensor], count: int, name: str) -> Tensor:
        if value is None:
            return torch.ones(count, dtype=dtype, device=device)
        value = torch.as_tensor(value, dtype=dtype, device=device)
        if value.shape != (count,) or not torch.isfinite(value).all() or not (value >= 0).all():
            raise ValueError(f"{name} must be finite, non-negative, and have shape [{count}]")
        if float(value.sum().detach().cpu()) <= 0.0:
            raise ValueError(f"{name} must have positive support")
        return value

    static_weights = weights_or_ones(static_weights, num_static, "static_weights")
    mobile_weights = weights_or_ones(mobile_weights, num_mobile, "mobile_weights")
    pair_weights = mobile_weights[pair_index[0]] * static_weights[pair_index[1]]
    if float(pair_weights.sum().detach().cpu()) <= 0.0:
        raise ValueError("candidate pair weights must have positive support")

    static_radii = static_scales.square().mean(dim=-1).sqrt() * config.radius_scale
    mobile_radii = mobile_scales.square().mean(dim=-1).sqrt() * config.radius_scale
    all_means = torch.cat((static_means, mobile_means_state0), dim=0)
    scene_scale = torch.linalg.vector_norm(all_means.max(dim=0).values - all_means.min(dim=0).values)
    median_radius = torch.cat((static_radii, mobile_radii)).median()
    numerical_floor = torch.as_tensor(torch.finfo(dtype).eps ** 0.5, dtype=dtype, device=device)
    scene_scale = scene_scale.clamp_min(median_radius).clamp_min(numerical_floor)
    band = torch.maximum(config.contact_band_radius * median_radius, config.contact_band_scene * scene_scale)

    lower_scalars = torch.linspace(config.lower_scan, 0.0, config.samples_per_side + 1, dtype=dtype, device=device)[:-1]
    upper_scalars = torch.linspace(1.0, config.upper_scan, config.samples_per_side + 1, dtype=dtype, device=device)[1:]
    common = dict(
        mobile_means=mobile_means_state0,
        static_means=static_means,
        mobile_radii=mobile_radii,
        static_radii=static_radii,
        pair_index=pair_index,
        pair_weights=pair_weights,
        joint_kind=joint_kind,
        axis=axis,
        pivot=pivot,
        observed_displacement=displacement,
        band=band,
        config=config,
    )
    lower_field = _field_for_side(lower_scalars, -1.0, **common)
    upper_field = _field_for_side(upper_scalars, 1.0, **common)
    lower = _summarize_endpoint(lower_field, config)
    upper = _summarize_endpoint(upper_field, config)

    mass_sum = lower.contact_mass + upper.contact_mass
    mass_separation = (lower.contact_mass - upper.contact_mass).abs() / mass_sum.clamp_min(torch.finfo(dtype).eps)
    closed_identifiable = bool(
        lower.identifiable and upper.identifiable and float(mass_separation.detach().cpu()) >= config.closed_mass_margin
    )
    if not closed_identifiable:
        closed_end: ClosedEnd = "unknown"
    elif float(lower.contact_mass.detach().cpu()) > float(upper.contact_mass.detach().cpu()):
        closed_end = "lower"
    else:
        closed_end = "upper"
    closed_uncertainty = (1.0 - mass_separation).clamp(0.0, 1.0)
    identifiable = lower.identifiable and upper.identifiable and closed_identifiable
    return ContactEndpointPrediction(
        lower=lower,
        upper=upper,
        closed_end=closed_end,
        closed_end_identifiable=closed_identifiable,
        identifiable=identifiable,
        closed_end_uncertainty=closed_uncertainty,
        scene_scale=scene_scale,
        contact_band=band,
    )


__all__ = [
    "ContactEndpointPrediction",
    "EndpointEnergyField",
    "EndpointEstimate",
    "EndpointFieldConfig",
    "dense_pair_index",
    "infer_contact_feasible_endpoints",
]
