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
from itertools import product
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
    min_geometry_weight: float = 1.0e-2
    max_broad_phase_pairs: int = 65_536
    broad_phase_samples: int = 33
    broad_phase_neighbor_radius: int = 1
    pair_chunk_size: int = 16_384
    scalar_chunk_size: int = 32

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
        if not 0.0 <= self.min_geometry_weight <= 1.0:
            raise ValueError("min_geometry_weight must lie in [0, 1]")
        if self.max_broad_phase_pairs < 1:
            raise ValueError("max_broad_phase_pairs must be positive")
        if self.broad_phase_samples < 3:
            raise ValueError("broad_phase_samples must be at least 3")
        if self.broad_phase_neighbor_radius < 0:
            raise ValueError("broad_phase_neighbor_radius must be non-negative")
        if self.pair_chunk_size < 1 or self.scalar_chunk_size < 1:
            raise ValueError("chunk sizes must be positive")


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


@dataclass(frozen=True)
class EndpointCandidateCertificate:
    """Physical evidence recomputed at one published endpoint scalar."""

    value: Tensor
    identifiable: bool
    uncertainty: Tensor
    contact_gap: Tensor
    contact_mass: Tensor
    support_rise: Tensor
    inside_penetration_bands: Tensor
    boundary_selected: bool


@dataclass(frozen=True)
class CertifiedEndpointPair:
    """A pair of explicitly supplied endpoint scalars and their certificates."""

    lower: EndpointCandidateCertificate
    upper: EndpointCandidateCertificate
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


@torch.no_grad()
def trajectory_voxel_pair_index(
    static_means: Tensor,
    mobile_means_state0: Tensor,
    scalars: Tensor,
    *,
    joint_kind: JointKind,
    axis: Tensor,
    pivot: Tensor,
    observed_displacement: Tensor,
    voxel_size: Tensor | float,
    neighbor_radius: int = 1,
    max_pairs: int = 65_536,
) -> Tensor:
    """Build deterministic candidate pairs over the complete screw trajectory.

    Mobile centres are sampled at fixed scalar positions, then paired with
    static centres in neighboring voxels.  Candidate pairs are ranked by their
    closest sampled centre distance before applying the fixed memory cap.  The
    operation is deliberately detached: broad-phase membership is not a
    learnable or evaluator-dependent selection mechanism.
    """

    if scalars.ndim != 1 or scalars.numel() < 2:
        raise ValueError("scalars must be a one-dimensional tensor with at least two values")
    if neighbor_radius < 0 or max_pairs < 1:
        raise ValueError("neighbor_radius must be non-negative and max_pairs positive")
    cell = float(torch.as_tensor(voxel_size).detach().cpu())
    if not cell > 0.0:
        raise ValueError("voxel_size must be positive")

    static_cpu = static_means.detach().to(device="cpu", dtype=torch.float64)
    transformed_cpu = _transform_mobile(
        mobile_means_state0.detach(),
        scalars.detach(),
        joint_kind=joint_kind,
        axis=axis.detach(),
        pivot=pivot.detach(),
        observed_displacement=observed_displacement.detach(),
    ).to(device="cpu", dtype=torch.float64)
    static_cells = torch.floor(static_cpu / cell).to(torch.int64)
    buckets: dict[tuple[int, int, int], list[int]] = {}
    for static_id, key_tensor in enumerate(static_cells):
        key = tuple(int(v) for v in key_tensor.tolist())
        buckets.setdefault(key, []).append(static_id)

    offsets = tuple(product(range(-neighbor_radius, neighbor_radius + 1), repeat=3))
    best_distance: dict[tuple[int, int], float] = {}
    for positions in transformed_cpu:
        mobile_cells = torch.floor(positions / cell).to(torch.int64)
        for mobile_id, key_tensor in enumerate(mobile_cells):
            key = tuple(int(v) for v in key_tensor.tolist())
            for offset in offsets:
                neighbor = (key[0] + offset[0], key[1] + offset[1], key[2] + offset[2])
                for static_id in buckets.get(neighbor, ()):
                    pair = (mobile_id, static_id)
                    distance = float(torch.linalg.vector_norm(positions[mobile_id] - static_cpu[static_id]))
                    previous = best_distance.get(pair)
                    if previous is None or distance < previous:
                        best_distance[pair] = distance

    # A no-contact scene may produce no neighboring voxels.  Keep one explicit
    # deterministic far-field pair so downstream logic can certify "unknown"
    # instead of failing or silently expanding an unbounded dense allocation.
    if not best_distance:
        best_distance[(0, 0)] = float(torch.linalg.vector_norm(transformed_cpu[0, 0] - static_cpu[0]))
    ranked = sorted(best_distance, key=lambda pair: (best_distance[pair], pair[0], pair[1]))[:max_pairs]
    return torch.tensor(ranked, dtype=torch.long, device=static_means.device).t().contiguous()


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


def _chunked_gap_statistics(
    transformed_mobile: Tensor,
    static_means: Tensor,
    mobile_radii: Tensor,
    static_radii: Tensor,
    pair_index: Tensor,
    pair_weights: Tensor,
    band: Tensor,
    pair_chunk_size: int,
    *,
    compute_contact_mass: bool,
) -> tuple[Tensor, Tensor | None]:
    """Accumulate minimum gap and contact mass without a [S, P, 3] tensor."""

    min_gap = torch.full(
        (transformed_mobile.shape[0],), torch.inf, dtype=transformed_mobile.dtype, device=transformed_mobile.device
    )
    contact_numerator = torch.zeros_like(min_gap) if compute_contact_mass else None
    for start in range(0, pair_index.shape[1], pair_chunk_size):
        end = min(start + pair_chunk_size, pair_index.shape[1])
        chunk_index = pair_index[:, start:end]
        gaps = _pair_gaps(transformed_mobile, static_means, mobile_radii, static_radii, chunk_index)
        min_gap = torch.minimum(min_gap, gaps.min(dim=-1).values)
        if contact_numerator is not None:
            kernel = torch.exp(-0.5 * (gaps / band).square())
            contact_numerator = contact_numerator + (kernel * pair_weights[None, start:end]).sum(dim=-1)
    if contact_numerator is None:
        return min_gap, None
    denominator = pair_weights.sum().clamp_min(torch.finfo(transformed_mobile.dtype).eps)
    return min_gap, contact_numerator / denominator


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
    def stats_at(q: Tensor, *, compute_contact_mass: bool) -> tuple[Tensor, Tensor | None]:
        gaps: list[Tensor] = []
        masses: list[Tensor] = []
        for q_chunk in q.split(config.scalar_chunk_size):
            transformed = _transform_mobile(
                mobile_means,
                q_chunk,
                joint_kind=joint_kind,
                axis=axis,
                pivot=pivot,
                observed_displacement=observed_displacement,
            )
            chunk_gap, chunk_mass = _chunked_gap_statistics(
                transformed,
                static_means,
                mobile_radii,
                static_radii,
                pair_index,
                pair_weights,
                band,
                config.pair_chunk_size,
                compute_contact_mass=compute_contact_mass,
            )
            gaps.append(chunk_gap)
            if chunk_mass is not None:
                masses.append(chunk_mass)
        return torch.cat(gaps), torch.cat(masses) if masses else None

    signed_gap, contact_mass = stats_at(scalars, compute_contact_mass=True)
    outside_gap, _ = stats_at(scalars + direction * config.support_step, compute_contact_mass=False)
    inside_gap, _ = stats_at(scalars - direction * config.support_step, compute_contact_mass=False)
    assert contact_mass is not None
    normalized_gap = signed_gap / band

    # Surface contact, no penetration at the endpoint, a rising collision
    # barrier immediately beyond it, and a collision-free point just inside.
    contact_energy = normalized_gap.square()
    penetration_energy = F.relu(-normalized_gap).square()
    support_rise = F.relu((signed_gap - outside_gap) / band)
    support_energy = F.relu(config.support_margin_bands - support_rise).square()
    inside_penetration_energy = F.relu(-inside_gap / band).square()
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


def _summarize_endpoint(
    field: EndpointEnergyField, final_field: EndpointEnergyField, config: EndpointFieldConfig
) -> EndpointEstimate:
    """Summarize a scan while certifying the exact scalar that is published."""

    raw_value = final_field.scalars[0]
    eps = torch.finfo(field.posterior.dtype).eps
    entropy = -(field.posterior * field.posterior.clamp_min(eps).log()).sum()
    entropy = entropy / torch.log(torch.as_tensor(field.posterior.numel(), dtype=entropy.dtype, device=entropy.device))
    gap_bands = final_field.contact_energy[0].clamp_min(0.0).sqrt()
    inside_penetration_bands = final_field.inside_penetration_energy[0].clamp_min(0.0).sqrt()
    grid_step = (field.scalars[-1] - field.scalars[0]).abs() / max(field.scalars.numel() - 1, 1)
    boundary_selected = bool(
        (raw_value - field.scalars[0]).abs() <= grid_step / 2
        or (raw_value - field.scalars[-1]).abs() <= grid_step / 2
    )
    identifiable = bool(
        not boundary_selected
        and float(gap_bands.detach().cpu()) <= config.max_contact_gap_bands
        and float(final_field.support_rise[0].detach().cpu()) >= config.support_margin_bands
        and float(inside_penetration_bands.detach().cpu()) <= config.max_inside_penetration_bands
        and float(entropy.detach().cpu()) <= config.max_normalized_entropy
    )

    contact_confidence = torch.exp(-0.5 * gap_bands.square())
    support_confidence = torch.sigmoid(8.0 * (final_field.support_rise[0] - config.support_margin_bands))
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
        contact_gap=final_field.signed_gap[0],
        contact_mass=final_field.contact_mass[0],
        support_rise=final_field.support_rise[0],
        normalized_entropy=entropy,
        boundary_selected=boundary_selected,
        field=field,
    )


@dataclass(frozen=True)
class _PreparedContactScene:
    common: dict
    scene_scale: Tensor
    contact_band: Tensor


def _prepare_contact_scene(
    static_means: Tensor,
    static_scales: Tensor,
    mobile_means_state0: Tensor,
    mobile_scales: Tensor,
    *,
    joint_kind: JointKind,
    axis: Tensor,
    observed_displacement: Tensor | float,
    pivot: Optional[Tensor],
    static_weights: Optional[Tensor],
    mobile_weights: Optional[Tensor],
    pair_index: Optional[Tensor],
    config: EndpointFieldConfig,
) -> _PreparedContactScene:
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

    def weights_or_ones(value: Optional[Tensor], count: int, name: str) -> Tensor:
        if value is None:
            return torch.ones(count, dtype=dtype, device=device)
        value = torch.as_tensor(value, dtype=dtype, device=device)
        if value.shape != (count,) or not torch.isfinite(value).all() or not (value >= 0).all():
            raise ValueError(f"{name} must be finite, non-negative, and have shape [{count}]")
        if float(value.sum().detach().cpu()) <= 0.0:
            raise ValueError(f"{name} must have positive support")
        return value

    num_mobile, num_static = mobile_means_state0.shape[0], static_means.shape[0]
    static_weights = weights_or_ones(static_weights, num_static, "static_weights")
    mobile_weights = weights_or_ones(mobile_weights, num_mobile, "mobile_weights")
    active_static = torch.where(static_weights >= config.min_geometry_weight)[0]
    active_mobile = torch.where(mobile_weights >= config.min_geometry_weight)[0]
    if active_static.numel() == 0 or active_mobile.numel() == 0:
        raise ValueError("fixed geometry-weight threshold removed all static or mobile support")

    static_radii = static_scales.square().mean(dim=-1).sqrt() * config.radius_scale
    mobile_radii = mobile_scales.square().mean(dim=-1).sqrt() * config.radius_scale
    all_means = torch.cat((static_means[active_static], mobile_means_state0[active_mobile]), dim=0)
    scene_scale = torch.linalg.vector_norm(all_means.max(dim=0).values - all_means.min(dim=0).values)
    median_radius = torch.cat((static_radii[active_static], mobile_radii[active_mobile])).median()
    numerical_floor = torch.as_tensor(torch.finfo(dtype).eps ** 0.5, dtype=dtype, device=device)
    scene_scale = scene_scale.clamp_min(median_radius).clamp_min(numerical_floor)
    band = torch.maximum(config.contact_band_radius * median_radius, config.contact_band_scene * scene_scale)

    if pair_index is None:
        active_product = int(active_mobile.numel()) * int(active_static.numel())
        if active_product <= config.max_dense_pairs:
            local_pairs = dense_pair_index(int(active_mobile.numel()), int(active_static.numel()), device=device)
        else:
            coarse_scalars = torch.linspace(
                config.lower_scan,
                config.upper_scan,
                config.broad_phase_samples,
                dtype=dtype,
                device=device,
            )
            radius_sum = mobile_radii[active_mobile].median() + static_radii[active_static].median()
            voxel_size = torch.maximum(4.0 * band, 2.0 * radius_sum)
            local_pairs = trajectory_voxel_pair_index(
                static_means[active_static],
                mobile_means_state0[active_mobile],
                coarse_scalars,
                joint_kind=joint_kind,
                axis=axis,
                pivot=pivot,
                observed_displacement=displacement,
                voxel_size=voxel_size,
                neighbor_radius=config.broad_phase_neighbor_radius,
                max_pairs=config.max_broad_phase_pairs,
            )
        pair_index = torch.stack((active_mobile[local_pairs[0]], active_static[local_pairs[1]]))
    else:
        pair_index = torch.as_tensor(pair_index, dtype=torch.long, device=device)
        if pair_index.ndim != 2 or pair_index.shape[0] != 2 or pair_index.shape[1] == 0:
            raise ValueError("pair_index must have shape [2, P] with P > 0")
        if pair_index[0].min() < 0 or pair_index[0].max() >= num_mobile:
            raise ValueError("pair_index contains an invalid mobile index")
        if pair_index[1].min() < 0 or pair_index[1].max() >= num_static:
            raise ValueError("pair_index contains an invalid static index")
        pair_keep = (mobile_weights[pair_index[0]] >= config.min_geometry_weight) & (
            static_weights[pair_index[1]] >= config.min_geometry_weight
        )
        pair_index = pair_index[:, pair_keep]
        if pair_index.shape[1] == 0:
            raise ValueError("fixed geometry-weight threshold removed all candidate pairs")
        if pair_index.shape[1] > config.max_broad_phase_pairs:
            pair_index = pair_index[:, : config.max_broad_phase_pairs]

    pair_weights = mobile_weights[pair_index[0]] * static_weights[pair_index[1]]
    if float(pair_weights.sum().detach().cpu()) <= 0.0:
        raise ValueError("candidate pair weights must have positive support")
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
    return _PreparedContactScene(common=common, scene_scale=scene_scale, contact_band=band)


def _certificate_from_field(
    field: EndpointEnergyField, *, side: Literal["lower", "upper"], config: EndpointFieldConfig
) -> EndpointCandidateCertificate:
    if field.scalars.numel() != 1:
        raise ValueError("candidate certificate requires exactly one scalar")
    value = field.scalars[0]
    gap_bands = field.contact_energy[0].clamp_min(0.0).sqrt()
    inside_penetration_bands = field.inside_penetration_energy[0].clamp_min(0.0).sqrt()
    lower_boundary = torch.as_tensor(config.lower_scan, dtype=value.dtype, device=value.device)
    upper_boundary = torch.as_tensor(config.upper_scan, dtype=value.dtype, device=value.device)
    if side == "lower":
        boundary_selected = bool(value <= lower_boundary or value >= 0.0)
    else:
        boundary_selected = bool(value <= 1.0 or value >= upper_boundary)
    identifiable = bool(
        not boundary_selected
        and float(gap_bands.detach().cpu()) <= config.max_contact_gap_bands
        and float(field.support_rise[0].detach().cpu()) >= config.support_margin_bands
        and float(inside_penetration_bands.detach().cpu()) <= config.max_inside_penetration_bands
    )
    contact_confidence = torch.exp(-0.5 * gap_bands.square())
    support_confidence = torch.sigmoid(8.0 * (field.support_rise[0] - config.support_margin_bands))
    uncertainty = (1.0 - contact_confidence * support_confidence).clamp(0.0, 1.0)
    if boundary_selected:
        uncertainty = torch.ones_like(uncertainty)
    return EndpointCandidateCertificate(
        value=value,
        identifiable=identifiable,
        uncertainty=uncertainty,
        contact_gap=field.signed_gap[0],
        contact_mass=field.contact_mass[0],
        support_rise=field.support_rise[0],
        inside_penetration_bands=inside_penetration_bands,
        boundary_selected=boundary_selected,
    )


def certify_contact_feasible_endpoints(
    static_means: Tensor,
    static_scales: Tensor,
    mobile_means_state0: Tensor,
    mobile_scales: Tensor,
    *,
    lower_scalar: Tensor | float,
    upper_scalar: Tensor | float,
    joint_kind: JointKind,
    axis: Tensor,
    observed_displacement: Tensor | float,
    pivot: Optional[Tensor] = None,
    static_weights: Optional[Tensor] = None,
    mobile_weights: Optional[Tensor] = None,
    pair_index: Optional[Tensor] = None,
    config: EndpointFieldConfig = EndpointFieldConfig(),
) -> CertifiedEndpointPair:
    """Recompute physical evidence at an explicitly published scalar pair."""

    prepared = _prepare_contact_scene(
        static_means,
        static_scales,
        mobile_means_state0,
        mobile_scales,
        joint_kind=joint_kind,
        axis=axis,
        observed_displacement=observed_displacement,
        pivot=pivot,
        static_weights=static_weights,
        mobile_weights=mobile_weights,
        pair_index=pair_index,
        config=config,
    )
    dtype, device = static_means.dtype, static_means.device
    lower_value = torch.as_tensor(lower_scalar, dtype=dtype, device=device).reshape(1)
    upper_value = torch.as_tensor(upper_scalar, dtype=dtype, device=device).reshape(1)
    lower_field = _field_for_side(lower_value, -1.0, **prepared.common)
    upper_field = _field_for_side(upper_value, 1.0, **prepared.common)
    lower = _certificate_from_field(lower_field, side="lower", config=config)
    upper = _certificate_from_field(upper_field, side="upper", config=config)
    mass_sum = lower.contact_mass + upper.contact_mass
    eps = torch.finfo(dtype).eps
    separation = (lower.contact_mass - upper.contact_mass).abs() / mass_sum.clamp_min(eps)
    closed_identifiable = bool(
        lower.identifiable and upper.identifiable and float(separation.detach().cpu()) >= config.closed_mass_margin
    )
    if not closed_identifiable:
        closed_end: ClosedEnd = "unknown"
    elif float(lower.contact_mass.detach().cpu()) > float(upper.contact_mass.detach().cpu()):
        closed_end = "lower"
    else:
        closed_end = "upper"
    return CertifiedEndpointPair(
        lower=lower,
        upper=upper,
        closed_end=closed_end,
        closed_end_identifiable=closed_identifiable,
        identifiable=lower.identifiable and upper.identifiable and closed_identifiable,
        closed_end_uncertainty=(1.0 - separation).clamp(0.0, 1.0),
        scene_scale=prepared.scene_scale,
        contact_band=prepared.contact_band,
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

    prepared = _prepare_contact_scene(
        static_means,
        static_scales,
        mobile_means_state0,
        mobile_scales,
        joint_kind=joint_kind,
        axis=axis,
        observed_displacement=observed_displacement,
        pivot=pivot,
        static_weights=static_weights,
        mobile_weights=mobile_weights,
        pair_index=pair_index,
        config=config,
    )
    common = prepared.common
    dtype, device = static_means.dtype, static_means.device
    lower_scalars = torch.linspace(config.lower_scan, 0.0, config.samples_per_side + 1, dtype=dtype, device=device)[:-1]
    upper_scalars = torch.linspace(1.0, config.upper_scan, config.samples_per_side + 1, dtype=dtype, device=device)[1:]
    lower_field = _field_for_side(lower_scalars, -1.0, **common)
    upper_field = _field_for_side(upper_scalars, 1.0, **common)
    lower_raw = (lower_field.scalars * lower_field.posterior).sum()
    upper_raw = (upper_field.scalars * upper_field.posterior).sum()
    lower_final_field = _field_for_side(lower_raw.reshape(1), -1.0, **common)
    upper_final_field = _field_for_side(upper_raw.reshape(1), 1.0, **common)
    lower = _summarize_endpoint(lower_field, lower_final_field, config)
    upper = _summarize_endpoint(upper_field, upper_final_field, config)

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
        scene_scale=prepared.scene_scale,
        contact_band=prepared.contact_band,
    )


__all__ = [
    "CertifiedEndpointPair",
    "ContactEndpointPrediction",
    "EndpointCandidateCertificate",
    "EndpointEnergyField",
    "EndpointEstimate",
    "EndpointFieldConfig",
    "JointKind",
    "certify_contact_feasible_endpoints",
    "dense_pair_index",
    "infer_contact_feasible_endpoints",
    "trajectory_voxel_pair_index",
]
