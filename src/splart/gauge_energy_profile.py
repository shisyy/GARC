"""Gauge-equivariant endpoint distances from frozen D2 energy profiles."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Iterable

import torch
from torch import Tensor, nn
from torch.nn import functional as F

PROFILE_CHANNELS = (
    "signed_gap", "contact_energy", "penetration_energy", "support_energy",
    "inside_penetration_energy", "contact_mass", "support_rise", "total_energy", "posterior",
)


def canonical_outward_coordinates(scalars: Tensor) -> Tensor:
    """Convert [lower,upper] scalar grids to non-negative outward distance."""
    if scalars.ndim != 4 or scalars.shape[1] != 2:
        raise ValueError("scalars must be [B,2,R,S]")
    return torch.stack((-scalars[:, 0], scalars[:, 1] - 1.0), dim=1)


def swap_profiles(features: Tensor, scalars: Tensor) -> tuple[Tensor, Tensor]:
    """Exact observation-order action in raw scalar coordinates."""
    if features.ndim != 5 or features.shape[1] != 2 or scalars.shape != features.shape[:-1]:
        raise ValueError("features/scalars must be [B,2,R,S,C]/[B,2,R,S]")
    swapped_features = features.flip(1)
    swapped_scalars = torch.stack((1.0 - scalars[:, 1], 1.0 - scalars[:, 0]), dim=1)
    return swapped_features, swapped_scalars


def make_profile_null(features: Tensor, kind: str, permutation: Tensor | None = None) -> Tensor:
    """Construct preregistered nulls without changing coordinates or labels."""
    if kind in {"zero_geometry", "order_only"}:
        return torch.zeros_like(features)
    if kind == "profile_permutation":
        if permutation is None or permutation.shape != (features.shape[-2],):
            raise ValueError("profile_permutation requires one complete sample permutation")
        return features[..., permutation, :]
    raise ValueError(f"unknown null: {kind}")


class GaugeEquivariantProfileHead(nn.Module):
    """One shared side function; equivariance follows structurally, not by loss."""

    def __init__(self, channels: int = len(PROFILE_CHANNELS), hidden: int = 64) -> None:
        super().__init__()
        self.channels = channels
        self.point = nn.Sequential(nn.Linear(channels + 1, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU())
        self.sequence = nn.Conv1d(hidden, hidden, kernel_size=5, padding=2, bias=True)
        self.readout = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))

    def forward(self, features: Tensor, scalars: Tensor) -> Tensor:
        if features.ndim != 5 or features.shape[-1] != self.channels or scalars.shape != features.shape[:-1]:
            raise ValueError("features/scalars have incompatible profile shapes")
        outward = canonical_outward_coordinates(scalars)
        if not torch.isfinite(features).all() or not torch.isfinite(outward).all() or (outward < 0).any():
            raise ValueError("profiles must be finite and outside the observed interval")
        b, sides, radii, samples, _ = features.shape
        x = torch.cat((features, outward[..., None]), dim=-1).reshape(b * sides * radii, samples, -1)
        x = self.point(x)
        x = F.silu(self.sequence(x.transpose(1, 2))).transpose(1, 2)
        pooled = torch.cat((x.mean(1), x.amax(1)), dim=-1).reshape(b, sides, radii, -1).mean(2)
        return F.softplus(self.readout(pooled).squeeze(-1))

    def endpoints(self, features: Tensor, scalars: Tensor) -> Tensor:
        distance = self(features, scalars)
        return torch.stack((-distance[:, 0], 1.0 + distance[:, 1]), dim=1)


def object_macro_nmae(predicted_distance: Tensor, target_distance: Tensor, object_ids: Iterable[str]) -> Tensor:
    if predicted_distance.shape != target_distance.shape or predicted_distance.ndim != 2 or predicted_distance.shape[1] != 2:
        raise ValueError("distances must be matching [N,2]")
    ids = tuple(object_ids)
    if len(ids) != predicted_distance.shape[0]:
        raise ValueError("object_ids length mismatch")
    losses = (predicted_distance - target_distance).abs().mean(1)
    values = [losses[[i for i, key in enumerate(ids) if key == object_id]].mean() for object_id in sorted(set(ids))]
    return torch.stack(values).mean()


def _finite_quantile(scores: Tensor, coverage: float) -> Tensor:
    if scores.ndim != 1 or scores.numel() == 0 or not 0.0 < coverage < 1.0:
        raise ValueError("scores must be non-empty and coverage in (0,1)")
    rank = min(scores.numel(), ceil((scores.numel() + 1) * coverage))
    return scores.sort().values[rank - 1]


@dataclass(frozen=True)
class ObjectSplitConformal:
    radius: Tensor
    coverage: float
    calibration_objects: tuple[str, ...]

    def distance_interval(self, prediction: Tensor) -> tuple[Tensor, Tensor]:
        return (prediction - self.radius).clamp_min(0.0), prediction + self.radius

    def endpoint_interval(self, prediction: Tensor) -> Tensor:
        lo, hi = self.distance_interval(prediction)
        return torch.stack((torch.stack((-hi[:, 0], -lo[:, 0]), 1), torch.stack((1.0 + lo[:, 1], 1.0 + hi[:, 1]), 1)), 1)


def fit_object_split_conformal(
    predicted_distance: Tensor, target_distance: Tensor, object_ids: Iterable[str], coverage: float = 0.9
) -> ObjectSplitConformal:
    ids = tuple(object_ids)
    residual = (predicted_distance - target_distance).abs()
    if residual.ndim != 2 or residual.shape[1] != 2 or len(ids) != residual.shape[0]:
        raise ValueError("calibration distances must be [N,2] with aligned object_ids")
    unique = tuple(sorted(set(ids)))
    # One exchangeable score per object; max gives simultaneous two-boundary coverage.
    scores = torch.stack([residual[[i for i, key in enumerate(ids) if key == obj]].amax() for obj in unique])
    return ObjectSplitConformal(_finite_quantile(scores, coverage), coverage, unique)


def fit_marginal_conformal(predicted: Tensor, target: Tensor, coverage: float = 0.9) -> Tensor:
    residual = (predicted - target).abs()
    return torch.stack([_finite_quantile(residual[:, side], coverage) for side in range(2)])


def fit_global_conformal(predicted: Tensor, target: Tensor, coverage: float = 0.9) -> Tensor:
    return _finite_quantile((predicted - target).abs().reshape(-1), coverage)


def fields_to_profile(lower_fields: tuple, upper_fields: tuple) -> tuple[Tensor, Tensor]:
    """Stack complete frozen D2 fields as [2,R,S,C] and scalar grids."""
    if not lower_fields or len(lower_fields) != len(upper_fields):
        raise ValueError("paired non-empty multi-radius fields required")
    sides = []
    grids = []
    for fields in (lower_fields, upper_fields):
        side_values, side_grids = [], []
        for field in fields:
            side_values.append(torch.stack([getattr(field, name) for name in PROFILE_CHANNELS], dim=-1))
            side_grids.append(field.scalars)
        sides.append(torch.stack(side_values))
        grids.append(torch.stack(side_grids))
    return torch.stack(sides), torch.stack(grids)


def predictions_to_profile(predictions: tuple, num_radii: int) -> tuple[Tensor, Tensor]:
    """Aggregate dual reconstructed states without privileging observation order."""
    if num_radii < 1 or not predictions or len(predictions) % num_radii:
        raise ValueError("predictions must contain a complete geometry-by-radius grid")
    num_geometries = len(predictions) // num_radii
    lower, upper = [], []
    for radius in range(num_radii):
        lower.append(tuple(predictions[g * num_radii + radius].lower.field for g in range(num_geometries)))
        upper.append(tuple(predictions[g * num_radii + radius].upper.field for g in range(num_geometries)))

    def aggregate(radius_groups: list[tuple]) -> tuple:
        values = []
        for group in radius_groups:
            reference = group[0]
            if any(not torch.equal(field.scalars, reference.scalars) for field in group[1:]):
                raise ValueError("dual-state D2 fields do not share a scalar gauge")
            payload = {
                name: torch.stack([getattr(field, name) for field in group]).mean(0)
                for name in PROFILE_CHANNELS
            }
            values.append(type("AggregatedField", (), {"scalars": reference.scalars, **payload})())
        return tuple(values)

    return fields_to_profile(aggregate(lower), aggregate(upper))


__all__ = [
    "GaugeEquivariantProfileHead", "ObjectSplitConformal", "PROFILE_CHANNELS",
    "canonical_outward_coordinates", "fields_to_profile", "fit_global_conformal",
    "fit_marginal_conformal", "fit_object_split_conformal", "make_profile_null", "object_macro_nmae",
    "predictions_to_profile", "swap_profiles",
]
