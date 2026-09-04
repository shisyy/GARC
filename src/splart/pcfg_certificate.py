"""Proxy-aligned physical contact-fraction-gain (PCFG) certificate.

The implementation mirrors the frozen v4 evaluator's Gaussian proxy: maximum
axis radius, opacity 0.1, hard mobility partition at 0.5, 4096-point original-
index even-stride sampling, 5 mm contact tolerance, and 2 mm penetration
tolerance.  It adds only predeclared counterfactuals: radii 0.75/1/1.25 and a
fixed 0.04 observed-span step just inside/outside each predicted endpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import Tensor
from torch.nn.functional import normalize

RADIUS_MULTIPLIERS = (0.75, 1.0, 1.25)
COUNTERFACTUAL_DELTA = 0.04
SAMPLE_CAP = 4096
OPACITY_THRESHOLD = 0.1
MOBILITY_THRESHOLD = 0.5
CONTACT_TOLERANCE_M = 0.005
PENETRATION_TOLERANCE_M = 0.002
MIN_CONTACT_FRACTION = 0.01
MIN_CONTACT_GAIN = 0.005


@dataclass(frozen=True)
class PCFGEndpointEvidence:
    direction: int
    anchor_scalar: float
    endpoint_scalar: float
    per_radius: tuple[dict[str, float | bool], ...]
    terminal_votes: int
    terminal_valid: bool


@dataclass(frozen=True)
class PCFGPrediction:
    lower: PCFGEndpointEvidence
    upper: PCFGEndpointEvidence
    closed_end: Literal["lower", "upper", "unknown"]
    closed_votes_lower: int
    closed_votes_upper: int
    closed_mean_gain_lower: float
    closed_mean_gain_upper: float
    closed_identifiable: bool
    static_count: int
    mobile_count: int
    static_sample_count: int
    mobile_sample_count: int


def deterministic_sample_indices(count: int, cap: int = SAMPLE_CAP) -> Tensor:
    if count <= 0:
        raise ValueError("sample source must be non-empty")
    if count <= cap:
        return torch.arange(count, dtype=torch.long)
    # Same integer even-stride rule as the v4 candidate-bound evaluator.
    return torch.div(torch.arange(cap, dtype=torch.long) * count, cap, rounding_mode="floor")


def _quantiles(gap_m: Tensor) -> dict[str, float]:
    penetration = torch.relu(-gap_m)
    return {
        "contact_fraction": float((gap_m.abs() <= CONTACT_TOLERANCE_M).float().mean()),
        "penetration_fraction": float((penetration > PENETRATION_TOLERANCE_M).float().mean()),
        "penetration_q99_m": float(torch.quantile(penetration, 0.99)),
        "penetration_depth_m": float(penetration.max()),
        "surface_gap_q01_m": float(torch.quantile(gap_m, 0.01)),
        "surface_gap_q50_m": float(torch.quantile(gap_m, 0.50)),
        "surface_gap_q99_m": float(torch.quantile(gap_m, 0.99)),
    }


@torch.inference_mode()
def pcfg_certificate(model: Any, lower_scalar: float, upper_scalar: float, ns_scale: float) -> PCFGPrediction:
    """Certify fixed endpoint predictions without endpoint labels or images."""

    if not lower_scalar < 0.0 or not upper_scalar > 1.0 or ns_scale <= 0.0:
        raise ValueError("PCFG expects ordered outside-observation scalars and positive scene scale")
    states = model.states.detach().squeeze(-1)
    means = model.means.detach()
    radii = model.scales.detach().exp().amax(dim=-1)
    opacity = model.opacities.detach().sigmoid().squeeze(-1)
    mobility = model.mobilities.detach().sigmoid().squeeze(-1)
    target = ~model.mobilities.detach().squeeze(-1).isnan()
    static_mask = target & (opacity >= OPACITY_THRESHOLD) & (mobility < MOBILITY_THRESHOLD)
    mobile_mask = target & (opacity >= OPACITY_THRESHOLD) & (mobility >= MOBILITY_THRESHOLD)
    if not static_mask.any() or not mobile_mask.any():
        raise ValueError("PCFG hard partition has empty static or mobile support")
    static_means = means[static_mask]
    static_radii = radii[static_mask]
    mobile_means = means[mobile_mask]
    mobile_radii = radii[mobile_mask]
    mobile_states = states[mobile_mask]
    static_count, mobile_count = int(static_means.shape[0]), int(mobile_means.shape[0])
    static_ids = deterministic_sample_indices(static_count).to(means.device)
    mobile_ids = deterministic_sample_indices(mobile_count).to(means.device)
    static_means, static_radii = static_means[static_ids], static_radii[static_ids]
    mobile_means, mobile_radii, mobile_states = (
        mobile_means[mobile_ids],
        mobile_radii[mobile_ids],
        mobile_states[mobile_ids],
    )
    axis = normalize(model.articulation_params.axis.detach(), dim=0)
    articulation_type = int(model.articulation_params.articulation_type.item())

    def center_neighbors(query: float) -> tuple[Tensor, Tensor]:
        transformed = mobile_means.clone()
        factors = torch.where(mobile_states == 0, float(query), float(query) - 1.0)
        if articulation_type in {1, 3}:  # REVOLUTE, CYLINDRICAL
            pivot = model.articulation_params.pivot.detach()
            theta = model.articulation_params.angle.detach() * factors
            centered = transformed - pivot
            cross = torch.linalg.cross(axis.expand_as(centered), centered, dim=-1)
            projection = (centered * axis).sum(dim=-1, keepdim=True) * axis
            transformed = (
                centered * torch.cos(theta)[:, None]
                + cross * torch.sin(theta)[:, None]
                + projection * (1.0 - torch.cos(theta))[:, None]
                + pivot
            )
        if articulation_type in {2, 3}:  # PRISMATIC, CYLINDRICAL
            transformed = transformed + axis * (model.articulation_params.dist.detach() * factors)[:, None]
        nearest_distance, nearest_index = torch.cdist(transformed, static_means).min(dim=1)
        return nearest_distance, nearest_index

    endpoints = (("lower", lower_scalar, 0.0, -1), ("upper", upper_scalar, 1.0, 1))
    evidence: dict[str, PCFGEndpointEvidence] = {}
    gains: dict[str, list[float]] = {"lower": [], "upper": []}
    for name, endpoint, anchor, direction in endpoints:
        samples = {
            "endpoint": center_neighbors(endpoint),
            "anchor": center_neighbors(anchor),
            "inside": center_neighbors(endpoint - direction * COUNTERFACTUAL_DELTA),
            "outside": center_neighbors(endpoint + direction * COUNTERFACTUAL_DELTA),
        }
        records = []
        for multiplier in RADIUS_MULTIPLIERS:
            stats = {}
            for location, (distance, nearest_index) in samples.items():
                gap = (distance - multiplier * (mobile_radii + static_radii[nearest_index])) / float(ns_scale)
                stats[location] = _quantiles(gap)
            contact_gain = stats["endpoint"]["contact_fraction"] - stats["anchor"]["contact_fraction"]
            outside_vs_inside = (
                stats["outside"]["penetration_q99_m"] > stats["inside"]["penetration_q99_m"]
            )
            terminal = bool(
                stats["endpoint"]["contact_fraction"] >= MIN_CONTACT_FRACTION
                and contact_gain >= MIN_CONTACT_GAIN
                and stats["endpoint"]["penetration_fraction"] <= 0.01
                and stats["endpoint"]["penetration_q99_m"] <= PENETRATION_TOLERANCE_M
                and outside_vs_inside
            )
            gains[name].append(contact_gain)
            records.append(
                {
                    "radius_multiplier": multiplier,
                    "contact_fraction": stats["endpoint"]["contact_fraction"],
                    "anchor_contact_fraction": stats["anchor"]["contact_fraction"],
                    "contact_fraction_gain": contact_gain,
                    "penetration_fraction": stats["endpoint"]["penetration_fraction"],
                    "penetration_q99_m": stats["endpoint"]["penetration_q99_m"],
                    "penetration_depth_m": stats["endpoint"]["penetration_depth_m"],
                    "inside_penetration_q99_m": stats["inside"]["penetration_q99_m"],
                    "outside_penetration_q99_m": stats["outside"]["penetration_q99_m"],
                    "outside_vs_inside": outside_vs_inside,
                    "terminal_valid": terminal,
                }
            )
        votes = sum(bool(record["terminal_valid"]) for record in records)
        # The central radius is definition-identical to v4. Neighboring radii
        # are counterfactual evidence, not additional tunable acceptance gates.
        evidence[name] = PCFGEndpointEvidence(direction, anchor, endpoint, tuple(records), votes, bool(records[1]["terminal_valid"]))

    lower_votes = sum(
        lower_gain - upper_gain >= MIN_CONTACT_GAIN for lower_gain, upper_gain in zip(gains["lower"], gains["upper"])
    )
    upper_votes = sum(
        upper_gain - lower_gain >= MIN_CONTACT_GAIN for lower_gain, upper_gain in zip(gains["lower"], gains["upper"])
    )
    mean_lower_gain = sum(gains["lower"]) / len(RADIUS_MULTIPLIERS)
    mean_upper_gain = sum(gains["upper"]) / len(RADIUS_MULTIPLIERS)
    closed_identifiable = abs(mean_lower_gain - mean_upper_gain) >= MIN_CONTACT_GAIN
    closed_end: Literal["lower", "upper", "unknown"]
    if not closed_identifiable:
        closed_end = "unknown"
    else:
        closed_end = "lower" if mean_lower_gain > mean_upper_gain else "upper"
    return PCFGPrediction(
        lower=evidence["lower"],
        upper=evidence["upper"],
        closed_end=closed_end,
        closed_votes_lower=lower_votes,
        closed_votes_upper=upper_votes,
        closed_mean_gain_lower=mean_lower_gain,
        closed_mean_gain_upper=mean_upper_gain,
        closed_identifiable=closed_identifiable,
        static_count=static_count,
        mobile_count=mobile_count,
        static_sample_count=int(static_ids.numel()),
        mobile_sample_count=int(mobile_ids.numel()),
    )


__all__ = [
    "COUNTERFACTUAL_DELTA",
    "PCFGEndpointEvidence",
    "PCFGPrediction",
    "RADIUS_MULTIPLIERS",
    "deterministic_sample_indices",
    "pcfg_certificate",
]
