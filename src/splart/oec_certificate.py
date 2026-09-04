"""Oriented-Ellipsoid Contact certificate with frozen PCFG decision gates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import Tensor
from torch.nn.functional import normalize

from .pcfg_certificate import (
    CONTACT_TOLERANCE_M, COUNTERFACTUAL_DELTA, MIN_CONTACT_FRACTION,
    MIN_CONTACT_GAIN, PENETRATION_TOLERANCE_M, RADIUS_MULTIPLIERS,
    SAMPLE_CAP, deterministic_sample_indices,
)

GEOMETRY_WEIGHT_THRESHOLD = 0.01
PAIR_CHUNK = 256


@dataclass(frozen=True)
class OECEndpointEvidence:
    direction: int
    anchor_scalar: float
    endpoint_scalar: float
    per_radius: tuple[dict[str, float | bool], ...]
    terminal_votes: int
    terminal_valid: bool


@dataclass(frozen=True)
class OECPrediction:
    lower: OECEndpointEvidence
    upper: OECEndpointEvidence
    closed_end: Literal["lower", "upper", "unknown"]
    closed_mean_gain_lower: float
    closed_mean_gain_upper: float
    closed_identifiable: bool
    static_count: int
    mobile_count: int
    static_sample_count: int
    mobile_sample_count: int


def _quat_to_matrix(quat: Tensor) -> Tensor:
    """Convert gsplat wxyz quaternions to rotation matrices."""
    q = normalize(quat, dim=-1)
    w, x, y, z = q.unbind(-1)
    return torch.stack((
        1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
        2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
        2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y),
    ), -1).reshape(q.shape[:-1] + (3, 3))


def _axis_angle_matrix(axis: Tensor, theta: Tensor) -> Tensor:
    axis = normalize(axis, dim=0)
    x, y, z = axis
    zero = x.new_zeros(())
    skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero)).reshape(3, 3)
    eye = torch.eye(3, dtype=axis.dtype, device=axis.device)
    return eye + torch.sin(theta)[..., None, None] * skew + (1-torch.cos(theta))[..., None, None] * (skew @ skew)


def _surface_gaps(mobile_means: Tensor, mobile_rot: Tensor, mobile_scales: Tensor,
                  static_means: Tensor, static_rot: Tensor, static_scales: Tensor,
                  multiplier: float) -> Tensor:
    """Minimum pair surface gap using exact radial support of both ellipsoids."""
    outputs = []
    inv_m = mobile_rot @ torch.diag_embed(mobile_scales.square().reciprocal()) @ mobile_rot.transpose(-1, -2)
    inv_s = static_rot @ torch.diag_embed(static_scales.square().reciprocal()) @ static_rot.transpose(-1, -2)
    for start in range(0, len(mobile_means), PAIR_CHUNK):
        delta = static_means[None] - mobile_means[start:start+PAIR_CHUNK, None]
        distance = delta.norm(dim=-1).clamp_min(torch.finfo(delta.dtype).eps)
        u = delta / distance[..., None]
        m_inv = inv_m[start:start+PAIR_CHUNK]
        m_quad = torch.einsum("mnj,mjk,mnk->mn", u, m_inv, u)
        s_quad = torch.einsum("mnj,njk,mnk->mn", u, inv_s, u)
        support = multiplier * (m_quad.rsqrt() + s_quad.rsqrt())
        outputs.append((distance - support).min(dim=1).values)
    return torch.cat(outputs)


def _stats(gap: Tensor) -> dict[str, float]:
    penetration = torch.relu(-gap)
    return {
        "contact_fraction": float((gap.abs() <= CONTACT_TOLERANCE_M).float().mean()),
        "penetration_fraction": float((penetration > PENETRATION_TOLERANCE_M).float().mean()),
        "penetration_q99_m": float(torch.quantile(penetration, .99)),
        "penetration_depth_m": float(penetration.max()),
    }


@torch.inference_mode()
def oec_certificate(model: Any, lower_scalar: float, upper_scalar: float, ns_scale: float) -> OECPrediction:
    if not lower_scalar < 0 or not upper_scalar > 1 or ns_scale <= 0:
        raise ValueError("OEC expects extrapolated ordered endpoints and positive scene scale")
    states = model.states.detach().squeeze(-1)
    means = model.means.detach()
    scales = model.scales.detach().exp()
    rotations = _quat_to_matrix(model.quats.detach())
    opacity = model.opacities.detach().sigmoid().squeeze(-1)
    mobility = model.mobilities.detach().sigmoid().squeeze(-1)
    target = ~model.mobilities.detach().squeeze(-1).isnan()
    static_mask = target & (opacity * (1-mobility) >= GEOMETRY_WEIGHT_THRESHOLD)
    mobile_mask = target & (opacity * mobility >= GEOMETRY_WEIGHT_THRESHOLD)
    if not static_mask.any() or not mobile_mask.any():
        raise ValueError("OEC weighted partition has empty support")
    sm, ss, sr = means[static_mask], scales[static_mask], rotations[static_mask]
    mm, ms, mr, mst = means[mobile_mask], scales[mobile_mask], rotations[mobile_mask], states[mobile_mask]
    sc, mc = len(sm), len(mm)
    si = deterministic_sample_indices(sc, SAMPLE_CAP).to(means.device)
    mi = deterministic_sample_indices(mc, SAMPLE_CAP).to(means.device)
    sm, ss, sr = sm[si], ss[si], sr[si]
    mm, ms, mr, mst = mm[mi], ms[mi], mr[mi], mst[mi]
    axis = normalize(model.articulation_params.axis.detach(), dim=0)
    kind = int(model.articulation_params.articulation_type.item())

    def gaps(query: float, radius_multiplier: float) -> Tensor:
        factors = torch.where(mst == 0, float(query), float(query)-1.)
        moved, moved_rot = mm.clone(), mr.clone()
        if kind in {1, 3}:
            pivot = model.articulation_params.pivot.detach()
            motion = _axis_angle_matrix(axis, model.articulation_params.angle.detach() * factors)
            moved = torch.einsum("nij,nj->ni", motion, moved-pivot) + pivot
            moved_rot = motion @ moved_rot
        if kind in {2, 3}:
            moved = moved + axis * (model.articulation_params.dist.detach()*factors)[:, None]
        return _surface_gaps(moved, moved_rot, ms, sm, sr, ss, radius_multiplier) / float(ns_scale)

    evidence = {}
    gains = {"lower": [], "upper": []}
    for name, endpoint, anchor, direction in (("lower", lower_scalar, 0., -1), ("upper", upper_scalar, 1., 1)):
        records = []
        for multiplier in RADIUS_MULTIPLIERS:
            stats = {key: _stats(gaps(q, multiplier)) for key, q in {
                "endpoint": endpoint, "anchor": anchor,
                "inside": endpoint-direction*COUNTERFACTUAL_DELTA,
                "outside": endpoint+direction*COUNTERFACTUAL_DELTA,
            }.items()}
            gain = stats["endpoint"]["contact_fraction"] - stats["anchor"]["contact_fraction"]
            outside = stats["outside"]["penetration_q99_m"] > stats["inside"]["penetration_q99_m"]
            terminal = bool(stats["endpoint"]["contact_fraction"] >= MIN_CONTACT_FRACTION and gain >= MIN_CONTACT_GAIN
                            and stats["endpoint"]["penetration_fraction"] <= .01
                            and stats["endpoint"]["penetration_q99_m"] <= PENETRATION_TOLERANCE_M and outside)
            gains[name].append(gain)
            records.append({"radius_multiplier": multiplier, "contact_fraction": stats["endpoint"]["contact_fraction"],
                "anchor_contact_fraction": stats["anchor"]["contact_fraction"], "contact_fraction_gain": gain,
                "penetration_fraction": stats["endpoint"]["penetration_fraction"],
                "penetration_q99_m": stats["endpoint"]["penetration_q99_m"], "penetration_depth_m": stats["endpoint"]["penetration_depth_m"],
                "inside_penetration_q99_m": stats["inside"]["penetration_q99_m"],
                "outside_penetration_q99_m": stats["outside"]["penetration_q99_m"], "outside_vs_inside": outside,
                "terminal_valid": terminal})
        evidence[name] = OECEndpointEvidence(direction, anchor, endpoint, tuple(records), sum(bool(x["terminal_valid"]) for x in records), bool(records[1]["terminal_valid"]))
    lg, ug = sum(gains["lower"])/3, sum(gains["upper"])/3
    identifiable = max(lg, ug) >= MIN_CONTACT_GAIN and abs(lg-ug) >= MIN_CONTACT_GAIN
    closed = "unknown" if not identifiable else ("lower" if lg > ug else "upper")
    return OECPrediction(evidence["lower"], evidence["upper"], closed, lg, ug, identifiable, sc, mc, len(si), len(mi))


__all__ = ["GEOMETRY_WEIGHT_THRESHOLD", "OECPrediction", "oec_certificate"]
