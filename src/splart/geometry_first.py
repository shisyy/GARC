"""Frozen node9.4 geometry-first proposals and semantic verification.

This module consumes only the strict target-free whitelist. Penetration profiles
are coarse proxies, not a physical collision oracle; abstentions retain finite
benchmark predictions without turning a semantic guess into a certified stop.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor

from splart.relative_search import CONFIG, METHODS, predict, validate_input


MODES = ("raw", "geometry_first", "donor_semantics", "shuffled_verification")
NUMERICAL_FLOOR = torch.finfo(torch.float32).eps ** 2


def fit_positive_hinge(d: Tensor, curve: Tensor) -> dict:
    """Compare affine and searched continuous positive-slope-change models."""
    x = d.to(dtype=torch.float64, device="cpu")
    y = curve.to(dtype=torch.float64, device="cpu")
    if x.ndim != 1 or y.shape != x.shape or len(x) < 5:
        raise ValueError("hinge requires paired one-dimensional grids of length >=5")
    if not torch.isfinite(x).all() or not torch.isfinite(y).all() or not torch.all(x[1:] > x[:-1]):
        raise ValueError("hinge requires finite increasing coordinates")
    x = (x - x[0]) / (x[-1] - x[0])
    y = (y - y[0]) / max(1.0, float(y.abs().max()))
    design = torch.stack((torch.ones_like(x), x), dim=1)
    # Full-rank affine projection and one-dimensional residualized hinge fit:
    # equivalent to constrained least squares, without iterative optimizers.
    gram = design.T @ design
    affine = design @ torch.linalg.solve(gram, design.T @ y)
    residual = y - affine
    affine_mse = float(residual.square().mean())
    n = len(x)
    affine_bic = n * math.log(affine_mse + NUMERICAL_FLOOR) + 2 * math.log(n)
    knots = torch.arange(2, n - 2)
    hinge = (x[:, None] - x[knots][None, :]).clamp_min(0)
    residual_hinge = hinge - design @ torch.linalg.solve(gram, design.T @ hinge)
    coefficients = (residual_hinge.T @ residual) / residual_hinge.square().sum(0)
    coefficients = coefficients.clamp_min(0)
    errors = residual[:, None] - residual_hinge * coefficients[None, :]
    mse = errors.square().mean(0)
    bic = n * torch.log(mse + NUMERICAL_FLOOR) + 4 * math.log(n)
    best = int(bic.argmin())
    accepted = bool(float(bic[best]) < affine_bic and float(coefficients[best]) > 0)
    return {"accepted": accepted, "knot_index": int(knots[best]) if accepted else None,
            "affine_bic": affine_bic, "hinge_bic": float(bic[best]),
            "slope_increase": float(coefficients[best]), "affine_mse": affine_mse,
            "hinge_mse": float(mse[best])}


def geometry_proposals(value: dict) -> list[dict]:
    """Deduplicate exact raw evidence, then require majority and compact spread."""
    validate_input(value)
    g = value["geometry"]
    origins = []
    for origin in g:
        if not any(torch.equal(origin, previous) for previous in origins):
            origins.append(origin)
    output = []
    for side in range(2):
        curves = []
        for origin in origins:
            for curve in origin[side, :, :, 2]:
                if not any(torch.equal(curve, previous) for previous in curves):
                    curves.append(curve)
        fits = [fit_positive_hinge(value["coordinates"][side], curve) for curve in curves]
        supported = [fit["knot_index"] for fit in fits if fit["accepted"]]
        candidates = sorted(set(supported))
        d = value["coordinates"][side]
        spread = float(d[max(supported)] - d[min(supported)]) if supported else 0.0
        majority = len(supported) * 2 > len(curves)
        compact = bool(supported) and spread <= CONFIG.disagreement_fraction * float(d[-1] - d[0])
        confirmed = majority and compact
        medoid = min(candidates, key=lambda k: (sum(abs(float(d[k]) - float(d[j])) for j in supported), k)) if candidates else None
        output.append({"geometric_confirmed": confirmed, "unique_origin_count": len(origins),
                       "unique_curve_count": len(curves), "supporting_curve_count": len(supported),
                       "supported_knot_indices": supported, "candidate_indices": candidates,
                       "candidate_distances": [float(d[i]) for i in candidates], "proposal_spread": spread,
                       "strict_majority": majority, "compact_spread": compact,
                       "medoid_index": medoid, "fits": fits})
    return output


def semantic_role(owner: dict, side: int) -> dict:
    """A single consensus direction; zero votes never count as negative."""
    a, t = owner["observed_embeddings"], owner["text_direction"]
    delta = ((a[side] - a[1 - side]) * t).sum(-1)
    positive, negative = int((delta > 0).sum()), int((delta < 0).sum())
    sign = 1 if positive * 2 > len(delta) else (-1 if negative * 2 > len(delta) else 0)
    strength = float(delta.abs().mean())
    eligible = sign != 0 and strength >= CONFIG.minimum_direction_strength
    return {"role": ("closure_like" if sign > 0 else "opening_like") if eligible else "unknown",
            "role_sign": sign if eligible else 0, "role_eligible": eligible,
            "positive_votes": positive, "negative_votes": negative,
            "zero_votes": int((delta == 0).sum()), "view_count": len(delta),
            "direction_strength": strength,
            "weak_direction": strength < CONFIG.minimum_direction_strength,
            "direction_has_majority": sign != 0}


def _validate_donor(value: dict, donor: dict | None, purpose: str) -> dict:
    if donor is None:
        raise ValueError(f"{purpose} requires a distinct deterministic donor")
    validate_input(donor)
    if donor["object_id"] == value["object_id"] or donor["split"] != value["split"]:
        raise ValueError(f"{purpose} requires a distinct same-split donor")
    if not torch.equal(donor["coordinates"], value["coordinates"]):
        raise ValueError("donor grid must match exactly")
    if donor["image_embeddings"].shape != value["image_embeddings"].shape:
        raise ValueError("donor semantic shape must match")
    return donor


def _candidate_verification(owner: dict, side: int, candidates: list[int], sign: int) -> list[dict]:
    a = (owner["observed_embeddings"] * owner["text_direction"]).sum(-1)
    scores = (owner["image_embeddings"][side] * owner["text_direction"]).sum(-1)
    output = []
    for index in candidates:
        margin = scores[index] - a.max(0).values if sign > 0 else a.min(0).values - scores[index]
        positive = int((margin > 0).sum())
        output.append({"index": index, "verified": positive * 2 > len(margin),
                       "positive_views": positive, "median_margin": float(torch.quantile(margin, 0.5)),
                       "view_margins": margin.tolist()})
    return output


def geometry_first_predict(value: dict, method: str, mode: str = "geometry_first",
                           semantic_donor: dict | None = None, route_donor: dict | None = None) -> dict:
    """Original prediction schema plus per-side ``geometry_first`` diagnostics."""
    if mode not in MODES or method not in METHODS:
        raise ValueError("unknown preregistered mode/method")
    if mode == "raw":
        return predict(value, method, semantic_donor)
    if method in ("coordinate_only", "semantic_only"):
        return predict(value, method)
    validate_input(value)
    owner = value
    if method == "object_shuffle" or (method == "joint" and mode == "donor_semantics"):
        owner = _validate_donor(value, semantic_donor, "semantic control")
    role_owner = owner
    if method in ("joint", "object_shuffle") and mode == "shuffled_verification":
        role_owner = _validate_donor(value, route_donor, "routing control")
    proposals = geometry_proposals(value)
    fallback = predict(value, "geometry_only")
    results = []
    for side, proposal in enumerate(proposals):
        d = value["coordinates"][side]
        diagnostic = dict(proposal, semantic_owner=owner["object_id"], routing_owner=role_owner["object_id"],
                          role="not_evaluated", role_eligible=False, semantic_verified=False,
                          semantic_activation=False, selection_changed=False, verification=[],
                          fallback_reason=None, cost_interpretation="geometry_medoid_distance")
        reasons = []
        if not proposal["geometric_confirmed"]:
            result = dict(fallback["sides"][side])
            reasons = result["reasons"] + ["no_stable_geometric_boundary"]
            diagnostic["fallback_reason"] = "original_geometry_unconfirmed"
            diagnostic["cost_interpretation"] = "original_geometry_cost"
        else:
            candidates = proposal["candidate_indices"]
            selected = proposal["medoid_index"]
            medoid = selected
            cost = torch.tensor([sum(abs(float(position) - float(d[j])) for j in proposal["supported_knot_indices"])
                                 for position in d], dtype=torch.float64)
            # The optimization domain is only the finite supported proposal set.
            noncandidate = torch.ones(len(d), dtype=torch.bool)
            noncandidate[candidates] = False
            cost[noncandidate] = float(cost.max()) + 1.0
            if method != "geometry_only":
                role = semantic_role(role_owner, side)
                diagnostic.update(role)
                if not role["role_eligible"]:
                    reasons.append("unknown_semantic_role")
                    if role["weak_direction"]:
                        reasons.append("weak_observed_semantic_direction")
                    diagnostic["fallback_reason"] = "geometric_medoid_unknown_role"
                else:
                    diagnostic["semantic_activation"] = True
                    verification = _candidate_verification(owner, side, candidates, role["role_sign"])
                    diagnostic["verification"] = verification
                    verified = [r for r in verification if r["verified"]]
                    if role["role_sign"] < 0:
                        reasons.append("opening_limit_not_identifiable_from_closure_semantics")
                    if not verified:
                        reasons.append("semantic_verification_failed")
                        diagnostic["fallback_reason"] = "geometric_medoid_failed_verification"
                    else:
                        selected = min(verified, key=lambda r: (-r["median_margin"], abs(float(d[r["index"]]) - float(d[medoid])), r["index"]))["index"]
                        diagnostic["semantic_verified"] = True
                        diagnostic["selection_changed"] = selected != medoid
                        diagnostic["cost_interpretation"] = "negative_median_margin_over_verified_candidates"
                        # Exact selection (including preregistered tie-breaking)
                        # is recorded explicitly; costs retain the real margins.
                        cost = torch.full((len(d),), 1.0 + max(abs(r["median_margin"]) for r in verification), dtype=torch.float64)
                        for row in verified:
                            cost[row["index"]] = -row["median_margin"]
            hull = [float(d[min(candidates)]), float(d[max(candidates)])]
            result = {"distance": float(d[selected]), "near_optimum_hull": hull,
                      "component_optima": [float(d[j]) for j in proposal["supported_knot_indices"]],
                      "cost": cost.tolist(), "evidence_interval": hull}
            diagnostic["selected_index"] = selected
        result["reasons"] = sorted(set(reasons))
        result["abstain"] = bool(reasons)
        if result["abstain"]:
            result["evidence_interval"] = [float(d[0]), float(d[-1])]
        result["geometry_first"] = diagnostic
        results.append(result)
    return {"method": method, "object_id": value["object_id"], "split": value["split"],
            "distance": [result["distance"] for result in results], "sides": results,
            "uncertainty": "heuristic_not_calibrated_not_physical_certification"}
