"""Node 9.2: one fixed, label-free observed-motion reliability formula."""
from __future__ import annotations

import torch
from torch import Tensor

from splart.relative_search import CONFIG, METHODS, _summarize_side, evidence, predict, validate_input


MODES = ("uniform", "reliability", "inverse_noise_only", "shuffled_weights")


def view_weights(value: dict, mode: str) -> tuple[Tensor, dict]:
    """Compute shared weights, pooling both outward sides symmetrically.

    r_v = ((a1-a0).t)^2 / (mean_{side,position}(second_diff(e.t)^2) + eps).
    Only IEEE float64 epsilon is used; no tuned threshold/temperature is fitted.
    """
    if mode not in MODES:
        raise ValueError("unknown preregistered weighting mode")
    validate_input(value)
    e = value["image_embeddings"].double()
    a = value["observed_embeddings"].double()
    t = value["text_direction"].double()
    projection = ((a[1] - a[0]) * t).sum(-1)
    score = (e * t).sum(-1)
    second = score[:, 2:] - 2.0 * score[:, 1:-1] + score[:, :-2]
    side_noise = second.square().mean(dim=1)
    # Summing two pre-reduced sides is exactly invariant to side reversal.
    noise = 0.5 * (side_noise[0] + side_noise[1])
    eps = torch.finfo(torch.float64).eps
    if mode == "uniform":
        raw = torch.ones_like(noise)
    elif mode == "inverse_noise_only":
        raw = 1.0 / (noise + eps)
    else:
        raw = projection.square() / (noise + eps)
    all_unobservable = bool(raw.sum() == 0)
    if all_unobservable:
        weights = torch.ones_like(raw) / raw.numel()
    else:
        weights = raw / raw.sum()
    if not torch.isfinite(weights).all():
        raise ValueError("nonfinite reliability weights")
    return weights, {
        "observed_projection_squared": projection.square().tolist(),
        "projected_second_difference_noise": noise.tolist(),
        "weights": weights.tolist(),
        "effective_view_count": float(1.0 / weights.square().sum()),
        "all_unobservable_uniform_fallback": all_unobservable,
        "numerical_epsilon": eps,
    }


def weighted_predict(value: dict, method: str, mode: str,
                     semantic_donor: dict | None = None,
                     weight_donor: dict | None = None) -> dict:
    if mode not in MODES or method not in METHODS:
        raise ValueError("unknown fixed mode or method")
    # Delegate unchanged controls and the uniform path to the exact old code.
    if mode == "uniform" or method in ("coordinate_only", "geometry_only"):
        return predict(value, method, semantic_donor)
    geo, sem, direction, _ = evidence(value)
    semantic_owner = value
    if method == "object_shuffle":
        if semantic_donor is None or semantic_donor["object_id"] == value["object_id"]:
            raise ValueError("distinct semantic donor required")
        if semantic_donor["split"] != value["split"] or not torch.equal(semantic_donor["coordinates"], value["coordinates"]):
            raise ValueError("semantic donor must share split and grid")
        semantic_owner = semantic_donor
        _, sem, direction, _ = evidence(semantic_owner)
    weight_owner = semantic_owner
    if mode == "shuffled_weights":
        if weight_donor is None or weight_donor["object_id"] == semantic_owner["object_id"]:
            raise ValueError("weights must come from a distinct semantic-owner donor")
        if weight_donor["split"] != semantic_owner["split"] or not torch.equal(weight_donor["coordinates"], semantic_owner["coordinates"]):
            raise ValueError("weight donor must share split and grid")
        weight_owner = weight_donor
    weights, _ = view_weights(weight_owner, mode)
    semantic_mean = (sem.double() * weights[None, :, None]).sum(1)
    results = []
    for side in range(2):
        reasons = []
        if method == "semantic_only":
            cost, components = semantic_mean[side], sem[side]
        else:
            cost = geo[side].mean(0) + CONFIG.joint_semantic_weight * semantic_mean[side]
            components = torch.cat((geo[side], sem[side]))
        # Preserve every original uncertainty rule and its unweighted evidence.
        # Weighting cannot silently suppress a disagreeing view for acceptance.
        if float(direction[side].abs().mean()) < CONFIG.minimum_direction_strength:
            reasons.append("weak_observed_semantic_direction")
        if float(direction[side].mean()) <= 0:
            reasons.append("opening_limit_not_identifiable_from_closure_semantics")
        results.append(_summarize_side(cost, components, value["coordinates"][side], reasons))
    return {"method": method, "object_id": value["object_id"], "split": value["split"],
            "distance": [r["distance"] for r in results], "sides": results,
            "uncertainty": "heuristic_not_calibrated_not_physical_certification"}
