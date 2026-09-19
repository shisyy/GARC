"""Shared controls and metrics for the CSTR source gates."""

from __future__ import annotations

from typing import Any, Iterable

import torch
from torch import Tensor

from splart.cr_fpl import CR_FPL_EPSILON
from splart.clip_limit import CSTR_CHANNELS, counterfactual_semantic_tangent_residual


CONTROLS = ("cstr", "zero", "object_trajectory_shuffle", "coordinate_only", "sign_flip")


def normalize_semantic_by_train_rms(
    data: dict[str, tuple[Tensor, Tensor, tuple[str, ...], tuple[str, ...]]]
) -> dict[str, tuple[Tensor, Tensor, tuple[str, ...], tuple[str, ...]]]:
    """Use per-channel, sign/zero-preserving scales fitted on source train."""

    rms = data["source_train"][0].square().mean(dim=(0, 1, 2), keepdim=True).sqrt().clamp_min(1e-6)
    return {
        split: (evidence / rms, coordinates, object_ids, episode_ids)
        for split, (evidence, coordinates, object_ids, episode_ids) in data.items()
    }


def apply_semantic_control(
    evidence: Tensor, coordinates: Tensor, object_ids: Iterable[str], episode_ids: Iterable[str], control: str
) -> tuple[Tensor, Tensor]:
    """Build true CSTR or one of its parameter-matched executable controls."""

    if control not in CONTROLS:
        raise ValueError(f"unknown semantic control: {control}")
    if evidence.ndim != 4 or evidence.shape[-1] != 1 or coordinates.shape != evidence.shape[:-1]:
        raise ValueError("raw semantic trajectory shape mismatch")
    if control == "coordinate_only":
        minimum = coordinates[..., :1]
        span = (coordinates[..., -1:] - minimum).clamp_min(1e-6)
        unit = (coordinates - minimum) / span
        return torch.stack((unit, unit.square(), unit * (1.0 - unit)), dim=-1), coordinates
    if control in ("cstr", "sign_flip"):
        signed = -evidence if control == "sign_flip" else evidence
        return counterfactual_semantic_tangent_residual(signed, coordinates), coordinates
    if control == "zero":
        shape = (*evidence.shape[:-1], CSTR_CHANNELS)
        return evidence.new_zeros(shape), coordinates

    objects = tuple(object_ids)
    episodes = tuple(episode_ids)
    if len(objects) != evidence.shape[0] or len(episodes) != evidence.shape[0]:
        raise ValueError("CSTR shuffle identities are misaligned")
    unique = sorted(set(objects))
    if len(unique) < 2:
        raise ValueError("object-trajectory shuffle requires at least two objects")
    donor_object = {value: unique[(index + 1) % len(unique)] for index, value in enumerate(unique)}
    per_object = {
        value: sorted(
            (index for index, object_id in enumerate(objects) if object_id == value), key=lambda index: episodes[index]
        )
        for value in unique
    }
    counts = {len(indices) for indices in per_object.values()}
    if len(counts) != 1:
        raise ValueError("object-trajectory shuffle requires equal trajectories per object")
    donor_indices = [0] * len(objects)
    for recipient in unique:
        donor = donor_object[recipient]
        for recipient_index, source_index in zip(per_object[recipient], per_object[donor]):
            donor_indices[recipient_index] = source_index
    index = torch.tensor(donor_indices, device=evidence.device)
    shuffled_coordinates = coordinates.index_select(0, index)
    shuffled = counterfactual_semantic_tangent_residual(evidence.index_select(0, index), shuffled_coordinates)
    return shuffled, shuffled_coordinates


def _side_nmae(prediction: Tensor, target: Tensor) -> list[float]:
    return [float(value) for value in (prediction - target).abs().mean(dim=0)]


def _target_relative_p99_side(prediction: Tensor, target: Tensor) -> list[float]:
    relative = (prediction - target).abs() / target.clamp_min(CR_FPL_EPSILON)
    return [float(torch.quantile(relative[:, side], 0.99)) for side in range(2)]


def summarize_candidate(output: Any, anchor: Tensor, target: Tensor) -> dict[str, Any]:
    anchor_side = _side_nmae(anchor, target)
    one_loop = output.loop_distances[..., 0]
    one_loop_side = _side_nmae(one_loop, target)
    final_side = _side_nmae(output.distance, target)
    tail_side = _target_relative_p99_side(output.distance, target)
    metrics = {
        "d2_anchor_side_nmae": anchor_side,
        "d2_anchor_mean_side_nmae": sum(anchor_side) / 2.0,
        "d2_anchor_worst_side_nmae": max(anchor_side),
        "one_loop_same_parameters_side_nmae": one_loop_side,
        "one_loop_same_parameters_mean_side_nmae": sum(one_loop_side) / 2.0,
        "one_loop_same_parameters_worst_side_nmae": max(one_loop_side),
        "candidate_side_nmae": final_side,
        "candidate_mean_side_nmae": sum(final_side) / 2.0,
        "candidate_worst_side_nmae": max(final_side),
        "candidate_target_relative_p99_side": tail_side,
        "candidate_target_relative_p99_max_side": max(tail_side),
        "loop_prediction_residual_mean": output.loop_prediction_residuals.mean((0, 1)).cpu().tolist(),
        "loop_state_residual_l2_mean": output.loop_state_residual_l2.mean((0, 1)).cpu().tolist(),
        "loop_fixed_point_residual_mean": output.loop_fixed_point_residuals.mean((0, 1)).cpu().tolist(),
        "loop_query_residual_l2_mean": output.loop_query_residual_l2.mean((0, 1)).cpu().tolist(),
        "loop_semantic_evidence_mean": output.loop_semantic_evidence.mean((0, 1)).cpu().tolist(),
        "loop_semantic_residual_l2_mean": output.loop_semantic_residual_l2.mean((0, 1)).cpu().tolist(),
    }
    return metrics


def summarize_cr_fpl_control(output: Any, target: Tensor) -> dict[str, Any]:
    """Report the frozen geometry baseline under the identical max-side tail metric."""

    side = _side_nmae(output.distance, target)
    one_loop_side = _side_nmae(output.loop_distances[..., 0], target)
    tail_side = _target_relative_p99_side(output.distance, target)
    return {
        "cr_fpl_control_side_nmae": side,
        "cr_fpl_control_mean_side_nmae": sum(side) / 2.0,
        "cr_fpl_control_worst_side_nmae": max(side),
        "cr_fpl_control_one_loop_mean_side_nmae": sum(one_loop_side) / 2.0,
        "cr_fpl_control_target_relative_p99_side": tail_side,
        "cr_fpl_control_target_relative_p99_max_side": max(tail_side),
    }


def binary_auroc(scores: Tensor, labels: Tensor) -> float:
    """Exact rank AUROC with averaged ranks for ties."""

    scores = scores.detach().double().flatten().cpu()
    labels = labels.detach().bool().flatten().cpu()
    if scores.shape != labels.shape or not torch.isfinite(scores).all():
        raise ValueError("AUROC scores/labels are invalid")
    positives = int(labels.sum())
    negatives = labels.numel() - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    order = torch.argsort(scores, stable=True)
    sorted_scores = scores[order]
    ranks = torch.arange(1, scores.numel() + 1, dtype=torch.float64)
    start = 0
    while start < scores.numel():
        end = start + 1
        while end < scores.numel() and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[start:end] = ranks[start:end].mean()
        start = end
    original_ranks = torch.empty_like(ranks)
    original_ranks[order] = ranks
    positive_rank_sum = original_ranks[labels].sum()
    return float((positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives))


def semantic_endpoint_auroc(evidence: Tensor, coordinates: Tensor, target: Tensor) -> float:
    if evidence.shape[:-1] != coordinates.shape or evidence.shape[-1] < 1:
        raise ValueError("semantic AUROC trajectory shape mismatch")
    if target.shape != coordinates.shape[:2]:
        raise ValueError("semantic AUROC target shape mismatch")
    labels = coordinates >= target.unsqueeze(-1)
    return binary_auroc(evidence[..., 0], labels)


def semantic_endpoint_object_macro_auroc(
    evidence: Tensor, coordinates: Tensor, target: Tensor, object_ids: Iterable[str]
) -> float:
    object_ids = tuple(object_ids)
    values = []
    for object_id in sorted(set(object_ids)):
        indices = [index for index, value in enumerate(object_ids) if value == object_id]
        values.append(semantic_endpoint_auroc(evidence[indices], coordinates[indices], target[indices]))
    if not values or not all(torch.isfinite(torch.tensor(value)) for value in values):
        return float("nan")
    return sum(values) / len(values)


def add_object_macro_metrics(metrics: dict[str, Any], output: Any, target: Tensor, object_ids: Iterable[str]) -> None:
    object_ids = tuple(object_ids)
    values = []
    for object_id in sorted(set(object_ids)):
        indices = [index for index, value in enumerate(object_ids) if value == object_id]
        values.append((output.distance[indices] - target[indices]).abs().mean(dim=0))
    side = [float(value) for value in torch.stack(values).mean(dim=0)]
    metrics["candidate_object_macro_side_nmae"] = side
    metrics["candidate_object_macro_mean_side_nmae"] = sum(side) / 2.0
    metrics["candidate_object_macro_worst_side_nmae"] = max(side)
    one_loop_values = []
    for object_id in sorted(set(object_ids)):
        indices = [index for index, value in enumerate(object_ids) if value == object_id]
        one_loop_values.append((output.loop_distances[indices, :, 0] - target[indices]).abs().mean(dim=0))
    one_loop_side = [float(value) for value in torch.stack(one_loop_values).mean(dim=0)]
    metrics["candidate_object_macro_one_loop_side_nmae"] = one_loop_side
    metrics["candidate_object_macro_one_loop_mean_side_nmae"] = sum(one_loop_side) / 2.0
    metrics["candidate_object_macro_one_loop_worst_side_nmae"] = max(one_loop_side)


def add_cr_fpl_object_macro_metrics(
    metrics: dict[str, Any], output: Any, target: Tensor, object_ids: Iterable[str]
) -> None:
    object_ids = tuple(object_ids)
    values = []
    for object_id in sorted(set(object_ids)):
        indices = [index for index, value in enumerate(object_ids) if value == object_id]
        values.append((output.distance[indices] - target[indices]).abs().mean(dim=0))
    side = [float(value) for value in torch.stack(values).mean(dim=0)]
    metrics["cr_fpl_control_object_macro_side_nmae"] = side
    metrics["cr_fpl_control_object_macro_mean_side_nmae"] = sum(side) / 2.0
    metrics["cr_fpl_control_object_macro_worst_side_nmae"] = max(side)
    one_loop_values = []
    for object_id in sorted(set(object_ids)):
        indices = [index for index, value in enumerate(object_ids) if value == object_id]
        one_loop_values.append((output.loop_distances[indices, :, 0] - target[indices]).abs().mean(dim=0))
    one_loop_side = [float(value) for value in torch.stack(one_loop_values).mean(dim=0)]
    metrics["cr_fpl_control_object_macro_one_loop_side_nmae"] = one_loop_side
    metrics["cr_fpl_control_object_macro_one_loop_mean_side_nmae"] = sum(one_loop_side) / 2.0
    metrics["cr_fpl_control_object_macro_one_loop_worst_side_nmae"] = max(one_loop_side)


__all__ = [
    "CONTROLS",
    "add_cr_fpl_object_macro_metrics",
    "add_object_macro_metrics",
    "apply_semantic_control",
    "binary_auroc",
    "normalize_semantic_by_train_rms",
    "semantic_endpoint_auroc",
    "semantic_endpoint_object_macro_auroc",
    "summarize_candidate",
    "summarize_cr_fpl_control",
]
