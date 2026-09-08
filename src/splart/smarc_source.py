"""Deterministic source-only fitting and audit utilities for SMARC."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import torch
from torch import Tensor

from .smarc import SMARC


def hash_ids(values: list[str]) -> str:
    return hashlib.sha256(("\n".join(sorted(values)) + "\n").encode()).hexdigest()


def object_domain_weights(rows: list[dict], indices: list[int]) -> Tensor:
    """Equal domain -> object -> joint -> gauge weights; IDs stay sidecars."""
    domains = sorted({rows[i]["domain"] for i in indices})
    weights = torch.zeros(len(indices), dtype=torch.float64)
    for domain in domains:
        objects = sorted({rows[i]["object_group_id"] for i in indices if rows[i]["domain"] == domain})
        for obj in objects:
            joints = sorted({rows[i]["joint_id"] for i in indices
                             if rows[i]["domain"] == domain and rows[i]["object_group_id"] == obj})
            for joint in joints:
                locations = [k for k, i in enumerate(indices) if rows[i]["domain"] == domain
                             and rows[i]["object_group_id"] == obj and rows[i]["joint_id"] == joint]
                weights[locations] = 1.0 / len(domains) / len(objects) / len(joints) / len(locations)
    if not torch.allclose(weights.sum(), torch.tensor(1., dtype=weights.dtype), atol=1e-12, rtol=0):
        raise RuntimeError("object/domain weights do not sum to one")
    return weights


@dataclass
class Preprocessor:
    mechanical_mean: Tensor
    mechanical_scale: Tensor
    mechanical_keep: Tensor
    semantic_mean: Tensor
    semantic_components: Tensor
    global_prior_logit: float
    train_object_hash: str

    def transform_mechanical(self, value: Tensor) -> Tensor:
        return ((value - self.mechanical_mean) / self.mechanical_scale)[..., self.mechanical_keep]

    def transform_semantic(self, value: Tensor) -> Tensor:
        return (value - self.semantic_mean) @ self.semantic_components.T


def _fix_svd_sign(components: Tensor) -> Tensor:
    result = components.clone()
    for row in result:
        pivot = int(row.abs().argmax())
        if row[pivot] < 0:
            row.mul_(-1)
    return result


def fit_preprocessor(rows: list[dict], targets: Tensor, indices: list[int], pca_dim: int = 16) -> Preprocessor:
    weights = object_domain_weights(rows, indices)
    mechanical = torch.stack([rows[i]["mechanical"].double() for i in indices])
    semantic = torch.stack([rows[i]["semantic"].double() for i in indices])
    mean_m = (weights[:, None] * mechanical).sum(0)
    variance = (weights[:, None] * (mechanical - mean_m).square()).sum(0)
    keep = variance > 1e-12
    if int(keep.sum()) == 0:
        raise ValueError("all mechanical columns are constant")
    scale = variance.sqrt().clamp_min(1e-8)
    mean_s = (weights[:, None] * semantic).sum(0)
    centered = (semantic - mean_s) * weights.sqrt()[:, None]
    _, singular, vh = torch.linalg.svd(centered, full_matrices=False)
    if vh.shape[0] < pca_dim or float(singular[pca_dim-1]) <= 1e-10:
        raise ValueError("not enough source rank for frozen PCA dimension")
    components = _fix_svd_sign(vh[:pca_dim])
    d = torch.tensor([abs(float(rows[i]["observed_displacement"])) for i in indices], dtype=torch.float64)
    y = targets[indices].double()
    if torch.any(d<=0) or torch.any(d>=2*math.pi) or torch.any(y<d-1e-8) or torch.any(y>2*math.pi+1e-8):
        raise ValueError("source revolute range must satisfy 0 < d <= y <= 2pi")
    fraction = ((y - d) / (2 * math.pi - d)).clamp(1e-6, 1 - 1e-6)
    mean_fraction = float((weights * fraction).sum())
    prior = math.log(mean_fraction / (1 - mean_fraction))
    objects=sorted({rows[i]["object_group_id"] for i in indices})
    return Preprocessor(mean_m, scale, keep, mean_s, components, prior,hash_ids(objects))


def transform(preprocessor: Preprocessor, rows: list[dict], indices: list[int],
              semantic_override: Tensor | None = None, mechanical_override: Tensor | None = None) -> tuple[Tensor, Tensor, Tensor]:
    mechanical = (torch.stack([rows[i]["mechanical"].double() for i in indices])
                  if mechanical_override is None else mechanical_override.double())
    semantic = (torch.stack([rows[i]["semantic"].double() for i in indices])
                if semantic_override is None else semantic_override.double())
    displacement = torch.tensor([abs(float(rows[i]["observed_displacement"])) for i in indices], dtype=torch.float64)
    return preprocessor.transform_mechanical(mechanical), preprocessor.transform_semantic(semantic), displacement


def analytic_linear_baseline(displacement: Tensor, targets: Tensor, train_weights: Tensor,
                             held_displacement: Tensor) -> Tensor:
    x = torch.stack((torch.ones_like(displacement), displacement), -1).double()
    lhs = x.T @ (train_weights[:, None] * x) + torch.eye(2, dtype=torch.float64) * 1e-8
    beta = torch.linalg.solve(lhs, x.T @ (train_weights * targets.double()))
    prediction=torch.stack((torch.ones_like(held_displacement), held_displacement), -1).double() @ beta
    return torch.maximum(prediction,held_displacement).clamp_max(2*math.pi)


def deterministic_object_donors(rows: list[dict], train_indices: list[int], recipient_indices: list[int],
                                values: dict[str, float], edges: list[float]) -> dict[str, str]:
    """Frozen object-level donor mapping inside preregistered bins."""
    train_objects = sorted({rows[i]["object_group_id"] for i in train_indices})
    recipient_objects = sorted({rows[i]["object_group_id"] for i in recipient_indices})

    def bucket(value: float) -> int:
        for index in range(len(edges) - 1):
            if edges[index] <= value < edges[index + 1] or (index == len(edges)-2 and value == edges[-1]):
                return index
        raise ValueError(f"shuffle value {value} outside frozen bins")

    domains={row["object_group_id"]:row["domain"] for row in rows}
    bins: dict[tuple[str,int], list[str]] = {}
    for obj in train_objects:
        bins.setdefault((domains[obj],bucket(values[obj])), []).append(obj)
    for members in bins.values():
        if len(members) < 2:
            raise ValueError("occupied training shuffle bin has fewer than two objects")
    donors = {}
    for obj in recipient_objects:
        members = bins.get((domains[obj],bucket(values[obj])), [])
        if not members:
            raise ValueError("recipient shuffle bin has no training donor")
        if obj in members:
            donors[obj] = members[(members.index(obj) + 1) % len(members)]
        else:
            donors[obj] = members[0]
        if donors[obj] == obj:
            raise RuntimeError("shuffle donor is not a derangement")
    return donors


def apply_object_donors(rows: list[dict], indices: list[int], field: str,
                        donors: dict[str, str], train_indices: list[int]) -> Tensor:
    by_object: dict[str, list[Tensor]] = {}
    for i in train_indices:
        by_object.setdefault(rows[i]["object_group_id"], []).append(rows[i][field])
    outputs = []
    object_counter: dict[str, int] = {}
    for i in indices:
        obj = rows[i]["object_group_id"]
        donor = donors[obj]
        cursor = object_counter.get(obj, 0)
        bank = by_object[donor]
        outputs.append(bank[cursor % len(bank)])
        object_counter[obj] = cursor + 1
    return torch.stack(outputs)


def train_model(rows: list[dict], targets: Tensor, train_indices: list[int], preprocessor: Preprocessor,
                semantic_override: Tensor | None = None, mechanical_override: Tensor | None = None,
                steps: int = 1200, device: str = "cuda") -> SMARC:
    mechanical, semantic, displacement = transform(preprocessor, rows, train_indices,
                                                    semantic_override, mechanical_override)
    train_targets = targets[train_indices].double()
    weights = object_domain_weights(rows, train_indices)
    model = SMARC(mechanical.shape[-1], semantic.shape[-1], preprocessor.global_prior_logit).double().to(device)
    mechanical, semantic = mechanical.to(device), semantic.to(device)
    displacement, train_targets, weights = displacement.to(device), train_targets.to(device), weights.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01, weight_decay=.001)
    for _ in range(steps):
        prediction, details = model(mechanical, semantic, displacement)
        relative = (prediction - train_targets) / train_targets
        loss = (weights * torch.sqrt(relative.square() + 1e-6)).sum()
        load = (weights[:, None] * details["gate"]).sum(0)
        loss = loss + .01 * (load - .25).square().sum()
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
    return model.cpu()


def predict(model: SMARC, preprocessor: Preprocessor, rows: list[dict], indices: list[int],
            semantic_override: Tensor | None = None, mechanical_override: Tensor | None = None) -> Tensor:
    mechanical, semantic, displacement = transform(preprocessor, rows, indices,
                                                    semantic_override, mechanical_override)
    with torch.no_grad():
        return model(mechanical, semantic, displacement)[0]


def object_macro_mare(rows: list[dict], indices: list[int], prediction: Tensor, targets: Tensor) -> float:
    values = []
    for obj in sorted({rows[i]["object_group_id"] for i in indices}):
        locations = [k for k, i in enumerate(indices) if rows[i]["object_group_id"] == obj]
        truth = targets[[indices[k] for k in locations]]
        values.append(((prediction[locations] - truth).abs() / truth).mean())
    return float(torch.stack(values).mean())


__all__ = ["Preprocessor", "analytic_linear_baseline", "apply_object_donors",
           "deterministic_object_donors", "fit_preprocessor", "hash_ids", "object_domain_weights",
           "object_macro_mare", "predict", "train_model", "transform"]
