"""Deterministic source-only fitting and audit utilities for SMARC."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import torch
from torch import Tensor

from .smarc import SMARC, extensions_to_endpoints, project_extensions_to_range


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
                                values: dict[str, float]) -> tuple[dict[str, str], dict]:
    """Minimum-cost domain-preserving scalar matching with a train-only scale."""
    import scipy
    from scipy.optimize import linear_sum_assignment

    train_objects = sorted({rows[i]["object_group_id"] for i in train_indices})
    recipient_objects = sorted({rows[i]["object_group_id"] for i in recipient_indices})
    domains={row["object_group_id"]:row["domain"] for row in rows}
    donors={}; distances=[]; regrets=[]; held_distances=[]; held_load={}
    for domain in sorted(set(domains[obj] for obj in train_objects)):
        members=[obj for obj in train_objects if domains[obj]==domain]
        recipients=[obj for obj in recipient_objects if domains[obj]==domain]
        if len(members)<2: raise ValueError("matching domain has fewer than two train objects")
        raw=torch.tensor([values[obj] for obj in members],dtype=torch.float64)
        if not torch.isfinite(raw).all(): raise ValueError("nonfinite matching scalar")
        median=torch.quantile(raw,.5); scale=1.4826*torch.quantile((raw-median).abs(),.5)
        if float(scale)<=1e-12: raise ValueError("matching robust scale is degenerate")
        z=(raw-median)/scale
        cost=(z[:,None]-z[None,:]).abs().numpy(); cost[range(len(members)),range(len(members))]=1e12
        row_index,col_index=linear_sum_assignment(cost)
        if list(row_index)!=list(range(len(members))): raise RuntimeError("unexpected Hungarian row order")
        assigned={members[i]:members[int(col_index[i])] for i in row_index}
        if set(assigned.values())!=set(members) or any(key==value for key,value in assigned.items()):
            raise RuntimeError("Hungarian assignment is not a derangement")
        for obj,donor in assigned.items():
            if obj in recipients:
                donors[obj]=donor; i=members.index(obj); j=members.index(donor); distances.append(float(abs(z[i]-z[j])))
                nearest=min(float(abs(z[i]-z[k])) for k in range(len(members)) if k!=i)
                regrets.append(float(abs(z[i]-z[j]))-nearest)
        for obj in recipients:
            if obj in assigned: continue
            value=(float(values[obj])-float(median))/float(scale)
            donor=min(members,key=lambda candidate:(abs(value-float(z[members.index(candidate)])),candidate))
            donors[obj]=donor; distance=abs(value-float(z[members.index(donor)])); held_distances.append(distance); held_load[donor]=held_load.get(donor,0)+1
    if set(donors)!=set(recipient_objects): raise RuntimeError("matching did not cover recipients")
    def stats(data):
        value=torch.tensor(data or [0.],dtype=torch.float64)
        return {"median":float(value.median()),"p90":float(torch.quantile(value,.9)),"max":float(value.max())}
    payload="\n".join(f"{key}->{donors[key]}" for key in sorted(donors))+"\n"
    receipt={"solver":"scipy.optimize.linear_sum_assignment","scipy_version":scipy.__version__,
             "mapping_sha256":hashlib.sha256(payload.encode()).hexdigest(),"train_distance":stats(distances),
             "held_distance":stats(held_distances),"nearest_other_regret_p90":float(torch.quantile(torch.tensor(regrets or [0.]),.9)),
             "held_donor_load":held_load,"held_effective_donor_count":len(held_load)}
    return donors,receipt


def apply_object_donors(rows: list[dict], indices: list[int], field: str,
                        donors: dict[str, str], train_indices: list[int], preserve_last: bool = False) -> Tensor:
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
        donated=bank[cursor % len(bank)]
        if preserve_last:
            donated=torch.cat((donated[:-1],rows[i][field][-1:]))
        outputs.append(donated)
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


def swap_audit(rows: list[dict], indices: list[int], prediction: Tensor) -> dict[str, float]:
    pairs: dict[str, list[int]]={}
    for location,index in enumerate(indices): pairs.setdefault(rows[index]["swap_pair_id"],[]).append(location)
    range_error=[]; projection_error=[]; endpoint_error=[]
    for locations in pairs.values():
        if len(locations)!=2: raise ValueError("held swap pair is incomplete")
        forward=next(k for k in locations if rows[indices[k]]["order"]=="forward")
        reverse=next(k for k in locations if rows[indices[k]]["order"]=="reverse")
        fr,rr=rows[indices[forward]],rows[indices[reverse]]
        range_error.append((prediction[forward]-prediction[reverse]).abs())
        fext=project_extensions_to_range(fr["base_extension"][None],prediction[forward][None],torch.tensor([fr["observed_displacement"]],dtype=prediction.dtype))[0]
        rext=project_extensions_to_range(rr["base_extension"][None],prediction[reverse][None],torch.tensor([rr["observed_displacement"]],dtype=prediction.dtype))[0]
        projection_error.append((rext-fext.flip(0)).abs().max())
        fpoint=extensions_to_endpoints(fext[None])[0]; rpoint=extensions_to_endpoints(rext[None])[0]
        expected=torch.stack((1-fpoint[1],1-fpoint[0]))
        endpoint_error.append((rpoint-expected).abs().max())
    return {"range_max":float(torch.stack(range_error).max()),
            "projection_max":float(torch.stack(projection_error).max()),
            "endpoint_max":float(torch.stack(endpoint_error).max())}


__all__ = ["Preprocessor", "analytic_linear_baseline", "apply_object_donors",
           "deterministic_object_donors", "fit_preprocessor", "hash_ids", "object_domain_weights",
           "object_macro_mare", "predict", "swap_audit", "train_model", "transform"]
