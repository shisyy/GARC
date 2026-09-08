"""Raw-z anchored anisotropic conditional whitening transport.

This is the unique node-8.11 label-free null.  It intentionally does not
import, mutate, or alias the node-8.10 RQ-LSOT implementation history.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math

import torch
from torch import Tensor

from .conditional_residual import _distance_correlation, _rank, _spearman
from .smarc_source import fit_feature_preprocessor, transform


SCHEMA = "splart-rza-acwt-null/v1"
CROSSFIT_SCHEMA = "splart-rza-acwt-fold-local-crossfit/v1"


def _sha_tensor(value: Tensor) -> str:
    data = value.detach().cpu().contiguous().double().numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def _sha_ids(values: list[str]) -> str:
    return hashlib.sha256(("\n".join(values) + "\n").encode()).hexdigest()


def _canonical_sha(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    return hashlib.sha256(payload.encode()).hexdigest()


def global_row_identity(row: dict) -> str:
    """A stable row identity whose namespace is global across source domains."""
    domain = str(row.get("domain", ""))
    if not domain or ":" in domain:
        raise ValueError("nonempty colon-free domain required")
    for key in ("gauge_id", "row_id"):
        if key in row:
            local = str(row[key])
            break
    else:
        if "joint_id" not in row or "order" not in row:
            raise ValueError("stable gauge_id/row_id sidecar required")
        local = f"{row['joint_id']}:{row['order']}"
    if not local:
        raise ValueError("empty local row identity")
    return f"{domain}:{local}"


def _canonical_indices(rows: list[dict], indices: list[int]) -> list[int]:
    result = sorted(indices, key=lambda i: global_row_identity(rows[i]))
    identities = [global_row_identity(rows[i]) for i in result]
    if len(identities) != len(set(identities)):
        raise ValueError("global row identity is not unique")
    return result


def canonical_object_scalar(rows: list[dict], indices: list[int], kind: str,
                            reference: list[int] | None = None) -> dict[str, float]:
    """Canonical object reductions, independent of incoming row order."""
    indices = _canonical_indices(rows, indices)
    objects = sorted({str(rows[i]["object_group_id"]) for i in indices})
    if kind == "semantic":
        result = {}
        for obj in objects:
            locations = [i for i in indices if rows[i]["object_group_id"] == obj]
            values = torch.tensor([abs(float(rows[i]["observed_displacement"]))
                                   for i in locations], dtype=torch.float64)
            result[obj] = float(values.mean())
        return result
    if kind != "mechanical":
        raise ValueError("unknown RZA-ACWT field kind")
    reference = indices if reference is None else _canonical_indices(rows, reference)
    result: dict[str, float] = {}
    domains = sorted({str(rows[i]["domain"]) for i in indices})
    for domain in domains:
        ref_objects = sorted({str(rows[i]["object_group_id"]) for i in reference
                              if rows[i]["domain"] == domain})
        if not ref_objects:
            raise ValueError("mechanical scalar reference domain is empty")
        ref_means = []
        for obj in ref_objects:
            locations = [i for i in reference if rows[i]["domain"] == domain
                         and rows[i]["object_group_id"] == obj]
            ref_means.append(torch.stack([rows[i]["semantic"].double()
                                          for i in locations]).mean(0))
        reference_mean = torch.stack(ref_means).mean(0)
        for obj in [value for value in objects if any(
                rows[i]["domain"] == domain and rows[i]["object_group_id"] == value
                for i in indices)]:
            locations = [i for i in indices if rows[i]["domain"] == domain
                         and rows[i]["object_group_id"] == obj]
            value = torch.stack([rows[i]["semantic"].double() for i in locations]).mean(0)
            result[obj] = float(1 - torch.nn.functional.cosine_similarity(
                value[None], reference_mean[None]).item())
    return result


def empirical_rank(train_z: Tensor, held_z: Tensor | None = None) -> tuple[Tensor, Tensor | None, float]:
    train_z = train_z.double()
    if train_z.ndim != 1 or len(train_z) < 4 or not torch.isfinite(train_z).all():
        raise ValueError("invalid rank reference")
    n = len(train_z)
    train_x = 2 * ((_rank(train_z) + .5) / n) - 1
    if held_z is None:
        return train_x, None, 0.
    held_z = held_z.double()
    if held_z.ndim != 1 or not torch.isfinite(held_z).all():
        raise ValueError("invalid held rank values")
    raw = []
    for value in held_z:
        less = (train_z < value).sum()
        equal = (train_z == value).sum()
        raw.append((less.double() + .5 * equal.double()) / n)
    raw_tensor = torch.stack(raw)
    low, high = .5 / n, 1 - .5 / n
    clipped = float(((raw_tensor < low) | (raw_tensor > high)).double().mean())
    return train_x, 2 * raw_tensor.clamp(low, high) - 1, clipped


def _positive_qr(design: Tensor, rank_tolerance: float,
                 condition_max: float) -> tuple[Tensor, Tensor, dict]:
    design = design.double()
    if (design.ndim != 2 or design.shape[0] < design.shape[1]
            or not torch.isfinite(design).all()):
        raise ValueError("invalid QR design")
    q, r = torch.linalg.qr(design, mode="reduced")
    diagonal = torch.diag(r)
    signs = torch.where(diagonal < 0, -torch.ones_like(diagonal), torch.ones_like(diagonal))
    q, r = q * signs[None], signs[:, None] * r
    singular = torch.linalg.svdvals(r)
    threshold = float(rank_tolerance) * float(singular.max())
    rank = int((singular > threshold).sum())
    condition = float(singular.max() / singular.min())
    if rank != design.shape[1] or not math.isfinite(condition) or condition > condition_max:
        raise ValueError("RZA-ACWT QR rank/condition gate failed")
    if torch.any(torch.diag(r) <= 0):
        raise RuntimeError("positive QR diagonal contract failed")
    return q, r, {"rank": rank, "condition": condition, "positive_diagonal": True,
                  "rank_threshold": threshold}


def _p2(x: Tensor) -> Tensor:
    return (3 * x.square() - 1) / 2


def _solve(design: Tensor, target: Tensor, contract: dict) -> tuple[Tensor, dict]:
    q, r, receipt = _positive_qr(design, float(contract["rank_relative_tolerance"]),
                                 float(contract["condition_max"]))
    beta = torch.linalg.solve_triangular(r, q.T @ target, upper=True)
    return beta, receipt


def _fix_axis_sign(axis: Tensor) -> tuple[Tensor, int]:
    pivot = int(axis.abs().argmax())
    if axis[pivot] < 0:
        axis = -axis
    return axis, pivot


def _fit_model(z: Tensor, x: Tensor, means: Tensor, contract: dict) -> dict:
    z, x, means = z.double(), x.double(), means.double()
    if (z.ndim != 1 or x.shape != z.shape or means.ndim != 2
            or means.shape[0] != len(z) or means.shape[1] < 2
            or not all(torch.isfinite(v).all() for v in (z, x, means))):
        raise ValueError("invalid RZA-ACWT fit input")
    z_mean = z.mean()
    z_centered = z - z_mean
    linear = torch.stack((torch.ones_like(z), z_centered), -1)
    q_projection, q_projection_qr = _solve(linear, _p2(x)[:, None], contract)
    q_residual = _p2(x) - (linear @ q_projection)[:, 0]
    q_rms = q_residual.square().mean().sqrt()
    if not torch.isfinite(q_rms) or float(q_rms) <= 1e-12:
        raise ValueError("RZA-ACWT q_perp is degenerate")
    q_perp = q_residual / q_rms
    location_basis = torch.stack((torch.ones_like(z), z_centered, q_perp), -1)
    location_beta, location_qr = _solve(location_basis, means, contract)
    location = location_basis @ location_beta
    residual = means - location
    centered = means - means.mean(0)
    centered_rms = float(centered.square().mean().sqrt())
    epsilon0 = max(1e-12, 1e-6 * centered_rms)

    radial = residual.square().mean(-1).sqrt()
    radial_basis = torch.stack((torch.ones_like(x), x), -1)
    radial_gamma, radial_qr = _solve(radial_basis,
                                     torch.log(radial + epsilon0)[:, None], contract)
    radial_gamma = radial_gamma[:, 0]
    radial_scale = torch.exp(radial_basis @ radial_gamma)
    if not torch.isfinite(radial_scale).all() or torch.any(radial_scale <= 0):
        raise ValueError("invalid scalar radial scale")
    g = residual / radial_scale[:, None]

    covariance = torch.einsum("ni,nj->nij", g, g)
    c0 = covariance.mean(0)
    centered_covariance = covariance - c0
    a1 = torch.einsum("n,nij->ij", x, centered_covariance) / len(x)
    a2 = torch.einsum("n,nij->ij", _p2(x), centered_covariance) / len(x)
    h = a1 @ a1 + a2 @ a2
    h = (h + h.T) * .5
    eigenvalues, eigenvectors = torch.linalg.eigh(h)
    operator_norm = float(eigenvalues.abs().max())
    eigengap = float(eigenvalues[-1] - eigenvalues[-2])
    eigengap_threshold = (float(contract["eigengap_multiplier"])
                          * torch.finfo(torch.float64).eps * operator_norm)
    if not math.isfinite(eigengap) or eigengap <= eigengap_threshold:
        raise ValueError("RZA-ACWT covariance eigengap gate failed")
    axis, axis_pivot = _fix_axis_sign(eigenvectors[:, -1])

    axial = g @ axis
    orthogonal = g - axial[:, None] * axis
    orthogonal_rms = (orthogonal.square().sum(-1) / (means.shape[1] - 1)).sqrt()
    epsilon_g = max(1e-12, 1e-6 * float(g.square().mean().sqrt()))
    block_basis = torch.stack((torch.ones_like(x), x, _p2(x)), -1)
    block_target = torch.stack((torch.log(axial.abs() + epsilon_g),
                                torch.log(orthogonal_rms + epsilon_g)), -1)
    block_gamma, block_qr = _solve(block_basis, block_target, contract)
    block_scale = torch.exp(block_basis @ block_gamma)
    if not torch.isfinite(block_scale).all() or torch.any(block_scale <= 0):
        raise ValueError("invalid anisotropic block scale")
    standardized = ((axial / block_scale[:, 0])[:, None] * axis
                    + orthogonal / block_scale[:, 1, None])
    reconstructed_g = ((standardized @ axis) * block_scale[:, 0])[:, None] * axis
    reconstructed_g += (standardized - (standardized @ axis)[:, None] * axis) * block_scale[:, 1, None]
    reconstruction = location + radial_scale[:, None] * reconstructed_g

    field_scale = means.std(0, unbiased=False).clamp_min(1e-12)
    raw_z_scale = z_centered.std(unbiased=False).clamp_min(1e-12)
    raw_z_correlation = float(((z_centered[:, None] * residual).mean(0).abs()
                               / (raw_z_scale * field_scale)).max())
    return {
        "z_mean": z_mean, "q_projection": q_projection[:, 0], "q_rms": q_rms,
        "location_beta": location_beta, "radial_gamma": radial_gamma,
        "axis": axis, "axis_pivot": axis_pivot, "block_gamma": block_gamma,
        "location": location, "residual": residual, "radial_scale": radial_scale,
        "standardized": standardized, "c0": c0, "a1": a1, "a2": a2, "h": h,
        "eigenvalues": eigenvalues, "eigengap": eigengap,
        "eigengap_threshold": eigengap_threshold, "operator_norm": operator_norm,
        "epsilon0": epsilon0, "epsilon_g": epsilon_g,
        "q_projection_qr": q_projection_qr, "location_qr": location_qr,
        "radial_qr": radial_qr, "block_qr": block_qr,
        "q_constant_orthogonality": float(q_perp.mean().abs()),
        "q_raw_z_orthogonality": float((q_perp * z_centered).mean().abs()),
        "location_basis_orthogonality_max": float((location_basis.T @ residual / len(z)).abs().max()),
        "radial_basis_orthogonality_max": float((radial_basis.T @
            (torch.log(radial + epsilon0) - radial_basis @ radial_gamma)[:, None] / len(z)).abs().max()),
        "block_basis_orthogonality_max": float((block_basis.T @
            (block_target - block_basis @ block_gamma) / len(z)).abs().max()),
        "normalized_residual_mean_max": float((residual.mean(0).abs() / field_scale).max()),
        "normalized_residual_raw_z_correlation_max": raw_z_correlation,
        "reconstruction_max_error": float((reconstruction - means).abs().max()),
        "residual_energy_ratio": float(residual.square().mean()
                                       / centered.square().mean().clamp_min(1e-12)),
        "radial_scale_min": float(radial_scale.min()),
        "radial_scale_max": float(radial_scale.max()),
        "parallel_scale_min": float(block_scale[:, 0].min()),
        "parallel_scale_max": float(block_scale[:, 0].max()),
        "perpendicular_scale_min": float(block_scale[:, 1].min()),
        "perpendicular_scale_max": float(block_scale[:, 1].max()),
    }


def _predict(model: dict, z: Tensor, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    z, x = z.double(), x.double()
    z_centered = z - model["z_mean"]
    linear = torch.stack((torch.ones_like(z), z_centered), -1)
    q_perp = (_p2(x) - linear @ model["q_projection"]) / model["q_rms"]
    basis = torch.stack((torch.ones_like(z), z_centered, q_perp), -1)
    location = basis @ model["location_beta"]
    radial_scale = torch.exp(torch.stack((torch.ones_like(x), x), -1) @ model["radial_gamma"])
    block_scale = torch.exp(torch.stack((torch.ones_like(x), x, _p2(x)), -1) @ model["block_gamma"])
    if not all(torch.isfinite(value).all() for value in (location, radial_scale, block_scale)):
        raise ValueError("nonfinite RZA-ACWT prediction")
    if torch.any(radial_scale <= 0) or torch.any(block_scale <= 0):
        raise ValueError("nonpositive RZA-ACWT prediction scale")
    return location, radial_scale, block_scale[:, 0], block_scale[:, 1]


def _reconstruct(location: Tensor, radial: Tensor, parallel: Tensor, perpendicular: Tensor,
                 standardized: Tensor, axis: Tensor) -> Tensor:
    coefficient = standardized @ axis
    orthogonal = standardized - coefficient[:, None] * axis
    g = (parallel * coefficient)[:, None] * axis + perpendicular[:, None] * orthogonal
    return location + radial[:, None] * g


def _balanced_folds(domain: str, object_ids: list[str], folds: int) -> list[int]:
    if len(set(object_ids)) != len(object_ids) or len(object_ids) < 2 * folds:
        raise ValueError("cross-fit needs two objects per fold")
    order = sorted(range(len(object_ids)), key=lambda i: (
        hashlib.sha256(("splart-rza-acwt-crossfit-v1:" + domain + ":" + object_ids[i]).encode()).hexdigest(),
        object_ids[i]))
    result = [-1] * len(object_ids)
    for rank, index in enumerate(order):
        result[index] = rank % folds
    return result


def _preprocessor_parts(preprocessor) -> dict:
    return {"mechanical_mean": _sha_tensor(preprocessor.mechanical_mean),
            "mechanical_scale": _sha_tensor(preprocessor.mechanical_scale),
            "mechanical_keep": hashlib.sha256(preprocessor.mechanical_keep.cpu().numpy().tobytes()).hexdigest(),
            "semantic_mean": _sha_tensor(preprocessor.semantic_mean),
            "semantic_components": _sha_tensor(preprocessor.semantic_components),
            "train_object_hash": preprocessor.train_object_hash,
            "global_prior_logit": preprocessor.global_prior_logit}


def _model_provenance(model: dict) -> dict:
    tensor_names = ("q_projection", "location_beta", "radial_gamma", "axis", "block_gamma",
                    "c0", "a1", "a2", "h", "eigenvalues")
    result = {name + "_sha256": _sha_tensor(model[name]) for name in tensor_names}
    result.update({"z_mean": float(model["z_mean"]), "q_rms": float(model["q_rms"]),
                   "axis_pivot": model["axis_pivot"], "eigengap": model["eigengap"],
                   "eigengap_threshold": model["eigengap_threshold"],
                   "operator_norm": model["operator_norm"], "epsilon0": model["epsilon0"],
                   "epsilon_g": model["epsilon_g"], "q_projection_qr": model["q_projection_qr"],
                   "location_qr": model["location_qr"], "radial_qr": model["radial_qr"],
                   "block_qr": model["block_qr"]})
    return result


def feature_crossfit_rza_acwt(rows: list[dict], train_indices: list[int], field_kind: str,
                              contract: dict) -> dict[str, dict]:
    """Fold-local OOF E with a joint-domain preprocessor and domain-local RZA fits."""
    train_indices = _canonical_indices(rows, train_indices)
    folds = int(contract["crossfit_folds"])
    domains = sorted({rows[i]["domain"] for i in train_indices})
    domain_objects = {domain: sorted({rows[i]["object_group_id"] for i in train_indices
                                      if rows[i]["domain"] == domain}) for domain in domains}
    assignment = {domain: {obj: fold for obj, fold in zip(
        domain_objects[domain], _balanced_folds(domain, domain_objects[domain], folds))}
        for domain in domains}
    raw_geometry_dim = int(rows[train_indices[0]]["mechanical"].numel()) - 1
    raw_semantic_dim = int(rows[train_indices[0]]["semantic"].numel())
    outputs = {domain: {
        "standardized": (torch.full((len(domain_objects[domain]), raw_semantic_dim), float("nan"), dtype=torch.float64)
                         if field_kind == "semantic" else
                         torch.zeros((len(domain_objects[domain]), raw_geometry_dim), dtype=torch.float64)),
        "rank_x": torch.full((len(domain_objects[domain]),), float("nan"), dtype=torch.float64),
        "folds": []} for domain in domains}

    for fold in range(folds):
        fit_qualified = {f"{domain}:{obj}" for domain in domains for obj in domain_objects[domain]
                         if assignment[domain][obj] != fold}
        held_qualified = {f"{domain}:{obj}" for domain in domains for obj in domain_objects[domain]
                          if assignment[domain][obj] == fold}
        fit_indices = _canonical_indices(rows, [i for i in train_indices
            if f"{rows[i]['domain']}:{rows[i]['object_group_id']}" in fit_qualified])
        held_indices = _canonical_indices(rows, [i for i in train_indices
            if f"{rows[i]['domain']}:{rows[i]['object_group_id']}" in held_qualified])
        preprocessor = fit_feature_preprocessor(rows, fit_indices, 16)
        fit_mech, fit_sem, _ = transform(preprocessor, rows, fit_indices)
        held_mech, held_sem, _ = transform(preprocessor, rows, held_indices)
        if field_kind == "semantic":
            fit_field, held_field, kept = fit_sem, held_sem, None
        elif field_kind == "mechanical":
            kept = torch.nonzero(preprocessor.mechanical_keep).flatten().tolist()
            if not kept or kept[-1] != len(preprocessor.mechanical_keep) - 1:
                raise ValueError("fold mechanical displacement column missing")
            fit_field, held_field, kept = fit_mech[:, :-1], held_mech[:, :-1], kept[:-1]
        else:
            raise ValueError("unknown field kind")
        prep = _preprocessor_parts(preprocessor)
        common = {"fold": fold, "joint_domain_fit_object_hash": _sha_ids(sorted(fit_qualified)),
                  "joint_domain_held_object_hash": _sha_ids(sorted(held_qualified)),
                  "preprocessor_sha256": _canonical_sha(prep), "preprocessor_provenance": prep,
                  "raw_mechanical_dim": len(preprocessor.mechanical_keep),
                  "mechanical_kept_indices": torch.nonzero(preprocessor.mechanical_keep).flatten().tolist(),
                  "mechanical_keep_mask": [bool(v) for v in preprocessor.mechanical_keep.tolist()]}
        for domain in domains:
            fit_objects = [obj for obj in domain_objects[domain] if assignment[domain][obj] != fold]
            held_objects = [obj for obj in domain_objects[domain] if assignment[domain][obj] == fold]
            fi = [k for k, i in enumerate(fit_indices) if rows[i]["domain"] == domain]
            hi = [k for k, i in enumerate(held_indices) if rows[i]["domain"] == domain]
            def means_for(indices, locations, field, objects):
                return torch.stack([field[[k for k in locations
                    if rows[indices[k]]["object_group_id"] == obj]].mean(0) for obj in objects])
            fit_means = means_for(fit_indices, fi, fit_field, fit_objects)
            held_means = means_for(held_indices, hi, held_field, held_objects)
            domain_fit = [fit_indices[k] for k in fi]
            domain_held = [held_indices[k] for k in hi]
            fit_z_map = canonical_object_scalar(rows, domain_fit, field_kind, domain_fit)
            held_z_map = canonical_object_scalar(rows, domain_held, field_kind, domain_fit)
            fit_z = torch.tensor([fit_z_map[obj] for obj in fit_objects], dtype=torch.float64)
            held_z = torch.tensor([held_z_map[obj] for obj in held_objects], dtype=torch.float64)
            fit_x, held_x, clipped = empirical_rank(fit_z, held_z)
            model = _fit_model(fit_z, fit_x, fit_means, contract)
            location, radial, parallel, perpendicular = _predict(model, held_z, held_x)
            residual = (held_means - location) / radial[:, None]
            axis = model["axis"]
            coefficient = residual @ axis
            local = ((coefficient / parallel)[:, None] * axis
                     + (residual - coefficient[:, None] * axis) / perpendicular[:, None])
            for j, obj in enumerate(held_objects):
                target = domain_objects[domain].index(obj)
                outputs[domain]["rank_x"][target] = held_x[j]
                if field_kind == "semantic":
                    outputs[domain]["standardized"][target] = local[j] @ preprocessor.semantic_components
                else:
                    outputs[domain]["standardized"][target, kept] = local[j]
            outputs[domain]["folds"].append({**common,
                "domain_fit_object_hash": _sha_ids(fit_objects),
                "domain_held_object_hash": _sha_ids(held_objects),
                "boundary_clipped_fraction": clipped,
                "model_provenance": _model_provenance(model)})

    result = {}
    for domain in domains:
        objects = domain_objects[domain]
        standardized = outputs[domain]["standardized"]
        domain_indices = [i for i in train_indices if rows[i]["domain"] == domain]
        unified_z_map = canonical_object_scalar(rows, domain_indices, field_kind, domain_indices)
        raw_z = torch.tensor([unified_z_map[obj] for obj in objects], dtype=torch.float64)
        rank_x = outputs[domain]["rank_x"]
        if not all(torch.isfinite(value).all() for value in (standardized, raw_z, rank_x)):
            raise RuntimeError("fold-local crossfit output uninitialized")
        assign = assignment[domain]
        payload = {"object_ids": objects, "fold_assignment": assign,
                   "standardized": standardized.tolist(), "raw_z": raw_z.tolist(),
                   "rank_x": rank_x.tolist()}
        norm = standardized.norm(dim=-1)
        result[domain] = {"schema": CROSSFIT_SCHEMA, "field_kind": field_kind,
            "fold_counts": [list(assign.values()).count(fold) for fold in range(folds)],
            "fold_assignment_sha256": _canonical_sha(assign), "folds": outputs[domain]["folds"],
            "raw_z_diagnostic_scope": "one unified canonical raw-z per domain across all OOF objects",
            "fold_fit_z_scope": "fold-train raw-z is independently ECDF-ranked; held uses that fold train ECDF",
            "residual_norm_raw_z_absolute_spearman": _spearman(norm, raw_z),
            "residual_raw_z_distance_correlation": _distance_correlation(raw_z, standardized),
            "residual_norm_rank_x_absolute_spearman_extra": _spearman(norm, rank_x),
            "residual_rank_x_distance_correlation_extra": _distance_correlation(rank_x, standardized),
            "standardized_sha256": _sha_tensor(standardized), "raw_z_sha256": _sha_tensor(raw_z),
            "rank_x_sha256": _sha_tensor(rank_x), "audit_payload": payload}
    return result


def rza_acwt_ablation(rows: list[dict], train_indices: list[int], held_indices: list[int],
                      train_field: Tensor, held_field: Tensor,
                      train_z: dict[str, float], held_z: dict[str, float], contract: dict,
                      context: str, crossfit_by_domain: dict[str, dict]) -> tuple[Tensor, Tensor, dict]:
    if (train_field.ndim != 2 or train_field.shape[0] != len(train_indices)
            or held_field.shape[0] != len(held_indices)
            or train_field.shape[1:] != held_field.shape[1:] or not context):
        raise ValueError("field alignment mismatch")
    # Reorder tensors and indices together into their one canonical global order.
    train_order = sorted(range(len(train_indices)), key=lambda k: global_row_identity(rows[train_indices[k]]))
    held_order = sorted(range(len(held_indices)), key=lambda k: global_row_identity(rows[held_indices[k]]))
    train_indices = [train_indices[k] for k in train_order]
    held_indices = [held_indices[k] for k in held_order]
    train_field = train_field[train_order].double()
    held_field = held_field[held_order].double()
    all_keys = [global_row_identity(rows[i]) for i in train_indices + held_indices]
    if len(all_keys) != len(set(all_keys)):
        raise ValueError("global row identity is not unique")
    train_domains = {rows[i]["domain"] for i in train_indices}
    if train_domains != {rows[i]["domain"] for i in held_indices}:
        raise ValueError("held/train domain sets differ")
    owners = {}
    for split, indices in (("train", train_indices), ("held", held_indices)):
        for i in indices:
            qualified = f"{rows[i]['domain']}:{rows[i]['object_group_id']}"
            if qualified in owners and owners[qualified] != split:
                raise ValueError("object crosses split")
            owners[qualified] = split

    train_out = torch.full_like(train_field, float("nan"))
    held_out = torch.full_like(held_field, float("nan"))
    receipt = {"schema": SCHEMA, "context": context,
               "contract_sha256": _canonical_sha(contract), "domains": {}}
    for domain in sorted(train_domains):
        train_objects = sorted({rows[i]["object_group_id"] for i in train_indices if rows[i]["domain"] == domain})
        held_objects = sorted({rows[i]["object_group_id"] for i in held_indices if rows[i]["domain"] == domain})
        if not train_objects or not held_objects:
            raise ValueError("each domain needs train and held objects")
        def locations(indices, objects):
            return {obj: [k for k, i in enumerate(indices)
                          if rows[i]["domain"] == domain and rows[i]["object_group_id"] == obj]
                    for obj in objects}
        train_locations, held_locations = locations(train_indices, train_objects), locations(held_indices, held_objects)
        means = torch.stack([train_field[train_locations[obj]].mean(0) for obj in train_objects])
        held_means = torch.stack([held_field[held_locations[obj]].mean(0) for obj in held_objects])
        z = torch.tensor([train_z[obj] for obj in train_objects], dtype=torch.float64)
        hz = torch.tensor([held_z[obj] for obj in held_objects], dtype=torch.float64)
        x, hx, held_clipped = empirical_rank(z, hz)
        model = _fit_model(z, x, means, contract)
        held_location, held_radial, held_parallel, held_perpendicular = _predict(model, hz, hx)
        train_location, train_radial, train_parallel, train_perpendicular = _predict(model, z, x)
        offsets = {obj: train_field[train_locations[obj]] - means[j] for j, obj in enumerate(train_objects)}
        held_offsets = {obj: held_field[held_locations[obj]] - held_means[j] for j, obj in enumerate(held_objects)}
        reconstruction = _reconstruct(train_location, train_radial, train_parallel,
                                      train_perpendicular, model["standardized"], model["axis"])
        train_donor = {obj: train_objects[(j + 1) % len(train_objects)] for j, obj in enumerate(train_objects)}
        held_donor = {obj: train_objects[j % len(train_objects)] for j, obj in enumerate(held_objects)}
        standardized = {obj: model["standardized"][j] for j, obj in enumerate(train_objects)}
        for j, obj in enumerate(train_objects):
            mean = _reconstruct(train_location[j:j+1], train_radial[j:j+1],
                                train_parallel[j:j+1], train_perpendicular[j:j+1],
                                standardized[train_donor[obj]][None], model["axis"])[0]
            train_out[train_locations[obj]] = mean + offsets[obj]
        for j, obj in enumerate(held_objects):
            mean = _reconstruct(held_location[j:j+1], held_radial[j:j+1],
                                held_parallel[j:j+1], held_perpendicular[j:j+1],
                                standardized[held_donor[obj]][None], model["axis"])[0]
            held_out[held_locations[obj]] = mean + held_offsets[obj]
        original = torch.cat([train_field[train_locations[obj]] for obj in train_objects])
        shuffled = torch.cat([train_out[train_locations[obj]] for obj in train_objects])
        held_shuffled = torch.cat([held_out[held_locations[obj]] for obj in held_objects])
        train_u = max(float(((train_out[train_locations[obj]] - train_out[train_locations[obj]].mean(0))
                             - offsets[obj]).abs().max()) for obj in train_objects)
        held_u = max(float(((held_out[held_locations[obj]] - held_out[held_locations[obj]].mean(0))
                            - held_offsets[obj]).abs().max()) for obj in held_objects)
        original_sd = original.std(unbiased=False).clamp_min(1e-12)
        original_p99 = float(torch.quantile(original.norm(dim=-1), .99).clamp_min(1e-12))
        shuffled_p99 = max(float(torch.quantile(shuffled.norm(dim=-1), .99)),
                           float(torch.quantile(held_shuffled.norm(dim=-1), .99)))
        crossfit = copy.deepcopy(crossfit_by_domain[domain])
        payload = {"train_object_ids": train_objects, "held_object_ids": held_objects,
                   "z": z.tolist(), "train_rank_x": x.tolist(),
                   "model": {name: model[name].tolist() for name in (
                       "q_projection", "location_beta", "radial_gamma", "axis", "block_gamma",
                       "c0", "a1", "a2", "h", "eigenvalues")}}
        mapping_payload = {"context": context, "domain": domain,
                           "train": train_donor, "held": held_donor}
        receipt["domains"][domain] = {
            "train_objects": len(train_objects), "held_objects": len(held_objects),
            "train_object_hash": _sha_ids(train_objects), "held_object_hash": _sha_ids(held_objects),
            "train_mapping": train_donor, "held_mapping": held_donor,
            "mapping_sha256": _canonical_sha(mapping_payload), "z_sha256": _sha_tensor(z),
            "train_rank_x_sha256": _sha_tensor(x), "model_provenance": _model_provenance(model),
            "q_constant_orthogonality": model["q_constant_orthogonality"],
            "q_raw_z_orthogonality": model["q_raw_z_orthogonality"],
            "location_basis_orthogonality_max": model["location_basis_orthogonality_max"],
            "radial_basis_orthogonality_max": model["radial_basis_orthogonality_max"],
            "block_basis_orthogonality_max": model["block_basis_orthogonality_max"],
            "normalized_residual_mean_max": model["normalized_residual_mean_max"],
            "normalized_residual_raw_z_correlation_max": model["normalized_residual_raw_z_correlation_max"],
            "reconstruction_max_error": max(model["reconstruction_max_error"],
                                               float((reconstruction - means).abs().max())),
            "residual_energy_ratio": model["residual_energy_ratio"],
            "radial_scale_min": model["radial_scale_min"], "radial_scale_max": model["radial_scale_max"],
            "parallel_scale_min": model["parallel_scale_min"], "parallel_scale_max": model["parallel_scale_max"],
            "perpendicular_scale_min": model["perpendicular_scale_min"],
            "perpendicular_scale_max": model["perpendicular_scale_max"], "scale_clipped": False,
            "held_boundary_clipped_fraction": held_clipped,
            "shuffle_rms_over_original_sd": float((shuffled - original).square().mean().sqrt() / original_sd),
            "shuffled_norm_p99_ratio": shuffled_p99 / original_p99,
            "train_coverage": len(train_donor) / len(train_objects),
            "held_coverage": len(held_donor) / len(held_objects),
            "train_self_rate": sum(k == v for k, v in train_donor.items()) / len(train_objects),
            "train_donor_marginal_exact": sorted(train_donor.values()) == train_objects,
            "held_effective_donors_per_object": len(set(held_donor.values())) / len(held_objects),
            "held_max_donor_load": max(sum(v == donor for v in held_donor.values()) for donor in train_objects),
            "held_load_bound": math.ceil(len(held_objects) / len(train_objects)),
            "recipient_u_train_max_error": train_u, "recipient_u_held_max_error": held_u,
            "crossfit": crossfit, "audit_payload": payload}
    if not torch.isfinite(train_out).all() or not torch.isfinite(held_out).all():
        raise RuntimeError("RZA-ACWT output uninitialized")
    # Restore the caller's row order while keeping every reduction canonical.
    restored_train = torch.empty_like(train_out)
    restored_held = torch.empty_like(held_out)
    for canonical_position, original_position in enumerate(train_order):
        restored_train[original_position] = train_out[canonical_position]
    for canonical_position, original_position in enumerate(held_order):
        restored_held[original_position] = held_out[canonical_position]
    return restored_train, restored_held, receipt


def _receipt_passes_checked(receipt: dict, expected: dict, contract: dict,
                            expected_context: str, expected_domains: set[str]) -> bool:
    if receipt is expected or _canonical_sha(receipt) != _canonical_sha(expected):
        return False
    if (set(receipt) != {"schema", "context", "contract_sha256", "domains"}
            or receipt.get("schema") != SCHEMA or receipt.get("context") != expected_context
            or receipt.get("contract_sha256") != _canonical_sha(contract)
            or set(receipt.get("domains", {})) != expected_domains):
        return False
    expected_field_kind = ("semantic" if expected_context.startswith("semantic_rza_acwt/")
                           else "mechanical" if expected_context.startswith("mechanical_rza_acwt/")
                           else None)
    if expected_field_kind is None:
        return False
    for domain, value in receipt["domains"].items():
        train_ids = value["audit_payload"]["train_object_ids"]
        held_ids = value["audit_payload"]["held_object_ids"]
        if train_ids != sorted(train_ids) or held_ids != sorted(held_ids) or set(train_ids) & set(held_ids):
            return False
        if value["train_mapping"] != {obj: train_ids[(j + 1) % len(train_ids)] for j, obj in enumerate(train_ids)}:
            return False
        if value["held_mapping"] != {obj: train_ids[j % len(train_ids)] for j, obj in enumerate(held_ids)}:
            return False
        if value["train_object_hash"] != _sha_ids(train_ids) or value["held_object_hash"] != _sha_ids(held_ids):
            return False
        if value["mapping_sha256"] != _canonical_sha({"context": expected_context, "domain": domain,
                                                       "train": value["train_mapping"], "held": value["held_mapping"]}):
            return False
        provenance = value["model_provenance"]
        if (provenance["axis_pivot"] < 0 or provenance["axis_pivot"] >= len(value["audit_payload"]["model"]["axis"])
                or provenance["eigengap"] <= provenance["eigengap_threshold"]
                or provenance["eigengap_threshold"] != float(contract["eigengap_multiplier"]) * torch.finfo(torch.float64).eps * provenance["operator_norm"]):
            return False
        for name, payload in value["audit_payload"]["model"].items():
            tensor = torch.tensor(payload, dtype=torch.float64)
            if provenance.get(name + "_sha256") != _sha_tensor(tensor) or not torch.isfinite(tensor).all():
                return False
        axis = torch.tensor(value["audit_payload"]["model"]["axis"], dtype=torch.float64)
        pivot = provenance["axis_pivot"]
        if abs(float(axis.norm()) - 1) > 1e-12 or axis[pivot] < 0 or pivot != int(axis.abs().argmax()):
            return False
        numeric = ("q_constant_orthogonality", "q_raw_z_orthogonality",
                   "location_basis_orthogonality_max", "radial_basis_orthogonality_max",
                   "block_basis_orthogonality_max", "normalized_residual_mean_max",
                   "normalized_residual_raw_z_correlation_max", "reconstruction_max_error",
                   "residual_energy_ratio", "radial_scale_min", "radial_scale_max",
                   "parallel_scale_min", "parallel_scale_max", "perpendicular_scale_min",
                   "perpendicular_scale_max", "held_boundary_clipped_fraction",
                   "shuffle_rms_over_original_sd", "shuffled_norm_p99_ratio",
                   "recipient_u_train_max_error", "recipient_u_held_max_error")
        if any(not math.isfinite(float(value[name])) for name in numeric):
            return False
        if (value["q_constant_orthogonality"] > contract["orthogonality_max"]
                or value["q_raw_z_orthogonality"] > contract["orthogonality_max"]
                or value["location_basis_orthogonality_max"] > contract["orthogonality_max"]
                or value["radial_basis_orthogonality_max"] > contract["orthogonality_max"]
                or value["block_basis_orthogonality_max"] > contract["orthogonality_max"]
                or value["normalized_residual_mean_max"] > contract["normalized_residual_mean_max"]
                or value["normalized_residual_raw_z_correlation_max"] > contract["normalized_residual_raw_z_correlation_max"]
                or value["reconstruction_max_error"] > contract["reconstruction_max_error"]
                or value["held_boundary_clipped_fraction"] > contract["held_boundary_clipped_fraction_max"]
                or value["residual_energy_ratio"] < contract["residual_energy_ratio_at_least"]
                or value["shuffle_rms_over_original_sd"] < contract["shuffle_rms_over_original_sd_at_least"]
                or value["shuffled_norm_p99_ratio"] > contract["shuffled_norm_p99_ratio_at_most"]
                or value["train_coverage"] != 1 or value["held_coverage"] != 1
                or value["train_self_rate"] != 0 or value["train_donor_marginal_exact"] is not True
                or value["held_effective_donors_per_object"] != 1
                or value["held_max_donor_load"] > value["held_load_bound"]
                or value["recipient_u_train_max_error"] > contract["recipient_u_max_error"]
                or value["recipient_u_held_max_error"] > contract["recipient_u_max_error"]
                or value["scale_clipped"] is not False):
            return False
        crossfit = value["crossfit"]
        if (crossfit["schema"] != CROSSFIT_SCHEMA
                or crossfit["field_kind"] != expected_field_kind
                or crossfit.get("raw_z_diagnostic_scope") != "one unified canonical raw-z per domain across all OOF objects"
                or crossfit.get("fold_fit_z_scope") != "fold-train raw-z is independently ECDF-ranked; held uses that fold train ECDF"
                or crossfit["residual_norm_raw_z_absolute_spearman"] > contract["crossfit_absolute_spearman_at_most"]
                or crossfit["residual_raw_z_distance_correlation"] > contract["crossfit_distance_correlation_at_most"]
                or len(crossfit["folds"]) != int(contract["crossfit_folds"])):
            return False
        payload = crossfit["audit_payload"]
        ids = payload["object_ids"]
        expected_assignment = {obj: fold for obj, fold in zip(
            ids, _balanced_folds(domain, ids, int(contract["crossfit_folds"]))) }
        if (ids != sorted(ids) or len(ids) != len(set(ids))
                or payload["fold_assignment"] != expected_assignment
                or crossfit["fold_assignment_sha256"] != _canonical_sha(expected_assignment)
                or crossfit["fold_counts"] != [list(expected_assignment.values()).count(fold)
                                               for fold in range(int(contract["crossfit_folds"]))]):
            return False
        standardized = torch.tensor(payload["standardized"], dtype=torch.float64)
        raw_z = torch.tensor(payload["raw_z"], dtype=torch.float64)
        rank_x = torch.tensor(payload["rank_x"], dtype=torch.float64)
        if (crossfit["standardized_sha256"] != _sha_tensor(standardized)
                or crossfit["raw_z_sha256"] != _sha_tensor(raw_z)
                or crossfit["rank_x_sha256"] != _sha_tensor(rank_x)):
            return False
        norm = standardized.norm(dim=-1)
        diagnostics = {
            "residual_norm_raw_z_absolute_spearman": _spearman(norm, raw_z),
            "residual_raw_z_distance_correlation": _distance_correlation(raw_z, standardized),
            "residual_norm_rank_x_absolute_spearman_extra": _spearman(norm, rank_x),
            "residual_rank_x_distance_correlation_extra": _distance_correlation(rank_x, standardized),
        }
        if any(abs(float(crossfit[name]) - float(score)) > 1e-12
               for name, score in diagnostics.items()):
            return False
        if any(fold["model_provenance"]["eigengap"] <= fold["model_provenance"]["eigengap_threshold"]
               for fold in crossfit["folds"]):
            return False
    # Cross-domain fold-local preprocessing must be one identical joint fit.
    ordered = sorted(expected_domains)
    if len(ordered) > 1:
        reference = {v["fold"]: v for v in receipt["domains"][ordered[0]]["crossfit"]["folds"]}
        common = ("joint_domain_fit_object_hash", "joint_domain_held_object_hash",
                  "preprocessor_sha256", "preprocessor_provenance", "raw_mechanical_dim",
                  "mechanical_kept_indices", "mechanical_keep_mask")
        for domain in ordered[1:]:
            candidate = {v["fold"]: v for v in receipt["domains"][domain]["crossfit"]["folds"]}
            if set(candidate) != set(reference):
                return False
            if any(candidate[fold][key] != reference[fold][key]
                   for fold in reference for key in common):
                return False
    return True


def receipt_passes(receipt: dict, expected: dict, contract: dict,
                   expected_context: str, expected_domains: set[str]) -> bool:
    """Fail closed against a separately computed receipt object graph."""
    try:
        return _receipt_passes_checked(receipt, expected, contract, expected_context, expected_domains)
    except (KeyError, TypeError, ValueError, IndexError, RuntimeError, ZeroDivisionError):
        return False


def preserve_recipient_displacement(shuffled_geometry: Tensor, recipient_mechanical: Tensor) -> Tensor:
    if (shuffled_geometry.shape[:-1] != recipient_mechanical.shape[:-1]
            or shuffled_geometry.shape[-1] + 1 != recipient_mechanical.shape[-1]):
        raise ValueError("mechanical geometry/displacement shape mismatch")
    result = torch.cat((shuffled_geometry, recipient_mechanical[..., -1:]), -1)
    if not torch.equal(result[..., -1], recipient_mechanical[..., -1]):
        raise RuntimeError("recipient displacement changed")
    return result


__all__ = ["CROSSFIT_SCHEMA", "SCHEMA", "canonical_object_scalar", "empirical_rank",
           "feature_crossfit_rza_acwt", "global_row_identity", "preserve_recipient_displacement",
           "receipt_passes", "rza_acwt_ablation"]
