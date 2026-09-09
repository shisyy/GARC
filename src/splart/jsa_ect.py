"""Joint-domain shared-axis, energy-conserving transport (JSA-ECT)."""

from __future__ import annotations

import copy
import hashlib
import json
import math

import torch
from torch import Tensor

from .conditional_residual import _distance_correlation, _spearman
from .rza_acwt import (canonical_object_scalar, empirical_rank,
                       global_row_identity)
from .smarc_source import fit_feature_preprocessor, transform


SCHEMA = "splart-jsa-ect-null/v1"
CROSSFIT_SCHEMA = "splart-jsa-ect-fold-local-crossfit/v1"


def _canonical_sha(value) -> str:
    return hashlib.sha256((json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()).hexdigest()


def _sha_tensor(value: Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().double().numpy().tobytes()).hexdigest()


def _sha_ids(values: list[str]) -> str:
    return hashlib.sha256(("\n".join(values) + "\n").encode()).hexdigest()


def _shares_container_identity(left, right) -> bool:
    if isinstance(left,(dict,list)) and left is right: return True
    if isinstance(left,dict) and isinstance(right,dict):
        return any(k in right and _shares_container_identity(v,right[k]) for k,v in left.items())
    if isinstance(left,list) and isinstance(right,list):
        return any(_shares_container_identity(a,b) for a,b in zip(left,right))
    return False


def _canonical_indices(rows: list[dict], indices: list[int]) -> list[int]:
    result = sorted(indices, key=lambda i: global_row_identity(rows[i]))
    identities = [global_row_identity(rows[i]) for i in result]
    if len(identities) != len(set(identities)):
        raise ValueError("global row identity collision")
    return result


def _positive_qr(design: Tensor, contract: dict) -> tuple[Tensor, Tensor, dict]:
    design = design.double()
    if design.ndim != 2 or design.shape[0] < design.shape[1] or not torch.isfinite(design).all():
        raise ValueError("invalid pooled QR design")
    q, r = torch.linalg.qr(design, mode="reduced")
    signs = torch.where(torch.diag(r) < 0, -torch.ones(r.shape[0], dtype=r.dtype), torch.ones(r.shape[0], dtype=r.dtype))
    q, r = q * signs[None], signs[:, None] * r
    singular = torch.linalg.svdvals(r)
    threshold = float(contract["rank_relative_tolerance"]) * float(singular.max())
    rank = int((singular > threshold).sum())
    condition = float(singular.max() / singular.min())
    if rank != design.shape[1] or not math.isfinite(condition) or condition > float(contract["condition_max"]):
        raise ValueError("pooled QR rank/condition gate failed")
    if torch.any(torch.diag(r) <= 0):
        raise RuntimeError("positive QR diagonal failed")
    return q, r, {"rank": rank, "condition": condition, "rank_threshold": threshold, "positive_diagonal": True}


def _solve(design: Tensor, target: Tensor, contract: dict) -> tuple[Tensor, dict]:
    q, r, receipt = _positive_qr(design, contract)
    return torch.linalg.solve_triangular(r, q.T @ target.double(), upper=True), receipt


def _domain_anchor(z: Tensor, x: Tensor, contract: dict) -> dict:
    z = z.double(); x = x.double(); centered = z - z.mean()
    q0 = 6 * ((x + 1) / 2).square() - 6 * ((x + 1) / 2) + 1
    design = torch.stack((torch.ones_like(z), centered), -1)
    projection, qr = _solve(design, q0[:, None], contract)
    residual = q0 - (design @ projection)[:, 0]
    infinity = residual.abs().max()
    if not torch.isfinite(infinity) or float(infinity) <= 1e-12:
        raise ValueError("degenerate rank-quadratic anchor")
    q = residual / infinity
    return {"z_mean": z.mean(), "q_projection": projection[:, 0], "q_infinity": infinity,
            "q": q, "qr": qr, "q_mean": float(q.mean().abs()),
            "q_raw_z_covariance": float((q * centered).mean().abs())}


def _anchor_predict(anchor: dict, z: Tensor, x: Tensor) -> Tensor:
    z = z.double(); x = x.double(); centered = z - anchor["z_mean"]
    q0 = 6 * ((x + 1) / 2).square() - 6 * ((x + 1) / 2) + 1
    return (q0 - torch.stack((torch.ones_like(z), centered), -1) @ anchor["q_projection"]) / anchor["q_infinity"]


def fit_shared_model(by_domain: dict[str, tuple[Tensor, Tensor, Tensor]], contract: dict) -> dict:
    """Fit one pooled fixed-effect mean and one shared rank-one energy axis."""
    domains = sorted(by_domain)
    if len(domains) < 2:
        raise ValueError("JSA-ECT requires at least two domains")
    width = {values[2].shape[1] for values in by_domain.values()}
    if len(width) != 1 or next(iter(width)) < 2:
        raise ValueError("domain field widths differ or are rank-one")
    anchors = {}; blocks = []; targets = []; row_domains = []
    columns = 2 * len(domains) + 1
    for domain_index, domain in enumerate(domains):
        z, x, means = (value.double() for value in by_domain[domain])
        if z.ndim != 1 or x.shape != z.shape or means.ndim != 2 or means.shape[0] != len(z):
            raise ValueError("invalid domain block")
        anchor = _domain_anchor(z, x, contract); anchors[domain] = anchor
        design = torch.zeros((len(z), columns), dtype=torch.float64)
        design[:, 2 * domain_index] = 1
        design[:, 2 * domain_index + 1] = z - anchor["z_mean"]
        design[:, -1] = anchor["q"]
        blocks.append(design); targets.append(means); row_domains.extend([domain] * len(z))
    design = torch.cat(blocks); means = torch.cat(targets)
    beta, mean_qr = _solve(design, means, contract)
    location = design @ beta; residual = means - location
    pooled_orthogonality = float((design.T @ residual / len(design)).abs().max())
    domain_slices = {}; start = 0
    for domain in domains:
        count = len(by_domain[domain][0]); domain_slices[domain] = slice(start, start + count); start += count
    covariances = {}
    for domain in domains:
        r = residual[domain_slices[domain]]; covariances[domain] = torch.einsum("ni,nj->ij", r, r) / len(r)
    q_all = torch.cat([anchors[d]["q"] for d in domains]); centered_outer = []
    for domain in domains:
        r = residual[domain_slices[domain]]
        outer = torch.einsum("ni,nj->nij", r, r)
        centered_outer.append(torch.einsum("n,nij->ij", anchors[domain]["q"], outer - covariances[domain]))
    h = sum(centered_outer) / q_all.square().sum()
    h = (h + h.T) * .5
    g = h @ h; g = (g + g.T) * .5
    eigenvalues, eigenvectors = torch.linalg.eigh(g)
    operator = float(eigenvalues.abs().max()); gap = float(eigenvalues[-1] - eigenvalues[-2])
    threshold = float(contract["eigengap_multiplier"]) * torch.finfo(torch.float64).eps * operator
    if not math.isfinite(gap) or gap <= threshold:
        raise ValueError("shared-axis eigengap gate failed")
    axis = eigenvectors[:, -1]; pivot = int(axis.abs().argmax())
    if axis[pivot] < 0: axis = -axis
    sigmas = {}; energy_floors={}; c_parts = []; energy_error = 0.
    for domain in domains:
        r = residual[domain_slices[domain]]; axial = r @ axis; bulk = r - axial[:, None] * axis
        parallel = axial.square().mean(); perpendicular = bulk.square().sum(-1).mean() / (r.shape[1] - 1)
        total=parallel+(r.shape[1]-1)*perpendicular
        centered_reference=(by_domain[domain][2].double()-by_domain[domain][2].double().mean(0)).square().sum(-1).mean()
        reference=max(float(total),float(centered_reference)); floor=float(contract["rank_relative_tolerance"])*reference
        if not torch.isfinite(parallel + perpendicular) or float(parallel) <= floor or float(perpendicular) <= floor:
            raise ValueError("unidentifiable domain energy intercept")
        sigmas[domain] = (parallel, perpendicular)
        energy_floors[domain]=(floor,reference)
        c_parts.append(axial.square() / parallel - bulk.square().sum(-1) / ((r.shape[1] - 1) * perpendicular))
    c_all = torch.cat(c_parts); denominator = torch.sqrt(q_all.square().sum() * c_all.square().sum())
    if not torch.isfinite(denominator) or float(denominator) <= 0:
        raise ValueError("degenerate theta denominator")
    theta = (q_all * c_all).sum() / denominator
    tolerance = float(contract["numeric_tolerance"])
    if not torch.isfinite(theta) or abs(float(theta)) > 1 + tolerance:
        raise ValueError("theta bound failed")
    first_standardized = {}; scales = {}
    for domain in domains:
        sl = domain_slices[domain]; r = residual[sl]; qd = anchors[domain]["q"]
        axial = r @ axis; bulk = r - axial[:, None] * axis; sp0, so0 = sigmas[domain]
        modulation = torch.exp(theta * qd)
        normalizer = (sp0 * modulation + (r.shape[1] - 1) * so0) / (sp0 + (r.shape[1] - 1) * so0)
        sp = torch.sqrt(sp0 * modulation / normalizer); so = torch.sqrt(so0 / normalizer)
        energy = sp.square() + (r.shape[1] - 1) * so.square()
        baseline = sp0 + (r.shape[1] - 1) * so0
        energy_error = max(energy_error, float((energy - baseline).abs().max()))
        bulk_standardized = bulk / so[:, None]
        bulk_standardized = bulk_standardized - (bulk_standardized @ axis)[:, None] * axis
        first_standardized[domain] = (axial / sp)[:, None] * axis + bulk_standardized
        scales[domain] = (sp, so)
    # One-shot post-scale FWL.  The first pass above is used only to freeze
    # axis/scales; the actually transported E is orthogonal after scaling.
    parallel_target=[]; perpendicular_target=[]
    for domain in domains:
        y=by_domain[domain][2].double(); sp,so=scales[domain]; y_parallel=y@axis; y_perp=y-y_parallel[:,None]*axis
        parallel_target.append((y_parallel/sp)[:,None]); perpendicular_target.append(y_perp/so[:,None])
    parallel_beta, parallel_qr=_solve(design,torch.cat(parallel_target),contract)
    perpendicular_beta, perpendicular_qr=_solve(design,torch.cat(perpendicular_target),contract)
    perpendicular_beta=perpendicular_beta-(perpendicular_beta@axis)[:,None]*axis
    parallel_location=(design@parallel_beta)[:,0]; perpendicular_location=design@perpendicular_beta
    standardized={}; reconstruction_error=0.
    for domain in domains:
        sl=domain_slices[domain]; y=by_domain[domain][2].double(); sp,so=scales[domain]
        y_parallel=y@axis; y_perp=y-y_parallel[:,None]*axis
        e_parallel=y_parallel/sp-parallel_location[sl]; e_perp=y_perp/so[:,None]-perpendicular_location[sl]
        e_perp=e_perp-(e_perp@axis)[:,None]*axis
        standardized[domain]=e_parallel[:,None]*axis+e_perp
        rebuilt=(sp*(parallel_location[sl]+e_parallel))[:,None]*axis+so[:,None]*(perpendicular_location[sl]+e_perp)
        reconstruction_error=max(reconstruction_error,float((rebuilt-y).abs().max()))
    field_scale = {d: by_domain[d][2].double().std(0, unbiased=False).clamp_min(1e-12) for d in domains}
    diagnostics = {}
    for domain in domains:
        r = residual[domain_slices[domain]]; e=standardized[domain]; z = by_domain[domain][0].double(); centered = z - z.mean()
        centered_means = by_domain[domain][2].double() - by_domain[domain][2].double().mean(0)
        e_scale=e.std(0,unbiased=False).clamp_min(1e-12)
        diagnostics[domain] = {"first_pass_normalized_residual_mean_max": float((r.mean(0).abs() / field_scale[domain]).max()),
            "first_pass_normalized_residual_raw_z_correlation_max": float(((centered[:, None] * r).mean(0).abs() /
                (centered.std(unbiased=False) * field_scale[domain]).clamp_min(1e-12)).max()),
            "normalized_residual_mean_max":float((e.mean(0).abs()/e_scale).max()),
            "normalized_residual_raw_z_correlation_max":float(((centered[:,None]*e).mean(0).abs()/(centered.std(unbiased=False)*e_scale).clamp_min(1e-12)).max()),
            "residual_energy_ratio": float(r.square().mean() / centered_means.square().mean().clamp_min(1e-12))}
    if energy_error > float(contract["energy_conservation_max_error"]):
        raise ValueError("energy conservation gate failed")
    return {"domains": domains, "anchors": anchors, "beta": beta, "mean_qr": mean_qr,
            "residual": residual, "location": location, "domain_slices": domain_slices,
            "h": h, "g": g, "axis": axis, "axis_pivot": pivot, "eigenvalues": eigenvalues,
            "eigengap": gap, "eigengap_threshold": threshold, "theta": theta, "covariances": covariances,
            "sigmas": sigmas,"energy_floors":energy_floors, "standardized": standardized, "first_standardized":first_standardized,"scales": scales,
            "parallel_beta":parallel_beta,"perpendicular_beta":perpendicular_beta,"parallel_qr":parallel_qr,"perpendicular_qr":perpendicular_qr,
            "parallel_location":parallel_location,"perpendicular_location":perpendicular_location,"postscale_reconstruction_max_error":reconstruction_error,
            "energy_conservation_max_error": energy_error, "pooled_design_orthogonality_max": pooled_orthogonality, "diagnostics": diagnostics}


def predict_shared(model: dict, domain: str, z: Tensor, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    if domain not in model["domains"]: raise ValueError("unknown held domain")
    q = _anchor_predict(model["anchors"][domain], z, x); index = model["domains"].index(domain)
    design = torch.zeros((len(z), 2 * len(model["domains"]) + 1), dtype=torch.float64)
    design[:, 2 * index] = 1; design[:, 2 * index + 1] = z.double() - model["anchors"][domain]["z_mean"]; design[:, -1] = q
    parallel_location=(design@model["parallel_beta"])[:,0]; perpendicular_location=design@model["perpendicular_beta"]; sp0, so0 = model["sigmas"][domain]
    modulation = torch.exp(model["theta"] * q); width = model["axis"].numel()
    normalizer = (sp0 * modulation + (width - 1) * so0) / (sp0 + (width - 1) * so0)
    return parallel_location,perpendicular_location,torch.sqrt(sp0 * modulation / normalizer), torch.sqrt(so0 / normalizer)


def reconstruct(parallel_location:Tensor,perpendicular_location:Tensor,parallel: Tensor, perpendicular: Tensor, standardized: Tensor, axis: Tensor) -> Tensor:
    axial = standardized @ axis; bulk = standardized - axial[:, None] * axis
    return (parallel*(parallel_location+axial))[:,None]*axis+perpendicular[:,None]*(perpendicular_location+bulk)


def model_provenance(model: dict) -> dict:
    anchors = {d: {"z_mean": float(model["anchors"][d]["z_mean"]), "q_projection_sha256": _sha_tensor(model["anchors"][d]["q_projection"]),
        "q_infinity": float(model["anchors"][d]["q_infinity"]), "q_mean_absolute": model["anchors"][d]["q_mean"],
        "q_raw_z_covariance_absolute": model["anchors"][d]["q_raw_z_covariance"], "q_projection_qr": model["anchors"][d]["qr"]} for d in model["domains"]}
    energies = {d: {"sigma_parallel_squared": float(model["sigmas"][d][0]), "sigma_perpendicular_squared": float(model["sigmas"][d][1]),
        "total_residual_energy":float(model["sigmas"][d][0]+(model["axis"].numel()-1)*model["sigmas"][d][1]),
        "relative_identifiability_reference_energy":model["energy_floors"][d][1],
        "relative_identifiability_floor":model["energy_floors"][d][0]} for d in model["domains"]}
    return {"beta_sha256": _sha_tensor(model["beta"]),"parallel_beta_sha256":_sha_tensor(model["parallel_beta"]),"perpendicular_beta_sha256":_sha_tensor(model["perpendicular_beta"]), "h_sha256": _sha_tensor(model["h"]),
        "g_sha256": _sha_tensor(model["g"]), "axis_sha256": _sha_tensor(model["axis"]),
        "axis_pivot": model["axis_pivot"], "theta": float(model["theta"]),
        "eigengap": model["eigengap"], "eigengap_threshold": model["eigengap_threshold"],
        "energy_conservation_max_error": model["energy_conservation_max_error"],
        "h_symmetry_max_error": float((model["h"] - model["h"].T).abs().max()),
        "g_symmetry_max_error": float((model["g"] - model["g"].T).abs().max()),
        "pooled_design_orthogonality_max": model["pooled_design_orthogonality_max"],
        "mean_qr": model["mean_qr"],"parallel_qr":model["parallel_qr"],"perpendicular_qr":model["perpendicular_qr"],"postscale_reconstruction_max_error":model["postscale_reconstruction_max_error"], "domain_anchor": anchors, "domain_anchor_sha256": {d: _canonical_sha(anchors[d]) for d in model["domains"]},
        "domain_energy_intercepts": energies, "domain_energy_sha256": {d: _canonical_sha(energies[d]) for d in model["domains"]}}


def _balanced_folds(domain: str, objects: list[str], folds: int) -> dict[str, int]:
    if len(objects) < 2 * folds: raise ValueError("two objects per fold required")
    order = sorted(objects, key=lambda obj: (hashlib.sha256(("splart-jsa-ect-fold-v1:" + domain + ":" + obj).encode()).hexdigest(), obj))
    return {obj: rank % folds for rank, obj in enumerate(order)}


def feature_crossfit_jsa_ect(rows: list[dict], train_indices: list[int], field_kind: str, contract: dict) -> dict[str, dict]:
    train_indices = _canonical_indices(rows, train_indices); domains = sorted({rows[i]["domain"] for i in train_indices}); folds = int(contract["crossfit_folds"])
    objects = {d: sorted({rows[i]["object_group_id"] for i in train_indices if rows[i]["domain"] == d}) for d in domains}
    assignments = {d: _balanced_folds(d, objects[d], folds) for d in domains}
    raw_sem = rows[train_indices[0]]["semantic"].numel(); raw_mech = rows[train_indices[0]]["mechanical"].numel() - 1
    output = {d: torch.full((len(objects[d]), raw_sem), float("nan"), dtype=torch.float64) if field_kind == "semantic" else torch.zeros((len(objects[d]), raw_mech), dtype=torch.float64) for d in domains}
    fold_receipts = {d: [] for d in domains}
    for fold in range(folds):
        fit = _canonical_indices(rows, [i for i in train_indices if assignments[rows[i]["domain"]][rows[i]["object_group_id"]] != fold])
        held = _canonical_indices(rows, [i for i in train_indices if assignments[rows[i]["domain"]][rows[i]["object_group_id"]] == fold])
        prep = fit_feature_preprocessor(rows, fit, 16); fm, fs, _ = transform(prep, rows, fit); hm, hs, _ = transform(prep, rows, held)
        fit_field, held_field = (fs, hs) if field_kind == "semantic" else (fm[:, :-1], hm[:, :-1])
        if field_kind == "mechanical":
            if prep.mechanical_keep.ndim != 1 or not bool(prep.mechanical_keep[-1]):
                raise ValueError("displacement must be the final kept mechanical coordinate")
            kept_all = torch.nonzero(prep.mechanical_keep).flatten().tolist()
            if kept_all[-1] != prep.mechanical_keep.numel() - 1:
                raise ValueError("displacement index is not last")
            kept = kept_all[:-1]
        else:
            kept = None
        blocks = {}; held_blocks = {}
        for domain in domains:
            fit_objects = [o for o in objects[domain] if assignments[domain][o] != fold]; held_objects = [o for o in objects[domain] if assignments[domain][o] == fold]
            fi = [k for k, i in enumerate(fit) if rows[i]["domain"] == domain]; hi = [k for k, i in enumerate(held) if rows[i]["domain"] == domain]
            means = lambda indices, positions, field, selected: torch.stack([field[[k for k in positions if rows[indices[k]]["object_group_id"] == obj]].mean(0) for obj in selected])
            fit_means = means(fit, fi, fit_field, fit_objects); held_means = means(held, hi, held_field, held_objects)
            dfit = [fit[k] for k in fi]; dheld = [held[k] for k in hi]
            fz = canonical_object_scalar(rows, dfit, field_kind, dfit); hz = canonical_object_scalar(rows, dheld, field_kind, dfit)
            z = torch.tensor([fz[o] for o in fit_objects], dtype=torch.float64); zh = torch.tensor([hz[o] for o in held_objects], dtype=torch.float64)
            x, xh, clipped = empirical_rank(z, zh); blocks[domain] = (z, x, fit_means); held_blocks[domain] = (held_objects, zh, xh, held_means, clipped)
        model = fit_shared_model(blocks, contract); provenance = model_provenance(model)
        prep_payload={"train_object_hash": prep.train_object_hash, "mechanical_mean": _sha_tensor(prep.mechanical_mean), "mechanical_scale": _sha_tensor(prep.mechanical_scale), "mechanical_keep":_sha_tensor(prep.mechanical_keep.to(torch.float64)),"mechanical_raw_dim":int(prep.mechanical_keep.numel()),"mechanical_kept_indices":torch.nonzero(prep.mechanical_keep).flatten().tolist(), "semantic_mean": _sha_tensor(prep.semantic_mean), "semantic_components": _sha_tensor(prep.semantic_components)}
        prep_hash = _canonical_sha(prep_payload)
        for domain in domains:
            held_objects, zh, xh, held_means, clipped = held_blocks[domain]; ploc, oloc, sp, so = predict_shared(model, domain, zh, xh)
            axis = model["axis"]; yparallel=held_means@axis; yperp=held_means-yparallel[:,None]*axis
            eparallel=yparallel/sp-ploc; eperp=yperp/so[:,None]-oloc; eperp=eperp-(eperp@axis)[:,None]*axis
            local=eparallel[:,None]*axis+eperp
            for j, obj in enumerate(held_objects):
                target = objects[domain].index(obj); output[domain][target] = local[j] @ prep.semantic_components if field_kind == "semantic" else output[domain][target].scatter(0, torch.tensor(kept), local[j])
            fold_receipts[domain].append({"fold": fold, "joint_model_sha256": _canonical_sha(provenance), "joint_model": provenance,
                "joint_preprocessor_sha256": prep_hash,"joint_preprocessor":prep_payload, "boundary_clipped_fraction": clipped,
                "domain_fit_object_hash": _sha_ids([o for o in objects[domain] if assignments[domain][o] != fold]),
                "domain_held_object_hash": _sha_ids(held_objects)})
    result = {}
    for domain in domains:
        zmap = canonical_object_scalar(rows, [i for i in train_indices if rows[i]["domain"] == domain], field_kind,
                                       [i for i in train_indices if rows[i]["domain"] == domain])
        z = torch.tensor([zmap[o] for o in objects[domain]], dtype=torch.float64); e = output[domain]
        if not torch.isfinite(e).all(): raise RuntimeError("OOF output uninitialized")
        result[domain] = {"schema": CROSSFIT_SCHEMA, "field_kind": field_kind, "folds": fold_receipts[domain],
            "fold_assignment": assignments[domain], "fold_counts": [list(assignments[domain].values()).count(f) for f in range(folds)],
            "standardized_sha256": _sha_tensor(e), "raw_z_sha256": _sha_tensor(z),
            "residual_norm_raw_z_absolute_spearman": _spearman(e.norm(dim=-1), z),
            "residual_raw_z_distance_correlation": _distance_correlation(z, e)}
    return result


def jsa_ect_ablation(rows: list[dict], train_indices: list[int], held_indices: list[int],
                     train_field: Tensor, held_field: Tensor, train_z: dict[str, float],
                     held_z: dict[str, float], contract: dict, context: str,
                     crossfit: dict[str, dict]) -> tuple[Tensor, Tensor, dict]:
    train_order = sorted(range(len(train_indices)), key=lambda k: global_row_identity(rows[train_indices[k]]))
    held_order = sorted(range(len(held_indices)), key=lambda k: global_row_identity(rows[held_indices[k]]))
    train_indices = [train_indices[k] for k in train_order]; held_indices = [held_indices[k] for k in held_order]
    train_field = train_field[train_order].double(); held_field = held_field[held_order].double()
    domains = sorted({rows[i]["domain"] for i in train_indices})
    if set(domains) != {rows[i]["domain"] for i in held_indices}: raise ValueError("domain mismatch")
    blocks = {}; held_blocks = {}; locations = {}
    for domain in domains:
        train_objects = sorted({rows[i]["object_group_id"] for i in train_indices if rows[i]["domain"] == domain})
        held_objects = sorted({rows[i]["object_group_id"] for i in held_indices if rows[i]["domain"] == domain})
        tl = {obj: [k for k, i in enumerate(train_indices) if rows[i]["domain"] == domain and rows[i]["object_group_id"] == obj] for obj in train_objects}
        hl = {obj: [k for k, i in enumerate(held_indices) if rows[i]["domain"] == domain and rows[i]["object_group_id"] == obj] for obj in held_objects}
        means = torch.stack([train_field[tl[o]].mean(0) for o in train_objects]); hmeans = torch.stack([held_field[hl[o]].mean(0) for o in held_objects])
        z = torch.tensor([train_z[o] for o in train_objects], dtype=torch.float64); hz = torch.tensor([held_z[o] for o in held_objects], dtype=torch.float64)
        x, hx, clipped = empirical_rank(z, hz); blocks[domain] = (z, x, means); held_blocks[domain] = (hz, hx, hmeans, clipped)
        locations[domain] = (train_objects, held_objects, tl, hl)
    model = fit_shared_model(blocks, contract); train_out = torch.full_like(train_field, float("nan")); held_out = torch.full_like(held_field, float("nan"))
    receipt = {"schema": SCHEMA, "context": context, "contract_sha256": _canonical_sha(contract), "shared_model": model_provenance(model), "domains": {}}
    for domain in domains:
        train_objects, held_objects, tl, hl = locations[domain]; z, x, means = blocks[domain]; hz, hx, hmeans, clipped = held_blocks[domain]
        ploc, oloc, sp, so = predict_shared(model, domain, z, x); hploc, holoc, hsp, hso = predict_shared(model, domain, hz, hx)
        standardized = model["standardized"][domain]; offsets = {o: train_field[tl[o]] - means[j] for j, o in enumerate(train_objects)}; hoffsets = {o: held_field[hl[o]] - hmeans[j] for j, o in enumerate(held_objects)}
        donors = {o: train_objects[(j + 1) % len(train_objects)] for j, o in enumerate(train_objects)}; hdonors = {o: train_objects[j % len(train_objects)] for j, o in enumerate(held_objects)}
        by_obj = {o: standardized[j] for j, o in enumerate(train_objects)}
        reconstruction = reconstruct(ploc,oloc,sp,so,standardized,model["axis"])
        for j, o in enumerate(train_objects): train_out[tl[o]] = reconstruct(ploc[j:j+1],oloc[j:j+1],sp[j:j+1],so[j:j+1],by_obj[donors[o]][None],model["axis"])[0] + offsets[o]
        for j, o in enumerate(held_objects): held_out[hl[o]] = reconstruct(hploc[j:j+1],holoc[j:j+1],hsp[j:j+1],hso[j:j+1],by_obj[hdonors[o]][None],model["axis"])[0] + hoffsets[o]
        train_u = max(float(((train_out[tl[o]] - train_out[tl[o]].mean(0)) - offsets[o]).abs().max()) for o in train_objects)
        held_u = max(float(((held_out[hl[o]] - held_out[hl[o]].mean(0)) - hoffsets[o]).abs().max()) for o in held_objects)
        original = torch.cat([train_field[tl[o]] for o in train_objects]); shuffled = torch.cat([train_out[tl[o]] for o in train_objects]); hshuffled = torch.cat([held_out[hl[o]] for o in held_objects])
        original_sd = original.std(unbiased=False).clamp_min(1e-12); original_p99 = torch.quantile(original.norm(dim=-1), .99).clamp_min(1e-12)
        receipt["domains"][domain] = {**model["diagnostics"][domain], "reconstruction_max_error": float((reconstruction - means).abs().max()),
            "recipient_u_max_error": max(train_u, held_u), "held_boundary_clipped_fraction": clipped,
            "shuffle_rms_over_original_sd": float((shuffled - original).square().mean().sqrt() / original_sd),
            "shuffled_norm_p99_ratio": float(max(torch.quantile(shuffled.norm(dim=-1), .99), torch.quantile(hshuffled.norm(dim=-1), .99)) / original_p99),
            "train_mapping": donors, "held_mapping": hdonors, "train_coverage": 1., "held_coverage": 1., "train_self_rate": 0.,
            "train_donor_marginal_exact": sorted(donors.values()) == train_objects, "held_effective_donors_per_object": len(set(hdonors.values())) / len(held_objects),
            "crossfit": copy.deepcopy(crossfit[domain]), "shared_axis_sha256": _sha_tensor(model["axis"])}
    if not torch.isfinite(train_out).all() or not torch.isfinite(held_out).all(): raise RuntimeError("transport output uninitialized")
    return train_out, held_out, receipt


def preserve_recipient_displacement(geometry: Tensor, mechanical: Tensor) -> Tensor:
    if geometry.shape[:-1] != mechanical.shape[:-1] or geometry.shape[-1] + 1 != mechanical.shape[-1]: raise ValueError("mechanical shape mismatch")
    result = torch.cat((geometry, mechanical[..., -1:]), -1)
    if not torch.equal(result[..., -1], mechanical[..., -1]): raise RuntimeError("recipient displacement changed")
    return result


def global_receipt_passes(candidate: dict, expected: dict, expected_sha256: str,
                           contract: dict, context: str, domains: set[str]) -> bool:
    try:
        if candidate is expected or _shares_container_identity(candidate,expected) or not isinstance(expected_sha256, str) or len(expected_sha256) != 64: return False
        if _canonical_sha(expected) != expected_sha256 or _canonical_sha(candidate) != expected_sha256: return False
        if set(candidate) != {"schema", "context", "contract_sha256", "shared_model", "domains"}: return False
        if candidate["schema"] != SCHEMA or candidate["context"] != context or candidate["contract_sha256"] != _canonical_sha(contract) or set(candidate["domains"]) != domains: return False
        shared = candidate["shared_model"]
        shared_keys = {"beta_sha256","parallel_beta_sha256","perpendicular_beta_sha256","h_sha256","g_sha256","axis_sha256","axis_pivot","theta","eigengap","eigengap_threshold",
                       "energy_conservation_max_error","h_symmetry_max_error","g_symmetry_max_error","pooled_design_orthogonality_max","mean_qr",
                       "parallel_qr","perpendicular_qr","postscale_reconstruction_max_error","domain_anchor","domain_anchor_sha256","domain_energy_intercepts","domain_energy_sha256"}
        if set(shared) != shared_keys or set(shared["domain_anchor_sha256"]) != domains or set(shared["domain_energy_sha256"]) != domains: return False
        scalar_fields=("theta","eigengap","eigengap_threshold","energy_conservation_max_error","h_symmetry_max_error","g_symmetry_max_error","pooled_design_orthogonality_max","postscale_reconstruction_max_error")
        if not all(math.isfinite(float(shared[k])) for k in scalar_fields): return False
        for qr_name in ("mean_qr","parallel_qr","perpendicular_qr"):
            qr=shared[qr_name]
            if set(qr)!={"rank","condition","rank_threshold","positive_diagonal"} or qr["positive_diagonal"] is not True or qr["rank"]<=0 or not math.isfinite(float(qr["condition"])): return False
        for domain in domains:
            if shared["domain_anchor_sha256"][domain] != _canonical_sha(shared["domain_anchor"][domain]) or shared["domain_energy_sha256"][domain] != _canonical_sha(shared["domain_energy_intercepts"][domain]): return False
            anchor=shared["domain_anchor"][domain]; energy=shared["domain_energy_intercepts"][domain]
            if not all(math.isfinite(float(anchor[k])) for k in ("z_mean","q_infinity","q_mean_absolute","q_raw_z_covariance_absolute")): return False
            if anchor["q_infinity"] <= 0 or anchor["q_mean_absolute"] > contract["orthogonality_max"] or anchor["q_raw_z_covariance_absolute"] > contract["orthogonality_max"]: return False
            if not all(math.isfinite(float(energy[k])) for k in ("sigma_parallel_squared","sigma_perpendicular_squared","total_residual_energy","relative_identifiability_reference_energy","relative_identifiability_floor")): return False
            if energy["sigma_parallel_squared"] <= energy["relative_identifiability_floor"] or energy["sigma_perpendicular_squared"] <= energy["relative_identifiability_floor"]: return False
        if abs(shared["theta"]) > 1 + contract["numeric_tolerance"] or shared["energy_conservation_max_error"] > contract["energy_conservation_max_error"]: return False
        if shared["h_symmetry_max_error"] > contract["symmetry_max_error"] or shared["g_symmetry_max_error"] > contract["symmetry_max_error"]: return False
        if shared["pooled_design_orthogonality_max"] > contract["orthogonality_max"]: return False
        if shared["postscale_reconstruction_max_error"]>contract["reconstruction_max_error"]:return False
        domain_keys = {"first_pass_normalized_residual_mean_max","first_pass_normalized_residual_raw_z_correlation_max",
                       "normalized_residual_mean_max","normalized_residual_raw_z_correlation_max","residual_energy_ratio","reconstruction_max_error",
                       "recipient_u_max_error","held_boundary_clipped_fraction","shuffle_rms_over_original_sd","shuffled_norm_p99_ratio",
                       "train_mapping","held_mapping","train_coverage","held_coverage","train_self_rate","train_donor_marginal_exact",
                       "held_effective_donors_per_object","crossfit","shared_axis_sha256"}
        reference = candidate["domains"][sorted(domains)[0]]["crossfit"]["folds"]
        for domain, value in candidate["domains"].items():
            if set(value) != domain_keys or value["shared_axis_sha256"] != shared["axis_sha256"]: return False
            train_ids=sorted(value["train_mapping"]); held_ids=sorted(value["held_mapping"])
            if value["train_mapping"] != {o: train_ids[(i+1)%len(train_ids)] for i,o in enumerate(train_ids)}: return False
            if value["held_mapping"] != {o: train_ids[i%len(train_ids)] for i,o in enumerate(held_ids)}: return False
            cf = value["crossfit"]
            if sorted(f["fold"] for f in cf["folds"]) != list(range(contract["crossfit_folds"])): return False
            if any(f["joint_model_sha256"] != reference[f["fold"]]["joint_model_sha256"] or
                   f["joint_preprocessor_sha256"] != reference[f["fold"]]["joint_preprocessor_sha256"] for f in cf["folds"]): return False
            for fold in cf["folds"]:
                joint=fold["joint_model"]
                if fold["joint_model_sha256"] != _canonical_sha(joint) or fold["joint_preprocessor_sha256"]!=_canonical_sha(fold["joint_preprocessor"]) or abs(joint["theta"]) > 1+contract["numeric_tolerance"]: return False
                pp=fold["joint_preprocessor"]
                if pp["mechanical_kept_indices"][-1]!=pp["mechanical_raw_dim"]-1: return False
                if joint["energy_conservation_max_error"] > contract["energy_conservation_max_error"] or joint["h_symmetry_max_error"] > contract["symmetry_max_error"] or joint["g_symmetry_max_error"] > contract["symmetry_max_error"]: return False
                if joint["postscale_reconstruction_max_error"]>contract["reconstruction_max_error"]: return False
                for energy in joint["domain_energy_intercepts"].values():
                    if energy["sigma_parallel_squared"]<=energy["relative_identifiability_floor"] or energy["sigma_perpendicular_squared"]<=energy["relative_identifiability_floor"]: return False
        return True
    except (KeyError, TypeError, ValueError, IndexError, RuntimeError, ZeroDivisionError):
        return False


def domain_receipt_passes(candidate: dict, expected: dict, expected_sha256: str, contract: dict,
                          context: str, domains: set[str], domain: str) -> bool:
    if not global_receipt_passes(candidate, expected, expected_sha256, contract, context, domains) or domain not in domains:
        return False
    try:
        value = candidate["domains"][domain]; cf = value["crossfit"]
        numeric = [value[k] for k in ("first_pass_normalized_residual_mean_max", "first_pass_normalized_residual_raw_z_correlation_max",
                   "normalized_residual_mean_max", "normalized_residual_raw_z_correlation_max", "residual_energy_ratio",
                   "reconstruction_max_error", "recipient_u_max_error", "held_boundary_clipped_fraction",
                   "shuffle_rms_over_original_sd", "shuffled_norm_p99_ratio")]
        if not all(math.isfinite(float(x)) for x in numeric): return False
        if value["first_pass_normalized_residual_mean_max"] > contract["normalized_residual_mean_max"] or value["first_pass_normalized_residual_raw_z_correlation_max"] > contract["normalized_residual_raw_z_correlation_max"]: return False
        if value["normalized_residual_mean_max"] > contract["normalized_residual_mean_max"] or value["normalized_residual_raw_z_correlation_max"] > contract["normalized_residual_raw_z_correlation_max"]: return False
        if value["reconstruction_max_error"] > contract["reconstruction_max_error"] or value["recipient_u_max_error"] > contract["recipient_u_max_error"]: return False
        if value["held_boundary_clipped_fraction"] > contract["held_boundary_clipped_fraction_max"] or value["residual_energy_ratio"] < contract["residual_energy_ratio_at_least"]: return False
        if value["shuffle_rms_over_original_sd"] < contract["shuffle_rms_over_original_sd_at_least"] or value["shuffled_norm_p99_ratio"] > contract["shuffled_norm_p99_ratio_at_most"]: return False
        if value["train_coverage"] != 1 or value["held_coverage"] != 1 or value["train_self_rate"] != 0 or value["train_donor_marginal_exact"] is not True or value["held_effective_donors_per_object"] != 1: return False
        if cf["residual_norm_raw_z_absolute_spearman"] > contract["crossfit_absolute_spearman_at_most"] or cf["residual_raw_z_distance_correlation"] > contract["crossfit_distance_correlation_at_most"]: return False
        return True
    except (KeyError, TypeError, ValueError, IndexError, RuntimeError, ZeroDivisionError):
        return False


def receipt_passes(candidate: dict, expected: dict, expected_sha256: str, contract: dict,
                   context: str, domains: set[str]) -> bool:
    return global_receipt_passes(candidate, expected, expected_sha256, contract, context, domains) and all(
        domain_receipt_passes(candidate, expected, expected_sha256, contract, context, domains, domain) for domain in domains)


__all__ = ["SCHEMA", "canonical_object_scalar", "feature_crossfit_jsa_ect", "fit_shared_model", "jsa_ect_ablation",
           "global_row_identity", "model_provenance", "predict_shared", "preserve_recipient_displacement",
           "domain_receipt_passes", "global_receipt_passes", "receipt_passes", "reconstruct"]
