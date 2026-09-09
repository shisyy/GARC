"""Pooled Kernel Sliced Rosenblatt Transport (PKSRT), node 8.13.

This module is deliberately label- and score-free.  Every fit is object-local,
float64, deterministic, and fail-closed.  Exact pooled marginal ties are handled
by the frozen grouped-weight ECDF erratum; no jitter is permitted.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math

import torch
from torch import Tensor

from .conditional_residual import _distance_correlation, _spearman
from .rza_acwt import canonical_object_scalar, empirical_rank, global_row_identity
from .smarc_source import fit_feature_preprocessor, transform


SCHEMA = "splart-pksrt-null/v1"
CROSSFIT_SCHEMA = "splart-pksrt-fold-local-crossfit/v1"


def _canonical_sha(value) -> str:
    return hashlib.sha256((json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()).hexdigest()


def _sha_tensor(value: Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().double().numpy().tobytes()).hexdigest()


def _sha_ids(values: list[str]) -> str:
    return hashlib.sha256(("\n".join(values) + "\n").encode()).hexdigest()


def _container_ids(value) -> set[int]:
    if isinstance(value, dict):
        return {id(value)}.union(*(_container_ids(v) for v in value.values()), set())
    if isinstance(value, (list, set)):
        return {id(value)}.union(*(_container_ids(v) for v in value), set())
    return set()


def _shares_container_identity(left, right) -> bool:
    return bool(_container_ids(left) & _container_ids(right))


def _canonical_indices(rows: list[dict], indices: list[int]) -> list[int]:
    result = sorted(indices, key=lambda i: global_row_identity(rows[i]))
    identities = [global_row_identity(rows[i]) for i in result]
    if len(identities) != len(set(identities)):
        raise ValueError("global row identity collision")
    return result


def _positive_qr(design: Tensor, target: Tensor, contract: dict) -> tuple[Tensor, dict]:
    design, target = design.double(), target.double()
    if design.ndim != 2 or target.shape[0] != design.shape[0] or design.shape[0] < design.shape[1]:
        raise ValueError("invalid QR shapes")
    if not torch.isfinite(design).all() or not torch.isfinite(target).all():
        raise ValueError("nonfinite QR input")
    q, r = torch.linalg.qr(design, mode="reduced")
    signs = torch.where(torch.diag(r) < 0, -torch.ones(r.shape[0], dtype=r.dtype), torch.ones(r.shape[0], dtype=r.dtype))
    q, r = q * signs[None], signs[:, None] * r
    singular = torch.linalg.svdvals(r)
    threshold = float(contract["rank_relative_tolerance"]) * float(singular.max())
    rank = int((singular > threshold).sum())
    condition = float(singular.max() / singular.min())
    if rank != design.shape[1] or not math.isfinite(condition) or condition > float(contract["condition_max"]):
        raise ValueError("QR rank/condition gate failed")
    if torch.any(torch.diag(r) <= 0):
        raise RuntimeError("positive QR diagonal failed")
    beta = torch.linalg.solve_triangular(r, q.T @ target, upper=True)
    return beta, {"rank": rank, "condition": condition, "rank_threshold": threshold, "positive_diagonal": True}


def _logit(p: Tensor) -> Tensor:
    if torch.any((p <= 0) | (p >= 1)):
        raise ValueError("logit probability outside open unit interval")
    return torch.log(p) - torch.log1p(-p)


def _linear_bijection(x: Tensor, y: Tensor) -> dict:
    x, y = x.double(), y.double()
    if x.ndim != 1 or y.shape != x.shape or len(x) < 2 or not torch.isfinite(x).all() or not torch.isfinite(y).all():
        raise ValueError("invalid bijection knots")
    dx, dy = x[1:] - x[:-1], y[1:] - y[:-1]
    if torch.any(dx <= 0) or torch.any(dy <= 0):
        raise ValueError("bijection knots must be strictly increasing")
    slopes = dy / dx
    if torch.any(slopes <= 0) or not torch.isfinite(slopes).all():
        raise ValueError("nonpositive bijection slope")
    return {"x": x.clone(), "y": y.clone(), "slopes": slopes.clone()}


def _linear_forward(model: dict, value: Tensor) -> Tensor:
    value = value.double(); x, y, slopes = model["x"], model["y"], model["slopes"]
    index = torch.searchsorted(x, value, right=True) - 1
    index = index.clamp(0, len(x) - 2)
    return y[index] + slopes[index] * (value - x[index])


def _linear_inverse(model: dict, value: Tensor) -> Tensor:
    inverse = {"x": model["y"], "y": model["x"], "slopes": 1 / model["slopes"]}
    return _linear_forward(inverse, value)


def fixed_slices(width: int) -> tuple[Tensor, dict]:
    if width < 1:
        raise ValueError("positive width required")
    eye = torch.eye(width, dtype=torch.float64)
    n = torch.arange(width, dtype=torch.float64)
    dct = torch.stack([
        (torch.ones(width, dtype=torch.float64) / math.sqrt(width)) if k == 0
        else math.sqrt(2 / width) * torch.cos(math.pi * (n + .5) * k / width)
        for k in range(width)
    ])
    rows = []
    for row in torch.cat((eye, dct)):
        pivot = int(row.abs().argmax())
        rows.append(-row if row[pivot] < 0 else row)
    slices = torch.stack(rows)
    orth = max(float((eye @ eye.T - torch.eye(width)).abs().max()),
               float((slices[width:] @ slices[width:].T - torch.eye(width)).abs().max()))
    receipt = {"width": width, "count": 2 * width, "sequence": "coordinate_then_dct2",
               "basis_sha256": _sha_tensor(slices), "orthonormal_max_error": orth,
               "sign_pivots": [int(v.abs().argmax()) for v in slices]}
    return slices, receipt


def fit_domain_marginal(t: Tensor) -> dict:
    t = t.double()
    if t.ndim != 1 or len(t) < 2 or not torch.isfinite(t).all():
        raise ValueError("invalid domain marginal")
    x = torch.sort(t).values
    if torch.any(x[1:] <= x[:-1]):
        raise ValueError("domain marginal tie or nonpositive gap")
    n = len(x)
    y = _logit((torch.arange(n, dtype=torch.float64) + .5) / n)
    return _linear_bijection(x, y)


def _bandwidth(count: int) -> float:
    if count < 2:
        raise ValueError("at least two pooled objects required")
    return (1 / math.sqrt(3)) * (4 / (3 * count)) ** (1 / 5)


def fit_conditional_pool(records: list[tuple[float, float, str]]) -> dict:
    """Canonical exact-a grouping for the pooled conditional ECDF."""
    if len(records) < 2:
        raise ValueError("insufficient conditional pool")
    ordered = sorted(records, key=lambda item: (item[0], item[2]))
    keys = [str(item[2]) for item in ordered]
    if len(keys) != len(set(keys)):
        raise ValueError("pooled object key collision")
    a = torch.tensor([item[0] for item in ordered], dtype=torch.float64)
    u = torch.tensor([item[1] for item in ordered], dtype=torch.float64)
    if not torch.isfinite(a).all() or not torch.isfinite(u).all():
        raise ValueError("nonfinite conditional pool")
    groups: list[list[int]] = []
    for index in range(len(a)):
        if not groups or a[index] != a[groups[-1][0]]:
            groups.append([index])
        else:
            groups[-1].append(index)
    if len(groups) < 2:
        raise ValueError("conditional pool needs two unique a knots")
    grouping = [[keys[i] for i in group] for group in groups]
    return {"a": a, "u": u, "keys": keys, "groups": groups, "bandwidth": _bandwidth(len(records)),
            "raw_count": len(records), "unique_count": len(groups),
            "tie_group_count": sum(len(g) > 1 for g in groups),
            "max_multiplicity": max(len(g) for g in groups), "grouping_sha256": _canonical_sha(grouping)}


def conditional_bijection(pool: dict, query_u: Tensor | float, minimum_effective_sample_size: float = 2.) -> tuple[dict, float]:
    query = torch.as_tensor(query_u, dtype=torch.float64)
    if query.ndim != 0 or not torch.isfinite(query):
        raise ValueError("query u must be finite scalar")
    h = float(pool["bandwidth"])
    weights = torch.exp(-((query - pool["u"]) ** 2) / (2 * h * h))
    sumw = weights.sum(); neff = sumw.square() / weights.square().sum()
    if not torch.isfinite(neff) or float(neff) < float(minimum_effective_sample_size):
        raise ValueError("conditional effective sample size gate failed")
    # Each exact-a group is reduced in canonical (domain, object) key order.
    grouped = torch.stack([torch.stack([weights[i] for i in group]).sum() for group in pool["groups"]])
    if torch.any(grouped <= 0):
        raise ValueError("conditional group underflow")
    before = torch.cat((torch.zeros(1, dtype=torch.float64), torch.cumsum(grouped, 0)[:-1]))
    probabilities = (.5 + before + .5 * grouped) / (1 + sumw)
    eta = _logit(probabilities)
    unique_a = torch.stack([pool["a"][group[0]] for group in pool["groups"]])
    model = _linear_bijection(unique_a, eta)
    return model, float(neff)


def _anchor_design(z: Tensor, z_mean: Tensor) -> Tensor:
    return torch.stack((torch.ones_like(z, dtype=torch.float64), z.double() - z_mean), -1)


def _fit_first_stage(z: Tensor, means: Tensor, contract: dict) -> dict:
    z, means = z.double(), means.double()
    if z.ndim != 1 or means.ndim != 2 or len(z) != len(means):
        raise ValueError("invalid first-stage block")
    x, _, _ = empirical_rank(z)
    u = (x + 1) / 2
    z_mean = z.mean(); design = _anchor_design(z, z_mean)
    beta, qr = _positive_qr(design, means, contract)
    anchor = design @ beta; residual = means - anchor
    scale_squared = residual.square().mean(); centered_energy = (means - means.mean(0)).square().mean()
    floor = float(contract["rank_relative_tolerance"]) * float(centered_energy)
    if not torch.isfinite(scale_squared) or float(scale_squared) <= floor:
        raise ValueError("unidentifiable first-stage residual scale")
    scale = torch.sqrt(scale_squared)
    return {"z": z.clone(), "x": x, "u": u, "z_mean": z_mean, "beta": beta, "qr": qr,
            "anchor": anchor, "residual": residual / scale, "scale": scale,
            "scale_squared": float(scale_squared), "centered_energy": float(centered_energy), "identifiability_floor": floor}


def _first_stage_predict(stage: dict, z: Tensor, x: Tensor, means: Tensor) -> Tensor:
    return (means.double() - _anchor_design(z.double(), stage["z_mean"]) @ stage["beta"]) / stage["scale"]


def fit_model(by_domain: dict[str, tuple[Tensor, Tensor]], contract: dict,
              object_ids: dict[str, list[str]] | None = None) -> dict:
    domains = sorted(by_domain)
    if len(domains) < 2:
        raise ValueError("PKSRT requires at least two domains")
    widths = {values[1].shape[1] for values in by_domain.values()}
    if len(widths) != 1:
        raise ValueError("domain width mismatch")
    width = next(iter(widths)); slices, slice_receipt = fixed_slices(width)
    stages = {}; current = {}; ids = {}
    for domain in domains:
        z, means = by_domain[domain]
        stages[domain] = _fit_first_stage(z, means, contract)
        current[domain] = stages[domain]["residual"].clone()
        ids[domain] = list(object_ids[domain]) if object_ids is not None else [f"{domain}:{i:08d}" for i in range(len(z))]
        if len(ids[domain]) != len(z) or len(ids[domain]) != len(set(ids[domain])):
            raise ValueError("invalid object identities")
    layers = []; minimum = float(contract["minimum_effective_sample_size"])
    for layer_index, vector in enumerate(slices):
        marginals = {}; records = []; preupdate = {}
        for domain in domains:
            t = current[domain] @ vector
            preupdate[domain] = t.clone()
            marginal = fit_domain_marginal(t); marginals[domain] = marginal
            a = _linear_forward(marginal, t)
            records.extend((float(a[i]), float(stages[domain]["u"][i]), f"{domain}:{ids[domain][i]}") for i in range(len(t)))
        pool = fit_conditional_pool(records)
        min_neff = math.inf; min_conditional_slope = math.inf; query_receipts = []
        for domain in domains:
            t = current[domain] @ vector; a = _linear_forward(marginals[domain], t); eta_values = []
            for i in range(len(t)):
                conditional, neff = conditional_bijection(pool, stages[domain]["u"][i], minimum)
                eta_values.append(_linear_forward(conditional, a[i:i + 1])[0]); min_neff = min(min_neff, neff)
                min_conditional_slope = min(min_conditional_slope, float(conditional["slopes"].min()))
                query_receipts.append({"key": f"{domain}:{ids[domain][i]}", "u": float(stages[domain]["u"][i]),
                    "effective_sample_size": neff, "conditional_x_sha256": _sha_tensor(conditional["x"]),
                    "conditional_y_sha256": _sha_tensor(conditional["y"]),
                    "minimum_slope": float(conditional["slopes"].min())})
            eta = torch.stack(eta_values)
            current[domain] = current[domain] + (eta - t)[:, None] * vector[None]
        layers.append({"index": layer_index, "vector": vector.clone(), "marginals": marginals, "pool": pool,
                       "preupdate": preupdate, "query_receipts": query_receipts,
                       "minimum_observed_effective_sample_size": min_neff,
                       "minimum_observed_conditional_slope": min_conditional_slope})
    finals = {}; standardized = {}; diagnostics = {}; inverse_error = 0.
    for domain in domains:
        stage = stages[domain]; design = _anchor_design(stage["z"], stage["z_mean"])
        beta, qr = _positive_qr(design, current[domain], contract)
        e = current[domain] - design @ beta; standardized[domain] = e
        e_scale = e.std(0, unbiased=False).clamp_min(torch.finfo(torch.float64).tiny)
        centered_z = stage["z"] - stage["z_mean"]
        mean_max = float((e.mean(0).abs() / e_scale).max())
        z_max = float(((centered_z[:, None] * e).mean(0).abs() /
                       (centered_z.std(unbiased=False) * e_scale).clamp_min(torch.finfo(torch.float64).tiny)).max())
        finals[domain] = {"beta": beta, "qr": qr}
        diagnostics[domain] = {"normalized_residual_mean_max": mean_max,
            "normalized_residual_raw_z_correlation_max": z_max,
            # Both numerator and denominator are object-by-coordinate mean-square.
            # Using the standardized residual here would make the numerator one.
            "residual_energy_ratio": stage["scale_squared"] / stage["centered_energy"]}
    model = {"domains": domains, "width": width, "slices": slices, "slice_receipt": slice_receipt,
             "stages": stages, "object_ids": ids, "input_means": {d: by_domain[d][1].double().clone() for d in domains},
             "layers": layers, "finals": finals, "standardized": standardized,
             "diagnostics": diagnostics, "minimum_effective_sample_size": minimum}
    for domain in domains:
        recovered = reconstruct_means(model, domain, stages[domain]["z"], stages[domain]["x"], standardized[domain])
        inverse_error = max(inverse_error, float((recovered - by_domain[domain][1].double()).abs().max()))
    model["inverse_max_error"] = inverse_error
    return model


def forward(model: dict, domain: str, z: Tensor, x: Tensor, means: Tensor) -> Tensor:
    stage = model["stages"][domain]; current = _first_stage_predict(stage, z, x, means)
    u = (x.double() + 1) / 2
    for layer in model["layers"]:
        vector = layer["vector"]; t = current @ vector
        a = _linear_forward(layer["marginals"][domain], t); eta_values = []
        for i in range(len(t)):
            conditional, _ = conditional_bijection(layer["pool"], u[i], model["minimum_effective_sample_size"])
            eta_values.append(_linear_forward(conditional, a[i:i + 1])[0])
        eta = torch.stack(eta_values); current = current + (eta - t)[:, None] * vector[None]
    design = _anchor_design(z.double(), stage["z_mean"])
    return current - design @ model["finals"][domain]["beta"]


def reconstruct_means(model: dict, domain: str, z: Tensor, x: Tensor, standardized: Tensor) -> Tensor:
    stage = model["stages"][domain]; design = _anchor_design(z.double(), stage["z_mean"])
    current = standardized.double() + design @ model["finals"][domain]["beta"]
    u = (x.double() + 1) / 2
    for layer in reversed(model["layers"]):
        vector = layer["vector"]; eta = current @ vector; t_values = []
        for i in range(len(eta)):
            conditional, _ = conditional_bijection(layer["pool"], u[i], model["minimum_effective_sample_size"])
            a = _linear_inverse(conditional, eta[i:i + 1])
            t_values.append(_linear_inverse(layer["marginals"][domain], a)[0])
        t = torch.stack(t_values); current = current + (t - eta)[:, None] * vector[None]
    return current * stage["scale"] + design @ stage["beta"]


def _bijection_provenance(model: dict) -> dict:
    return {"x": model["x"].tolist(), "y": model["y"].tolist(), "slopes": model["slopes"].tolist(),
            "x_sha256": _sha_tensor(model["x"]), "y_sha256": _sha_tensor(model["y"])}


def model_provenance(model: dict) -> dict:
    stages = {}; finals = {}
    for domain in model["domains"]:
        stage = model["stages"][domain]
        stages[domain] = {"train_z": stage["z"].tolist(), "train_u": stage["u"].tolist(),
            "object_ids": list(model["object_ids"][domain]), "object_ids_sha256": _sha_ids(model["object_ids"][domain]),
            "train_means": model["input_means"][domain].tolist(), "train_means_sha256": _sha_tensor(model["input_means"][domain]),
            "z_mean": float(stage["z_mean"]), "beta": stage["beta"].tolist(), "beta_sha256": _sha_tensor(stage["beta"]), "qr": stage["qr"],
            "scale": float(stage["scale"]), "scale_squared": stage["scale_squared"],
            "centered_energy": stage["centered_energy"], "identifiability_floor": stage["identifiability_floor"]}
        finals[domain] = {"beta": model["finals"][domain]["beta"].tolist(), "beta_sha256": _sha_tensor(model["finals"][domain]["beta"]),
            "standardized": model["standardized"][domain].tolist(), "standardized_sha256": _sha_tensor(model["standardized"][domain]),
            "qr": model["finals"][domain]["qr"]}
    layers = []
    for layer in model["layers"]:
        pool = layer["pool"]
        layers.append({"index": layer["index"], "vector": layer["vector"].tolist(),
            "vector_sha256": _sha_tensor(layer["vector"]),
            "domain_marginals": {d: _bijection_provenance(layer["marginals"][d]) for d in model["domains"]},
            "domain_preupdate_t": {d: {"values": layer["preupdate"][d].tolist(), "sha256": _sha_tensor(layer["preupdate"][d])} for d in model["domains"]},
            "conditional_pool": {"a": pool["a"].tolist(), "u": pool["u"].tolist(), "keys": pool["keys"],
                "groups": pool["groups"], "bandwidth": pool["bandwidth"], "raw_count": pool["raw_count"],
                "unique_count": pool["unique_count"], "tie_group_count": pool["tie_group_count"],
                "max_multiplicity": pool["max_multiplicity"], "grouping_sha256": pool["grouping_sha256"]},
            "query_receipts": copy.deepcopy(layer["query_receipts"]),
            "minimum_observed_effective_sample_size": layer["minimum_observed_effective_sample_size"],
            "minimum_observed_conditional_slope": layer["minimum_observed_conditional_slope"]})
    return {"width": model["width"], "slice_receipt": model["slice_receipt"], "stages": stages,
            "layers": layers, "finals": finals, "inverse_max_error": model["inverse_max_error"]}


def _balanced_folds(domain: str, objects: list[str], folds: int) -> dict[str, int]:
    if len(objects) < 2 * folds:
        raise ValueError("two objects per fold required")
    # Frozen unchanged from node 8.12; changing this salt changes the held sets.
    order = sorted(objects, key=lambda obj: (hashlib.sha256(("splart-jsa-ect-fold-v1:" + domain + ":" + obj).encode()).hexdigest(), obj))
    return {obj: rank % folds for rank, obj in enumerate(order)}


def feature_crossfit_pksrt(rows: list[dict], train_indices: list[int], field_kind: str, contract: dict) -> dict[str, dict]:
    train_indices = _canonical_indices(rows, train_indices); domains = sorted({rows[i]["domain"] for i in train_indices})
    folds = int(contract["crossfit_folds"])
    objects = {d: sorted({rows[i]["object_group_id"] for i in train_indices if rows[i]["domain"] == d}) for d in domains}
    assignments = {d: _balanced_folds(d, objects[d], folds) for d in domains}
    raw_sem = rows[train_indices[0]]["semantic"].numel(); raw_mech = rows[train_indices[0]]["mechanical"].numel() - 1
    output = {d: torch.full((len(objects[d]), raw_sem), float("nan"), dtype=torch.float64) if field_kind == "semantic"
              else torch.zeros((len(objects[d]), raw_mech), dtype=torch.float64) for d in domains}
    receipts = {d: [] for d in domains}
    for fold in range(folds):
        fit = _canonical_indices(rows, [i for i in train_indices if assignments[rows[i]["domain"]][rows[i]["object_group_id"]] != fold])
        held = _canonical_indices(rows, [i for i in train_indices if assignments[rows[i]["domain"]][rows[i]["object_group_id"]] == fold])
        prep = fit_feature_preprocessor(rows, fit, 16); fm, fs, _ = transform(prep, rows, fit); hm, hs, _ = transform(prep, rows, held)
        fit_field, held_field = (fs, hs) if field_kind == "semantic" else (fm[:, :-1], hm[:, :-1])
        kept = None
        if field_kind == "mechanical":
            kept_all = torch.nonzero(prep.mechanical_keep).flatten().tolist()
            if not bool(prep.mechanical_keep[-1]) or kept_all[-1] != prep.mechanical_keep.numel() - 1:
                raise ValueError("displacement must be final kept coordinate")
            kept = kept_all[:-1]
        blocks = {}; held_blocks = {}; object_ids = {}
        for domain in domains:
            fit_objects = [o for o in objects[domain] if assignments[domain][o] != fold]
            held_objects = [o for o in objects[domain] if assignments[domain][o] == fold]
            fi = [k for k, i in enumerate(fit) if rows[i]["domain"] == domain]; hi = [k for k, i in enumerate(held) if rows[i]["domain"] == domain]
            means = lambda indices, positions, field, selected: torch.stack([field[[k for k in positions if rows[indices[k]]["object_group_id"] == obj]].mean(0) for obj in selected])
            fit_means = means(fit, fi, fit_field, fit_objects); held_means = means(held, hi, held_field, held_objects)
            dfit, dheld = [fit[k] for k in fi], [held[k] for k in hi]
            fz = canonical_object_scalar(rows, dfit, field_kind, dfit); hz = canonical_object_scalar(rows, dheld, field_kind, dfit)
            z = torch.tensor([fz[o] for o in fit_objects], dtype=torch.float64); zh = torch.tensor([hz[o] for o in held_objects], dtype=torch.float64)
            x, xh, clipped = empirical_rank(z, zh)
            blocks[domain] = (z, fit_means); held_blocks[domain] = (held_objects, zh, xh, held_means, clipped); object_ids[domain] = fit_objects
        model = fit_model(blocks, contract, object_ids); provenance = model_provenance(model); joint_hash = _canonical_sha(provenance)
        prep_payload = {"train_object_hash": prep.train_object_hash, "mechanical_mean": _sha_tensor(prep.mechanical_mean),
            "mechanical_scale": _sha_tensor(prep.mechanical_scale), "mechanical_keep": _sha_tensor(prep.mechanical_keep.to(torch.float64)),
            "mechanical_raw_dim": int(prep.mechanical_keep.numel()), "mechanical_kept_indices": torch.nonzero(prep.mechanical_keep).flatten().tolist(),
            "semantic_mean": _sha_tensor(prep.semantic_mean), "semantic_components": _sha_tensor(prep.semantic_components)}
        prep_hash = _canonical_sha(prep_payload)
        for domain in domains:
            held_objects, zh, xh, held_means, clipped = held_blocks[domain]
            local = forward(model, domain, zh, xh, held_means)
            for j, obj in enumerate(held_objects):
                target = objects[domain].index(obj)
                if field_kind == "semantic": output[domain][target] = local[j] @ prep.semantic_components
                else: output[domain][target].scatter_(0, torch.tensor(kept), local[j])
            fit_ids = [o for d in domains for o in objects[d] if assignments[d][o] != fold]
            held_ids = [o for d in domains for o in objects[d] if assignments[d][o] == fold]
            receipts[domain].append({"fold": fold, "joint_model_sha256": joint_hash, "joint_model": copy.deepcopy(provenance),
                "joint_preprocessor_sha256": prep_hash, "joint_preprocessor": copy.deepcopy(prep_payload),
                "joint_fit_object_ids": [f"{d}:{o}" for d in domains for o in objects[d] if assignments[d][o] != fold],
                "joint_held_object_ids": [f"{d}:{o}" for d in domains for o in objects[d] if assignments[d][o] == fold],
                "joint_fit_object_hash": _sha_ids([f"{d}:{o}" for d in domains for o in objects[d] if assignments[d][o] != fold]),
                "joint_held_object_hash": _sha_ids([f"{d}:{o}" for d in domains for o in objects[d] if assignments[d][o] == fold]),
                "domain_fit_object_ids": [o for o in objects[domain] if assignments[domain][o] != fold],
                "domain_held_object_ids": [o for o in objects[domain] if assignments[domain][o] == fold],
                "domain_fit_object_hash": _sha_ids([o for o in objects[domain] if assignments[domain][o] != fold]),
                "domain_held_object_hash": _sha_ids([o for o in objects[domain] if assignments[domain][o] == fold]),
                "boundary_clipped_fraction": clipped, "fit_count": len(fit_ids), "held_count": len(held_ids)})
    result = {}
    for domain in domains:
        dindices = [i for i in train_indices if rows[i]["domain"] == domain]
        zmap = canonical_object_scalar(rows, dindices, field_kind, dindices)
        z = torch.tensor([zmap[o] for o in objects[domain]], dtype=torch.float64); e = output[domain]
        if not torch.isfinite(e).all(): raise RuntimeError("OOF output uninitialized")
        result[domain] = {"schema": CROSSFIT_SCHEMA, "field_kind": field_kind, "folds": receipts[domain],
            "fold_assignment": assignments[domain], "fold_assignment_sha256": _canonical_sha(assignments[domain]),
            "fold_counts": [list(assignments[domain].values()).count(f) for f in range(folds)],
            "standardized_sha256": _sha_tensor(e), "raw_z_sha256": _sha_tensor(z),
            "residual_norm_raw_z_absolute_spearman": _spearman(e.norm(dim=-1), z),
            "residual_raw_z_distance_correlation": _distance_correlation(z, e)}
    return result


def pksrt_ablation(rows: list[dict], train_indices: list[int], held_indices: list[int], train_field: Tensor,
                    held_field: Tensor, train_z: dict[str, float], held_z: dict[str, float], contract: dict,
                    context: str, crossfit: dict[str, dict]) -> tuple[Tensor, Tensor, dict]:
    train_order = sorted(range(len(train_indices)), key=lambda k: global_row_identity(rows[train_indices[k]]))
    held_order = sorted(range(len(held_indices)), key=lambda k: global_row_identity(rows[held_indices[k]]))
    train_indices = [train_indices[k] for k in train_order]; held_indices = [held_indices[k] for k in held_order]
    train_field, held_field = train_field[train_order].double(), held_field[held_order].double()
    domains = sorted({rows[i]["domain"] for i in train_indices})
    if set(domains) != {rows[i]["domain"] for i in held_indices}: raise ValueError("domain mismatch")
    blocks = {}; held_blocks = {}; locations = {}; object_ids = {}
    for domain in domains:
        train_objects = sorted({rows[i]["object_group_id"] for i in train_indices if rows[i]["domain"] == domain})
        held_objects = sorted({rows[i]["object_group_id"] for i in held_indices if rows[i]["domain"] == domain})
        tl = {o: [k for k, i in enumerate(train_indices) if rows[i]["domain"] == domain and rows[i]["object_group_id"] == o] for o in train_objects}
        hl = {o: [k for k, i in enumerate(held_indices) if rows[i]["domain"] == domain and rows[i]["object_group_id"] == o] for o in held_objects}
        means = torch.stack([train_field[tl[o]].mean(0) for o in train_objects]); hmeans = torch.stack([held_field[hl[o]].mean(0) for o in held_objects])
        z = torch.tensor([train_z[o] for o in train_objects], dtype=torch.float64); hz = torch.tensor([held_z[o] for o in held_objects], dtype=torch.float64)
        x, hx, clipped = empirical_rank(z, hz); blocks[domain] = (z, means); held_blocks[domain] = (hz, hx, hmeans, clipped)
        locations[domain] = (train_objects, held_objects, tl, hl); object_ids[domain] = train_objects
    model = fit_model(blocks, contract, object_ids); train_out = torch.full_like(train_field, float("nan")); held_out = torch.full_like(held_field, float("nan"))
    provenance = model_provenance(model)
    receipt = {"schema": SCHEMA, "context": context, "contract_sha256": _canonical_sha(contract),
               "shared_model": provenance, "domains": {}}
    for domain in domains:
        train_objects, held_objects, tl, hl = locations[domain]; z, means = blocks[domain]; hz, hx, hmeans, clipped = held_blocks[domain]
        x = model["stages"][domain]["x"]; standardized = model["standardized"][domain]
        offsets = {o: train_field[tl[o]] - means[j] for j, o in enumerate(train_objects)}
        hoffsets = {o: held_field[hl[o]] - hmeans[j] for j, o in enumerate(held_objects)}
        donors = {o: train_objects[(j + 1) % len(train_objects)] for j, o in enumerate(train_objects)}
        hdonors = {o: train_objects[j % len(train_objects)] for j, o in enumerate(held_objects)}
        by_obj = {o: standardized[j] for j, o in enumerate(train_objects)}
        reconstruction = reconstruct_means(model, domain, z, x, standardized)
        for j, o in enumerate(train_objects):
            train_out[tl[o]] = reconstruct_means(model, domain, z[j:j+1], x[j:j+1], by_obj[donors[o]][None])[0] + offsets[o]
        for j, o in enumerate(held_objects):
            held_out[hl[o]] = reconstruct_means(model, domain, hz[j:j+1], hx[j:j+1], by_obj[hdonors[o]][None])[0] + hoffsets[o]
        train_u = max(float(((train_out[tl[o]] - train_out[tl[o]].mean(0)) - offsets[o]).abs().max()) for o in train_objects)
        held_u = max(float(((held_out[hl[o]] - held_out[hl[o]].mean(0)) - hoffsets[o]).abs().max()) for o in held_objects)
        original = torch.cat([train_field[tl[o]] for o in train_objects]); shuffled = torch.cat([train_out[tl[o]] for o in train_objects]); hshuffled = torch.cat([held_out[hl[o]] for o in held_objects])
        original_sd = original.std(unbiased=False).clamp_min(1e-12); original_p99 = torch.quantile(original.norm(dim=-1), .99).clamp_min(1e-12)
        receipt["domains"][domain] = {**model["diagnostics"][domain], "reconstruction_max_error": float((reconstruction - means).abs().max()),
            "inverse_max_error": model["inverse_max_error"], "recipient_u_max_error": max(train_u, held_u),
            "held_boundary_clipped_fraction": clipped, "shuffle_rms_over_original_sd": float((shuffled - original).square().mean().sqrt() / original_sd),
            "shuffled_norm_p99_ratio": float(max(torch.quantile(shuffled.norm(dim=-1), .99), torch.quantile(hshuffled.norm(dim=-1), .99)) / original_p99),
            "train_mapping": donors, "held_mapping": hdonors, "train_coverage": 1., "held_coverage": 1., "train_self_rate": 0.,
            "train_donor_marginal_exact": sorted(donors.values()) == train_objects,
            "held_effective_donors_per_object": len(set(hdonors.values())) / len(held_objects), "crossfit": copy.deepcopy(crossfit[domain])}
    if not torch.isfinite(train_out).all() or not torch.isfinite(held_out).all(): raise RuntimeError("transport output uninitialized")
    return train_out, held_out, receipt


def preserve_recipient_displacement(geometry: Tensor, mechanical: Tensor) -> Tensor:
    if geometry.shape[:-1] != mechanical.shape[:-1] or geometry.shape[-1] + 1 != mechanical.shape[-1]: raise ValueError("mechanical shape mismatch")
    result = torch.cat((geometry, mechanical[..., -1:]), -1)
    if not torch.equal(result[..., -1], mechanical[..., -1]): raise RuntimeError("recipient displacement changed")
    return result


def _valid_sha(value) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _validate_qr(value: dict) -> bool:
    return (set(value) == {"rank", "condition", "rank_threshold", "positive_diagonal"}
            and value["positive_diagonal"] is True and isinstance(value["rank"], int) and value["rank"] > 0
            and math.isfinite(float(value["condition"])) and math.isfinite(float(value["rank_threshold"]))
            and float(value["condition"]) > 0 and float(value["rank_threshold"]) >= 0)


def _pool_provenance(pool: dict) -> dict:
    return {"a": pool["a"].tolist(), "u": pool["u"].tolist(), "keys": pool["keys"], "groups": pool["groups"],
            "bandwidth": pool["bandwidth"], "raw_count": pool["raw_count"], "unique_count": pool["unique_count"],
            "tie_group_count": pool["tie_group_count"], "max_multiplicity": pool["max_multiplicity"],
            "grouping_sha256": pool["grouping_sha256"]}


def _validate_shared_provenance_uncached(shared: dict, contract: dict) -> bool:
    """Replay every deterministic receipt field; self-consistent mutations fail."""
    try:
        if set(shared) != {"width", "slice_receipt", "stages", "layers", "finals", "inverse_max_error"}: return False
        width = shared["width"]
        if not isinstance(width, int) or width < 1 or not math.isfinite(float(shared["inverse_max_error"])): return False
        slices, slice_receipt = fixed_slices(width)
        if shared["slice_receipt"] != slice_receipt or len(shared["layers"]) != 2 * width: return False
        domains = sorted(shared["stages"])
        if len(domains) < 2 or set(shared["finals"]) != set(domains): return False
        stages = {}; ids = {}; means = {}; current = {}
        stage_keys = {"train_z", "train_u", "object_ids", "object_ids_sha256", "train_means", "train_means_sha256",
                      "z_mean", "beta", "beta_sha256", "qr", "scale", "scale_squared", "centered_energy", "identifiability_floor"}
        for domain in domains:
            value = shared["stages"][domain]
            if set(value) != stage_keys or not _validate_qr(value["qr"]): return False
            object_ids = value["object_ids"]
            if object_ids != sorted(object_ids) or len(object_ids) != len(set(object_ids)) or value["object_ids_sha256"] != _sha_ids(object_ids): return False
            z = torch.tensor(value["train_z"], dtype=torch.float64); y = torch.tensor(value["train_means"], dtype=torch.float64)
            if y.ndim != 2 or y.shape != (len(z), width) or len(object_ids) != len(z): return False
            if not torch.isfinite(z).all() or not torch.isfinite(y).all() or value["train_means_sha256"] != _sha_tensor(y): return False
            replay = _fit_first_stage(z, y, contract); beta = torch.tensor(value["beta"], dtype=torch.float64)
            if value["beta_sha256"] != _sha_tensor(beta) or not torch.equal(beta, replay["beta"]): return False
            if value["qr"] != replay["qr"] or value["train_u"] != replay["u"].tolist() or value["z_mean"] != float(replay["z_mean"]): return False
            for key in ("scale", "scale_squared", "centered_energy", "identifiability_floor"):
                expected = float(replay["scale"]) if key == "scale" else replay[key]
                if not math.isfinite(float(value[key])) or value[key] != expected: return False
            stages[domain], ids[domain], means[domain], current[domain] = replay, object_ids, y, replay["residual"].clone()
        internal_layers = []
        for index, layer in enumerate(shared["layers"]):
            layer_keys = {"index", "vector", "vector_sha256", "domain_marginals", "domain_preupdate_t", "conditional_pool",
                          "query_receipts", "minimum_observed_effective_sample_size", "minimum_observed_conditional_slope"}
            if set(layer) != layer_keys or layer["index"] != index: return False
            vector = torch.tensor(layer["vector"], dtype=torch.float64)
            if not torch.equal(vector, slices[index]) or layer["vector_sha256"] != _sha_tensor(vector): return False
            if set(layer["domain_marginals"]) != set(domains) or set(layer["domain_preupdate_t"]) != set(domains): return False
            marginals = {}; records = []; preupdate = {}
            for domain in domains:
                t = current[domain] @ vector; stored_t = layer["domain_preupdate_t"][domain]
                if set(stored_t) != {"values", "sha256"}: return False
                receipt_t = torch.tensor(stored_t["values"], dtype=torch.float64)
                if not torch.equal(t, receipt_t) or stored_t["sha256"] != _sha_tensor(receipt_t): return False
                marginal = fit_domain_marginal(t)
                if layer["domain_marginals"][domain] != _bijection_provenance(marginal): return False
                a = _linear_forward(marginal, t)
                records.extend((float(a[i]), float(stages[domain]["u"][i]), f"{domain}:{ids[domain][i]}") for i in range(len(t)))
                marginals[domain], preupdate[domain] = marginal, t
            pool = fit_conditional_pool(records)
            if layer["conditional_pool"] != _pool_provenance(pool): return False
            query_receipts = []; min_neff = math.inf; min_slope = math.inf
            for domain in domains:
                t = preupdate[domain]; a = _linear_forward(marginals[domain], t); eta_values = []
                for i in range(len(t)):
                    conditional, neff = conditional_bijection(pool, stages[domain]["u"][i], contract["minimum_effective_sample_size"])
                    slope = float(conditional["slopes"].min()); min_neff = min(min_neff, neff); min_slope = min(min_slope, slope)
                    query_receipts.append({"key": f"{domain}:{ids[domain][i]}", "u": float(stages[domain]["u"][i]),
                        "effective_sample_size": neff, "conditional_x_sha256": _sha_tensor(conditional["x"]),
                        "conditional_y_sha256": _sha_tensor(conditional["y"]), "minimum_slope": slope})
                    eta_values.append(_linear_forward(conditional, a[i:i + 1])[0])
                eta = torch.stack(eta_values); current[domain] = current[domain] + (eta - t)[:, None] * vector[None]
            if layer["query_receipts"] != query_receipts or layer["minimum_observed_effective_sample_size"] != min_neff or layer["minimum_observed_conditional_slope"] != min_slope: return False
            if min_neff < contract["minimum_effective_sample_size"] or not math.isfinite(min_slope) or min_slope <= 0: return False
            internal_layers.append({"vector": vector, "marginals": marginals, "pool": pool})
        finals = {}; standardized = {}
        final_keys = {"beta", "beta_sha256", "standardized", "standardized_sha256", "qr"}
        for domain in domains:
            value = shared["finals"][domain]
            if set(value) != final_keys or not _validate_qr(value["qr"]): return False
            design = _anchor_design(stages[domain]["z"], stages[domain]["z_mean"])
            beta, qr = _positive_qr(design, current[domain], contract); e = current[domain] - design @ beta
            stored_beta = torch.tensor(value["beta"], dtype=torch.float64); stored_e = torch.tensor(value["standardized"], dtype=torch.float64)
            if not torch.equal(stored_beta, beta) or not torch.equal(stored_e, e) or value["qr"] != qr: return False
            if value["beta_sha256"] != _sha_tensor(stored_beta) or value["standardized_sha256"] != _sha_tensor(stored_e): return False
            scale = e.std(0, unbiased=False).clamp_min(torch.finfo(torch.float64).tiny); centered = stages[domain]["z"] - stages[domain]["z_mean"]
            if float((e.mean(0).abs()/scale).max()) > contract["normalized_residual_mean_max"]: return False
            if float(((centered[:,None]*e).mean(0).abs()/(centered.std(unbiased=False)*scale)).max()) > contract["normalized_residual_raw_z_correlation_max"]: return False
            finals[domain], standardized[domain] = {"beta": beta, "qr": qr}, e
        internal = {"domains": domains, "stages": stages, "layers": internal_layers, "finals": finals,
                    "minimum_effective_sample_size": contract["minimum_effective_sample_size"]}
        inverse = 0.
        for domain in domains:
            recovered = reconstruct_means(internal, domain, stages[domain]["z"], stages[domain]["x"], standardized[domain])
            inverse = max(inverse, float((recovered - means[domain]).abs().max()))
        return shared["inverse_max_error"] == inverse and inverse <= contract["inverse_max_error"]
    except (KeyError, TypeError, ValueError, IndexError, RuntimeError, ZeroDivisionError, OverflowError):
        return False


_VALID_SHARED_PROVENANCE: set[tuple[str, str]] = set()


def _validate_shared_provenance(shared: dict, contract: dict) -> bool:
    """Content-addressed success cache avoids replaying duplicate cross-domain folds."""
    key = (_canonical_sha(shared), _canonical_sha(contract))
    if key in _VALID_SHARED_PROVENANCE:
        return True
    valid = _validate_shared_provenance_uncached(shared, contract)
    if valid:
        _VALID_SHARED_PROVENANCE.add(key)
    return valid


def global_receipt_passes(candidate: dict, expected: dict, expected_sha256: str, contract: dict, context: str, domains: set[str]) -> bool:
    try:
        if candidate is expected or _shares_container_identity(candidate, expected): return False
        if not isinstance(expected_sha256, str) or len(expected_sha256) != 64: return False
        if _canonical_sha(candidate) != expected_sha256 or _canonical_sha(expected) != expected_sha256: return False
        if set(candidate) != {"schema", "context", "contract_sha256", "shared_model", "domains"}: return False
        if candidate["schema"] != SCHEMA or candidate["context"] != context or candidate["contract_sha256"] != _canonical_sha(contract) or set(candidate["domains"]) != domains: return False
        shared = candidate["shared_model"]
        if not _validate_shared_provenance(shared, contract): return False
        if shared["slice_receipt"]["count"] != 2 * shared["width"] or shared["slice_receipt"]["sequence"] != "coordinate_then_dct2": return False
        if shared["slice_receipt"]["orthonormal_max_error"] > contract["numeric_tolerance"]: return False
        if shared["inverse_max_error"] > contract["inverse_max_error"] or len(shared["layers"]) != 2 * shared["width"]: return False
        for index, layer in enumerate(shared["layers"]):
            if layer["index"] != index or layer["conditional_pool"]["raw_count"] < layer["conditional_pool"]["unique_count"]: return False
            if layer["conditional_pool"]["unique_count"] < 2 or layer["minimum_observed_effective_sample_size"] < contract["minimum_effective_sample_size"]: return False
            if not math.isfinite(layer["minimum_observed_conditional_slope"]) or layer["minimum_observed_conditional_slope"] <= 0: return False
            expected_h = _bandwidth(layer["conditional_pool"]["raw_count"])
            if layer["conditional_pool"]["bandwidth"] != expected_h: return False
            for marginal in layer["domain_marginals"].values():
                if min(marginal["slopes"]) <= 0: return False
        crossfit_keys = {"schema", "field_kind", "folds", "fold_assignment", "fold_assignment_sha256", "fold_counts",
                         "standardized_sha256", "raw_z_sha256", "residual_norm_raw_z_absolute_spearman", "residual_raw_z_distance_correlation"}
        crossfits = {d: candidate["domains"][d]["crossfit"] for d in domains}
        for domain, cf in crossfits.items():
            if set(cf) != crossfit_keys or cf["schema"] != CROSSFIT_SCHEMA: return False
            expected_kind = "semantic" if context.startswith("semantic_pksrt/") else "mechanical" if context.startswith("mechanical_pksrt/") else None
            if cf["field_kind"] != expected_kind or not _valid_sha(cf["standardized_sha256"]) or not _valid_sha(cf["raw_z_sha256"]): return False
            if not math.isfinite(float(cf["residual_norm_raw_z_absolute_spearman"])) or not math.isfinite(float(cf["residual_raw_z_distance_correlation"])): return False
            if cf["fold_assignment_sha256"] != _canonical_sha(cf["fold_assignment"]): return False
            objects = sorted(cf["fold_assignment"])
            if cf["fold_assignment"] != _balanced_folds(domain, objects, contract["crossfit_folds"]): return False
            counts = [list(cf["fold_assignment"].values()).count(i) for i in range(contract["crossfit_folds"])]
            if cf["fold_counts"] != counts or min(counts) < 2: return False
        reference = crossfits[sorted(domains)[0]]["folds"]
        domain_keys = {"normalized_residual_mean_max", "normalized_residual_raw_z_correlation_max", "residual_energy_ratio",
                       "reconstruction_max_error", "inverse_max_error", "recipient_u_max_error", "held_boundary_clipped_fraction",
                       "shuffle_rms_over_original_sd", "shuffled_norm_p99_ratio", "train_mapping", "held_mapping", "train_coverage",
                       "held_coverage", "train_self_rate", "train_donor_marginal_exact", "held_effective_donors_per_object", "crossfit"}
        fold_keys = {"fold", "joint_model_sha256", "joint_model", "joint_preprocessor_sha256", "joint_preprocessor",
                     "joint_fit_object_ids", "joint_held_object_ids", "joint_fit_object_hash", "joint_held_object_hash",
                     "domain_fit_object_ids", "domain_held_object_ids", "domain_fit_object_hash", "domain_held_object_hash",
                     "boundary_clipped_fraction", "fit_count", "held_count"}
        for domain, value in candidate["domains"].items():
            if set(value) != domain_keys: return False
            numeric = [value[k] for k in ("normalized_residual_mean_max", "normalized_residual_raw_z_correlation_max",
                       "residual_energy_ratio", "reconstruction_max_error", "inverse_max_error", "recipient_u_max_error",
                       "held_boundary_clipped_fraction", "shuffle_rms_over_original_sd", "shuffled_norm_p99_ratio",
                       "train_coverage", "held_coverage", "train_self_rate", "held_effective_donors_per_object")]
            if not all(math.isfinite(float(item)) for item in numeric): return False
            train_ids, held_ids = sorted(value["train_mapping"]), sorted(value["held_mapping"])
            if train_ids != sorted(crossfits[domain]["fold_assignment"]): return False
            if value["train_mapping"] != {o: train_ids[(i + 1) % len(train_ids)] for i, o in enumerate(train_ids)}: return False
            if value["held_mapping"] != {o: train_ids[i % len(train_ids)] for i, o in enumerate(held_ids)}: return False
            cf = value["crossfit"]
            if sorted(f["fold"] for f in cf["folds"]) != list(range(contract["crossfit_folds"])): return False
            for fold in cf["folds"]:
                if set(fold) != fold_keys: return False
                peer = reference[fold["fold"]]
                if fold["joint_model_sha256"] != peer["joint_model_sha256"] or fold["joint_preprocessor_sha256"] != peer["joint_preprocessor_sha256"]: return False
                if fold["joint_fit_object_ids"] != peer["joint_fit_object_ids"] or fold["joint_held_object_ids"] != peer["joint_held_object_ids"]: return False
                if fold["joint_model_sha256"] != _canonical_sha(fold["joint_model"]) or fold["joint_preprocessor_sha256"] != _canonical_sha(fold["joint_preprocessor"]): return False
                if not _validate_shared_provenance(fold["joint_model"], contract): return False
                if set(fold["joint_model"]["stages"]) != domains: return False
                if not _valid_sha(fold["joint_model_sha256"]) or not _valid_sha(fold["joint_preprocessor_sha256"]): return False
                pp = fold["joint_preprocessor"]
                if set(pp) != {"train_object_hash", "mechanical_mean", "mechanical_scale", "mechanical_keep", "mechanical_raw_dim",
                               "mechanical_kept_indices", "semantic_mean", "semantic_components"}: return False
                if not all(_valid_sha(pp[k]) for k in ("train_object_hash", "mechanical_mean", "mechanical_scale", "mechanical_keep", "semantic_mean", "semantic_components")): return False
                if not isinstance(pp["mechanical_raw_dim"], int) or pp["mechanical_raw_dim"] < 1 or not pp["mechanical_kept_indices"]: return False
                if pp["mechanical_kept_indices"][-1] != pp["mechanical_raw_dim"] - 1: return False
                if set(fold["joint_fit_object_ids"]) & set(fold["joint_held_object_ids"]): return False
                if fold["joint_fit_object_hash"] != _sha_ids(fold["joint_fit_object_ids"]) or fold["joint_held_object_hash"] != _sha_ids(fold["joint_held_object_ids"]): return False
                if fold["domain_fit_object_hash"] != _sha_ids(fold["domain_fit_object_ids"]) or fold["domain_held_object_hash"] != _sha_ids(fold["domain_held_object_ids"]): return False
                if set(fold["domain_fit_object_ids"]) & set(fold["domain_held_object_ids"]): return False
                expected_fit = sorted(o for o, assigned in cf["fold_assignment"].items() if assigned != fold["fold"])
                expected_held = sorted(o for o, assigned in cf["fold_assignment"].items() if assigned == fold["fold"])
                if fold["domain_fit_object_ids"] != expected_fit or fold["domain_held_object_ids"] != expected_held: return False
                joint_fit = [f"{d}:{o}" for d in sorted(domains) for o, assigned in sorted(crossfits[d]["fold_assignment"].items()) if assigned != fold["fold"]]
                joint_held = [f"{d}:{o}" for d in sorted(domains) for o, assigned in sorted(crossfits[d]["fold_assignment"].items()) if assigned == fold["fold"]]
                if fold["joint_fit_object_ids"] != joint_fit or fold["joint_held_object_ids"] != joint_held: return False
                if fold["fit_count"] != len(joint_fit) or fold["held_count"] != len(joint_held): return False
                for joined_domain in domains:
                    expected_stage_ids = sorted(o for o, assigned in crossfits[joined_domain]["fold_assignment"].items() if assigned != fold["fold"])
                    if fold["joint_model"]["stages"][joined_domain]["object_ids"] != expected_stage_ids: return False
                if not math.isfinite(float(fold["boundary_clipped_fraction"])) or not 0 <= float(fold["boundary_clipped_fraction"]) <= 1: return False
        return True
    except (KeyError, TypeError, ValueError, IndexError, RuntimeError, ZeroDivisionError):
        return False


def domain_receipt_passes(candidate: dict, expected: dict, expected_sha256: str, contract: dict, context: str, domains: set[str], domain: str) -> bool:
    if not global_receipt_passes(candidate, expected, expected_sha256, contract, context, domains) or domain not in domains: return False
    try:
        value = candidate["domains"][domain]; cf = value["crossfit"]
        numeric = [value[k] for k in ("normalized_residual_mean_max", "normalized_residual_raw_z_correlation_max",
                   "residual_energy_ratio", "reconstruction_max_error", "inverse_max_error", "recipient_u_max_error",
                   "held_boundary_clipped_fraction", "shuffle_rms_over_original_sd", "shuffled_norm_p99_ratio",
                   "train_coverage", "held_coverage", "train_self_rate", "held_effective_donors_per_object")]
        if not all(math.isfinite(float(item)) for item in numeric): return False
        if value["normalized_residual_mean_max"] > contract["normalized_residual_mean_max"]: return False
        if value["normalized_residual_raw_z_correlation_max"] > contract["normalized_residual_raw_z_correlation_max"]: return False
        if value["reconstruction_max_error"] > contract["reconstruction_max_error"] or value["inverse_max_error"] > contract["inverse_max_error"]: return False
        if value["recipient_u_max_error"] > contract["recipient_u_max_error"] or value["held_boundary_clipped_fraction"] > contract["held_boundary_clipped_fraction_max"]: return False
        if value["residual_energy_ratio"] < contract["residual_energy_ratio_at_least"]: return False
        if value["shuffle_rms_over_original_sd"] < contract["shuffle_rms_over_original_sd_at_least"] or value["shuffled_norm_p99_ratio"] > contract["shuffled_norm_p99_ratio_at_most"]: return False
        if value["train_coverage"] != 1 or value["held_coverage"] != 1 or value["train_self_rate"] != 0 or value["train_donor_marginal_exact"] is not True or value["held_effective_donors_per_object"] != 1: return False
        if cf["residual_norm_raw_z_absolute_spearman"] > contract["crossfit_absolute_spearman_at_most"] or cf["residual_raw_z_distance_correlation"] > contract["crossfit_distance_correlation_at_most"]: return False
        return True
    except (KeyError, TypeError, ValueError, IndexError, RuntimeError, ZeroDivisionError): return False


def receipt_passes(candidate: dict, expected: dict, expected_sha256: str, contract: dict, context: str, domains: set[str]) -> bool:
    return global_receipt_passes(candidate, expected, expected_sha256, contract, context, domains) and all(
        domain_receipt_passes(candidate, expected, expected_sha256, contract, context, domains, d) for d in domains)


__all__ = ["SCHEMA", "conditional_bijection", "domain_receipt_passes", "feature_crossfit_pksrt", "fit_conditional_pool",
           "fit_domain_marginal", "fit_model", "fixed_slices", "forward", "global_receipt_passes", "model_provenance",
           "pksrt_ablation", "preserve_recipient_displacement", "receipt_passes", "reconstruct_means"]
