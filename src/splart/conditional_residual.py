"""Object-level conditional-residual nulls for OCR-SMARC."""

from __future__ import annotations

import hashlib
import math

import torch
from torch import Tensor


def _fit_ols(z: Tensor, field: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    z,field=z.double(),field.double()
    if z.ndim!=1 or field.ndim!=2 or len(z)!=len(field) or len(z)<3:
        raise ValueError("OLS needs at least three object rows")
    if not torch.isfinite(z).all() or not torch.isfinite(field).all() or float(z.var(unbiased=False))<=1e-12:
        raise ValueError("conditional scalar is nonfinite or degenerate")
    design=torch.stack((torch.ones_like(z),z),-1)
    u,s,vh=torch.linalg.svd(design,full_matrices=False)
    if len(s)<2 or float(s[-1])<=1e-12: raise ValueError("conditional design rank below two")
    beta=vh.T@torch.diag(1/s)@u.T@field
    fitted=design@beta; residual=field-fitted
    return beta,fitted,residual


def _rank(value: Tensor) -> Tensor:
    order=torch.argsort(value,stable=True); result=torch.empty_like(value,dtype=torch.float64)
    result[order]=torch.arange(len(value),dtype=torch.float64)
    return result


def _spearman(left: Tensor,right: Tensor) -> float:
    x=_rank(left); y=_rank(right); x=x-x.mean(); y=y-y.mean()
    denominator=x.norm()*y.norm()
    return 0. if float(denominator)==0 else float((x@y/denominator).abs())


def _distance_correlation(z: Tensor,residual: Tensor) -> float:
    a=torch.cdist(z[:,None].double(),z[:,None].double()); b=torch.cdist(residual.double(),residual.double())
    a=a-a.mean(0,keepdim=True)-a.mean(1,keepdim=True)+a.mean()
    b=b-b.mean(0,keepdim=True)-b.mean(1,keepdim=True)+b.mean()
    covariance=(a*b).mean().clamp_min(0); variance_a=(a*a).mean().clamp_min(0); variance_b=(b*b).mean().clamp_min(0)
    denominator=torch.sqrt(variance_a*variance_b)
    return 0. if float(denominator)==0 else float(torch.sqrt(covariance/denominator))


def crossfit_diagnostics(object_ids: list[str],z: Tensor,field: Tensor,folds: int=5) -> dict:
    assignments=[int(hashlib.sha256(("splart-ocrsmarc-crossfit-v1:"+obj).encode()).hexdigest()[:16],16)%folds for obj in object_ids]
    if set(assignments)!=set(range(folds)): raise ValueError("cross-fit fold is empty")
    residual=torch.empty_like(field,dtype=torch.float64)
    for fold in range(folds):
        train=[i for i,value in enumerate(assignments) if value!=fold]; held=[i for i,value in enumerate(assignments) if value==fold]
        beta,_,_=_fit_ols(z[train],field[train]); design=torch.stack((torch.ones_like(z[held]),z[held]),-1).double()
        residual[held]=field[held]-design@beta
    norm=residual.norm(dim=-1)
    return {"fold_counts":[assignments.count(fold) for fold in range(folds)],
            "residual_norm_z_absolute_spearman":_spearman(norm,z),
            "residual_z_distance_correlation":_distance_correlation(z,residual)}


def conditional_residual_ablation(rows: list[dict],train_indices: list[int],held_indices: list[int],
                                  train_field: Tensor,held_field: Tensor,train_z: dict[str,float],
                                  held_z: dict[str,float],folds: int=5) -> tuple[Tensor,Tensor,dict]:
    """Swap train residuals while preserving conditional means and offsets."""
    if train_field.shape[0]!=len(train_indices) or held_field.shape[0]!=len(held_indices) or train_field.shape[1:]!=held_field.shape[1:]:
        raise ValueError("conditional-residual field alignment mismatch")
    train_out=torch.empty_like(train_field,dtype=torch.float64); held_out=torch.empty_like(held_field,dtype=torch.float64)
    domains=sorted({rows[i]["domain"] for i in train_indices}); receipt={"domains":{}}
    for domain in domains:
        train_objects=sorted({rows[i]["object_group_id"] for i in train_indices if rows[i]["domain"]==domain})
        held_objects=sorted({rows[i]["object_group_id"] for i in held_indices if rows[i]["domain"]==domain})
        if not train_objects or not held_objects: raise ValueError("each domain needs train and held objects")
        train_locations={obj:[k for k,i in enumerate(train_indices) if rows[i]["object_group_id"]==obj] for obj in train_objects}
        held_locations={obj:[k for k,i in enumerate(held_indices) if rows[i]["object_group_id"]==obj] for obj in held_objects}
        means=torch.stack([train_field[train_locations[obj]].double().mean(0) for obj in train_objects]); z=torch.tensor([train_z[obj] for obj in train_objects],dtype=torch.float64)
        beta,fitted,residual=_fit_ols(z,means); offsets={obj:train_field[train_locations[obj]].double()-means[j] for j,obj in enumerate(train_objects)}
        reconstruction=torch.cat([fitted[j][None]+residual[j][None]+offsets[obj] for j,obj in enumerate(train_objects)])
        original=torch.cat([train_field[train_locations[obj]].double() for obj in train_objects])
        reconstruction_error=float((reconstruction-original).abs().max())
        scale=means.std(0,unbiased=False).clamp_min(1e-12)
        mean_error=float((residual.mean(0).abs()/scale).max())
        centered_z=z-z.mean(); correlation_error=float(((centered_z[:,None]*residual).mean(0).abs()/(centered_z.std(unbiased=False)*scale).clamp_min(1e-12)).max())
        centered=means-means.mean(0); residual_energy=float(residual.square().mean()/centered.square().mean().clamp_min(1e-12))
        train_donor={obj:train_objects[(j+1)%len(train_objects)] for j,obj in enumerate(train_objects)}
        donor_residual={obj:residual[j] for j,obj in enumerate(train_objects)}
        for obj in train_objects:
            locations=train_locations[obj]; train_out[locations]=fitted[train_objects.index(obj)][None]+donor_residual[train_donor[obj]][None]+offsets[obj]
        held_load={obj:0 for obj in train_objects}; held_donor={}
        for j,obj in enumerate(held_objects): held_donor[obj]=train_objects[j%len(train_objects)]; held_load[held_donor[obj]]+=1
        for obj in held_objects:
            locations=held_locations[obj]; own=held_field[locations].double(); offsets_held=own-own.mean(0); value=torch.tensor([1.,held_z[obj]],dtype=torch.float64)@beta
            held_out[locations]=value[None]+donor_residual[held_donor[obj]][None]+offsets_held
        train_domain=[k for k,i in enumerate(train_indices) if rows[i]["domain"]==domain]; held_domain=[k for k,i in enumerate(held_indices) if rows[i]["domain"]==domain]
        original_sd=train_field[train_domain].double().std(unbiased=False).clamp_min(1e-12)
        shuffle_ratio=float((train_out[train_domain]-train_field[train_domain]).square().mean().sqrt()/original_sd)
        original_p99=float(torch.quantile(train_field[train_domain].double().norm(dim=-1),.99).clamp_min(1e-12))
        shuffled_p99=max(float(torch.quantile(train_out[train_domain].norm(dim=-1),.99)),float(torch.quantile(held_out[held_domain].norm(dim=-1),.99)))
        crossfit=crossfit_diagnostics(train_objects,z,means,folds)
        mapping="\n".join(f"{obj}->{train_donor[obj]}" for obj in train_objects)+"\n"+"\n".join(f"{obj}->{held_donor[obj]}" for obj in held_objects)+"\n"
        receipt["domains"][domain]={"train_objects":len(train_objects),"held_objects":len(held_objects),"train_coverage":1.0,"held_coverage":1.0,
            "train_self_rate":0.0,"train_donor_marginal_exact":set(train_donor.values())==set(train_objects),
            "held_effective_donors":len({*held_donor.values()}),"held_effective_donors_per_object":len({*held_donor.values()})/len(held_objects),
            "held_max_donor_load":max(held_load.values()),"held_load_bound":math.ceil(len(held_objects)/len(train_objects)),
            "reconstruction_max_error":reconstruction_error,"normalized_residual_mean_max":mean_error,
            "normalized_residual_z_correlation_max":correlation_error,"residual_energy_ratio":residual_energy,
            "shuffle_rms_over_original_sd":shuffle_ratio,"shuffled_norm_p99_ratio":shuffled_p99/original_p99,
            "crossfit":crossfit,"mapping_sha256":hashlib.sha256(mapping.encode()).hexdigest()}
    if not torch.isfinite(train_out).all() or not torch.isfinite(held_out).all(): raise ValueError("conditional residual output nonfinite")
    return train_out,held_out,receipt


def receipt_passes(receipt: dict,contract: dict) -> bool:
    gate=contract["gates"]; cross=contract["crossfit"]
    for value in receipt["domains"].values():
        if not (value["reconstruction_max_error"]<=gate["reconstruction_max_error"] and
                value["normalized_residual_mean_max"]<=gate["normalized_residual_mean_max"] and
                value["normalized_residual_z_correlation_max"]<=gate["normalized_residual_z_correlation_max"] and
                value["train_coverage"]==gate["coverage"] and value["held_coverage"]==gate["coverage"] and
                value["train_self_rate"]==gate["train_self_rate"] and value["train_donor_marginal_exact"] and
                value["held_effective_donors_per_object"]==gate["held_effective_donors_per_object"] and
                value["held_max_donor_load"]<=value["held_load_bound"] and
                value["residual_energy_ratio"]>=gate["residual_energy_ratio_at_least"] and
                value["shuffle_rms_over_original_sd"]>=gate["shuffle_rms_over_original_sd_at_least"] and
                value["shuffled_norm_p99_ratio"]<=gate["shuffled_norm_p99_ratio_at_most"] and
                value["crossfit"]["residual_norm_z_absolute_spearman"]<=cross["absolute_spearman_at_most"] and
                value["crossfit"]["residual_z_distance_correlation"]<=cross["distance_correlation_at_most"]): return False
    return True


__all__=["conditional_residual_ablation","crossfit_diagnostics","receipt_passes"]
