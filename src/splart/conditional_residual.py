"""Object-level conditional-residual nulls for OCR-SMARC."""

from __future__ import annotations

import hashlib
import math

import torch
from torch import Tensor


def _tensor_sha256(value: Tensor) -> str:
    value=value.detach().cpu().contiguous().double()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


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
    """Deterministic average ranks (ties receive the same mid-rank)."""
    value=value.double(); order=torch.argsort(value,stable=True); result=torch.empty_like(value)
    sorted_value=value[order]; start=0
    while start<len(value):
        stop=start+1
        while stop<len(value) and sorted_value[stop]==sorted_value[start]: stop+=1
        result[order[start:stop]]=(start+stop-1)/2
        start=stop
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
    if len(set(object_ids))!=len(object_ids) or len(object_ids)<2*folds: raise ValueError("cross-fit needs unique IDs and at least two objects per fold")
    ordered=sorted(range(len(object_ids)),key=lambda i:(hashlib.sha256(("splart-ocrsmarc-crossfit-v1:"+object_ids[i]).encode()).hexdigest(),object_ids[i]))
    assignments=[-1]*len(object_ids)
    for rank,index in enumerate(ordered): assignments[index]=rank%folds
    counts=[assignments.count(fold) for fold in range(folds)]
    if min(counts)<len(object_ids)//folds: raise ValueError("cross-fit balance contract failed")
    residual=torch.empty_like(field,dtype=torch.float64)
    for fold in range(folds):
        train=[i for i,value in enumerate(assignments) if value!=fold]; held=[i for i,value in enumerate(assignments) if value==fold]
        beta,_,_=_fit_ols(z[train],field[train]); design=torch.stack((torch.ones_like(z[held]),z[held]),-1).double()
        residual[held]=field[held]-design@beta
    norm=residual.norm(dim=-1)
    return {"fold_counts":counts,
            "residual_norm_z_absolute_spearman":_spearman(norm,z),
            "residual_z_distance_correlation":_distance_correlation(z,residual)}


def conditional_residual_ablation(rows: list[dict],train_indices: list[int],held_indices: list[int],
                                   train_field: Tensor,held_field: Tensor,train_z: dict[str,float],
                                   held_z: dict[str,float],folds: int=5,
                                   context: str="unspecified") -> tuple[Tensor,Tensor,dict]:
    """Swap train residuals while preserving conditional means and offsets."""
    if train_field.shape[0]!=len(train_indices) or held_field.shape[0]!=len(held_indices) or train_field.shape[1:]!=held_field.shape[1:]:
        raise ValueError("conditional-residual field alignment mismatch")
    if not train_indices or not held_indices or not context: raise ValueError("nonempty split and context required")
    train_out=torch.full_like(train_field,float("nan"),dtype=torch.float64); held_out=torch.full_like(held_field,float("nan"),dtype=torch.float64)
    train_domains={rows[i]["domain"] for i in train_indices}; held_domains={rows[i]["domain"] for i in held_indices}
    if train_domains!=held_domains: raise ValueError("held/train domain sets differ")
    ownership={}
    for split,indices in (("train",train_indices),("held",held_indices)):
        for i in indices:
            obj=rows[i]["object_group_id"]; pair=(rows[i]["domain"],split)
            if obj in ownership and ownership[obj]!=pair: raise ValueError("object ID crosses domain or split")
            ownership[obj]=pair
    domains=sorted(train_domains); receipt={"context":context,"domains":{}}
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
        train_coverage=len(train_donor)/len(train_objects); held_coverage=len(held_donor)/len(held_objects)
        self_rate=sum(obj==donor for obj,donor in train_donor.items())/len(train_objects)
        marginal=(sorted(train_donor.values())==train_objects)
        mapping=f"context={context}\ndomain={domain}\n"+"\n".join(f"train:{obj}->{train_donor[obj]}" for obj in train_objects)+"\n"+"\n".join(f"held:{obj}->{held_donor[obj]}" for obj in held_objects)+"\n"
        receipt["domains"][domain]={"train_objects":len(train_objects),"held_objects":len(held_objects),"train_object_hash":hashlib.sha256(("\n".join(train_objects)+"\n").encode()).hexdigest(),
            "held_object_hash":hashlib.sha256(("\n".join(held_objects)+"\n").encode()).hexdigest(),"z_sha256":_tensor_sha256(z),
            "ols_beta_sha256":_tensor_sha256(beta),"train_mapping":train_donor,"held_mapping":held_donor,
            "train_coverage":train_coverage,"held_coverage":held_coverage,
            "train_self_rate":self_rate,"train_donor_marginal_exact":marginal,
            "held_effective_donors":len({*held_donor.values()}),"held_effective_donors_per_object":len({*held_donor.values()})/len(held_objects),
            "held_max_donor_load":max(held_load.values()),"held_load_bound":math.ceil(len(held_objects)/len(train_objects)),
            "reconstruction_max_error":reconstruction_error,"normalized_residual_mean_max":mean_error,
            "normalized_residual_z_correlation_max":correlation_error,"residual_energy_ratio":residual_energy,
            "shuffle_rms_over_original_sd":shuffle_ratio,"shuffled_norm_p99_ratio":shuffled_p99/original_p99,
            "crossfit":crossfit,"mapping_sha256":hashlib.sha256(mapping.encode()).hexdigest()}
    if not torch.isfinite(train_out).all() or not torch.isfinite(held_out).all(): raise ValueError("conditional residual output nonfinite")
    return train_out,held_out,receipt


def receipt_passes(receipt: dict,contract: dict,expected_domains: set[str] | None=None) -> bool:
    gate=contract["gates"]; cross=contract["crossfit"]
    if not receipt.get("context") or not receipt.get("domains"): return False
    if expected_domains is not None and set(receipt["domains"])!=set(expected_domains): return False
    for domain,value in receipt["domains"].items():
        train_mapping=value.get("train_mapping",{}); held_mapping=value.get("held_mapping",{})
        if len(train_mapping)!=value["train_objects"] or len(held_mapping)!=value["held_objects"]: return False
        train_ids=sorted(train_mapping); held_ids=sorted(held_mapping)
        expected_train={obj:train_ids[(i+1)%len(train_ids)] for i,obj in enumerate(train_ids)}
        expected_held={obj:train_ids[i%len(train_ids)] for i,obj in enumerate(held_ids)}
        if set(train_ids)&set(held_ids) or any(donor not in train_ids for donor in train_mapping.values()) or any(donor not in train_ids for donor in held_mapping.values()): return False
        object_hash=lambda ids:hashlib.sha256(("\n".join(ids)+"\n").encode()).hexdigest()
        if (train_mapping!=expected_train or held_mapping!=expected_held or
                value.get("train_object_hash")!=object_hash(train_ids) or value.get("held_object_hash")!=object_hash(held_ids) or
                not all(isinstance(value.get(key),str) and len(value[key])==64 for key in ("z_sha256","ols_beta_sha256"))): return False
        actual_self=sum(obj==donor for obj,donor in train_mapping.items())/len(train_ids)
        actual_load={obj:sum(donor==obj for donor in held_mapping.values()) for obj in train_ids}
        actual_effective=len({*held_mapping.values()})
        mapping=f"context={receipt['context']}\ndomain={domain}\n"+"\n".join(f"train:{obj}->{train_mapping[obj]}" for obj in train_ids)+"\n"+"\n".join(f"held:{obj}->{held_mapping[obj]}" for obj in held_ids)+"\n"
        if (value.get("mapping_sha256")!=hashlib.sha256(mapping.encode()).hexdigest() or
                value["train_coverage"]!=len(train_mapping)/value["train_objects"] or
                value["held_coverage"]!=len(held_mapping)/value["held_objects"] or
                value["train_self_rate"]!=actual_self or
                value["train_donor_marginal_exact"]!=(sorted(train_mapping.values())==train_ids) or
                value["held_effective_donors"]!=actual_effective or
                value["held_effective_donors_per_object"]!=actual_effective/len(held_ids) or
                value["held_max_donor_load"]!=max(actual_load.values()) or
                value["held_load_bound"]!=math.ceil(len(held_ids)/len(train_ids))): return False
        numeric=(value["reconstruction_max_error"],value["normalized_residual_mean_max"],
                 value["normalized_residual_z_correlation_max"],value["residual_energy_ratio"],
                 value["shuffle_rms_over_original_sd"],value["shuffled_norm_p99_ratio"],
                 value["crossfit"]["residual_norm_z_absolute_spearman"],
                 value["crossfit"]["residual_z_distance_correlation"])
        if not all(math.isfinite(float(item)) for item in numeric): return False
        counts=value["crossfit"].get("fold_counts",[])
        if (len(counts)!=int(cross["folds"]) or min(counts,default=0)<value["train_objects"]//int(cross["folds"]) or
                sum(counts)!=value["train_objects"]): return False
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
