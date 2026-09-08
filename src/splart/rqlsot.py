"""Deterministic rank-quadratic location-scale orthogonal transport null."""

from __future__ import annotations

import hashlib
import math

import torch
from torch import Tensor

from .conditional_residual import _distance_correlation, _rank, _spearman


SCHEMA="splart-rqlsot-null/v1"


def _sha_tensor(value: Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().double().numpy().tobytes()).hexdigest()


def _sha_ids(values: list[str]) -> str:
    return hashlib.sha256(("\n".join(values)+"\n").encode()).hexdigest()


def _row_key(row: dict) -> str:
    for key in ("gauge_id","row_id"):
        if key in row: return str(row[key])
    if "joint_id" in row and "order" in row: return f"{row['joint_id']}:{row['order']}"
    raise ValueError("stable gauge_id/row_id sidecar required for row-order invariance")


def empirical_rank(train_z: Tensor,held_z: Tensor | None=None) -> tuple[Tensor,Tensor | None,float]:
    """Train mid-CDF ranks and held train-ECDF ranks on [-1,1]."""
    train_z=train_z.double()
    if train_z.ndim!=1 or len(train_z)<3 or not torch.isfinite(train_z).all(): raise ValueError("invalid rank reference")
    n=len(train_z); train_x=2*((_rank(train_z)+.5)/n)-1
    if held_z is None: return train_x,None,0.
    held_z=held_z.double()
    if held_z.ndim!=1 or not torch.isfinite(held_z).all(): raise ValueError("invalid held rank values")
    raw=[]
    for value in held_z:
        less=(train_z<value).sum(); equal=(train_z==value).sum()
        raw.append((less.double()+.5*equal.double())/n)
    raw=torch.stack(raw); low=.5/n; high=1-.5/n
    clipped=((raw<low)|(raw>high)).double().mean().item()
    held_x=2*raw.clamp(low,high)-1
    return train_x,held_x,float(clipped)


def _positive_qr(design: Tensor,rank_tolerance: float,condition_max: float) -> tuple[Tensor,Tensor,dict]:
    design=design.double()
    if not torch.isfinite(design).all() or design.ndim!=2 or design.shape[0]<design.shape[1]: raise ValueError("invalid QR design")
    q,r=torch.linalg.qr(design,mode="reduced")
    diagonal=torch.diag(r); signs=torch.where(diagonal<0,-torch.ones_like(diagonal),torch.ones_like(diagonal))
    q=q*signs[None]; r=signs[:,None]*r
    singular=torch.linalg.svdvals(r); threshold=rank_tolerance*float(singular.max())
    rank=int((singular>threshold).sum()); condition=float(singular.max()/singular.min())
    if rank!=design.shape[1] or not math.isfinite(condition) or condition>condition_max: raise ValueError("RQ-LSOT QR rank/condition gate failed")
    if torch.any(torch.diag(r)<=0): raise RuntimeError("positive QR diagonal contract failed")
    return q,r,{"rank":rank,"condition":condition,"positive_diagonal":True,"rank_threshold":threshold}


def _basis(x: Tensor) -> Tensor:
    return torch.stack((torch.ones_like(x),x,(3*x.square()-1)/2),-1).double()


def _fit_location_scale(x: Tensor,means: Tensor,contract: dict) -> dict:
    means=means.double(); basis=_basis(x)
    q,r,mean_qr=_positive_qr(basis,float(contract["rank_relative_tolerance"]),float(contract["condition_max"]))
    beta=torch.linalg.solve_triangular(r,q.T@means,upper=True); location=basis@beta; residual=means-location
    centered=means-means.mean(0); centered_rms=float(centered.square().mean().sqrt())
    eps=max(1e-12,1e-6*centered_rms)
    radial=residual.square().mean(-1).sqrt(); log_radial=torch.log(radial+eps)
    scale_basis=torch.stack((torch.ones_like(x),x),-1)
    sq,sr,scale_qr=_positive_qr(scale_basis,float(contract["rank_relative_tolerance"]),float(contract["condition_max"]))
    gamma=torch.linalg.solve_triangular(sr,sq.T@log_radial[:,None],upper=True)[:,0]
    scale=torch.exp(scale_basis@gamma)
    if not torch.isfinite(scale).all() or torch.any(scale<=0): raise ValueError("nonfinite or nonpositive fitted scale")
    standardized=residual/scale[:,None]
    basis_orth=float((basis.T@residual/len(x)).abs().max())
    scale_orth=float((scale_basis.T@(log_radial-scale_basis@gamma)/len(x)).abs().max())
    return {"beta":beta,"gamma":gamma,"location":location,"scale":scale,"standardized":standardized,
            "residual":residual,"epsilon":eps,"mean_qr":mean_qr,"scale_qr":scale_qr,
            "basis_orthogonality_max":basis_orth,"scale_basis_orthogonality_max":scale_orth,
            "residual_energy_ratio":float(residual.square().mean()/centered.square().mean().clamp_min(1e-12))}


def _predict(model: dict,x: Tensor) -> tuple[Tensor,Tensor]:
    location=_basis(x)@model["beta"]
    scale=torch.exp(torch.stack((torch.ones_like(x),x),-1)@model["gamma"])
    if not torch.isfinite(location).all() or not torch.isfinite(scale).all() or torch.any(scale<=0): raise ValueError("nonfinite RQ-LSOT prediction")
    return location,scale


def _balanced_folds(object_ids: list[str],folds: int) -> list[int]:
    if len(set(object_ids))!=len(object_ids) or len(object_ids)<2*folds: raise ValueError("cross-fit needs two objects per fold")
    order=sorted(range(len(object_ids)),key=lambda i:(hashlib.sha256(("splart-rqlsot-crossfit-v1:"+object_ids[i]).encode()).hexdigest(),object_ids[i]))
    result=[-1]*len(object_ids)
    for rank,index in enumerate(order): result[index]=rank%folds
    return result


def crossfit_rqlsot(object_ids: list[str],z: Tensor,means: Tensor,contract: dict) -> dict:
    folds=int(contract["crossfit_folds"]); assignments=_balanced_folds(object_ids,folds)
    standardized=torch.full_like(means,float("nan"),dtype=torch.float64); x_oof=torch.full_like(z,float("nan"),dtype=torch.float64)
    clipped=[]
    for fold in range(folds):
        train=[i for i,value in enumerate(assignments) if value!=fold]; held=[i for i,value in enumerate(assignments) if value==fold]
        train_x,held_x,fraction=empirical_rank(z[train],z[held]); model=_fit_location_scale(train_x,means[train],contract)
        location,scale=_predict(model,held_x); standardized[held]=(means[held]-location)/scale[:,None]; x_oof[held]=held_x; clipped.append(fraction)
    if not torch.isfinite(standardized).all() or not torch.isfinite(x_oof).all(): raise RuntimeError("cross-fit output uninitialized")
    norm=standardized.norm(dim=-1); counts=[assignments.count(fold) for fold in range(folds)]
    return {"fold_counts":counts,"fold_boundary_clipped_fraction":clipped,
            "residual_norm_x_absolute_spearman":_spearman(norm,x_oof),
            "residual_x_distance_correlation":_distance_correlation(x_oof,standardized),
            "standardized_sha256":_sha_tensor(standardized),"x_sha256":_sha_tensor(x_oof)}


def rqlsot_ablation(rows: list[dict],train_indices: list[int],held_indices: list[int],
                     train_field: Tensor,held_field: Tensor,train_z: dict[str,float],held_z: dict[str,float],
                     contract: dict,context: str) -> tuple[Tensor,Tensor,dict]:
    if train_field.shape[0]!=len(train_indices) or held_field.shape[0]!=len(held_indices) or train_field.shape[1:]!=held_field.shape[1:]: raise ValueError("field alignment mismatch")
    if train_field.ndim!=2 or not train_indices or not held_indices or not context: raise ValueError("invalid RQ-LSOT input")
    train_domains={rows[i]["domain"] for i in train_indices}; held_domains={rows[i]["domain"] for i in held_indices}
    if train_domains!=held_domains: raise ValueError("held/train domain sets differ")
    ownership={}; keys=set()
    for split,indices in (("train",train_indices),("held",held_indices)):
        for i in indices:
            obj=rows[i]["object_group_id"]; owner=(rows[i]["domain"],split); key=_row_key(rows[i])
            if obj in ownership and ownership[obj]!=owner: raise ValueError("object ID crosses domain or split")
            if key in keys: raise ValueError("row key is not unique")
            ownership[obj]=owner; keys.add(key)
    train_out=torch.full_like(train_field,float("nan"),dtype=torch.float64); held_out=torch.full_like(held_field,float("nan"),dtype=torch.float64)
    receipt={"schema":SCHEMA,"context":context,"contract_sha256":hashlib.sha256(str(sorted(contract.items())).encode()).hexdigest(),"domains":{}}
    for domain in sorted(train_domains):
        train_objects=sorted({rows[i]["object_group_id"] for i in train_indices if rows[i]["domain"]==domain})
        held_objects=sorted({rows[i]["object_group_id"] for i in held_indices if rows[i]["domain"]==domain})
        if not train_objects or not held_objects: raise ValueError("each domain needs train and held objects")
        def locations(indices,objects):
            return {obj:sorted([k for k,i in enumerate(indices) if rows[i]["object_group_id"]==obj],key=lambda k:_row_key(rows[indices[k]])) for obj in objects}
        train_locations=locations(train_indices,train_objects); held_locations=locations(held_indices,held_objects)
        means=torch.stack([train_field[train_locations[obj]].double().mean(0) for obj in train_objects])
        held_means=torch.stack([held_field[held_locations[obj]].double().mean(0) for obj in held_objects])
        z=torch.tensor([train_z[obj] for obj in train_objects],dtype=torch.float64); hz=torch.tensor([held_z[obj] for obj in held_objects],dtype=torch.float64)
        x,hx,held_clipped=empirical_rank(z,hz); model=_fit_location_scale(x,means,contract); held_location,held_scale=_predict(model,hx)
        offsets={obj:train_field[train_locations[obj]].double()-means[j] for j,obj in enumerate(train_objects)}
        held_offsets={obj:held_field[held_locations[obj]].double()-held_means[j] for j,obj in enumerate(held_objects)}
        reconstruction=torch.cat([model["location"][j][None]+model["scale"][j]*model["standardized"][j][None]+offsets[obj] for j,obj in enumerate(train_objects)])
        original=torch.cat([train_field[train_locations[obj]].double() for obj in train_objects])
        train_donor={obj:train_objects[(j+1)%len(train_objects)] for j,obj in enumerate(train_objects)}
        held_donor={obj:train_objects[j%len(train_objects)] for j,obj in enumerate(held_objects)}
        standardized={obj:model["standardized"][j] for j,obj in enumerate(train_objects)}
        for j,obj in enumerate(train_objects): train_out[train_locations[obj]]=model["location"][j][None]+model["scale"][j]*standardized[train_donor[obj]][None]+offsets[obj]
        for j,obj in enumerate(held_objects): held_out[held_locations[obj]]=held_location[j][None]+held_scale[j]*standardized[held_donor[obj]][None]+held_offsets[obj]
        shuffled_canonical=torch.cat([train_out[train_locations[obj]] for obj in train_objects])
        held_canonical=torch.cat([held_out[held_locations[obj]] for obj in held_objects])
        original_sd=original.std(unbiased=False).clamp_min(1e-12)
        p99=float(torch.quantile(original.norm(dim=-1),.99).clamp_min(1e-12))
        shuffled_p99=max(float(torch.quantile(shuffled_canonical.norm(dim=-1),.99)),float(torch.quantile(held_canonical.norm(dim=-1),.99)))
        crossfit=crossfit_rqlsot(train_objects,z,means,contract)
        mapping=f"context={context}\ndomain={domain}\n"+"\n".join(f"train:{k}->{train_donor[k]}" for k in train_objects)+"\n"+"\n".join(f"held:{k}->{held_donor[k]}" for k in held_objects)+"\n"
        receipt["domains"][domain]={"train_objects":len(train_objects),"held_objects":len(held_objects),"train_object_hash":_sha_ids(train_objects),"held_object_hash":_sha_ids(held_objects),
            "train_mapping":train_donor,"held_mapping":held_donor,"mapping_sha256":hashlib.sha256(mapping.encode()).hexdigest(),"z_sha256":_sha_tensor(z),"train_rank_x_sha256":_sha_tensor(x),
            "mean_beta_sha256":_sha_tensor(model["beta"]),"scale_gamma_sha256":_sha_tensor(model["gamma"]),"epsilon":model["epsilon"],"scale_clipped":False,
            "mean_qr":model["mean_qr"],"scale_qr":model["scale_qr"],"basis_orthogonality_max":model["basis_orthogonality_max"],"scale_basis_orthogonality_max":model["scale_basis_orthogonality_max"],
            "reconstruction_max_error":float((reconstruction-original).abs().max()),"held_boundary_clipped_fraction":held_clipped,
            "residual_energy_ratio":model["residual_energy_ratio"],"shuffle_rms_over_original_sd":float((shuffled_canonical-original).square().mean().sqrt()/original_sd),
            "shuffled_norm_p99_ratio":shuffled_p99/p99,"train_coverage":len(train_donor)/len(train_objects),"held_coverage":len(held_donor)/len(held_objects),
            "train_self_rate":sum(k==v for k,v in train_donor.items())/len(train_objects),"train_donor_marginal_exact":sorted(train_donor.values())==train_objects,
            "held_effective_donors_per_object":len(set(held_donor.values()))/len(held_objects),"held_max_donor_load":max(sum(v==donor for v in held_donor.values()) for donor in train_objects),
            "held_load_bound":math.ceil(len(held_objects)/len(train_objects)),"crossfit":crossfit}
    if not torch.isfinite(train_out).all() or not torch.isfinite(held_out).all(): raise RuntimeError("RQ-LSOT output uninitialized")
    return train_out,held_out,receipt


def receipt_passes(receipt: dict,contract: dict,contexts: set[str],expected_domains: set[str]) -> bool:
    if receipt.get("schema")!=SCHEMA or receipt.get("context") not in contexts or set(receipt.get("domains",{}))!=expected_domains: return False
    for value in receipt["domains"].values():
        train=value.get("train_mapping",{}); held=value.get("held_mapping",{}); ids=sorted(train); held_ids=sorted(held)
        if not ids or not held_ids: return False
        if train!={obj:ids[(i+1)%len(ids)] for i,obj in enumerate(ids)} or held!={obj:ids[i%len(ids)] for i,obj in enumerate(held_ids)}: return False
        if value.get("train_object_hash")!=_sha_ids(ids) or value.get("held_object_hash")!=_sha_ids(held_ids): return False
        numeric=[value.get(k,float("nan")) for k in ("basis_orthogonality_max","scale_basis_orthogonality_max","reconstruction_max_error","held_boundary_clipped_fraction","residual_energy_ratio","shuffle_rms_over_original_sd","shuffled_norm_p99_ratio")]
        cf=value.get("crossfit",{}); numeric += [cf.get("residual_norm_x_absolute_spearman",float("nan")),cf.get("residual_x_distance_correlation",float("nan"))]
        if not all(math.isfinite(float(v)) for v in numeric): return False
        if (value["mean_qr"]["rank"]!=3 or value["scale_qr"]["rank"]!=2 or not value["mean_qr"]["positive_diagonal"] or not value["scale_qr"]["positive_diagonal"] or
            value["mean_qr"]["condition"]>contract["condition_max"] or value["scale_qr"]["condition"]>contract["condition_max"] or value["scale_clipped"] is not False or
            value["basis_orthogonality_max"]>contract["orthogonality_max"] or value["scale_basis_orthogonality_max"]>contract["orthogonality_max"] or value["reconstruction_max_error"]>contract["reconstruction_max_error"] or
            value["held_boundary_clipped_fraction"]>contract["held_boundary_clipped_fraction_max"] or value["residual_energy_ratio"]<contract["residual_energy_ratio_at_least"] or
            value["shuffle_rms_over_original_sd"]<contract["shuffle_rms_over_original_sd_at_least"] or value["shuffled_norm_p99_ratio"]>contract["shuffled_norm_p99_ratio_at_most"] or
            value["train_coverage"]!=1 or value["held_coverage"]!=1 or value["train_self_rate"]!=0 or not value["train_donor_marginal_exact"] or
            value["held_effective_donors_per_object"]!=1 or value["held_max_donor_load"]>value["held_load_bound"] or
            cf["residual_norm_x_absolute_spearman"]>contract["crossfit_absolute_spearman_at_most"] or cf["residual_x_distance_correlation"]>contract["crossfit_distance_correlation_at_most"]): return False
    return True


def preserve_recipient_displacement(shuffled_geometry: Tensor,recipient_mechanical: Tensor) -> Tensor:
    if shuffled_geometry.shape[:-1]!=recipient_mechanical.shape[:-1] or shuffled_geometry.shape[-1]+1!=recipient_mechanical.shape[-1]: raise ValueError("mechanical geometry/displacement shape mismatch")
    result=torch.cat((shuffled_geometry,recipient_mechanical[...,-1:]),-1)
    if not torch.equal(result[...,-1],recipient_mechanical[...,-1]): raise RuntimeError("recipient displacement changed")
    return result


__all__=["SCHEMA","crossfit_rqlsot","empirical_rank","preserve_recipient_displacement","receipt_passes","rqlsot_ablation"]
