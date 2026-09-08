"""Deterministic rank-quadratic location-scale orthogonal transport null."""

from __future__ import annotations

import hashlib
import json
import math

import torch
from torch import Tensor

from .conditional_residual import _distance_correlation, _rank, _spearman
from .smarc_source import fit_feature_preprocessor, transform


SCHEMA="splart-rqlsot-null/v1"
CROSSFIT_SCHEMA="splart-rqlsot-fold-local-crossfit/v1"


def _sha_tensor(value: Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().double().numpy().tobytes()).hexdigest()


def _sha_ids(values: list[str]) -> str:
    return hashlib.sha256(("\n".join(values)+"\n").encode()).hexdigest()


def _canonical_sha(value) -> str:
    return hashlib.sha256((json.dumps(value,sort_keys=True,separators=(",",":"))+"\n").encode()).hexdigest()


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


def _object_scalar(rows: list[dict],indices: list[int],kind: str,reference: list[int] | None=None) -> dict[str,float]:
    objects=sorted({rows[i]["object_group_id"] for i in indices}); result={}
    if kind=="semantic":
        for obj in objects:
            values=[abs(float(rows[i]["observed_displacement"])) for i in indices if rows[i]["object_group_id"]==obj]
            result[obj]=sum(values)/len(values)
        return result
    if kind!="mechanical": raise ValueError("unknown RQ-LSOT field kind")
    reference=indices if reference is None else reference
    domains={rows[i]["domain"] for i in indices}
    for domain in domains:
        ref_objects=sorted({rows[i]["object_group_id"] for i in reference if rows[i]["domain"]==domain})
        ref_means=[torch.stack([rows[i]["semantic"].double() for i in reference if rows[i]["object_group_id"]==obj]).mean(0) for obj in ref_objects]
        reference_mean=torch.stack(ref_means).mean(0)
        for obj in [obj for obj in objects if any(rows[i]["object_group_id"]==obj and rows[i]["domain"]==domain for i in indices)]:
            value=torch.stack([rows[i]["semantic"].double() for i in indices if rows[i]["object_group_id"]==obj]).mean(0)
            result[obj]=float(1-torch.nn.functional.cosine_similarity(value[None],reference_mean[None]).item())
    return result


def _preprocessor_parts(preprocessor) -> dict:
    return {"mechanical_mean":_sha_tensor(preprocessor.mechanical_mean),"mechanical_scale":_sha_tensor(preprocessor.mechanical_scale),
             "mechanical_keep":hashlib.sha256(preprocessor.mechanical_keep.cpu().numpy().tobytes()).hexdigest(),
             "semantic_mean":_sha_tensor(preprocessor.semantic_mean),"semantic_components":_sha_tensor(preprocessor.semantic_components),
             "train_object_hash":preprocessor.train_object_hash,"global_prior_logit":preprocessor.global_prior_logit}


def _preprocessor_sha(preprocessor) -> str:
    return _canonical_sha(_preprocessor_parts(preprocessor))


def feature_crossfit_rqlsot(rows: list[dict],train_indices: list[int],field_kind: str,contract: dict) -> dict[str,dict]:
    """OOF diagnostics; each fold refits the joint-domain feature preprocessor.

    The feature preprocessor is fit on both source domains with domain-balanced
    weights, exactly like the final transform.  Conditional RQ fits remain
    domain-specific.  No held object participates in PCA/standardization/ECDF.
    """
    folds=int(contract["crossfit_folds"]); domains=sorted({rows[i]["domain"] for i in train_indices})
    domain_objects={d:sorted({rows[i]["object_group_id"] for i in train_indices if rows[i]["domain"]==d}) for d in domains}
    assignment={d:{obj:value for obj,value in zip(domain_objects[d],_balanced_folds(domain_objects[d],folds))} for d in domains}
    raw_geometry_dim=int(rows[train_indices[0]]["mechanical"].numel())-1
    raw_semantic_dim=int(rows[train_indices[0]]["semantic"].numel())
    outputs={d:{"standardized":torch.full((len(domain_objects[d]),raw_semantic_dim),float("nan"),dtype=torch.float64) if field_kind=="semantic" else torch.zeros((len(domain_objects[d]),raw_geometry_dim),dtype=torch.float64),
                "rank_x":torch.full((len(domain_objects[d]),),float("nan"),dtype=torch.float64),"folds":[]} for d in domains}
    canonical=lambda selected:sorted([i for i in train_indices if rows[i]["object_group_id"] in selected],key=lambda i:(rows[i]["domain"],rows[i]["object_group_id"],_row_key(rows[i])))
    for fold in range(folds):
        fit_set={obj for d in domains for obj in domain_objects[d] if assignment[d][obj]!=fold}
        held_set={obj for d in domains for obj in domain_objects[d] if assignment[d][obj]==fold}
        fit_indices=canonical(fit_set); held_indices=canonical(held_set)
        preprocessor=fit_feature_preprocessor(rows,fit_indices,16)
        fit_mech,fit_sem,_=transform(preprocessor,rows,fit_indices); held_mech,held_sem,_=transform(preprocessor,rows,held_indices)
        if field_kind=="semantic": fit_field,held_field=fit_sem,held_sem; kept=None
        else:
            kept=torch.nonzero(preprocessor.mechanical_keep).flatten().tolist()
            if not kept or kept[-1]!=len(preprocessor.mechanical_keep)-1: raise ValueError("fold mechanical displacement column missing")
            fit_field,held_field=fit_mech[:,:-1],held_mech[:,:-1]; kept=kept[:-1]
        prep_common={"fold":fold,"fit_object_hash":_sha_ids(sorted(fit_set)),"held_object_hash":_sha_ids(sorted(held_set)),
                     "preprocessor_sha256":_preprocessor_sha(preprocessor),"preprocessor_train_object_hash":preprocessor.train_object_hash,
                     "preprocessor_provenance":_preprocessor_parts(preprocessor),
                     "raw_mechanical_dim":len(preprocessor.mechanical_keep),
                     "mechanical_kept_indices":torch.nonzero(preprocessor.mechanical_keep).flatten().tolist(),
                     "mechanical_keep_mask":[bool(v) for v in preprocessor.mechanical_keep.tolist()],
                     "mechanical_keep_sha256":hashlib.sha256(preprocessor.mechanical_keep.cpu().numpy().tobytes()).hexdigest()}
        for domain in domains:
            fit_objects=[obj for obj in domain_objects[domain] if assignment[domain][obj]!=fold]
            held_objects=[obj for obj in domain_objects[domain] if assignment[domain][obj]==fold]
            fi=[k for k,i in enumerate(fit_indices) if rows[i]["domain"]==domain]
            hi=[k for k,i in enumerate(held_indices) if rows[i]["domain"]==domain]
            def object_means(indices,locations,field,selected):
                return torch.stack([field[[k for k in locations if rows[indices[k]]["object_group_id"]==obj]].mean(0) for obj in selected])
            fit_means=object_means(fit_indices,fi,fit_field,fit_objects); held_means=object_means(held_indices,hi,held_field,held_objects)
            domain_fit_indices=[fit_indices[k] for k in fi]; domain_held_indices=[held_indices[k] for k in hi]
            fit_z_map=_object_scalar(rows,domain_fit_indices,field_kind,domain_fit_indices)
            held_z_map=_object_scalar(rows,domain_held_indices,field_kind,domain_fit_indices)
            fit_z=torch.tensor([fit_z_map[obj] for obj in fit_objects],dtype=torch.float64); held_z=torch.tensor([held_z_map[obj] for obj in held_objects],dtype=torch.float64)
            fit_x,held_x,clipped=empirical_rank(fit_z,held_z); model=_fit_location_scale(fit_x,fit_means,contract); location,scale=_predict(model,held_x)
            local=(held_means-location)/scale[:,None]
            for j,obj in enumerate(held_objects):
                target=domain_objects[domain].index(obj); outputs[domain]["rank_x"][target]=held_x[j]
                if field_kind=="semantic": outputs[domain]["standardized"][target]=local[j]@preprocessor.semantic_components
                else: outputs[domain]["standardized"][target,kept]=local[j]
            outputs[domain]["folds"].append({**prep_common,"domain_fit_object_hash":_sha_ids(fit_objects),"domain_held_object_hash":_sha_ids(held_objects),
                "boundary_clipped_fraction":clipped,"mean_qr":model["mean_qr"],"scale_qr":model["scale_qr"]})
    result={}
    for domain in domains:
        objects=domain_objects[domain]; standardized=outputs[domain]["standardized"]; x_oof=outputs[domain]["rank_x"]
        if not torch.isfinite(standardized).all() or not torch.isfinite(x_oof).all(): raise RuntimeError("fold-local crossfit output uninitialized")
        unified_z_map=_object_scalar(rows,[i for i in train_indices if rows[i]["domain"]==domain],field_kind,
                                     [i for i in train_indices if rows[i]["domain"]==domain])
        unified_z=torch.tensor([unified_z_map[obj] for obj in objects],dtype=torch.float64); norm=standardized.norm(dim=-1)
        assign=assignment[domain]; assignment_hash=_canonical_sha(assign)
        payload={"object_ids":objects,"fold_assignment":assign,"standardized":standardized.tolist(),"raw_z":unified_z.tolist(),"rank_x":x_oof.tolist()}
        result[domain]={"schema":CROSSFIT_SCHEMA,"field_kind":field_kind,"fold_counts":[list(assign.values()).count(f) for f in range(folds)],
            "fold_assignment_sha256":assignment_hash,"folds":outputs[domain]["folds"],
            "raw_z_diagnostic_scope":"one unified original raw-z per domain across all OOF objects",
            "fold_fit_z_scope":"fold-train raw-z is independently ECDF-ranked; held uses that fold train ECDF",
            "residual_norm_raw_z_absolute_spearman":_spearman(norm,unified_z),"residual_raw_z_distance_correlation":_distance_correlation(unified_z,standardized),
            "residual_norm_rank_x_absolute_spearman_extra":_spearman(norm,x_oof),"residual_rank_x_distance_correlation_extra":_distance_correlation(x_oof,standardized),
            "standardized_sha256":_sha_tensor(standardized),"raw_z_sha256":_sha_tensor(unified_z),"rank_x_sha256":_sha_tensor(x_oof),"audit_payload":payload}
    return result


def rqlsot_ablation(rows: list[dict],train_indices: list[int],held_indices: list[int],
                     train_field: Tensor,held_field: Tensor,train_z: dict[str,float],held_z: dict[str,float],
                     contract: dict,context: str,crossfit_by_domain: dict[str,dict]) -> tuple[Tensor,Tensor,dict]:
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
    receipt={"schema":SCHEMA,"context":context,"contract_sha256":_canonical_sha(contract),"domains":{}}
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
        train_u=max(float(((train_out[train_locations[obj]]-train_out[train_locations[obj]].mean(0))-
                           (train_field[train_locations[obj]].double()-train_field[train_locations[obj]].double().mean(0))).abs().max()) for obj in train_objects)
        held_u=max(float(((held_out[held_locations[obj]]-held_out[held_locations[obj]].mean(0))-
                          (held_field[held_locations[obj]].double()-held_field[held_locations[obj]].double().mean(0))).abs().max()) for obj in held_objects)
        original_sd=original.std(unbiased=False).clamp_min(1e-12)
        p99=float(torch.quantile(original.norm(dim=-1),.99).clamp_min(1e-12))
        shuffled_p99=max(float(torch.quantile(shuffled_canonical.norm(dim=-1),.99)),float(torch.quantile(held_canonical.norm(dim=-1),.99)))
        if domain not in crossfit_by_domain: raise ValueError("fold-local crossfit receipt missing domain")
        crossfit=crossfit_by_domain[domain]
        field_scale=means.std(0,unbiased=False).clamp_min(1e-12); centered_z=z-z.mean()
        normalized_mean=float((model["residual"].mean(0).abs()/field_scale).max())
        raw_z_correlation=float(((centered_z[:,None]*model["residual"]).mean(0).abs()/(centered_z.std(unbiased=False)*field_scale).clamp_min(1e-12)).max())
        mapping_payload={"context":context,"domain":domain,"train":train_donor,"held":held_donor}
        audit_payload={"train_object_ids":train_objects,"held_object_ids":held_objects,"z":z.tolist(),"train_rank_x":x.tolist(),
                       "mean_beta":model["beta"].tolist(),"scale_gamma":model["gamma"].tolist()}
        receipt["domains"][domain]={"train_objects":len(train_objects),"held_objects":len(held_objects),"train_object_hash":_sha_ids(train_objects),"held_object_hash":_sha_ids(held_objects),
            "train_mapping":train_donor,"held_mapping":held_donor,"mapping_sha256":_canonical_sha(mapping_payload),"z_sha256":_sha_tensor(z),"train_rank_x_sha256":_sha_tensor(x),
            "mean_beta_sha256":_sha_tensor(model["beta"]),"scale_gamma_sha256":_sha_tensor(model["gamma"]),"epsilon":model["epsilon"],"scale_clipped":False,
            "mean_qr":model["mean_qr"],"scale_qr":model["scale_qr"],"basis_orthogonality_max":model["basis_orthogonality_max"],"scale_basis_orthogonality_max":model["scale_basis_orthogonality_max"],
            "reconstruction_max_error":float((reconstruction-original).abs().max()),"normalized_residual_mean_max":normalized_mean,
            "normalized_residual_raw_z_correlation_max":raw_z_correlation,"held_boundary_clipped_fraction":held_clipped,
            "residual_energy_ratio":model["residual_energy_ratio"],"shuffle_rms_over_original_sd":float((shuffled_canonical-original).square().mean().sqrt()/original_sd),
            "shuffled_norm_p99_ratio":shuffled_p99/p99,"train_coverage":len(train_donor)/len(train_objects),"held_coverage":len(held_donor)/len(held_objects),
            "train_self_rate":sum(k==v for k,v in train_donor.items())/len(train_objects),"train_donor_marginal_exact":sorted(train_donor.values())==train_objects,
            "held_effective_donors_per_object":len(set(held_donor.values()))/len(held_objects),"held_max_donor_load":max(sum(v==donor for v in held_donor.values()) for donor in train_objects),
            "held_load_bound":math.ceil(len(held_objects)/len(train_objects)),"recipient_u_train_max_error":train_u,"recipient_u_held_max_error":held_u,
            "crossfit":crossfit,"audit_payload":audit_payload}
    if not torch.isfinite(train_out).all() or not torch.isfinite(held_out).all(): raise RuntimeError("RQ-LSOT output uninitialized")
    return train_out,held_out,receipt


def _qr_valid(value: dict,rank: int,contract: dict) -> bool:
    if set(value)!={"rank","condition","positive_diagonal","rank_threshold"}: return False
    numeric=(value.get("condition"),value.get("rank_threshold"))
    return (value.get("rank")==rank and value.get("positive_diagonal") is True and
            all(isinstance(x,(int,float)) and math.isfinite(float(x)) for x in numeric) and
            0<=float(value["condition"])<=float(contract["condition_max"]) and float(value["rank_threshold"])>=0)


def _crossfit_valid(value: dict,contract: dict,expected_field_kind: str) -> bool:
    expected={"schema","field_kind","fold_counts","fold_assignment_sha256","folds","raw_z_diagnostic_scope","fold_fit_z_scope","residual_norm_raw_z_absolute_spearman",
              "residual_raw_z_distance_correlation","residual_norm_rank_x_absolute_spearman_extra",
              "residual_rank_x_distance_correlation_extra","standardized_sha256","raw_z_sha256","rank_x_sha256","audit_payload"}
    if set(value)!=expected or value.get("schema")!=CROSSFIT_SCHEMA or value.get("field_kind")!=expected_field_kind: return False
    if value.get("raw_z_diagnostic_scope")!="one unified original raw-z per domain across all OOF objects" or value.get("fold_fit_z_scope")!="fold-train raw-z is independently ECDF-ranked; held uses that fold train ECDF": return False
    payload=value.get("audit_payload",{})
    if set(payload)!={"object_ids","fold_assignment","standardized","raw_z","rank_x"}: return False
    ids=payload["object_ids"]
    if ids!=sorted(ids) or len(ids)!=len(set(ids)) or not ids: return False
    expected_assignment={obj:fold for obj,fold in zip(ids,_balanced_folds(ids,int(contract["crossfit_folds"])))}
    if payload["fold_assignment"]!=expected_assignment or value["fold_assignment_sha256"]!=_canonical_sha(expected_assignment): return False
    counts=[list(expected_assignment.values()).count(i) for i in range(int(contract["crossfit_folds"]))]
    if value["fold_counts"]!=counts or any(count<2 for count in counts): return False
    try:
        standardized=torch.tensor(payload["standardized"],dtype=torch.float64)
        raw_z=torch.tensor(payload["raw_z"],dtype=torch.float64); rank_x=torch.tensor(payload["rank_x"],dtype=torch.float64)
    except Exception: return False
    if standardized.ndim!=2 or standardized.shape[0]!=len(ids) or raw_z.shape!=(len(ids),) or rank_x.shape!=(len(ids),): return False
    if not torch.isfinite(standardized).all() or not torch.isfinite(raw_z).all() or not torch.isfinite(rank_x).all(): return False
    if value["standardized_sha256"]!=_sha_tensor(standardized) or value["raw_z_sha256"]!=_sha_tensor(raw_z) or value["rank_x_sha256"]!=_sha_tensor(rank_x): return False
    norm=standardized.norm(dim=-1)
    recomputed={"residual_norm_raw_z_absolute_spearman":_spearman(norm,raw_z),
                "residual_raw_z_distance_correlation":_distance_correlation(raw_z,standardized),
                "residual_norm_rank_x_absolute_spearman_extra":_spearman(norm,rank_x),
                "residual_rank_x_distance_correlation_extra":_distance_correlation(rank_x,standardized)}
    if any(not math.isfinite(float(value[k])) or abs(float(value[k])-float(v))>1e-12 for k,v in recomputed.items()): return False
    folds=value["folds"]
    fold_keys={"fold","fit_object_hash","held_object_hash","preprocessor_sha256","preprocessor_train_object_hash","preprocessor_provenance",
               "raw_mechanical_dim","mechanical_kept_indices","mechanical_keep_mask",
               "mechanical_keep_sha256","domain_fit_object_hash","domain_held_object_hash","boundary_clipped_fraction","mean_qr","scale_qr"}
    if not isinstance(folds,list) or len(folds)!=int(contract["crossfit_folds"]): return False
    for fold_record in folds:
        if set(fold_record)!=fold_keys or fold_record.get("fold") not in range(int(contract["crossfit_folds"])): return False
        fold=int(fold_record["fold"]); fit=[obj for obj in ids if expected_assignment[obj]!=fold]; held=[obj for obj in ids if expected_assignment[obj]==fold]
        if fold_record["domain_fit_object_hash"]!=_sha_ids(fit) or fold_record["domain_held_object_hash"]!=_sha_ids(held): return False
        if fold_record["preprocessor_train_object_hash"]!=fold_record["fit_object_hash"]: return False
        raw_dim=fold_record["raw_mechanical_dim"]; mask=fold_record["mechanical_keep_mask"]; kept=fold_record["mechanical_kept_indices"]
        if not isinstance(raw_dim,int) or raw_dim<2 or not isinstance(mask,list) or len(mask)!=raw_dim or any(type(v) is not bool for v in mask): return False
        if kept!=[i for i,v in enumerate(mask) if v] or not kept or kept[-1]!=raw_dim-1: return False
        mask_tensor=torch.tensor(mask,dtype=torch.bool)
        if fold_record["mechanical_keep_sha256"]!=hashlib.sha256(mask_tensor.numpy().tobytes()).hexdigest(): return False
        parts=fold_record["preprocessor_provenance"]
        if set(parts)!={"mechanical_mean","mechanical_scale","mechanical_keep","semantic_mean","semantic_components","train_object_hash","global_prior_logit"}: return False
        if parts["train_object_hash"]!=fold_record["fit_object_hash"] or parts["global_prior_logit"]!=0. or fold_record["preprocessor_sha256"]!=_canonical_sha(parts): return False
        if any(not isinstance(parts[key],str) or len(parts[key])!=64 for key in ("mechanical_mean","mechanical_scale","mechanical_keep","semantic_mean","semantic_components","train_object_hash")): return False
        for key in ("fit_object_hash","held_object_hash","preprocessor_sha256","preprocessor_train_object_hash","mechanical_keep_sha256","domain_fit_object_hash","domain_held_object_hash"):
            if not isinstance(fold_record[key],str) or len(fold_record[key])!=64: return False
        if not math.isfinite(float(fold_record["boundary_clipped_fraction"])) or not 0<=fold_record["boundary_clipped_fraction"]<=1: return False
        if not _qr_valid(fold_record["mean_qr"],3,contract) or not _qr_valid(fold_record["scale_qr"],2,contract): return False
    return True


def _receipt_passes_checked(receipt: dict,expected_receipt: dict,contract: dict,expected_context: str,expected_domains: set[str]) -> bool:
    # The expected receipt must come from an independent deterministic recomputation.
    # This binding rejects coordinated edits of a numeric payload and its self-hash.
    if receipt is expected_receipt or _canonical_sha(receipt)!=_canonical_sha(expected_receipt): return False
    if set(receipt)!={"schema","context","contract_sha256","domains"}: return False
    if receipt.get("schema")!=SCHEMA or receipt.get("context")!=expected_context or receipt.get("contract_sha256")!=_canonical_sha(contract): return False
    if set(receipt.get("domains",{}))!=expected_domains: return False
    domain_keys={"train_objects","held_objects","train_object_hash","held_object_hash","train_mapping","held_mapping","mapping_sha256",
        "z_sha256","train_rank_x_sha256","mean_beta_sha256","scale_gamma_sha256","epsilon","scale_clipped","mean_qr","scale_qr",
        "basis_orthogonality_max","scale_basis_orthogonality_max","reconstruction_max_error","normalized_residual_mean_max",
        "normalized_residual_raw_z_correlation_max","held_boundary_clipped_fraction","residual_energy_ratio","shuffle_rms_over_original_sd",
        "shuffled_norm_p99_ratio","train_coverage","held_coverage","train_self_rate","train_donor_marginal_exact",
        "held_effective_donors_per_object","held_max_donor_load","held_load_bound","recipient_u_train_max_error","recipient_u_held_max_error",
        "crossfit","audit_payload"}
    expected_field_kind="semantic" if expected_context.startswith("semantic_rqlsot/") else "mechanical" if expected_context.startswith("mechanical_rqlsot/") else None
    if expected_field_kind is None: return False
    # Cross-domain folds share one joint Articraft+NJC preprocessor.  Validate
    # that its union split and every preprocessing artifact are identical.
    if len(expected_domains)>1:
        ordered=sorted(expected_domains); reference=receipt["domains"][ordered[0]].get("crossfit",{}).get("folds",[])
        if sorted(f.get("fold") for f in reference)!=list(range(int(contract["crossfit_folds"]))): return False
        reference={f["fold"]:f for f in reference}
        common=("fit_object_hash","held_object_hash","preprocessor_sha256","preprocessor_train_object_hash","preprocessor_provenance",
                "raw_mechanical_dim","mechanical_kept_indices","mechanical_keep_mask","mechanical_keep_sha256")
        for domain in ordered:
            folds=receipt["domains"][domain].get("crossfit",{}).get("folds",[])
            if sorted(f.get("fold") for f in folds)!=list(range(int(contract["crossfit_folds"]))): return False
            by_fold={f["fold"]:f for f in folds}
            if len(by_fold)!=int(contract["crossfit_folds"]): return False
            for fold in range(int(contract["crossfit_folds"])):
                if any(by_fold[fold].get(key)!=reference[fold].get(key) for key in common): return False
    for domain,value in receipt["domains"].items():
        if set(value)!=domain_keys: return False
        payload=value.get("audit_payload",{})
        if set(payload)!={"train_object_ids","held_object_ids","z","train_rank_x","mean_beta","scale_gamma"}: return False
        payload_ids=payload["train_object_ids"]; payload_held_ids=payload["held_object_ids"]
        if payload_ids!=sorted(payload_ids) or payload_held_ids!=sorted(payload_held_ids) or not payload_ids or not payload_held_ids or set(payload_ids)&set(payload_held_ids): return False
        train=value.get("train_mapping",{}); held=value.get("held_mapping",{}); ids=sorted(train); held_ids=sorted(held)
        if payload_ids!=ids or payload_held_ids!=held_ids: return False
        if not ids or not held_ids: return False
        if train!={obj:ids[(i+1)%len(ids)] for i,obj in enumerate(ids)} or held!={obj:ids[i%len(ids)] for i,obj in enumerate(held_ids)}: return False
        if value.get("train_object_hash")!=_sha_ids(ids) or value.get("held_object_hash")!=_sha_ids(held_ids): return False
        if value.get("train_objects")!=len(ids) or value.get("held_objects")!=len(held_ids): return False
        if value.get("mapping_sha256")!=_canonical_sha({"context":expected_context,"domain":domain,"train":train,"held":held}): return False
        try:
            z=torch.tensor(payload["z"],dtype=torch.float64); x=torch.tensor(payload["train_rank_x"],dtype=torch.float64)
            beta=torch.tensor(payload["mean_beta"],dtype=torch.float64); gamma=torch.tensor(payload["scale_gamma"],dtype=torch.float64)
        except Exception: return False
        if z.shape!=(len(ids),) or x.shape!=(len(ids),) or beta.ndim!=2 or beta.shape[0]!=3 or gamma.shape!=(2,): return False
        if not all(torch.isfinite(t).all() for t in (z,x,beta,gamma)): return False
        if value["z_sha256"]!=_sha_tensor(z) or value["train_rank_x_sha256"]!=_sha_tensor(x) or value["mean_beta_sha256"]!=_sha_tensor(beta) or value["scale_gamma_sha256"]!=_sha_tensor(gamma): return False
        if not _qr_valid(value["mean_qr"],3,contract) or not _qr_valid(value["scale_qr"],2,contract): return False
        numeric=[value.get(k,float("nan")) for k in ("basis_orthogonality_max","scale_basis_orthogonality_max","reconstruction_max_error","normalized_residual_mean_max","normalized_residual_raw_z_correlation_max","held_boundary_clipped_fraction","residual_energy_ratio","shuffle_rms_over_original_sd","shuffled_norm_p99_ratio")]
        numeric += [value.get("recipient_u_train_max_error",float("nan")),value.get("recipient_u_held_max_error",float("nan"))]
        cf=value.get("crossfit",{}); numeric += [cf.get("residual_norm_raw_z_absolute_spearman",float("nan")),cf.get("residual_raw_z_distance_correlation",float("nan"))]
        if not all(math.isfinite(float(v)) for v in numeric): return False
        if (value["scale_clipped"] is not False or
            value["basis_orthogonality_max"]>contract["orthogonality_max"] or value["scale_basis_orthogonality_max"]>contract["orthogonality_max"] or value["reconstruction_max_error"]>contract["reconstruction_max_error"] or
            value["normalized_residual_mean_max"]>contract["normalized_residual_mean_max"] or value["normalized_residual_raw_z_correlation_max"]>contract["normalized_residual_raw_z_correlation_max"] or
            value["held_boundary_clipped_fraction"]>contract["held_boundary_clipped_fraction_max"] or value["residual_energy_ratio"]<contract["residual_energy_ratio_at_least"] or
            value["shuffle_rms_over_original_sd"]<contract["shuffle_rms_over_original_sd_at_least"] or value["shuffled_norm_p99_ratio"]>contract["shuffled_norm_p99_ratio_at_most"] or
            value["train_coverage"]!=1 or value["held_coverage"]!=1 or value["train_self_rate"]!=0 or not value["train_donor_marginal_exact"] or
            value["held_effective_donors_per_object"]!=1 or value["held_max_donor_load"]>value["held_load_bound"] or
            value["recipient_u_train_max_error"]>contract["recipient_u_max_error"] or value["recipient_u_held_max_error"]>contract["recipient_u_max_error"] or
            cf["residual_norm_raw_z_absolute_spearman"]>contract["crossfit_absolute_spearman_at_most"] or cf["residual_raw_z_distance_correlation"]>contract["crossfit_distance_correlation_at_most"]): return False
        if not _crossfit_valid(cf,contract,expected_field_kind): return False
    return True


def receipt_passes(receipt: dict,expected_receipt: dict,contract: dict,expected_context: str,expected_domains: set[str]) -> bool:
    """Fail-closed validation against an independent canonical recomputation."""
    try:
        return _receipt_passes_checked(receipt,expected_receipt,contract,expected_context,expected_domains)
    except (KeyError,TypeError,ValueError,IndexError,RuntimeError):
        return False


def preserve_recipient_displacement(shuffled_geometry: Tensor,recipient_mechanical: Tensor) -> Tensor:
    if shuffled_geometry.shape[:-1]!=recipient_mechanical.shape[:-1] or shuffled_geometry.shape[-1]+1!=recipient_mechanical.shape[-1]: raise ValueError("mechanical geometry/displacement shape mismatch")
    result=torch.cat((shuffled_geometry,recipient_mechanical[...,-1:]),-1)
    if not torch.equal(result[...,-1],recipient_mechanical[...,-1]): raise RuntimeError("recipient displacement changed")
    return result


__all__=["SCHEMA","crossfit_rqlsot","empirical_rank","feature_crossfit_rqlsot","preserve_recipient_displacement","receipt_passes","rqlsot_ablation"]
