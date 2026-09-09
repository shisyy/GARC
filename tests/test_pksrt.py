import copy
import hashlib
import json
import math
from pathlib import Path

import pytest
import torch

import splart.pksrt as pk
from scripts.run_pksrt_label_free import (FROZEN_CONFIG_SHA256,
                                           PREDECESSOR_CONFIG_SHA256,
                                           SOURCE_CONFIG_SHA256,
                                           canonical_sha256,
                                           verify_provenance_binding)
from splart.jsa_ect import _balanced_folds as jsa_balanced_folds


def contract():
    return {"rank_relative_tolerance": 1e-12, "condition_max": 1e6, "numeric_tolerance": 1e-12,
            "minimum_effective_sample_size": 2., "crossfit_folds": 5,
            "inverse_max_error": 1e-10, "reconstruction_max_error": 1e-10,
            "normalized_residual_mean_max": 1e-10, "normalized_residual_raw_z_correlation_max": 1e-10,
            "held_boundary_clipped_fraction_max": 1., "residual_energy_ratio_at_least": 0.,
            "shuffle_rms_over_original_sd_at_least": 0., "shuffled_norm_p99_ratio_at_most": 1e6,
            "recipient_u_max_error": 1e-12, "crossfit_absolute_spearman_at_most": 1.,
            "crossfit_distance_correlation_at_most": 1.}


def blocks(counts=(11, 13), width=4):
    result = {}; ids = {}
    for di, (domain, count) in enumerate(zip(("articraft", "njc"), counts)):
        z = torch.tensor([.17 + .07*i + .013*math.sin((i+1)*(di+2)) for i in range(count)], dtype=torch.float64)
        columns = []
        for k in range(width):
            columns.append(torch.tensor([.4*(k+1) + (.3-.04*k)*z[i] +
                math.sin((i+1)*(k+2)*.317 + di*.19) + .13*math.cos((i+2)*(k+3)*.113)
                for i in range(count)], dtype=torch.float64))
        result[domain] = (z, torch.stack(columns, -1)); ids[domain] = [f"{domain}-{i:03d}" for i in range(count)]
    return result, ids


def fake_crossfit(domains=("articraft", "njc")):
    result = {}; assignments={d:pk._balanced_folds(d,[f"{d}-train-{i:03d}" for i in range(12)],5) for d in domains}; common=[]
    for fold in range(5):
        fit_ids={d:sorted(o for o,f in assignments[d].items() if f!=fold) for d in domains}
        held_ids={d:sorted(o for o,f in assignments[d].items() if f==fold) for d in domains}
        data={}
        for di,d in enumerate(domains):
            n=len(fit_ids[d]); z=torch.tensor([.2+.071*i+.013*math.sin((i+1)*(di+2)) for i in range(n)],dtype=torch.float64)
            y=torch.stack((.4+.3*z+torch.tensor([math.sin((i+1)*.317+di*.2) for i in range(n)]),
                           -.2+.1*z+torch.tensor([math.cos((i+1)*.271+di*.3) for i in range(n)])),-1)
            data[d]=(z,y)
        model=pk.model_provenance(pk.fit_model(data,contract(),fit_ids)); prep = {
            "train_object_hash":hashlib.sha256(f"train{fold}".encode()).hexdigest(),
            "mechanical_mean":"1"*64,"mechanical_scale":"2"*64,"mechanical_keep":"3"*64,
            "mechanical_raw_dim":3,"mechanical_kept_indices":[0,2],"semantic_mean":"4"*64,"semantic_components":"5"*64}
        joint_fit=[f"{d}:{o}" for d in sorted(domains) for o in fit_ids[d]]; joint_held=[f"{d}:{o}" for d in sorted(domains) for o in held_ids[d]]
        common.append({"fold": fold, "joint_model": model, "joint_model_sha256": canonical_sha256(model),
            "joint_preprocessor": prep, "joint_preprocessor_sha256": canonical_sha256(prep),
            "joint_fit_object_ids": joint_fit, "joint_held_object_ids": joint_held,
            "domain_fit_object_ids": [], "domain_held_object_ids": [],
            "joint_fit_object_hash": pk._sha_ids(joint_fit), "joint_held_object_hash": pk._sha_ids(joint_held),
            "domain_fit_object_hash": "", "domain_held_object_hash": "",
            "boundary_clipped_fraction": 0., "fit_count": len(joint_fit), "held_count": len(joint_held)})
    for domain in domains:
        folds=copy.deepcopy(common)
        for fold in folds:
            fold["domain_fit_object_ids"]=sorted(o for o,f in assignments[domain].items() if f!=fold["fold"])
            fold["domain_held_object_ids"]=sorted(o for o,f in assignments[domain].items() if f==fold["fold"])
            fold["domain_fit_object_hash"]=pk._sha_ids(fold["domain_fit_object_ids"]); fold["domain_held_object_hash"]=pk._sha_ids(fold["domain_held_object_ids"])
        assignment=assignments[domain]
        result[domain] = {"schema": pk.CROSSFIT_SCHEMA, "field_kind": "semantic", "folds": folds,
            "fold_assignment": assignment, "fold_assignment_sha256": canonical_sha256(assignment),
            "fold_counts": [list(assignment.values()).count(f) for f in range(5)],
            "standardized_sha256": "3"*64, "raw_z_sha256": "4"*64,
            "residual_norm_raw_z_absolute_spearman": 0., "residual_raw_z_distance_correlation": 0.}
    return result


def row_fixture(train_count=12, held_count=4, width=4):
    rows=[]; train=[]; held=[]; train_field=[]; held_field=[]; tz={}; hz={}
    for di, domain in enumerate(("articraft", "njc")):
        for split, count, offset in (("train", train_count, 0), ("held", held_count, 50)):
            for local in range(count):
                obj=f"{domain}-{split}-{local:03d}"; oi=offset+local; z=.21+.041*oi+.07*di
                (tz if split == "train" else hz)[obj]=z
                mean=torch.tensor([.2*(k+1)+(.3-.02*k)*z+math.sin((oi+1)*(k+2)*.173+di*.31)+
                                   .07*math.cos((oi+3)*(k+1)*.271) for k in range(width)],dtype=torch.float64)
                for gauge in range(2):
                    semantic=torch.tensor([math.sin((oi+1)*(k+1)*.071+gauge*.013+di*.17)+
                                           .11*math.cos((oi+2)*(k+3)*.037) for k in range(20)],dtype=torch.float64)
                    mechanical=torch.tensor([math.sin((oi+1)*(k+1)*.091)+gauge*.003+di*.13 for k in range(7)]+[z],dtype=torch.float64)
                    index=len(rows); (train if split == "train" else held).append(index)
                    rows.append({"domain":domain,"object_group_id":obj,"joint_id":obj,"gauge_id":f"{split}-{local:03d}-g{gauge}",
                                 "semantic":semantic,"mechanical":mechanical,"observed_displacement":z+(gauge-1)*1e-15})
                    value=mean+torch.tensor([(.01*gauge)*(k+1)*(-1 if k%2 else 1) for k in range(width)],dtype=torch.float64)
                    (train_field if split == "train" else held_field).append(value)
    return rows,train,held,torch.stack(train_field),torch.stack(held_field),tz,hz


def test_fixed_coordinate_then_dct_basis_and_signs():
    slices, receipt = pk.fixed_slices(7)
    assert receipt["count"] == 14 and receipt["sequence"] == "coordinate_then_dct2"
    assert torch.equal(slices[:7], torch.eye(7, dtype=torch.float64))
    assert torch.allclose(slices[7:] @ slices[7:].T, torch.eye(7, dtype=torch.float64), atol=1e-14, rtol=0)
    assert all(float(v[int(v.abs().argmax())]) >= 0 for v in slices)


def test_strict_marginal_forward_inverse_and_linear_tails():
    marginal = pk.fit_domain_marginal(torch.tensor([-2., -.1, .7, 3.], dtype=torch.float64))
    query = torch.tensor([-8., -2., .2, 3., 11.], dtype=torch.float64)
    mapped = pk._linear_forward(marginal, query)
    assert torch.all(mapped[1:] > mapped[:-1])
    assert torch.allclose(pk._linear_inverse(marginal, mapped), query, atol=1e-12, rtol=0)
    assert marginal["slopes"][0] > 0 and marginal["slopes"][-1] > 0
    with pytest.raises(ValueError, match="tie"):
        pk.fit_domain_marginal(torch.tensor([0., 0., 1.]))


def test_grouped_exact_a_erratum_handles_odd_odd_medians_without_jitter():
    records=[]
    for domain, n in (("articraft", 97), ("njc", 11)):
        a=pk._logit((torch.arange(n,dtype=torch.float64)+.5)/n)
        records.extend((float(a[i]),float((i+.5)/n),f"{domain}:{i:03d}") for i in range(n))
    pool=pk.fit_conditional_pool(records)
    assert pool["raw_count"]==108 and pool["unique_count"]==107
    assert pool["tie_group_count"]==1 and pool["max_multiplicity"]==2
    zero=[g for g in pool["groups"] if pool["a"][g[0]] == 0]
    assert len(zero)==1 and len(zero[0])==2
    bijection,neff=pk.conditional_bijection(pool,.5,2.)
    assert neff>=2 and torch.all(bijection["slopes"]>0)
    tail_query=torch.tensor([-20.,float(bijection["x"][0]),0.,float(bijection["x"][-1]),20.],dtype=torch.float64)
    assert torch.allclose(pk._linear_inverse(bijection,pk._linear_forward(bijection,tail_query)),tail_query,atol=1e-12,rtol=0)
    reverse=pk.fit_conditional_pool(list(reversed(records)))
    assert {k:v for k,v in pool.items() if not isinstance(v,torch.Tensor)} == {k:v for k,v in reverse.items() if not isinstance(v,torch.Tensor)}
    assert torch.equal(pool["a"],reverse["a"]) and torch.equal(pool["u"],reverse["u"])


def test_conditional_effective_sample_size_fails_closed():
    pool=pk.fit_conditional_pool([(-1.,0.,"a:x"),(1.,1.,"b:y")])
    with pytest.raises(ValueError,match="effective sample size"):
        pk.conditional_bijection(pool,0.,2.)


def test_full_forward_inverse_actual_e_orthogonality_and_energy_ratio():
    data,ids=blocks(); model=pk.fit_model(data,contract(),ids)
    assert model["inverse_max_error"] < 1e-10
    assert model["slice_receipt"]["count"] == 2*model["width"]
    assert any(layer["pool"]["tie_group_count"] for layer in model["layers"])
    assert all(layer["minimum_observed_conditional_slope"] > 0 for layer in model["layers"])
    for domain,(z,y) in data.items():
        e=model["standardized"][domain]; centered=z-z.mean(); scale=e.std(0,unbiased=False)
        assert float((e.mean(0).abs()/scale).max()) < 1e-10
        assert float(((centered[:,None]*e).mean(0).abs()/(centered.std(unbiased=False)*scale)).max()) < 1e-10
        assert torch.allclose(pk.forward(model,domain,z,model["stages"][domain]["x"],y),e,atol=1e-11,rtol=0)
        assert torch.allclose(pk.reconstruct_means(model,domain,z,model["stages"][domain]["x"],e),y,atol=1e-10,rtol=0)
        expected=model["stages"][domain]["scale_squared"]/model["stages"][domain]["centered_energy"]
        assert model["diagnostics"][domain]["residual_energy_ratio"] == expected


def test_every_layer_uses_current_residual_and_reverse_order_is_required():
    data,ids=blocks(); model=pk.fit_model(data,contract(),ids); domain="articraft"; z,y=data[domain]
    current=model["stages"][domain]["residual"].clone()
    observed=[]
    for layer in model["layers"]:
        t=current@layer["vector"]; observed.append(t)
        a=pk._linear_forward(layer["marginals"][domain],t); eta=[]
        for i in range(len(t)):
            h,_=pk.conditional_bijection(layer["pool"],model["stages"][domain]["u"][i],2.)
            eta.append(pk._linear_forward(h,a[i:i+1])[0])
        eta=torch.stack(eta); current=current+(eta-t)[:,None]*layer["vector"]
    for t,layer in zip(observed,model["layers"]):
        assert torch.equal(torch.sort(t).values,layer["marginals"][domain]["x"])
    assert torch.allclose(current-pk._anchor_design(z,model["stages"][domain]["z_mean"])@model["finals"][domain]["beta"],
                          model["standardized"][domain],atol=1e-12,rtol=0)


def test_ablation_repeat_reverse_order_receipt_aliases_and_displacement():
    args=row_fixture(); rows,train,held,tf,hf,tz,hz=args; context="semantic_pksrt/final-source-features"; cf=fake_crossfit()
    first=pk.pksrt_ablation(*args,contract(),context,cf)
    second=pk.pksrt_ablation(*args,contract(),context,fake_crossfit())
    sha=canonical_sha256(second[2]); domains={"articraft","njc"}
    assert pk.global_receipt_passes(first[2],second[2],sha,contract(),context,domains)
    reverse=pk.pksrt_ablation(rows,list(reversed(train)),list(reversed(held)),tf.flip(0),hf.flip(0),tz,hz,contract(),context,fake_crossfit())
    assert first[2]==reverse[2] and torch.equal(first[0],reverse[0]) and torch.equal(first[1],reverse[1])
    assert not pk.global_receipt_passes(second[2],second[2],sha,contract(),context,domains)
    alias=copy.deepcopy(first[2]); expected=copy.deepcopy(second[2]); alias["shared_model"]["layers"][0]["conditional_pool"]["groups"]=expected["shared_model"]["layers"][0]["conditional_pool"]["groups"]
    assert canonical_sha256(alias)==canonical_sha256(expected)
    assert not pk.global_receipt_passes(alias,expected,canonical_sha256(expected),contract(),context,domains)
    def rejected(mutator):
        candidate=copy.deepcopy(first[2]); expected=copy.deepcopy(second[2]); mutator(candidate); mutator(expected)
        assert not pk.global_receipt_passes(candidate,expected,canonical_sha256(expected),contract(),context,domains)
    rejected(lambda r:r["shared_model"]["slice_receipt"].__setitem__("basis_sha256","0"*64))
    rejected(lambda r:r["shared_model"]["layers"][0].__setitem__("vector_sha256","0"*64))
    rejected(lambda r:r["shared_model"]["layers"][0]["conditional_pool"].__setitem__("grouping_sha256","0"*64))
    def invalid_groups(r):
        pool=r["shared_model"]["layers"][0]["conditional_pool"]; pool["groups"]=[[0],[0]]
        pool["grouping_sha256"]=canonical_sha256([[pool["keys"][0]],[pool["keys"][0]]])
    rejected(invalid_groups)
    rejected(lambda r:r["shared_model"]["layers"][0]["domain_marginals"]["articraft"].__setitem__("slopes",[1.]*11))
    rejected(lambda r:r["shared_model"]["layers"][0]["domain_marginals"]["articraft"].__setitem__("x_sha256","0"*64))
    rejected(lambda r:r["shared_model"].__setitem__("inverse_max_error",float("nan")))
    rejected(lambda r:r["domains"]["articraft"].__setitem__("residual_energy_ratio",float("nan")))
    def forged_assignment(r):
        cf=r["domains"]["articraft"]["crossfit"]; key=sorted(cf["fold_assignment"])[0]
        cf["fold_assignment"][key]=(cf["fold_assignment"][key]+1)%5
        cf["fold_assignment_sha256"]=canonical_sha256(cf["fold_assignment"])
    rejected(forged_assignment)
    rejected(lambda r:r["domains"]["articraft"]["crossfit"].__setitem__("standardized_sha256","bad"))
    rejected(lambda r:r["domains"]["articraft"]["crossfit"]["folds"][0]["joint_fit_object_ids"].append("articraft:forged"))
    mechanical=torch.randn(8,5,dtype=torch.float64); geometry=torch.randn(8,4,dtype=torch.float64)
    assert torch.equal(pk.preserve_recipient_displacement(geometry,mechanical)[:,-1],mechanical[:,-1])


def test_crossfit_reuses_jsa_folds_and_excludes_every_held_object(monkeypatch):
    objects=[f"o{i:02d}" for i in range(20)]
    for domain in ("articraft","njc"):
        assert pk._balanced_folds(domain,objects,5)==jsa_balanced_folds(domain,objects,5)
    rows,train,*_=row_fixture(train_count=15)
    calls=[]; original=pk.fit_model
    def wrapped(data,cfg,object_ids=None):
        calls.append(copy.deepcopy(object_ids)); return original(data,cfg,object_ids)
    monkeypatch.setattr(pk,"fit_model",wrapped)
    result=pk.feature_crossfit_pksrt(rows,train,"semantic",contract())
    assert len(calls)==5
    for fold,ids in enumerate(calls):
        for domain in ("articraft","njc"):
            assignment=result[domain]["fold_assignment"]
            assert set(ids[domain])=={o for o,f in assignment.items() if f!=fold}
            assert not set(ids[domain]) & {o for o,f in assignment.items() if f==fold}
            assert result[domain]["fold_assignment_sha256"]==canonical_sha256(assignment)
            assert result[domain]["folds"][fold]["joint_model_sha256"]==result["articraft"]["folds"][fold]["joint_model_sha256"]


def test_config_and_runner_are_frozen_score_free_and_provenance_bound():
    config=json.loads(Path("configs/pksrt_v1.json").read_text())
    assert config["model"]["candidate_search"] is False and config["model"]["jitter"] is False
    assert config["predecessor_commit"]=="7a57da0a0a9a438d3305c7dd1ea081346cd528b7"
    assert config["crossfit"]["assignment"].find("splart-jsa-ect-fold-v1:")>=0
    assert not config["source_labels_opened"] and not config["source_scores_computed"] and not config["training_started"] and not config["remote_execution_started"]
    source=Path("scripts/run_pksrt_label_free.py").read_text()
    assert "labels.pt" not in source and "--score" not in source and "--seed" not in source
    data=Path("configs/pksrt_v1.json").read_bytes().replace(b"\r\n",b"\n")
    assert hashlib.sha256(data).hexdigest()==FROZEN_CONFIG_SHA256
    assert config["source_config_sha256"]==SOURCE_CONFIG_SHA256 and config["predecessor_jsa_config_sha256"]==PREDECESSOR_CONFIG_SHA256
    result={"schema":"splart-pksrt-feasibility/v1","config_sha256":FROZEN_CONFIG_SHA256,"source_config_sha256":SOURCE_CONFIG_SHA256,"predecessor_config_sha256":PREDECESSOR_CONFIG_SHA256,
        "feature_provenance":{"opaque":1},"pksrt":{},"independent_recomputation_sha256":{},"runtime_audits":{},"global_receipt_pass":{},"domain_receipt_pass":{},"cell_pass":{},"all_pass":False,
        "object_list_hashes":{},"global_row_identity_sha256":"0"*64,"code_files_sha256":{"runner":"a"},"source_labels_opened":False,"source_labels_hashed":False,
        "source_scores_computed":False,"training_started":False,"remote_execution_started":True,"execution_environment":"CASIA_98","execution_phase":"label_free_feasibility","box_labels_read":[],"protected_splits_read":[]}
    result["feature_provenance_sha256"]=canonical_sha256(result["feature_provenance"]); result["code_sha256"]=canonical_sha256(result["code_files_sha256"])
    receipt={"schema":"splart-pksrt-feasibility-receipt/v1","config_sha256":FROZEN_CONFIG_SHA256,"source_config_sha256":SOURCE_CONFIG_SHA256,"predecessor_config_sha256":PREDECESSOR_CONFIG_SHA256,
        "feasibility_sha256":canonical_sha256(result),"decision":"PRUNE_LABEL_FREE","code_sha256":result["code_sha256"],"feature_provenance_sha256":result["feature_provenance_sha256"],
        "source_labels_opened":False,"source_labels_hashed":False,"source_scores_computed":False,"training_started":False,"remote_execution_started":True,
        "execution_environment":"CASIA_98","execution_phase":"label_free_feasibility","box_labels_read":[],"protected_splits_read":[]}
    assert verify_provenance_binding(result,receipt)
    bad=copy.deepcopy(receipt); bad["feasibility_sha256"]="0"*64; assert not verify_provenance_binding(result,bad)
    for key,value in (("remote_execution_started",False),("execution_environment","CASIA_102"),("execution_phase","score")):
        bad=copy.deepcopy(receipt); bad[key]=value; assert not verify_provenance_binding(result,bad)
    bad=copy.deepcopy(receipt); bad["extra"]=1; assert not verify_provenance_binding(result,bad)
