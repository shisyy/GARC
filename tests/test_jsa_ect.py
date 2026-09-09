import copy
import hashlib
import inspect
import json
from pathlib import Path

import torch
import pytest

from splart.jsa_ect import (_domain_anchor, _sha_tensor, canonical_object_scalar,
                            domain_receipt_passes,
                            feature_crossfit_jsa_ect, fit_shared_model,
                            global_receipt_passes,
                            jsa_ect_ablation, preserve_recipient_displacement,
                            receipt_passes, reconstruct)
from scripts.run_jsa_ect_label_free import (FROZEN_CONFIG_SHA256,
                                             PREDECESSOR_CONFIG_SHA256,
                                             SOURCE_CONFIG_SHA256,
                                             canonical_sha256,
                                             run_cell,
                                             verify_provenance_binding)


def contract():
    return {"rank_relative_tolerance": 1e-12, "condition_max": 1e6, "eigengap_multiplier": 100.,
            "numeric_tolerance": 1e-12, "crossfit_folds": 5, "energy_conservation_max_error": 1e-10,
            "symmetry_max_error": 1e-12, "orthogonality_max": 1e-10, "normalized_residual_mean_max": 1e-10,
            "normalized_residual_raw_z_correlation_max": 1e-10, "reconstruction_max_error": 1e-10,
            "recipient_u_max_error": 1e-12, "held_boundary_clipped_fraction_max": 1.,
            "residual_energy_ratio_at_least": 0., "shuffle_rms_over_original_sd_at_least": 0.,
            "shuffled_norm_p99_ratio_at_most": 100., "crossfit_absolute_spearman_at_most": 1.,
            "crossfit_distance_correlation_at_most": 1.}


def blocks(count=25, width=6):
    result = {}; axis = torch.tensor([1., .2, -.1, .05, .03, -.02], dtype=torch.float64); axis /= axis.norm()
    for di, domain in enumerate(("articraft", "njc")):
        z = torch.linspace(.2 + .1*di, 2.1 + .1*di, count, dtype=torch.float64)
        x = torch.linspace(-1 + 1/count, 1 - 1/count, count, dtype=torch.float64)
        q = _domain_anchor(z, x, contract())["q"]
        rows = []
        for i in range(count):
            base = torch.tensor([1 + z[i], 2 - .3*z[i], -.2 + .4*z[i], .7, -.5, .2], dtype=torch.float64)
            shared = q[i] * torch.tensor([.1, -.2, .05, .03, .02, -.01], dtype=torch.float64)
            bulk = torch.tensor([0., torch.sin(z[i]*2), torch.cos(z[i]*3), torch.sin(z[i]*5), torch.cos(z[i]*7), torch.sin(z[i]*11)])
            bulk -= (bulk @ axis) * axis
            sign = -1. if i % 2 else 1.
            rows.append(base + shared + sign * torch.exp(.35*q[i]) * axis + .23 * bulk)
        result[domain] = (z, x, torch.stack(rows))
    return result


def fixture():
    rows=[]; train=[]; held=[]; tf=[]; hf=[]; tz={}; hz={}
    for di,domain in enumerate(("articraft","njc")):
        for split,count,start in (("train",20,0),("held",5,40)):
            for local in range(count):
                oi=start+local; obj=f"{domain}-{split}-{local:02d}"; z=.3+.037*oi+.13*di
                (tz if split=="train" else hz)[obj]=z
                x=-1+2*local/max(count-1,1); axis=torch.tensor([1.,.2,-.1,.05,.03,-.02],dtype=torch.float64); axis/=axis.norm()
                bulk=torch.tensor([0.,torch.sin(torch.tensor(1.7*z)),torch.cos(torch.tensor(2.3*z)),torch.sin(torch.tensor(3.1*z)),torch.cos(torch.tensor(4.7*z)),torch.sin(torch.tensor(5.3*z))],dtype=torch.float64)
                bulk-=bulk.dot(axis)*axis
                mean=torch.tensor([1+z,2-.3*z,-.2+.4*z,.7,-.5,.2],dtype=torch.float64)+(1+.4*x*x)*(-1 if local%2 else 1)*axis+.2*bulk
                for gauge in range(3):
                    semantic=torch.tensor([torch.sin(torch.tensor((oi+1)*(k+1)*.137+gauge*.011+di*.23)).item()+.1*torch.cos(torch.tensor((oi+2)*(k+3)*.071)).item() for k in range(24)],dtype=torch.float64)
                    mechanical=torch.tensor([(oi+1)**((k%3)+1)*1e-3+(k+1)*gauge*.01+di*.2 for k in range(8)]+[z],dtype=torch.float64)
                    index=len(rows); (train if split=="train" else held).append(index)
                    rows.append({"domain":domain,"object_group_id":obj,"joint_id":obj,"gauge_id":f"{split}-{local}-g{gauge}","semantic":semantic,"mechanical":mechanical,"observed_displacement":z+(gauge-1)*1e-15})
                    value=mean+torch.tensor([gauge,-gauge,.5*gauge,-.25*gauge,.1*gauge,-.05*gauge],dtype=torch.float64)*.01
                    (tf if split=="train" else hf).append(value)
    return rows,train,held,torch.stack(tf),torch.stack(hf),tz,hz


def test_shared_axis_theta_energy_and_pooled_raw_z_orthogonality():
    model=fit_shared_model(blocks(),contract())
    assert abs(float(model["theta"]))<=1+1e-12
    assert model["eigengap"]>model["eigengap_threshold"]
    assert model["axis"][model["axis_pivot"]]>=0
    assert model["energy_conservation_max_error"]<1e-10
    assert model["pooled_design_orthogonality_max"]<1e-10
    for domain in model["domains"]:
        assert model["diagnostics"][domain]["normalized_residual_mean_max"]<1e-10
        assert model["diagnostics"][domain]["normalized_residual_raw_z_correlation_max"]<1e-10
        sl=model["domain_slices"][domain]; sp,so=model["scales"][domain]
        assert torch.allclose(reconstruct(model["parallel_location"][sl],model["perpendicular_location"][sl],sp,so,model["standardized"][domain],model["axis"]),blocks()[domain][2],atol=1e-10,rtol=0)


def test_postscale_fwl_repairs_scale_induced_raw_z_dependence_and_degenerate_blocks_fail():
    data=blocks(); model=fit_shared_model(data,contract())
    old_fails=False
    for domain in model["domains"]:
        sl=model["domain_slices"][domain]; z=data[domain][0]; centered=z-z.mean(); old=model["first_standardized"][domain]; scale=old.std(0,unbiased=False).clamp_min(1e-12)
        old_corr=float(((centered[:,None]*old).mean(0).abs()/(centered.std(unbiased=False)*scale).clamp_min(1e-12)).max())
        old_fails |= old_corr>1e-10
        assert model["diagnostics"][domain]["first_pass_normalized_residual_raw_z_correlation_max"]<1e-10
        assert model["diagnostics"][domain]["normalized_residual_raw_z_correlation_max"]<1e-10
    assert old_fails
    axis=torch.tensor([1.,0.,0.],dtype=torch.float64); bad={}
    for di,domain in enumerate(("articraft","njc")):
        z=torch.linspace(.2,2.,20,dtype=torch.float64)+di*.1; x=torch.linspace(-.95,.95,20,dtype=torch.float64); q=_domain_anchor(z,x,contract())["q"]
        y=torch.stack((1+z+(-1.)**torch.arange(20)*torch.exp(.2*q),2-.3*z,.5+.2*z),-1); bad[domain]=(z,x,y)
    with pytest.raises(ValueError,match="unidentifiable domain energy intercept"):
        fit_shared_model(bad,contract())
    affine={d:(z,x,torch.stack((1+z,2-.3*z,.5+.2*z),-1)) for d,(z,x,_) in bad.items()}
    with pytest.raises(ValueError): fit_shared_model(affine,contract())
    narrow={d:(z,x,y[:,:1]) for d,(z,x,y) in data.items()}
    with pytest.raises(ValueError): fit_shared_model(narrow,contract())


def test_fold_local_shared_model_is_finite_and_identical_across_domains():
    rows,train,*_=fixture(); result=feature_crossfit_jsa_ect(rows,train,"semantic",contract())
    reverse=feature_crossfit_jsa_ect(rows,list(reversed(train)),"semantic",contract())
    assert result==reverse
    for domain,value in result.items():
        assert torch.isfinite(torch.tensor([value["residual_norm_raw_z_absolute_spearman"],value["residual_raw_z_distance_correlation"]])).all()
        assert value["fold_counts"]==[4,4,4,4,4]
        for fold in range(5):
            assert value["folds"][fold]["joint_model_sha256"]==result["articraft"]["folds"][fold]["joint_model_sha256"]
    mechanical=feature_crossfit_jsa_ect(rows,train,"mechanical",contract())
    assert all(torch.isfinite(torch.tensor(value["residual_raw_z_distance_correlation"])) for value in mechanical.values())
    assert all(value["folds"][i]["joint_model_sha256"]==mechanical["articraft"]["folds"][i]["joint_model_sha256"] for value in mechanical.values() for i in range(5))


def test_transport_double_receipt_row_order_u_and_displacement():
    args=fixture(); rows,train,held,tf,hf,tz,hz=args; context="semantic_jsa_ect/final-source-features"
    cf=feature_crossfit_jsa_ect(rows,train,"semantic",contract()); first=jsa_ect_ablation(*args,contract(),context,cf)
    expected=jsa_ect_ablation(*args,contract(),context,feature_crossfit_jsa_ect(rows,train,"semantic",contract()))
    expected_sha=canonical_sha256(expected[2])
    assert receipt_passes(first[2],expected[2],expected_sha,contract(),context,{"articraft","njc"})
    assert all(domain_receipt_passes(first[2],expected[2],expected_sha,contract(),context,{"articraft","njc"},d) for d in ("articraft","njc"))
    reverse=jsa_ect_ablation(rows,list(reversed(train)),list(reversed(held)),tf.flip(0),hf.flip(0),tz,hz,contract(),context,feature_crossfit_jsa_ect(rows,list(reversed(train)),"semantic",contract()))
    assert first[2]==reverse[2]
    bad=copy.deepcopy(first[2]); bad["shared_model"]["theta"]=0.; assert not receipt_passes(bad,expected[2],expected_sha,contract(),context,{"articraft","njc"})
    coherent=copy.deepcopy(first[2]); coherent_expected=copy.deepcopy(expected[2]); coherent["domains"]["articraft"]["normalized_residual_mean_max"]=0.; coherent_expected["domains"]["articraft"]["normalized_residual_mean_max"]=0.
    assert not receipt_passes(coherent,coherent_expected,expected_sha,contract(),context,{"articraft","njc"})
    assert not receipt_passes(expected[2],expected[2],expected_sha,contract(),context,{"articraft","njc"})
    aliased=copy.deepcopy(first[2]); aliased_expected=copy.deepcopy(expected[2]); aliased["domains"]["articraft"]["crossfit"]=aliased_expected["domains"]["articraft"]["crossfit"]
    alias_sha=canonical_sha256(aliased_expected); assert not receipt_passes(aliased,aliased_expected,alias_sha,contract(),context,{"articraft","njc"})
    gate_bad=copy.deepcopy(first[2]); gate_bad["domains"]["articraft"]["normalized_residual_mean_max"]=1.
    gate_expected=copy.deepcopy(gate_bad); gate_sha=canonical_sha256(gate_expected)
    assert global_receipt_passes(gate_bad,gate_expected,gate_sha,contract(),context,{"articraft","njc"})
    assert not domain_receipt_passes(gate_bad,gate_expected,gate_sha,contract(),context,{"articraft","njc"},"articraft")
    geometry=torch.arange(12,dtype=torch.float64).reshape(3,4); mechanical=torch.randn(3,5,dtype=torch.float64)
    assert torch.equal(preserve_recipient_displacement(geometry,mechanical)[:,-1],mechanical[:,-1])


def test_runner_uses_immutable_independent_receipt_and_no_nested_aliases():
    rows,train,held,tf,hf,tz,hz=fixture(); context="semantic_jsa_ect/final-source-features"
    cell=run_cell(rows,train,held,tf,hf,tz,hz,contract(),context,"semantic")
    assert cell[4]==canonical_sha256(cell[3])
    assert cell[2] is not cell[3]
    for domain in ("articraft","njc"):
        assert cell[2]["domains"][domain]["crossfit"] is not cell[3]["domains"][domain]["crossfit"]
        assert cell[2]["domains"][domain]["crossfit"]["folds"] is not cell[3]["domains"][domain]["crossfit"]["folds"]
    assert cell[5]["repeat_bit_identical"] and cell[5]["row_order_invariant"]


def test_config_and_runner_are_score_free_and_frozen():
    config=json.loads(Path("configs/jsa_ect_v1.json").read_text())
    assert config["model"]["axis_rank"]==1 and config["model"]["candidate_search"] is False
    assert not config["source_labels_opened"] and not config["source_scores_computed"] and not config["training_started"] and not config["remote_execution_started"]
    source=Path("scripts/run_jsa_ect_label_free.py").read_text()
    assert "labels.pt" not in source and "--score" not in source and "--seed" not in source
    data=Path("configs/jsa_ect_v1.json").read_bytes().replace(b"\r\n",b"\n")
    assert hashlib.sha256(data).hexdigest()==FROZEN_CONFIG_SHA256
    assert config["module_invocation"]=="python -m scripts.run_jsa_ect_label_free"
    assert config["source_config_sha256"]==SOURCE_CONFIG_SHA256 and config["predecessor_rza_config_sha256"]==PREDECESSOR_CONFIG_SHA256
    result={"schema":"splart-jsa-ect-feasibility/v1","config_sha256":FROZEN_CONFIG_SHA256,"source_config_sha256":SOURCE_CONFIG_SHA256,"predecessor_config_sha256":PREDECESSOR_CONFIG_SHA256,
            "feature_provenance":{"opaque":1},"jsa_ect":{},"independent_recomputation_sha256":{},"runtime_audits":{},"global_receipt_pass":{},"domain_receipt_pass":{},"cell_pass":{},"all_pass":False,"object_list_hashes":{},"global_row_identity_sha256":"0"*64,"code_files_sha256":{"runner":"a"},"source_labels_opened":False,"source_labels_hashed":False,
            "source_scores_computed":False,"training_started":False,"remote_execution_started":True,"execution_environment":"CASIA_98","execution_phase":"label_free_feasibility","box_labels_read":[],"protected_splits_read":[]}
    result["feature_provenance_sha256"]=canonical_sha256(result["feature_provenance"]); result["code_sha256"]=canonical_sha256(result["code_files_sha256"])
    receipt={"schema":"splart-jsa-ect-feasibility-receipt/v1","config_sha256":FROZEN_CONFIG_SHA256,"source_config_sha256":SOURCE_CONFIG_SHA256,"predecessor_config_sha256":PREDECESSOR_CONFIG_SHA256,"feasibility_sha256":"1"*64,"decision":"PRUNE_LABEL_FREE","code_sha256":result["code_sha256"],"feature_provenance_sha256":result["feature_provenance_sha256"],
             "source_labels_opened":False,"source_labels_hashed":False,"source_scores_computed":False,"training_started":False,"remote_execution_started":True,
             "execution_environment":"CASIA_98","execution_phase":"label_free_feasibility","box_labels_read":[],"protected_splits_read":[]}
    assert verify_provenance_binding(result,receipt)
    bad=copy.deepcopy(receipt); bad["execution_environment"]="CASIA_102"; assert not verify_provenance_binding(result,bad)
    for key in ("schema","config_sha256","source_config_sha256","predecessor_config_sha256"):
        bad=copy.deepcopy(receipt); bad[key]="evil"; assert not verify_provenance_binding(result,bad)
    bad=copy.deepcopy(receipt); bad["extra"]=1; assert not verify_provenance_binding(result,bad)
    bad=copy.deepcopy(receipt); del bad["decision"]; assert not verify_provenance_binding(result,bad)
    bad_result=copy.deepcopy(result); bad_result["evil"]=1; assert not verify_provenance_binding(bad_result,receipt)
    bad_result=copy.deepcopy(result); del bad_result["object_list_hashes"]; assert not verify_provenance_binding(bad_result,receipt)
