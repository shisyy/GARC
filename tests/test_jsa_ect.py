import copy
import hashlib
import inspect
import json
from pathlib import Path

import torch

from splart.jsa_ect import (_domain_anchor, _sha_tensor, canonical_object_scalar,
                            feature_crossfit_jsa_ect, fit_shared_model,
                            jsa_ect_ablation, preserve_recipient_displacement,
                            receipt_passes, reconstruct)
from scripts.run_jsa_ect_label_free import (FROZEN_CONFIG_SHA256,
                                             canonical_sha256,
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
                x=-1+2*local/max(count-1,1); axis=torch.tensor([1.,.2,-.1,.05,.03,-.02]); axis/=axis.norm()
                mean=torch.tensor([1+z,2-.3*z,-.2+.4*z,.7,-.5,.2],dtype=torch.float64)+(1+.4*x*x)*(-1 if local%2 else 1)*axis
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
        assert torch.allclose(reconstruct(model["location"][sl],sp,so,model["standardized"][domain],model["axis"]),blocks()[domain][2],atol=1e-10,rtol=0)


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
    assert receipt_passes(first[2],expected[2],contract(),context,{"articraft","njc"})
    reverse=jsa_ect_ablation(rows,list(reversed(train)),list(reversed(held)),tf.flip(0),hf.flip(0),tz,hz,contract(),context,feature_crossfit_jsa_ect(rows,list(reversed(train)),"semantic",contract()))
    assert first[2]==reverse[2]
    bad=copy.deepcopy(first[2]); bad["shared_model"]["theta"]=0.; assert not receipt_passes(bad,expected[2],contract(),context,{"articraft","njc"})
    geometry=torch.arange(12,dtype=torch.float64).reshape(3,4); mechanical=torch.randn(3,5,dtype=torch.float64)
    assert torch.equal(preserve_recipient_displacement(geometry,mechanical)[:,-1],mechanical[:,-1])


def test_config_and_runner_are_score_free_and_frozen():
    config=json.loads(Path("configs/jsa_ect_v1.json").read_text())
    assert config["model"]["axis_rank"]==1 and config["model"]["candidate_search"] is False
    assert not config["source_labels_opened"] and not config["source_scores_computed"] and not config["training_started"] and not config["remote_execution_started"]
    source=Path("scripts/run_jsa_ect_label_free.py").read_text()
    assert "labels.pt" not in source and "--score" not in source and "--seed" not in source
    data=Path("configs/jsa_ect_v1.json").read_bytes().replace(b"\r\n",b"\n")
    assert hashlib.sha256(data).hexdigest()==FROZEN_CONFIG_SHA256
    result={"feature_provenance":{"opaque":1},"code_files_sha256":{"runner":"a"},"source_labels_opened":False,"source_labels_hashed":False,
            "source_scores_computed":False,"training_started":False,"remote_execution_started":True,"execution_environment":"CASIA_98","execution_phase":"label_free_feasibility","box_labels_read":[],"protected_splits_read":[]}
    result["feature_provenance_sha256"]=canonical_sha256(result["feature_provenance"]); result["code_sha256"]=canonical_sha256(result["code_files_sha256"])
    receipt={"schema":"splart-jsa-ect-feasibility-receipt/v1","config_sha256":FROZEN_CONFIG_SHA256,"code_sha256":result["code_sha256"],"feature_provenance_sha256":result["feature_provenance_sha256"],
             "source_labels_opened":False,"source_labels_hashed":False,"source_scores_computed":False,"training_started":False,"remote_execution_started":True,
             "execution_environment":"CASIA_98","execution_phase":"label_free_feasibility","box_labels_read":[],"protected_splits_read":[]}
    assert verify_provenance_binding(result,receipt)
    bad=copy.deepcopy(receipt); bad["execution_environment"]="CASIA_102"; assert not verify_provenance_binding(result,bad)
