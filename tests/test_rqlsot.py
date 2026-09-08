import copy
import inspect
import json
from pathlib import Path

import torch
import pytest

from splart.rqlsot import (empirical_rank, feature_crossfit_rqlsot,
                            preserve_recipient_displacement, receipt_passes,
                            rqlsot_ablation, _canonical_sha, _sha_tensor)
from splart.conditional_residual import _distance_correlation, _spearman
from scripts.run_smarc_source_gate import merge_features
from scripts.run_rqlsot_feasibility import (FROZEN_CONFIG_SHA256,
                                             canonical_sha256,
                                             verify_provenance_binding)


def contract():
    return {"rank_relative_tolerance":1e-12,"condition_max":1e6,"crossfit_folds":5,
            "crossfit_absolute_spearman_at_most":1.,"crossfit_distance_correlation_at_most":1.,
            "normalized_residual_mean_max":1e-10,"normalized_residual_raw_z_correlation_max":1e-10,"recipient_u_max_error":1e-12,
            "reconstruction_max_error":1e-10,"orthogonality_max":1e-10,"held_boundary_clipped_fraction_max":.25,
            "residual_energy_ratio_at_least":0.,"shuffle_rms_over_original_sd_at_least":0.,"shuffled_norm_p99_ratio_at_most":100.,
            "train_coverage":1.,"held_coverage":1.,"train_self_rate":0.,"held_effective_donors_per_object":1.}


def fixture():
    rows=[]; train=[]; held=[]; train_field=[]; held_field=[]; train_z={}; held_z={}
    for domain,domain_offset in (("art",0.),("njc",10.)):
        for object_index in range(20):
            obj=f"{domain}-train-{object_index:02d}"; z=-1+2*object_index/19+domain_offset; train_z[obj]=z
            x=-1+2*object_index/19; location=torch.tensor([1+x+x*x,2-x+.5*x*x,-1+.2*x*x,.5+x],dtype=torch.float64)
            direction=torch.tensor([(-1.)**object_index,(object_index%3)-1,(object_index%5)-2,.5*((object_index%4)-1.5)],dtype=torch.float64)
            mean=location+torch.exp(torch.tensor(.2*x))*direction
            for gauge in range(3):
                raw=torch.tensor([torch.sin(torch.tensor((object_index+1)*(k+1)*.13+domain_offset*.01+gauge*.001)).item() for k in range(24)],dtype=torch.float64)
                mechanical=torch.tensor([(object_index+1)**((k%3)+1)*1e-3+(k+1)*gauge*.01+domain_offset*.02 for k in range(8)]+[abs(z)],dtype=torch.float64)
                train.append(len(rows)); rows.append({"domain":domain,"object_group_id":obj,"joint_id":obj,"gauge_id":f"{obj}-g{gauge}","semantic":raw,"mechanical":mechanical,"observed_displacement":abs(z)+.1})
                train_field.append(mean+torch.tensor([gauge,-gauge,.5*gauge,.25*gauge],dtype=torch.float64)*.01)
        for object_index in range(5):
            obj=f"{domain}-held-{object_index:02d}"; z=-.8+.4*object_index+domain_offset; held_z[obj]=z
            x=z-domain_offset; mean=torch.tensor([1+x+x*x,2-x+.5*x*x,-1+.2*x*x,.5+x],dtype=torch.float64)
            for gauge in range(3):
                raw=torch.tensor([torch.sin(torch.tensor((object_index+31)*(k+1)*.13+domain_offset*.01+gauge*.001)).item() for k in range(24)],dtype=torch.float64)
                mechanical=torch.tensor([(object_index+31)**((k%3)+1)*1e-3+(k+1)*gauge*.01+domain_offset*.02 for k in range(8)]+[abs(z)],dtype=torch.float64)
                held.append(len(rows)); rows.append({"domain":domain,"object_group_id":obj,"joint_id":obj,"gauge_id":f"{obj}-g{gauge}","semantic":raw,"mechanical":mechanical,"observed_displacement":abs(z)+.1})
                held_field.append(mean+torch.tensor([gauge,-gauge,.5*gauge,.25*gauge],dtype=torch.float64)*.01)
    return rows,train,held,torch.stack(train_field),torch.stack(held_field),train_z,held_z


def run_fixture(args=None,context="semantic_rqlsot/final-source-features"):
    args=fixture() if args is None else args
    crossfit=feature_crossfit_rqlsot(args[0],args[1],"semantic",contract())
    return rqlsot_ablation(*args,contract(),context,crossfit)


def test_empirical_mid_rank_and_held_clipping_are_frozen():
    train=torch.tensor([0.,1.,1.,3.]); held=torch.tensor([-1.,1.,2.,4.])
    x,hx,clipped=empirical_rank(train,held)
    assert torch.equal(x,torch.tensor([-.75,0.,0.,.75],dtype=torch.float64))
    assert torch.equal(hx,torch.tensor([-.75,0.,.5,.75],dtype=torch.float64))
    assert clipped==.5


def test_rqlsot_exact_reconstruction_repeat_and_row_order_invariance():
    args=fixture(); context="semantic_rqlsot/final-source-features"
    first=run_fixture(args,context); second=run_fixture(args,context)
    assert torch.equal(first[0],second[0]) and torch.equal(first[1],second[1]) and first[2]==second[2]
    rows,train,held,train_field,held_field,train_z,held_z=args
    reverse_args=(rows,list(reversed(train)),list(reversed(held)),train_field.flip(0),held_field.flip(0),train_z,held_z)
    reverse=run_fixture(reverse_args,context)
    first_train={rows[i]["gauge_id"]:first[0][k] for k,i in enumerate(train)}; reversed_train={rows[i]["gauge_id"]:reverse[0][k] for k,i in enumerate(reversed(train))}
    first_held={rows[i]["gauge_id"]:first[1][k] for k,i in enumerate(held)}; reversed_held={rows[i]["gauge_id"]:reverse[1][k] for k,i in enumerate(reversed(held))}
    assert all(torch.equal(first_train[key],reversed_train[key]) for key in first_train)
    assert all(torch.equal(first_held[key],reversed_held[key]) for key in first_held)
    assert first[2]==reverse[2]
    for value in first[2]["domains"].values():
        assert value["reconstruction_max_error"]<1e-10
        assert value["basis_orthogonality_max"]<1e-10 and value["scale_basis_orthogonality_max"]<1e-10
        assert value["mean_qr"]["positive_diagonal"] and value["scale_qr"]["positive_diagonal"]
        assert value["mean_qr"]["rank"]==3 and value["scale_qr"]["rank"]==2


def test_receipt_mapping_mutation_and_rank_failure_are_rejected():
    args=fixture(); context="semantic_rqlsot/final-source-features"; result=run_fixture(args,context); expected=run_fixture(args,context)
    assert receipt_passes(result[2],expected[2],contract(),context,{"art","njc"})
    assert not receipt_passes(result[2],result[2],contract(),context,{"art","njc"})
    bad=copy.deepcopy(result[2]); ids=sorted(bad["domains"]["art"]["train_mapping"])
    bad["domains"]["art"]["train_mapping"]={obj:ids[(i+2)%len(ids)] for i,obj in enumerate(ids)}
    assert not receipt_passes(bad,expected[2],contract(),context,{"art","njc"})
    rows,train,held,train_field,held_field,train_z,held_z=args
    train_z={key:1. for key in train_z}
    try: rqlsot_ablation(rows,train,held,train_field,held_field,train_z,held_z,contract(),context,feature_crossfit_rqlsot(rows,train,"semantic",contract()))
    except ValueError as error: assert "rank/condition" in str(error)
    else: raise AssertionError("rank-deficient quadratic basis was accepted")


def test_mechanical_displacement_is_bit_exact_and_shape_mutation_fails():
    geometry=torch.arange(12,dtype=torch.float64).reshape(3,4); mechanical=torch.randn(3,5,dtype=torch.float64)
    result=preserve_recipient_displacement(geometry,mechanical)
    assert torch.equal(result[:,-1],mechanical[:,-1])
    try: preserve_recipient_displacement(geometry[:,:-1],mechanical)
    except ValueError as error: assert "shape" in str(error)
    else: raise AssertionError("bad mechanical shape was accepted")


def test_crossfit_and_fitted_train_transport_are_held_local():
    args=list(fixture()); context="semantic_rqlsot/final-source-features"
    first=run_fixture(args,context)
    args[4]=args[4]+1000; args[6]={key:value+.01 for key,value in args[6].items()}
    changed=run_fixture(args,context)
    assert torch.equal(first[0],changed[0])
    for domain in first[2]["domains"]:
        for key in ("mean_beta_sha256","scale_gamma_sha256","train_rank_x_sha256","z_sha256"):
            assert first[2]["domains"][domain][key]==changed[2]["domains"][domain][key]
        assert first[2]["domains"][domain]["crossfit"]==changed[2]["domains"][domain]["crossfit"]


def test_frozen_config_and_feature_merge_never_name_label_payloads():
    config=json.loads(Path("configs/rqlsot_v1.json").read_text())
    assert config["model"]["location_basis"]==["1","x","(3*x^2-1)/2"]
    assert not config["source_labels_opened"] and not config["source_labels_hashed"]
    assert "labels.pt" not in inspect.getsource(merge_features)
    assert "labels" not in config["source"]


def test_fold_local_preprocessor_is_fit_without_its_held_objects_and_is_deterministic():
    rows,train,*_=fixture(); first=feature_crossfit_rqlsot(rows,train,"semantic",contract())
    second=feature_crossfit_rqlsot(rows,list(reversed(train)),"semantic",contract())
    assert first==second
    domain="art"; target=first[domain]["audit_payload"]["object_ids"][0]
    held_fold=first[domain]["audit_payload"]["fold_assignment"][target]
    changed=copy.deepcopy(rows)
    for row in changed:
        if row["object_group_id"]==target: row["semantic"]+=1000
    altered=feature_crossfit_rqlsot(changed,train,"semantic",contract())
    before=first[domain]["folds"][held_fold]; after=altered[domain]["folds"][held_fold]
    assert before["preprocessor_sha256"]==after["preprocessor_sha256"]
    assert before["preprocessor_train_object_hash"]==after["preprocessor_train_object_hash"]
    assert first[domain]["standardized_sha256"]!=altered[domain]["standardized_sha256"]


def _set_path(value,path,replacement):
    target=value
    for key in path[:-1]: target=target[key]
    target[path[-1]]=replacement


@pytest.mark.parametrize("path,replacement",[
    (("schema",),"evil"),
    (("context",),"mechanical_rqlsot/final-source-features"),
    (("contract_sha256",),"0"*64),
    (("domains","art","mapping_sha256"),"0"*64),
    (("domains","art","z_sha256"),"0"*64),
    (("domains","art","train_rank_x_sha256"),"0"*64),
    (("domains","art","mean_beta_sha256"),"0"*64),
    (("domains","art","scale_gamma_sha256"),"0"*64),
    (("domains","art","crossfit","standardized_sha256"),"0"*64),
    (("domains","art","crossfit","raw_z_sha256"),"0"*64),
    (("domains","art","crossfit","rank_x_sha256"),"0"*64),
    (("domains","art","crossfit","fold_assignment_sha256"),"0"*64),
    (("domains","art","crossfit","fold_counts"),[1,1,1,1,16]),
    (("domains","art","audit_payload","z"),[0.]*20),
    (("domains","art","crossfit","audit_payload","raw_z"),[0.]*20),
    (("domains","art","crossfit","audit_payload","fold_assignment"),{}),
    (("domains","art","crossfit","folds",0,"preprocessor_sha256"),"0"*64),
    (("domains","art","mean_qr","rank"),2),
    (("domains","art","recipient_u_train_max_error"),1e-2),
])
def test_receipt_mutation(path,replacement):
    context="semantic_rqlsot/final-source-features"; receipt=run_fixture()[2]; expected=run_fixture()[2]
    bad=copy.deepcopy(receipt); _set_path(bad,path,replacement)
    assert not receipt_passes(bad,expected,contract(),context,{"art","njc"})


def test_receipt_rejects_unknown_missing_payload_crosswire_and_u_destruction():
    context="semantic_rqlsot/final-source-features"; result=run_fixture(); receipt=result[2]; expected=run_fixture()[2]
    bad=copy.deepcopy(receipt); bad["unknown"]=1
    assert not receipt_passes(bad,expected,contract(),context,{"art","njc"})
    bad=copy.deepcopy(receipt); del bad["domains"]["art"]["audit_payload"]["z"]
    assert not receipt_passes(bad,expected,contract(),context,{"art","njc"})
    bad=copy.deepcopy(receipt); bad["domains"]["art"]["crossfit"]["field_kind"]="mechanical"
    assert not receipt_passes(bad,expected,contract(),context,{"art","njc"})
    # Recipient within-object offsets are preserved; a one-row perturbation destroys it.
    rows,train,_,train_field,*_=fixture(); output=result[0].clone(); obj=rows[train[0]]["object_group_id"]
    loc=[k for k,i in enumerate(train) if rows[i]["object_group_id"]==obj]
    before=train_field[loc]-train_field[loc].mean(0); after=output[loc]-output[loc].mean(0)
    assert float((before-after).abs().max())<=1e-12
    output[loc[0],0]+=.1; destroyed=output[loc]-output[loc].mean(0)
    assert float((before-destroyed).abs().max())>1e-3


def test_independent_recomputation_rejects_coherent_numeric_and_hash_rewrites():
    context="semantic_rqlsot/final-source-features"; candidate=run_fixture()[2]; expected=run_fixture()[2]
    assert receipt_passes(candidate,expected,contract(),context,{"art","njc"})
    mutations=[]
    bad=copy.deepcopy(candidate); bad["domains"]["art"]["epsilon"]="1e-6"; mutations.append(bad)
    bad=copy.deepcopy(candidate); bad["domains"]["art"]["residual_energy_ratio"]=0.; mutations.append(bad)
    bad=copy.deepcopy(candidate); bad["domains"]["art"]["mean_qr"]["condition"]=0.; mutations.append(bad)
    bad=copy.deepcopy(candidate); bad["domains"]["art"]["held_max_donor_load"]=2; bad["domains"]["art"]["held_load_bound"]=2; mutations.append(bad)
    bad=copy.deepcopy(candidate); fold=copy.deepcopy(bad["domains"]["art"]["crossfit"]["folds"][0]); fold["fold"]=1; bad["domains"]["art"]["crossfit"]["folds"][1]=fold; mutations.append(bad)
    bad=copy.deepcopy(candidate); cf=bad["domains"]["art"]["crossfit"]; standardized=torch.zeros_like(torch.tensor(cf["audit_payload"]["standardized"],dtype=torch.float64))
    cf["audit_payload"]["standardized"]=standardized.tolist(); cf["standardized_sha256"]=_sha_tensor(standardized)
    raw_z=torch.tensor(cf["audit_payload"]["raw_z"],dtype=torch.float64); rank_x=torch.tensor(cf["audit_payload"]["rank_x"],dtype=torch.float64)
    cf["residual_norm_raw_z_absolute_spearman"]=_spearman(standardized.norm(dim=-1),raw_z); cf["residual_raw_z_distance_correlation"]=_distance_correlation(raw_z,standardized)
    cf["residual_norm_rank_x_absolute_spearman_extra"]=_spearman(standardized.norm(dim=-1),rank_x); cf["residual_rank_x_distance_correlation_extra"]=_distance_correlation(rank_x,standardized); mutations.append(bad)
    bad=copy.deepcopy(candidate); payload=bad["domains"]["art"]["audit_payload"]; beta=torch.zeros_like(torch.tensor(payload["mean_beta"],dtype=torch.float64))
    payload["mean_beta"]=beta.tolist(); bad["domains"]["art"]["mean_beta_sha256"]=_sha_tensor(beta); mutations.append(bad)
    assert all(not receipt_passes(bad,expected,contract(),context,{"art","njc"}) for bad in mutations)


def test_mechanical_crossfit_constant_columns_are_zero_filled_and_fold_masks_vary():
    rows,train,*_=fixture(); initial=feature_crossfit_rqlsot(rows,train,"mechanical",contract())
    assignments={d:initial[d]["audit_payload"]["fold_assignment"] for d in initial}
    for row in rows:
        if row["object_group_id"] not in assignments[row["domain"]]: continue
        fold=assignments[row["domain"]][row["object_group_id"]]
        row["mechanical"][0]=float(int(row["object_group_id"].split("-")[-1])+1) if fold==0 else 0.
        row["mechanical"][1]=1.  # Always constant and therefore always unkept.
    result=feature_crossfit_rqlsot(rows,train,"mechanical",contract())
    for domain,value in result.items():
        standardized=torch.tensor(value["audit_payload"]["standardized"],dtype=torch.float64)
        assert torch.isfinite(standardized).all()
        masks=[fold["mechanical_keep_mask"] for fold in value["folds"]]
        assert masks[0][0] is False and any(mask[0] for mask in masks[1:])
        assert all(mask[1] is False and mask[-1] is True for mask in masks)
        for index,obj in enumerate(value["audit_payload"]["object_ids"]):
            if value["audit_payload"]["fold_assignment"][obj]==0: assert standardized[index,0]==0


def test_cross_domain_joint_fold_preprocessor_mutation_is_rejected_even_if_expected_matches():
    context="semantic_rqlsot/final-source-features"; candidate=run_fixture()[2]; expected=copy.deepcopy(candidate)
    for receipt in (candidate,expected):
        fold=receipt["domains"]["njc"]["crossfit"]["folds"][0]
        fold["preprocessor_provenance"]["semantic_mean"]="f"*64
        fold["preprocessor_sha256"]=_canonical_sha(fold["preprocessor_provenance"])
    assert not receipt_passes(candidate,expected,contract(),context,{"art","njc"})


def test_runner_binds_complete_code_and_feature_provenance():
    source=Path("scripts/run_rqlsot_feasibility.py").read_text()
    for name in ("run_smarc_source_gate.py","smarc_source.py","smarc.py","frozen_visual.py","build_smarc_articraft_shard.py","build_smarc_smoke.py"):
        assert name in source
    assert '"code_sha256":code_sha256' in source and '"feature_provenance_sha256":feature_sha256' in source
    result={"schema":"splart-rqlsot-feasibility/v1","config_sha256":FROZEN_CONFIG_SHA256,
            "feature_provenance":{"source":"opaque"},"code_files_sha256":{"runner":"a"},
            "source_labels_opened":False,"source_labels_hashed":False,"source_scores_computed":False,"training_started":False,
            "box_labels_read":[],"protected_splits_read":[]}
    result["feature_provenance_sha256"]=canonical_sha256(result["feature_provenance"]); result["code_sha256"]=canonical_sha256(result["code_files_sha256"])
    receipt={"schema":"splart-rqlsot-feasibility-receipt/v1","config_sha256":FROZEN_CONFIG_SHA256,"feasibility_sha256":"0"*64,
             "decision":"PRUNE_LABEL_FREE","code_sha256":result["code_sha256"],"feature_provenance_sha256":result["feature_provenance_sha256"],
             "source_labels_opened":False,"source_labels_hashed":False,"source_scores_computed":False,"training_started":False,
             "box_labels_read":[],"protected_splits_read":[]}
    assert verify_provenance_binding(result,receipt)
    for key in ("code_sha256","feature_provenance_sha256"):
        bad=copy.deepcopy(receipt); bad[key]="f"*64
        assert not verify_provenance_binding(result,bad)
