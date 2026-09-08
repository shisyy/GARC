import copy
import inspect
import json
from pathlib import Path

import torch

from splart.rqlsot import (empirical_rank, preserve_recipient_displacement,
                            receipt_passes, rqlsot_ablation)
from scripts.run_smarc_source_gate import merge_features


def contract():
    return {"rank_relative_tolerance":1e-12,"condition_max":1e6,"crossfit_folds":5,
            "crossfit_absolute_spearman_at_most":1.,"crossfit_distance_correlation_at_most":1.,
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
                train.append(len(rows)); rows.append({"domain":domain,"object_group_id":obj,"joint_id":obj,"gauge_id":f"{obj}-g{gauge}"})
                train_field.append(mean+torch.tensor([gauge,-gauge,.5*gauge,.25*gauge],dtype=torch.float64)*.01)
        for object_index in range(5):
            obj=f"{domain}-held-{object_index:02d}"; z=-.8+.4*object_index+domain_offset; held_z[obj]=z
            x=z-domain_offset; mean=torch.tensor([1+x+x*x,2-x+.5*x*x,-1+.2*x*x,.5+x],dtype=torch.float64)
            for gauge in range(3):
                held.append(len(rows)); rows.append({"domain":domain,"object_group_id":obj,"joint_id":obj,"gauge_id":f"{obj}-g{gauge}"})
                held_field.append(mean+torch.tensor([gauge,-gauge,.5*gauge,.25*gauge],dtype=torch.float64)*.01)
    return rows,train,held,torch.stack(train_field),torch.stack(held_field),train_z,held_z


def test_empirical_mid_rank_and_held_clipping_are_frozen():
    train=torch.tensor([0.,1.,1.,3.]); held=torch.tensor([-1.,1.,2.,4.])
    x,hx,clipped=empirical_rank(train,held)
    assert torch.equal(x,torch.tensor([-.75,0.,0.,.75],dtype=torch.float64))
    assert torch.equal(hx,torch.tensor([-.75,0.,.5,.75],dtype=torch.float64))
    assert clipped==.5


def test_rqlsot_exact_reconstruction_repeat_and_row_order_invariance():
    args=fixture(); context="semantic_rqlsot/final-source-features"
    first=rqlsot_ablation(*args,contract(),context); second=rqlsot_ablation(*args,contract(),context)
    assert torch.equal(first[0],second[0]) and torch.equal(first[1],second[1]) and first[2]==second[2]
    rows,train,held,train_field,held_field,train_z,held_z=args
    reverse=rqlsot_ablation(rows,list(reversed(train)),list(reversed(held)),train_field.flip(0),held_field.flip(0),train_z,held_z,contract(),context)
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
    args=fixture(); context="semantic_rqlsot/final-source-features"; result=rqlsot_ablation(*args,contract(),context)
    assert receipt_passes(result[2],contract(),{context},{"art","njc"})
    bad=copy.deepcopy(result[2]); ids=sorted(bad["domains"]["art"]["train_mapping"])
    bad["domains"]["art"]["train_mapping"]={obj:ids[(i+2)%len(ids)] for i,obj in enumerate(ids)}
    assert not receipt_passes(bad,contract(),{context},{"art","njc"})
    rows,train,held,train_field,held_field,train_z,held_z=args
    train_z={key:1. for key in train_z}
    try: rqlsot_ablation(rows,train,held,train_field,held_field,train_z,held_z,contract(),context)
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
    first=rqlsot_ablation(*args,contract(),context)
    args[4]=args[4]+1000; args[6]={key:value+.01 for key,value in args[6].items()}
    changed=rqlsot_ablation(*args,contract(),context)
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
