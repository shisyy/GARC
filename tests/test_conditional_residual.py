import copy

import torch

from splart.conditional_residual import (_rank, conditional_residual_ablation,
                                         crossfit_diagnostics, receipt_passes)


def fixture():
    rows=[]; train=[]; held=[]; train_field=[]; held_field=[]; train_z={}; held_z={}
    for domain,offset in (("art",0.),("njc",10.)):
        for obj_index in range(40):
            obj=f"{domain}-train-{obj_index:02d}"; z=.1*obj_index+offset; train_z[obj]=z
            residual=torch.tensor([(-1.)**obj_index*.3,(obj_index%3-1)*.2,(obj_index%5-2)*.1],dtype=torch.float64)
            for gauge in range(2):
                train.append(len(rows)); rows.append({"domain":domain,"object_group_id":obj,"joint_id":obj})
                train_field.append(torch.tensor([1+2*z,2-z,.5*z],dtype=torch.float64)+residual+torch.tensor([gauge*.01,-gauge*.01,gauge*.02]))
        for obj_index in range(4):
            obj=f"{domain}-held-{obj_index:02d}"; z=.15*obj_index+offset; held_z[obj]=z
            for gauge in range(2):
                held.append(len(rows)); rows.append({"domain":domain,"object_group_id":obj,"joint_id":obj})
                held_field.append(torch.tensor([1+2*z,2-z,.5*z],dtype=torch.float64)+torch.tensor([gauge*.01,-gauge*.01,gauge*.02]))
    return rows,train,held,torch.stack(train_field),torch.stack(held_field),train_z,held_z


def test_conditional_residual_has_full_coverage_marginal_and_offsets():
    rows,train,held,train_field,held_field,train_z,held_z=fixture()
    shuffled_train,shuffled_held,receipt=conditional_residual_ablation(rows,train,held,train_field,held_field,train_z,held_z,context="semantic/test")
    assert shuffled_train.shape==train_field.shape and shuffled_held.shape==held_field.shape
    for domain,value in receipt["domains"].items():
        assert value["train_coverage"]==value["held_coverage"]==1
        assert value["train_self_rate"]==0 and value["train_donor_marginal_exact"]
        assert value["held_effective_donors_per_object"]==1 and value["held_max_donor_load"]==1
        assert value["reconstruction_max_error"]<1e-10
        assert value["normalized_residual_mean_max"]<1e-10
        assert value["normalized_residual_z_correlation_max"]<1e-10
    # Recipient within-object gauge offsets are exactly retained.
    assert torch.allclose(shuffled_train[1::2]-shuffled_train[0::2],train_field[1::2]-train_field[0::2],atol=1e-12,rtol=0)
    assert torch.allclose(shuffled_held[1::2]-shuffled_held[0::2],held_field[1::2]-held_field[0::2],atol=1e-12,rtol=0)


def test_conditional_fit_is_deterministic_and_degenerate_z_fails():
    args=fixture(); first=conditional_residual_ablation(*args,context="repeat"); second=conditional_residual_ablation(*args,context="repeat")
    assert torch.equal(first[0],second[0]) and torch.equal(first[1],second[1])
    assert first[2]==second[2]
    rows,train,held,train_field,held_field,train_z,held_z=args
    train_z={key:1. for key in train_z}
    try: conditional_residual_ablation(rows,train,held,train_field,held_field,train_z,held_z,context="degenerate")
    except ValueError as error: assert "degenerate" in str(error)
    else: raise AssertionError("constant conditioning scalar must fail closed")


def test_crossfit_is_opaque_hash_deterministic():
    ids=[f"object-{i:03d}" for i in range(100)]; z=torch.arange(100,dtype=torch.float64)
    field=torch.stack((z.sin(),z.cos()),-1)
    first=crossfit_diagnostics(ids,z,field); second=crossfit_diagnostics(ids,z,field)
    assert first==second and sum(first["fold_counts"])==100 and min(first["fold_counts"])>0


def test_average_rank_ties_are_equal():
    ranks=_rank(torch.tensor([3.,1.,1.,2.,3.]))
    assert torch.equal(ranks,torch.tensor([3.5,.5,.5,2.,3.5],dtype=torch.float64))


def test_domain_overlap_and_uninitialized_held_fail_closed():
    args=list(fixture()); rows,train,held=args[:3]
    rows[held[0]]["domain"]="held-only"
    try: conditional_residual_ablation(*args,context="held-only")
    except ValueError as error: assert "domain sets" in str(error)
    else: raise AssertionError("held-only domain was silently accepted")
    args=list(fixture()); rows,train,held=args[:3]
    rows[held[0]]["object_group_id"]=rows[train[0]]["object_group_id"]
    try: conditional_residual_ablation(*args,context="overlap")
    except ValueError as error: assert "crosses" in str(error)
    else: raise AssertionError("train/held object overlap was silently accepted")


def _contract():
    return {"gates":{"reconstruction_max_error":1e-10,"normalized_residual_mean_max":1e-10,
        "normalized_residual_z_correlation_max":1e-10,"coverage":1.,"train_self_rate":0.,
        "residual_energy_ratio_at_least":0.,"shuffle_rms_over_original_sd_at_least":0.,
        "shuffled_norm_p99_ratio_at_most":100.,"held_effective_donors_per_object":1.},
        "crossfit":{"folds":5,"absolute_spearman_at_most":1.,"distance_correlation_at_most":1.}}


def test_receipt_validation_recomputes_mapping_loads_and_rejects_empty():
    args=fixture(); receipt=conditional_residual_ablation(*args,context="receipt")[2]
    assert receipt_passes(receipt,_contract(),{"art","njc"})
    assert not receipt_passes({"context":"x","domains":{}},_contract(),{"art","njc"})
    for mutation in ("load","mapping","domain","fold"):
        bad=copy.deepcopy(receipt)
        if mutation=="load": bad["domains"]["art"]["held_max_donor_load"]+=1
        elif mutation=="mapping": bad["domains"]["art"]["held_mapping"]["art-held-00"]="art-train-01"
        elif mutation=="domain": del bad["domains"]["njc"]
        else: bad["domains"]["art"]["crossfit"]["fold_counts"][0]=0
        assert not receipt_passes(bad,_contract(),{"art","njc"})
