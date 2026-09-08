import torch

from splart.smarc_source import (deterministic_object_donors, fit_preprocessor,
                                 hash_ids, object_domain_weights, object_macro_mare, swap_audit, transform)


def rows_fixture():
    rows=[]
    for domain, objects in (("art", ("a", "b")), ("njc", ("c", "d"))):
        for obj in objects:
            count=2 if obj == "a" else 1
            for joint in range(count):
                rows.append({"domain":domain,"object_group_id":obj,"joint_id":f"{obj}{joint}",
                             "mechanical":torch.tensor([float(ord(obj)),float(joint)]),
                             "semantic":torch.tensor([float(ord(obj)),float(joint),1.]),
                             "observed_displacement":.5})
    return rows


def test_domain_object_weights_do_not_overweight_multi_joint_objects():
    rows=rows_fixture(); indices=list(range(len(rows))); w=object_domain_weights(rows,indices)
    assert torch.allclose(w.sum(),torch.tensor(1.,dtype=torch.float64))
    assert w[:2].sum() == w[2] == w[3] == w[4]


def test_hierarchy_equalizes_joints_even_with_different_gauge_counts():
    rows=[]
    for joint,count in (("j0",2),("j1",5)):
        for gauge in range(count):
            rows.append({"domain":"art","object_group_id":"a","joint_id":joint})
    w=object_domain_weights(rows,list(range(7)))
    assert torch.allclose(w[:2].sum(),torch.tensor(.5,dtype=torch.float64))
    assert torch.allclose(w[2:].sum(),torch.tensor(.5,dtype=torch.float64))


def test_preprocessor_deterministic_and_macro_metric():
    rows=rows_fixture(); targets=torch.full((len(rows),),2.)
    p0=fit_preprocessor(rows,targets,list(range(len(rows))),pca_dim=2)
    p1=fit_preprocessor(rows,targets,list(range(len(rows))),pca_dim=2)
    assert torch.equal(p0.semantic_components,p1.semantic_components)
    pred=torch.tensor([1.,3.,2.,2.,2.])
    assert object_macro_mare(rows,list(range(5)),pred,targets) == .125
    assert hash_ids(["b","a"]) == hash_ids(["a","b"])


def test_mechanical_only_raw_mean_becomes_exact_zero_post_pca():
    rows=rows_fixture(); targets=torch.full((len(rows),),2.); indices=list(range(len(rows)))
    prep=fit_preprocessor(rows,targets,indices,pca_dim=2)
    raw=prep.semantic_mean[None].repeat(len(rows),1)
    _,semantic,_=transform(prep,rows,indices,semantic_override=raw)
    assert torch.equal(semantic,torch.zeros_like(semantic))


def test_swap_audit_executes_and_detects_broken_predictions():
    base={"swap_pair_id":"p","observed_displacement":1.,"base_extension":torch.tensor([.2,.4],dtype=torch.float64)}
    rows=[dict(base,order="forward"),dict(base,order="reverse",base_extension=torch.tensor([.4,.2],dtype=torch.float64))]
    good=swap_audit(rows,[0,1],torch.tensor([2.,2.],dtype=torch.float64))
    assert max(good.values())<1e-12
    broken=swap_audit(rows,[0,1],torch.tensor([2.,2.1],dtype=torch.float64))
    assert broken["range_max"]>.09 and broken["endpoint_max"]>0


def test_shuffle_is_object_level_derangement_and_singletons_fail():
    rows=rows_fixture(); train=list(range(len(rows))); values={k:.25 for k in "abcd"}
    try:
        deterministic_object_donors(rows,train,train,values)
    except ValueError as error:
        assert "degenerate" in str(error)
    values={"a":.1,"b":.2,"c":.3,"d":.4}
    donors,receipt=deterministic_object_donors(rows,train,train,values)
    donors2,receipt2=deterministic_object_donors(rows,train,train,values)
    assert all(k != v for k,v in donors.items()) and donors["a"] in {"a","b"} and donors["c"] in {"c","d"}
    assert receipt["train_distance"]["max"]>0 and receipt["mapping_sha256"]
    assert donors==donors2 and receipt["mapping_sha256"]==receipt2["mapping_sha256"]


def test_matching_never_crosses_domains_when_other_domain_changes():
    rows=rows_fixture(); train=list(range(len(rows)))
    values={"a":.1,"b":.2,"c":.3,"d":.4}
    first,_=deterministic_object_donors(rows,train,train,values)
    values["c"],values["d"]=100.,200.
    second,_=deterministic_object_donors(rows,train,train,values)
    assert first["a"]==second["a"] and first["b"]==second["b"]
    assert all(next(r["domain"] for r in rows if r["object_group_id"]==k)==next(r["domain"] for r in rows if r["object_group_id"]==v) for k,v in second.items())
