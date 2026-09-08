import torch

from splart.smarc_source import (deterministic_object_donors, fit_preprocessor,
                                 hash_ids, object_domain_weights, object_macro_mare)


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


def test_preprocessor_deterministic_and_macro_metric():
    rows=rows_fixture(); targets=torch.full((len(rows),),2.)
    p0=fit_preprocessor(rows,targets,list(range(len(rows))),pca_dim=2)
    p1=fit_preprocessor(rows,targets,list(range(len(rows))),pca_dim=2)
    assert torch.equal(p0.semantic_components,p1.semantic_components)
    pred=torch.tensor([1.,3.,2.,2.,2.])
    assert object_macro_mare(rows,list(range(5)),pred,targets) == .125
    assert hash_ids(["b","a"]) == hash_ids(["a","b"])


def test_shuffle_is_object_level_derangement_and_singletons_fail():
    rows=rows_fixture(); train=list(range(len(rows))); values={k:.25 for k in "abcd"}
    donors=deterministic_object_donors(rows,train,train,values,[0.,.5,1.])
    assert all(k != v for k,v in donors.items())
    values={"a":.25,"b":.75,"c":.75,"d":.75}
    try:
        deterministic_object_donors(rows,train,train,values,[0.,.5,1.])
    except ValueError as error:
        assert "fewer than two" in str(error)
    else:
        raise AssertionError("singleton shuffle bin must fail closed")
