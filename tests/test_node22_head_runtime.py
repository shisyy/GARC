import pytest, torch
from splart.node22_head import *
from splart.gauge_energy_profile import canonical_outward_coordinates

def data(n=4,s=17):
    torch.manual_seed(9); f=torch.randn(n,2,3,s,9)
    lo=-torch.linspace(.05,2,s).repeat(n,3,1); hi=1+torch.linspace(.05,2,s).repeat(n,3,1)
    raw=torch.stack((lo,hi),1); norm=ProfileNormalizer(f,raw); return f,raw,norm(f,raw)

def test_shared_distance_scale_exact_swap():
    f,raw,(nf,nx)=data(); m=Node22Head(); initialize_frozen(m)
    p,s=m(nf,nx); assert (p>=0).all() and (s>=1e-4).all(); assert exact_swap_error(m,nf,nx)==0

def test_all_frozen_comparators_run_and_permutation_is_fixed():
    f,raw,(nf,nx)=data()
    for v in VARIANTS:
        m=Node22Head(v); initialize_frozen(m); p,s=m(nf,nx); assert p.shape==s.shape==(4,2)
    assert torch.equal(Node22Head('fixed_permutation')._side_input(nf[:,0],nx[:,0])[...,:9],nf[:,0].flip(-2))
    m=Node22Head('shared_distance_only');initialize_frozen(m);assert torch.equal(m(nf,nx)[1],torch.ones(4,2))

def test_joint_scaled_and_constant_conformal_and_aggregate():
    p=torch.ones(9,2); t=p+torch.arange(9)[:,None]/100; scale=torch.full_like(p,.5); ids=[f'o{i}' for i in range(9)]
    a=fit_joint_conformal(p,scale,t,ids); b=fit_joint_conformal(p,scale,t,ids,kind='constant')
    assert a.q==pytest.approx(.16) and b.q==pytest.approx(.08)
    rows=[{'object_id':x,'endpoint_nmae':.1,'joint_covered':1,'mean_joint_width':.2} for x in ids]
    out=aggregate_confirmatory(rows); assert out['objects']==9 and not out['per_object_targets_emitted']

def test_fixed_config_guard_and_private_denylist():
    assert FROZEN_CONFIG.steps==4000 and FROZEN_CONFIG.seed==2202
    g=RunGuard(); assert g.may_retry(); g.record_retry(); assert not g.may_retry(); g=RunGuard(); g.optimizer_initialized=True
    with pytest.raises(RuntimeError): g.record_retry()
    for bad in ({'target':[1,2]},{'meta':{'presentation_bit':1}},{'path':'x/B_test/y'}):
        with pytest.raises(ValueError): reject_private(bad)
    reject_private({'object_id':'opaque','profile_sha256':'abc'})

def test_train_contract_rejects_wrong_count_before_optimization():
    f,raw,_=data(); target=torch.ones(4,2)
    with pytest.raises(ValueError,match='exactly 18'):
        train_once(f,raw,target,['a','b','c','d'])
