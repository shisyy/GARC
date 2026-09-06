import pytest
import torch
from run_node22_sealed_evaluator import target_pair,metric_multiplier,normalized_object_metrics,validate_baselines,error_summary,validate_authorization,fsha,bootstrap_ci
def row():return {'endpoint_truth':{'local_endpoint_targets':{'extension0_joint_units':1.,'extension0_observation_units':.2,'extension1_joint_units':2.,'extension1_observation_units':.4,'local_lower_scalar':-.2,'local_upper_scalar':1.4}}}
def test_exact_builder_schema():assert target_pair(row())==[.2,.4]
def test_fail_closed_extra_or_missing():
 x=row();x['endpoint_truth']['local_endpoint_targets']['target']=9
 with pytest.raises(ValueError):target_pair(x)
 x=row();x['endpoint_truth']['local_endpoint_targets']['extension0_observation_units']='.2'
 with pytest.raises(ValueError):target_pair(x)
def test_direction_and_swap_q_orders():
 a=row();assert target_pair(a)==[.2,.4]
 b=row();b['endpoint_truth']['local_endpoint_targets']['extension0_observation_units'],b['endpoint_truth']['local_endpoint_targets']['extension1_observation_units']=.4,.2
 assert target_pair(b)==[.4,.2] and target_pair(b)[::-1]==target_pair(a)
def test_max_side_and_normalized_width():
 p=torch.tensor([[.1,.5]]);t=torch.tensor([[.2,.2]]);lo=torch.tensor([[0.,.1]]);hi=torch.tensor([[.2,.7]]);m=torch.tensor([2.])
 e,w=normalized_object_metrics(p,t,lo,hi,m);assert e.item()==pytest.approx(.6) and w.item()==pytest.approx(.8)
def test_normalizer_exact_schema_and_types():
 x=row();x['endpoint_truth']['target_nmae_normalization']={'local_scalar_error_multiplier':2.,'object_score_formula':'max(lower_endpoint_nmae,upper_endpoint_nmae)','per_endpoint_formula':'abs(predicted_local_scalar-target_local_scalar)*local_scalar_error_multiplier','physical_range':1.};assert metric_multiplier(x)==2
 x['endpoint_truth']['target_nmae_normalization']['local_scalar_error_multiplier']='2'
 with pytest.raises(ValueError):metric_multiplier(x)
def test_scaled_score_multiplier_cancels_and_baseline_gate():
 residual=torch.tensor([[.2,.4]]);scale=torch.tensor([[.1,.2]]);mult=torch.tensor([3.])
 assert torch.equal((residual*mult[:,None])/(scale*mult[:,None]),residual/scale)
 schema={'required_methods':['scratch'],'forbidden_fields':['score','target','split']};b={'schema':'splart-node22-target-free-baseline-predictions/v2','methods':{'scratch':{'rows':[{'object_id':str(i),'distances':[.1,.2]} for i in range(36)],'provenance':{'source_commit':'a','source_tree':'b','config_sha256':'c','prediction_tree_sha256':'d'}}}}
 assert len(validate_baselines(b,{str(i) for i in range(36)},schema)['scratch'])==36
 b['methods']['scratch']['rows'][0]['score']=1
 with pytest.raises(ValueError):validate_baselines(b,{str(i) for i in range(36)},schema)
def test_paper_error_summary_is_aggregate_only():
 p=torch.tensor([[0.,.4],[.4,0.]]);t=torch.zeros_like(p);m=torch.tensor([2.,1.]);s,rows=error_summary(p,t,m)
 assert s['lower_nmae']==pytest.approx(.2) and s['upper_nmae']==pytest.approx(.4) and s['endpoint_nmae']==pytest.approx(.6)
 assert set(s)=={'lower_nmae','upper_nmae','endpoint_nmae','median_endpoint_nmae'} and rows.shape==(2,)
def test_authorization_exact_hash_and_path(tmp_path):
 from types import SimpleNamespace
 truth=tmp_path/'truth';truth.write_text('x');base=tmp_path/'base';base.write_text('y');idx=tmp_path/'idx';idx.write_text('z');out=tmp_path/'fresh';rows=[{'object_id':str(i)} for i in range(36)];a=SimpleNamespace(truth=str(truth),baseline_export=str(base),index=[str(idx)],output=str(out));ids=sorted(r['object_id'] for r in rows)
 auth={'schema':'splart-node22-v2-authorization/v1','status':'AUTHORIZED_FOR_EVALUATOR','runner_sha256':fsha(__import__('run_node22_sealed_evaluator').__file__),'truth_sha256':fsha(truth),'profile_index_sha256':[fsha(idx)],'profile_set_sha256':__import__('hashlib').sha256('\n'.join(ids).encode()).hexdigest(),'baseline_export_sha256':fsha(base),'output_absolute_path':str(out.resolve()),'one_execution_only':True,'v3_allowed':False};validate_authorization(auth,a,rows)
 auth['output_absolute_path']+='-tamper'
 with pytest.raises(ValueError):validate_authorization(auth,a,rows)
def test_fixed_object_bootstrap_is_reproducible():
 g=torch.Generator().manual_seed(220290);idx=torch.randint(0,9,(10000,9),generator=g);a=bootstrap_ci(torch.arange(9,dtype=torch.float32),idx)
 g=torch.Generator().manual_seed(220290);assert a==bootstrap_ci(torch.arange(9,dtype=torch.float32),torch.randint(0,9,(10000,9),generator=g))
