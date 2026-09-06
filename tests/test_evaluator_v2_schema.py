import pytest
import torch
from run_node22_sealed_evaluator import target_pair,metric_multiplier,normalized_object_metrics,validate_baselines
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
