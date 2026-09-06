import pytest
from run_node22_sealed_evaluator import target_pair
def row():return {'endpoint_truth':{'local_endpoint_targets':{'extension0_joint_units':1.,'extension0_observation_units':.2,'extension1_joint_units':2.,'extension1_observation_units':.4,'local_lower_scalar':-.2,'local_upper_scalar':1.4}}}
def test_exact_builder_schema():assert target_pair(row())==[.2,.4]
def test_fail_closed_extra_or_missing():
 x=row();x['endpoint_truth']['local_endpoint_targets']['target']=9
 with pytest.raises(ValueError):target_pair(x)
