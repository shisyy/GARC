import json, math
from pathlib import Path
import audit_node22_head_protocol as audit

ROOT=Path(__file__).resolve().parents[1]
def test_protocol_passes_and_stays_locked(tmp_path):
    out=audit.audit(ROOT/'gauge_energy_preregister.json',ROOT/'node22_head_protocol_addendum.json')
    assert out['status']=='PASS' and out['launch_authorized'] is False

def test_conformal_rank_is_finite_and_joint():
    a=json.loads((ROOT/'node22_head_protocol_addendum.json').read_text())
    assert math.ceil((9+1)*a['conformal']['coverage'])==9
    assert a['conformal']['profile_conditioned_score'].startswith('max_side')

def test_no_private_read_receipts():
    a=json.loads((ROOT/'node22_head_protocol_addendum.json').read_text())
    assert a['target_files_read']==a['split_membership_read']==a['score_files_read']==[]
