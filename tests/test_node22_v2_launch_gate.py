import json
from pathlib import Path
from audit_node22_v2_launch_gate import audit

ROOT = Path(__file__).resolve().parents[1]

def test_final_public_and_sanitized_remote_gate_passes():
    value=audit(ROOT); assert value["status"]=="PASS"; assert value["launch_authorized"] is True; assert value["one_execution_only"] is True; assert value["v3_allowed"] is False; assert value["sealed_content_parsed"] is False; assert value["failures"]==[]

def test_remote_binding_tamper_fails_closed(tmp_path):
    value=json.loads((ROOT/"node22_v2_remote_preflight.json").read_text()); value["baseline_export"]["sha256"]="0"*64; path=tmp_path/"tampered.json"; path.write_text(json.dumps(value)); result=audit(ROOT,path); assert result["status"]=="FAIL"; assert result["launch_authorized"] is False; assert "REMOTE-BASELINE" in result["failures"]

def test_output_must_be_absent(tmp_path):
    value=json.loads((ROOT/"node22_v2_remote_preflight.json").read_text()); value["output"]["exists"]=True; path=tmp_path/"used-output.json"; path.write_text(json.dumps(value)); assert "REMOTE-OUTPUT-NOT-FRESH" in audit(ROOT,path)["failures"]
