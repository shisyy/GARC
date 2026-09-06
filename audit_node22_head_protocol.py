#!/usr/bin/env python3
import argparse, hashlib, json, math
from pathlib import Path

OLD_SHA = "2f85224478d1ded6683e2634e41aac7548f1fe793dcdb725a614823d599c3ee7"
FORBIDDEN = ("b_test", "full22")

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def audit(old_path, addendum_path):
    old=json.loads(old_path.read_text()); a=json.loads(addendum_path.read_text()); failures=[]
    def require(value, name):
        if not value: failures.append(name)
    require(sha(old_path)==OLD_SHA==a["supersedes"]["sha256"], "old-preregister-binding")
    require(a["status"]=="FROZEN_BEFORE_TARGET_OR_SPLIT_READ", "freeze-status")
    require(a["target_files_read"]==a["split_membership_read"]==a["score_files_read"]==[], "no-private-read")
    require(a["launch_authorized"] is False, "fail-closed-launch")
    t=a["training"]
    for key in ("optimizer","learning_rate","betas","weight_decay","steps","batch","checkpoint","rerun","seed","initialization"):
        require(key in t, "training-"+key)
    require(t["steps"]==4000 and t["seed"]==2202 and "exactly one" in t["checkpoint"], "fixed-training")
    require(set(a["fixed_comparators"]) >= {"coordinate_only","scalar_only_mlp","pooled_summary_mlp","fixed_permutation","unshared_head","evaluation"}, "strong-comparators")
    require("zero_geometry" not in a["fixed_comparators"] and "order_only" not in a["fixed_comparators"], "merged-coordinate-null")
    c=a["conformal"]; rank=math.ceil((9+1)*c["coverage"])
    require(rank==9 and "fail if rank>n_cal" in c["quantile"], "finite-sample-rank")
    require(c["primary_claim"]=="simultaneous two-endpoint object coverage only", "coverage-only-primary")
    require("no superiority gate" in c["width_reporting"], "no-width-claim")
    require(set(a["protected_inputs"]) >= {"B_test", "Full22"}, "protected-splits-explicit")
    return {"schema":"splart-node2.2-head-protocol-p0/v1","status":"PASS" if not failures else "FAIL","failures":failures,"old_sha256":sha(old_path),"addendum_sha256":sha(addendum_path),"launch_authorized":False}

def main():
    p=argparse.ArgumentParser(); p.add_argument("--old",type=Path,required=True); p.add_argument("--addendum",type=Path,required=True); p.add_argument("--output",type=Path,required=True); x=p.parse_args()
    if x.output.exists(): raise FileExistsError(x.output)
    result=audit(x.old,x.addendum); x.output.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
    if result["status"]!="PASS": raise SystemExit(1)
if __name__=="__main__": main()
