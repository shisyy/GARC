#!/usr/bin/env python3
"""Run the unique label-free RQ-LSOT feasibility candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from scripts.run_smarc_source_gate import (FROZEN_CONFIG_SHA256 as SOURCE_CONFIG_SHA256,
                                            _atomic_json, _begin_atomic_directory,
                                            _finish_atomic_directory, conditional_overrides,
                                            indices_for, merge_features, object_scalar)
from splart.frozen_visual import sha256_file
from splart.rqlsot import (feature_crossfit_rqlsot, preserve_recipient_displacement,
                           receipt_passes, rqlsot_ablation)
from splart.smarc_source import fit_feature_preprocessor, hash_ids, transform


FROZEN_CONFIG_SHA256="e3d6c1b3d59e2fd4c155fcff7e818cce0af95343da81c9d2f01bb2990c0a5acc"
CONTEXTS={"semantic":"semantic_rqlsot/final-source-features","mechanical":"mechanical_rqlsot/final-source-features"}


def text_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n",b"\n")).hexdigest()


def canonical_sha256(value) -> str:
    return hashlib.sha256((json.dumps(value,sort_keys=True,separators=(",",":"))+"\n").encode()).hexdigest()


def verify_provenance_binding(result: dict,receipt: dict) -> bool:
    receipt_keys={"schema","config_sha256","feasibility_sha256","decision","code_sha256","feature_provenance_sha256",
                  "source_labels_opened","source_labels_hashed","source_scores_computed","training_started","box_labels_read","protected_splits_read"}
    if set(receipt)!=receipt_keys or receipt.get("schema")!="splart-rqlsot-feasibility-receipt/v1" or receipt.get("config_sha256")!=FROZEN_CONFIG_SHA256: return False
    if result.get("schema")!="splart-rqlsot-feasibility/v1" or result.get("config_sha256")!=FROZEN_CONFIG_SHA256: return False
    if receipt.get("decision") not in {"REQUEST_SOURCE_SCORE_AUTHORIZATION","PRUNE_LABEL_FREE"} or not isinstance(receipt.get("feasibility_sha256"),str) or len(receipt["feasibility_sha256"])!=64: return False
    if receipt.get("code_sha256")!=canonical_sha256(result.get("code_files_sha256")) or receipt.get("feature_provenance_sha256")!=canonical_sha256(result.get("feature_provenance")): return False
    if result.get("code_sha256")!=receipt["code_sha256"] or result.get("feature_provenance_sha256")!=receipt["feature_provenance_sha256"]: return False
    protected=("source_labels_opened","source_labels_hashed","source_scores_computed","training_started")
    return (all(result.get(key) is False and receipt.get(key) is False for key in protected) and
            result.get("box_labels_read")==receipt.get("box_labels_read")==[] and result.get("protected_splits_read")==receipt.get("protected_splits_read")==[])


def contract(config: dict,source_config: dict) -> dict:
    unchanged=source_config["conditional_residual_contract"]["gates"]
    if unchanged.get("normalized_residual_mean_max")!=1e-10 or unchanged.get("normalized_residual_z_correlation_max")!=1e-10:
        raise RuntimeError("node8.9 unchanged-gate thresholds changed")
    return {"rank_relative_tolerance":config["numerics"]["rank_relative_tolerance"],
            "condition_max":config["numerics"]["condition_max"],"crossfit_folds":config["crossfit"]["folds"],
            "crossfit_absolute_spearman_at_most":config["crossfit"]["absolute_spearman_at_most"],
            "crossfit_distance_correlation_at_most":config["crossfit"]["distance_correlation_at_most"],
            "normalized_residual_mean_max":unchanged["normalized_residual_mean_max"],
            "normalized_residual_raw_z_correlation_max":unchanged["normalized_residual_z_correlation_max"],
            "recipient_u_max_error":1e-12,
            **{key:value for key,value in config["gates"].items() if isinstance(value,(int,float)) and not isinstance(value,bool)}}


def _remap(rows,indices,value):
    return {_row_identity(rows[i]):value[k] for k,i in enumerate(indices)}


def _row_identity(row):
    return str(row["gauge_id"])


def _bit_equal_mapping(left,right) -> bool:
    return set(left)==set(right) and all(torch.equal(left[key],right[key]) for key in left)


def run_cell(rows,train,held,train_field,held_field,train_z,held_z,cfg,context,field_kind):
    first_crossfit=feature_crossfit_rqlsot(rows,train,field_kind,cfg)
    second_crossfit=feature_crossfit_rqlsot(rows,train,field_kind,cfg)
    first=rqlsot_ablation(rows,train,held,train_field,held_field,train_z,held_z,cfg,context,first_crossfit)
    second=rqlsot_ablation(rows,train,held,train_field,held_field,train_z,held_z,cfg,context,second_crossfit)
    for domain in first[2]["domains"]:
        left=first[2]["domains"][domain]["crossfit"]; right=second[2]["domains"][domain]["crossfit"]
        if left is right or left["audit_payload"] is right["audit_payload"] or left["folds"] is right["folds"]:
            raise RuntimeError("independent crossfit receipts share nested identity")
    repeat=torch.equal(first[0],second[0]) and torch.equal(first[1],second[1]) and first[2]==second[2]
    reverse_train=list(reversed(train)); reverse_held=list(reversed(held))
    reverse_crossfit=feature_crossfit_rqlsot(rows,reverse_train,field_kind,cfg)
    reverse=rqlsot_ablation(rows,reverse_train,reverse_held,train_field.flip(0),held_field.flip(0),train_z,held_z,cfg,context,reverse_crossfit)
    invariant=(_bit_equal_mapping(_remap(rows,train,first[0]),_remap(rows,reverse_train,reverse[0])) and
               _bit_equal_mapping(_remap(rows,held,first[1]),_remap(rows,reverse_held,reverse[1])) and first[2]==reverse[2])
    return first[0],first[1],first[2],second[2],{"repeat_bit_identical":repeat,"row_order_invariant":invariant}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--articraft-shards",type=Path,nargs=3,required=True); parser.add_argument("--njc-cache",type=Path,required=True)
    parser.add_argument("--articraft-pilc-data",type=Path,required=True); parser.add_argument("--njc-pilc-data",type=Path,required=True)
    parser.add_argument("--render-logs",type=Path,nargs=4,required=True); parser.add_argument("--dino-checkpoint",type=Path,required=True)
    parser.add_argument("--source-config",type=Path,required=True); parser.add_argument("--config",type=Path,required=True); parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists() or (args.output.parent/(args.output.name+".atomic-tmp")).exists(): raise FileExistsError(args.output)
    config=json.loads(args.config.read_text()); source_config=json.loads(args.source_config.read_text())
    if text_sha256(args.config)!=FROZEN_CONFIG_SHA256 or text_sha256(args.source_config)!=SOURCE_CONFIG_SHA256: raise RuntimeError("frozen config hash mismatch")
    if any(key in config for key in ("labels","scores")) or config.get("source_labels_opened") or config.get("source_labels_hashed") or config.get("source_scores_computed"): raise RuntimeError("label-free config boundary violated")
    rows,provenance=merge_features(args,source_config)
    if provenance.get("target_payloads_read")!=[] or provenance.get("labels_opened") is not False: raise RuntimeError("target payload was accessed")
    log_hashes=[sha256_file(path) for path in args.render_logs]
    if log_hashes!=config["source"]["render_log_sha256"] or any("overflow" in path.read_text(errors="replace").lower() for path in args.render_logs): raise RuntimeError("renderer log contract failed")
    if sha256_file(args.dino_checkpoint)!=config["source"]["dino_checkpoint_sha256"]: raise RuntimeError("DINO hash mismatch")
    train=indices_for(rows,{"endpoint_pretrain","njc_train"}); held=indices_for(rows,{"endpoint_validation","njc_validation"})
    expected=config["source"]["object_list_hashes"]
    observed={"articraft_train":hash_ids(sorted({rows[i]["object_group_id"] for i in train if rows[i]["domain"]=="articraft"})),
              "articraft_held":hash_ids(sorted({rows[i]["object_group_id"] for i in held if rows[i]["domain"]=="articraft"})),
              "njc_train":hash_ids(sorted({rows[i]["object_group_id"] for i in train if rows[i]["domain"]=="njc"})),
              "njc_held":hash_ids(sorted({rows[i]["object_group_id"] for i in held if rows[i]["domain"]=="njc"}))}
    if observed!=expected: raise RuntimeError("frozen object-list hash mismatch")
    preprocessor=fit_feature_preprocessor(rows,train,16)
    train_mech,train_sem,_=transform(preprocessor,rows,train); held_mech,held_sem,_=transform(preprocessor,rows,held)
    _,linear_history=conditional_overrides(rows,train,held,preprocessor,source_config)
    cfg=contract(config,source_config)
    semantic=run_cell(rows,train,held,train_sem,held_sem,object_scalar(rows,train,"displacement"),object_scalar(rows,held,"displacement"),cfg,CONTEXTS["semantic"],"semantic")
    mechanical=run_cell(rows,train,held,train_mech[:,:-1],held_mech[:,:-1],object_scalar(rows,train,"semantic_distance",train),object_scalar(rows,held,"semantic_distance",train),cfg,CONTEXTS["mechanical"],"mechanical")
    mechanical_train=preserve_recipient_displacement(mechanical[0],train_mech); mechanical_held=preserve_recipient_displacement(mechanical[1],held_mech)
    d_equal=torch.equal(mechanical_train[:,-1],train_mech[:,-1]) and torch.equal(mechanical_held[:,-1],held_mech[:,-1])
    domains={"articraft","njc"}
    cell_pass={}; full_receipt_pass={}; domain_receipt_pass={}
    for name,result in (("semantic",semantic[2]),("mechanical",mechanical[2])):
        cell=semantic if name=="semantic" else mechanical; receipt=result; expected_receipt=cell[3]; audit=cell[4]; cell_pass[name]={}; domain_receipt_pass[name]={}
        full_receipt_pass[name]=receipt_passes(receipt,expected_receipt,cfg,CONTEXTS[name],domains)
        for domain in sorted(domains):
            single={"schema":receipt["schema"],"context":receipt["context"],"contract_sha256":receipt["contract_sha256"],"domains":{domain:receipt["domains"][domain]}}
            expected_single={"schema":expected_receipt["schema"],"context":expected_receipt["context"],"contract_sha256":expected_receipt["contract_sha256"],"domains":{domain:expected_receipt["domains"][domain]}}
            domain_gate=receipt_passes(single,expected_single,cfg,CONTEXTS[name],{domain})
            domain_receipt_pass[name][domain]=domain_gate
            cell_pass[name][domain]=(domain_gate and audit["repeat_bit_identical"] and
                                     audit["row_order_invariant"] and (name!="mechanical" or d_equal))
    code_root=Path(__file__).resolve().parents[1]
    code_files=(Path(__file__).resolve(),code_root/"src"/"splart"/"rqlsot.py",code_root/"src"/"splart"/"conditional_residual.py",
                code_root/"scripts"/"run_smarc_source_gate.py",code_root/"src"/"splart"/"smarc_source.py",code_root/"src"/"splart"/"smarc.py",
                code_root/"src"/"splart"/"frozen_visual.py",code_root/"scripts"/"build_smarc_articraft_shard.py",code_root/"scripts"/"build_smarc_smoke.py")
    code_hashes={path.relative_to(code_root).as_posix():sha256_file(path) for path in code_files}
    code_sha256=canonical_sha256(code_hashes); feature_sha256=canonical_sha256(provenance)
    result={"schema":"splart-rqlsot-feasibility/v1","config_sha256":FROZEN_CONFIG_SHA256,
            "source_config_sha256":SOURCE_CONFIG_SHA256,"feature_provenance":provenance,"feature_provenance_sha256":feature_sha256,
            "node89_linear_historical":linear_history,"rqlsot":{"semantic":semantic[2],"mechanical":mechanical[2]},"independent_recomputation_sha256":{"semantic":canonical_sha256(semantic[3]),"mechanical":canonical_sha256(mechanical[3])},
            "runtime_audits":{"semantic":semantic[4],"mechanical":{**mechanical[4],"recipient_displacement_bitwise_unchanged":d_equal}},"full_receipt_pass":full_receipt_pass,"domain_receipt_pass":domain_receipt_pass,
            "cell_pass":cell_pass,"all_pass":all(full_receipt_pass.values()) and all(value for part in cell_pass.values() for value in part.values()),"object_list_hashes":observed,
            "code_files_sha256":code_hashes,"code_sha256":code_sha256,
            "source_labels_opened":False,"source_labels_hashed":False,"source_scores_computed":False,"training_started":False,
            "box_labels_read":[],"protected_splits_read":[]}
    if canonical_sha256(result["feature_provenance"])!=feature_sha256 or canonical_sha256(result["code_files_sha256"])!=code_sha256: raise RuntimeError("provenance self-check failed")
    temporary=_begin_atomic_directory(args.output); _atomic_json(temporary/"feasibility.json",result)
    output_receipt={"schema":"splart-rqlsot-feasibility-receipt/v1","config_sha256":FROZEN_CONFIG_SHA256,
        "feasibility_sha256":sha256_file(temporary/"feasibility.json"),"decision":"REQUEST_SOURCE_SCORE_AUTHORIZATION" if result["all_pass"] else "PRUNE_LABEL_FREE",
        "code_sha256":code_sha256,"feature_provenance_sha256":feature_sha256,
        "source_labels_opened":False,"source_labels_hashed":False,"source_scores_computed":False,"training_started":False,"box_labels_read":[],"protected_splits_read":[]}
    if not verify_provenance_binding(result,output_receipt):
        raise RuntimeError("output receipt provenance binding failed")
    _atomic_json(temporary/"receipt.json",output_receipt)
    _finish_atomic_directory(temporary,args.output)
    print(json.dumps({"output":str(args.output),"all_pass":result["all_pass"],"cell_pass":cell_pass}))


if __name__=="__main__": main()
