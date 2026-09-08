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
from splart.rqlsot import preserve_recipient_displacement, receipt_passes, rqlsot_ablation
from splart.smarc_source import fit_feature_preprocessor, hash_ids, transform


FROZEN_CONFIG_SHA256="e3d6c1b3d59e2fd4c155fcff7e818cce0af95343da81c9d2f01bb2990c0a5acc"
CONTEXTS={"semantic":"semantic_rqlsot/final-source-features","mechanical":"mechanical_rqlsot/final-source-features"}


def text_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n",b"\n")).hexdigest()


def canonical_sha256(value) -> str:
    return hashlib.sha256((json.dumps(value,sort_keys=True,separators=(",",":"))+"\n").encode()).hexdigest()


def contract(config: dict) -> dict:
    return {"rank_relative_tolerance":config["numerics"]["rank_relative_tolerance"],
            "condition_max":config["numerics"]["condition_max"],"crossfit_folds":config["crossfit"]["folds"],
            "crossfit_absolute_spearman_at_most":config["crossfit"]["absolute_spearman_at_most"],
            "crossfit_distance_correlation_at_most":config["crossfit"]["distance_correlation_at_most"],
            **{key:value for key,value in config["gates"].items() if isinstance(value,(int,float)) and not isinstance(value,bool)}}


def _remap(rows,indices,value):
    return {_row_identity(rows[i]):value[k] for k,i in enumerate(indices)}


def _row_identity(row):
    return str(row["gauge_id"])


def _bit_equal_mapping(left,right) -> bool:
    return set(left)==set(right) and all(torch.equal(left[key],right[key]) for key in left)


def run_cell(rows,train,held,train_field,held_field,train_z,held_z,cfg,context):
    first=rqlsot_ablation(rows,train,held,train_field,held_field,train_z,held_z,cfg,context)
    second=rqlsot_ablation(rows,train,held,train_field,held_field,train_z,held_z,cfg,context)
    repeat=torch.equal(first[0],second[0]) and torch.equal(first[1],second[1]) and first[2]==second[2]
    reverse_train=list(reversed(train)); reverse_held=list(reversed(held))
    reverse=rqlsot_ablation(rows,reverse_train,reverse_held,train_field.flip(0),held_field.flip(0),train_z,held_z,cfg,context)
    invariant=(_bit_equal_mapping(_remap(rows,train,first[0]),_remap(rows,reverse_train,reverse[0])) and
               _bit_equal_mapping(_remap(rows,held,first[1]),_remap(rows,reverse_held,reverse[1])) and first[2]==reverse[2])
    receipt=first[2]; receipt["repeat_bit_identical"]=repeat; receipt["row_order_invariant"]=invariant
    return first[0],first[1],receipt


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
    cfg=contract(config)
    semantic=run_cell(rows,train,held,train_sem,held_sem,object_scalar(rows,train,"displacement"),object_scalar(rows,held,"displacement"),cfg,CONTEXTS["semantic"])
    mechanical=run_cell(rows,train,held,train_mech[:,:-1],held_mech[:,:-1],object_scalar(rows,train,"semantic_distance",train),object_scalar(rows,held,"semantic_distance",train),cfg,CONTEXTS["mechanical"])
    mechanical_train=preserve_recipient_displacement(mechanical[0],train_mech); mechanical_held=preserve_recipient_displacement(mechanical[1],held_mech)
    d_equal=torch.equal(mechanical_train[:,-1],train_mech[:,-1]) and torch.equal(mechanical_held[:,-1],held_mech[:,-1])
    mechanical[2]["recipient_displacement_bitwise_unchanged"]=d_equal
    contexts=set(CONTEXTS.values()); domains={"articraft","njc"}
    cell_pass={}
    for name,result in (("semantic",semantic[2]),("mechanical",mechanical[2])):
        cell_pass[name]={}
        for domain in sorted(domains):
            single={"schema":result["schema"],"context":result["context"],"domains":{domain:result["domains"][domain]}}
            cell_pass[name][domain]=(receipt_passes(single,cfg,contexts,{domain}) and result["repeat_bit_identical"] and
                                     result["row_order_invariant"] and (name!="mechanical" or d_equal))
    code_root=Path(__file__).resolve().parents[1]
    code_files=(Path(__file__).resolve(),code_root/"src"/"splart"/"rqlsot.py",code_root/"src"/"splart"/"conditional_residual.py")
    result={"schema":"splart-rqlsot-feasibility/v1","config_sha256":FROZEN_CONFIG_SHA256,
            "source_config_sha256":SOURCE_CONFIG_SHA256,"feature_provenance":provenance,"feature_provenance_sha256":canonical_sha256(provenance),
            "node89_linear_historical":linear_history,"rqlsot":{"semantic":semantic[2],"mechanical":mechanical[2]},
            "cell_pass":cell_pass,"all_pass":all(value for part in cell_pass.values() for value in part.values()),"object_list_hashes":observed,
            "code_sha256":{path.relative_to(code_root).as_posix():sha256_file(path) for path in code_files},
            "source_labels_opened":False,"source_labels_hashed":False,"source_scores_computed":False,"training_started":False,
            "box_labels_read":[],"protected_splits_read":[]}
    temporary=_begin_atomic_directory(args.output); _atomic_json(temporary/"feasibility.json",result)
    _atomic_json(temporary/"receipt.json",{"schema":"splart-rqlsot-feasibility-receipt/v1","config_sha256":FROZEN_CONFIG_SHA256,
        "feasibility_sha256":sha256_file(temporary/"feasibility.json"),"decision":"REQUEST_SOURCE_SCORE_AUTHORIZATION" if result["all_pass"] else "PRUNE_LABEL_FREE",
        "source_labels_opened":False,"source_labels_hashed":False,"source_scores_computed":False,"training_started":False,"box_labels_read":[],"protected_splits_read":[]})
    _finish_atomic_directory(temporary,args.output)
    print(json.dumps({"output":str(args.output),"all_pass":result["all_pass"],"cell_pass":cell_pass}))


if __name__=="__main__": main()
