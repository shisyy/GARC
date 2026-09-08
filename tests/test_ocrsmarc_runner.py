import copy
import hashlib
import json
from pathlib import Path

import torch

from scripts.run_smarc_source_gate import (NULL_FIELD_DEFINITIONS, aggregate_partials,
                                            canonical_sha256)
from splart.frozen_visual import sha256_file
from splart.smarc_source import hash_ids


def mapping_hash(context,domain,train,held):
    payload=f"context={context}\ndomain={domain}\n"+"\n".join(f"train:{key}->{train[key]}" for key in sorted(train))+"\n"+"\n".join(f"held:{key}->{held[key]}" for key in sorted(held))+"\n"
    return hashlib.sha256(payload.encode()).hexdigest()


def null_receipt(context):
    result={"context":context,"domains":{}}
    for domain in ("articraft","njc"):
        ids=[f"{domain}-train-{i}" for i in range(10)]; held_ids=[f"{domain}-held-{i}" for i in range(2)]
        train={obj:ids[(i+1)%10] for i,obj in enumerate(ids)}; held={obj:ids[i] for i,obj in enumerate(held_ids)}
        object_hash=lambda values:hashlib.sha256(("\n".join(values)+"\n").encode()).hexdigest()
        result["domains"][domain]={"train_objects":10,"held_objects":2,"train_object_hash":object_hash(ids),"held_object_hash":object_hash(held_ids),"z_sha256":"0"*64,"ols_beta_sha256":"1"*64,
            "train_mapping":train,"held_mapping":held,"train_coverage":1.,"held_coverage":1.,"train_self_rate":0.,"train_donor_marginal_exact":True,
            "held_effective_donors":2,"held_effective_donors_per_object":1.,"held_max_donor_load":1,"held_load_bound":1,
            "reconstruction_max_error":0.,"normalized_residual_mean_max":0.,"normalized_residual_z_correlation_max":0.,
            "residual_energy_ratio":1.,"shuffle_rms_over_original_sd":1.,"shuffled_norm_p99_ratio":1.,
            "crossfit":{"fold_counts":[2]*5,"residual_norm_z_absolute_spearman":0.,"residual_z_distance_correlation":0.},
            "mapping_sha256":mapping_hash(context,domain,train,held)}
    return result


def make_bundle():
    semantic=null_receipt("semantic_conditional_residual/final-source")
    mechanical=null_receipt("mechanical_conditional_residual/final-source")
    mechanical["recipient_displacement_bitwise_unchanged"]=True
    result={"schema":"splart-ocrsmarc-null-preflight/v1","semantic_conditional_residual":semantic,
            "mechanical_conditional_residual":mechanical,"field_definitions":NULL_FIELD_DEFINITIONS,
            "passes":{"semantic_conditional_residual":True,"mechanical_conditional_residual":True},"all_pass":True}
    return result,canonical_sha256(result)


def write_partial(root,task,record,config_sha,code):
    root.mkdir(parents=True); torch.save({},root/"models.pt")
    (root/"partial.json").write_text(json.dumps(record,sort_keys=True)+"\n")
    receipt={"task":task,"config_sha256":config_sha,"steps":1200,"code_sha256":code,
             "source_provenance_sha256":record["source_provenance_sha256"],"null_preflight_sha256":record["null_preflight_sha256"],
             "partial_sha256":sha256_file(root/"partial.json"),"models_sha256":sha256_file(root/"models.pt"),
             "box_labels_read":[],"protected_splits_read":[]}
    (root/"receipt.json").write_text(json.dumps(receipt,sort_keys=True)+"\n")


def fixture(tmp_path):
    config=json.loads(Path("configs/smarc_source_v1.json").read_text()); config_sha="synthetic-config"; code={"runner":"code"}
    config["data"]["articraft"].update(train_objects=10,validation_objects=2)
    config["data"]["njc"].update(train_objects=10,validation_objects=2)
    null,null_sha=make_bundle(); analytics={"stronger_lower_mare":{key:{"articraft":.2,"njc":.2} for key in ("global","displacement","category")}}
    label_hashes=[item["labels"] for item in config["data"]["frozen_render_payload_sha256"]]+[config["data"]["frozen_source_sha256"]["articraft_labels"],config["data"]["frozen_source_sha256"]["njc_labels"]]
    label_files={f"label-{i}.pt":value for i,value in enumerate(label_hashes)}
    merge={"labels_opened":True,"label_file_sha256":label_files,"target_payloads_read":sorted(label_files)}
    source_sha=canonical_sha256(merge)
    swap={key:{"range_max":0.,"projection_max":0.,"endpoint_max":0.} for key in ("full","mechanical_only","semantic_conditional_residual","mechanical_conditional_residual")}
    final={"schema":"splart-ocrsmarc-source-partial/v1","task":"final","config_sha256":config_sha,"steps":1200,"source_provenance_sha256":source_sha,"merge":merge,
           "null_preflight":null,"null_preflight_sha256":null_sha,"metrics":{key:{"articraft":value,"njc":value} for key,value in (("full",.1),("mechanical_only",.2),("semantic_conditional_residual",.13),("mechanical_conditional_residual",.13))},
           "analytic_controls":analytics,"swap":swap,"box_labels_read":[],"protected_splits_read":[]}
    records=[final]
    counts=(37,51,21)
    keys=("lofo_window_test37","lofo_sewing_test51","lofo_usb_test21_small")
    train_keys=("lofo_window_train72","lofo_sewing_train58","lofo_usb_train88")
    config["lofo_fold_contract"]=[]
    for index,(count,key,train_key) in enumerate(zip(counts,keys,train_keys)):
        ids=[f"held-{index}-{j}" for j in range(count)]; config["data"]["object_list_hashes"][key]=hash_ids(ids)
        config["data"]["object_list_hashes"][train_key]=f"train-{index}"
        config["lofo_fold_contract"].append({"family_audit_id":f"family-{index}","train_hash_key":train_key,"held_hash_key":key,"held_objects":count})
        records.append({"schema":"splart-ocrsmarc-source-partial/v1","task":f"lofo_{index}","config_sha256":config_sha,"steps":1200,
            "source_provenance_sha256":source_sha,"null_preflight":null,"null_preflight_sha256":null_sha,"held_object_ids":ids,
            "family_audit_id":f"family-{index}","train_object_hash":f"train-{index}","held_object_hash":hash_ids(ids),"held_objects":count,
            "full":.1,"mechanical_only":.2,"analytic_controls":analytics,"swap":{"full":swap["full"],"mechanical_only":swap["mechanical_only"]},
            "box_labels_read":[],"protected_splits_read":[]})
    dirs=[]
    for record in records:
        path=tmp_path/record["task"]; write_partial(path,record["task"],record,config_sha,code); dirs.append(path)
    return dirs,config,config_sha,code


def test_four_partial_aggregate_is_order_invariant_and_single_create(tmp_path):
    dirs,config,config_sha,code=fixture(tmp_path)
    first=tmp_path/"aggregate-a"; second=tmp_path/"aggregate-b"
    aggregate_partials(dirs,first,config,config_sha,code)
    aggregate_partials(list(reversed(dirs)),second,config,config_sha,code)
    assert (first/"metrics.json").read_bytes()==(second/"metrics.json").read_bytes()
    assert (first/"receipt.json").read_bytes()==(second/"receipt.json").read_bytes()
    assert not (tmp_path/"aggregate-a.atomic-tmp").exists()
    stale=tmp_path/"blocked.atomic-tmp"; stale.mkdir()
    try: aggregate_partials(dirs,tmp_path/"blocked",config,config_sha,code)
    except FileExistsError: pass
    else: raise AssertionError("stale transaction directory was accepted")


def test_aggregate_rejects_partial_and_null_mutation(tmp_path):
    dirs,config,config_sha,code=fixture(tmp_path)
    record=json.loads((dirs[0]/"partial.json").read_text()); record["null_preflight"]["mechanical_conditional_residual"]["recipient_displacement_bitwise_unchanged"]=False
    (dirs[0]/"partial.json").write_text(json.dumps(record,sort_keys=True)+"\n")
    receipt=json.loads((dirs[0]/"receipt.json").read_text()); receipt["partial_sha256"]=sha256_file(dirs[0]/"partial.json")
    (dirs[0]/"receipt.json").write_text(json.dumps(receipt,sort_keys=True)+"\n")
    try: aggregate_partials(dirs,tmp_path/"bad",config,config_sha,code)
    except RuntimeError as error: assert "conditional null" in str(error)
    else: raise AssertionError("weakened null was accepted")


def test_aggregate_rejects_stale_code_and_overlapping_lofo(tmp_path):
    dirs,config,config_sha,code=fixture(tmp_path/"stale-case")
    receipt=json.loads((dirs[1]/"receipt.json").read_text()); receipt["code_sha256"]={"runner":"stale"}
    (dirs[1]/"receipt.json").write_text(json.dumps(receipt,sort_keys=True)+"\n")
    try: aggregate_partials(dirs,tmp_path/"stale",config,config_sha,code)
    except RuntimeError as error: assert "stale" in str(error)
    else: raise AssertionError("stale code was accepted")
    dirs,config,config_sha,code=fixture(tmp_path/"overlap-case")
    first=json.loads((dirs[1]/"partial.json").read_text())
    second=json.loads((dirs[2]/"partial.json").read_text())
    second["held_object_ids"][0]=first["held_object_ids"][0]
    second["held_object_hash"]=hash_ids(second["held_object_ids"])
    config["data"]["object_list_hashes"]["lofo_sewing_test51"]=second["held_object_hash"]
    (dirs[2]/"partial.json").write_text(json.dumps(second,sort_keys=True)+"\n")
    receipt=json.loads((dirs[2]/"receipt.json").read_text()); receipt["partial_sha256"]=sha256_file(dirs[2]/"partial.json")
    (dirs[2]/"receipt.json").write_text(json.dumps(receipt,sort_keys=True)+"\n")
    try: aggregate_partials(dirs,tmp_path/"overlap",config,config_sha,code)
    except RuntimeError as error: assert "overlapping" in str(error)
    else: raise AssertionError("overlapping LOFO folds were accepted")


def test_aggregate_binds_receipt_provenance_and_all_six_lofo_hashes(tmp_path):
    for field,bad_value in (("steps",2),("source_provenance_sha256","other"),("null_preflight_sha256","other")):
        dirs,config,config_sha,code=fixture(tmp_path/field)
        receipt=json.loads((dirs[0]/"receipt.json").read_text()); receipt[field]=bad_value
        (dirs[0]/"receipt.json").write_text(json.dumps(receipt,sort_keys=True)+"\n")
        try: aggregate_partials(dirs,tmp_path/(field+"-out"),config,config_sha,code)
        except RuntimeError as error: assert "bound" in str(error)
        else: raise AssertionError(f"unbound {field} was accepted")
    dirs,config,config_sha,code=fixture(tmp_path/"family")
    record=json.loads((dirs[1]/"partial.json").read_text()); record["train_object_hash"]="tampered"
    (dirs[1]/"partial.json").write_text(json.dumps(record,sort_keys=True)+"\n")
    receipt=json.loads((dirs[1]/"receipt.json").read_text()); receipt["partial_sha256"]=sha256_file(dirs[1]/"partial.json")
    (dirs[1]/"receipt.json").write_text(json.dumps(receipt,sort_keys=True)+"\n")
    try: aggregate_partials(dirs,tmp_path/"family-out",config,config_sha,code)
    except RuntimeError as error: assert "family/train/held" in str(error)
    else: raise AssertionError("tampered LOFO train hash was accepted")


def test_each_lofo_fold_is_a_required_gate(tmp_path):
    dirs,config,config_sha,code=fixture(tmp_path/"fold-gate")
    record=json.loads((dirs[1]/"partial.json").read_text()); record["full"]=.19
    (dirs[1]/"partial.json").write_text(json.dumps(record,sort_keys=True)+"\n")
    receipt=json.loads((dirs[1]/"receipt.json").read_text()); receipt["partial_sha256"]=sha256_file(dirs[1]/"partial.json")
    (dirs[1]/"receipt.json").write_text(json.dumps(receipt,sort_keys=True)+"\n")
    result=aggregate_partials(dirs,tmp_path/"fold-gate-out",config,config_sha,code)
    assert not result["gates"]["lofo_each_fold_pass_15pct"]
    assert result["gates"]["decision"]=="PRUNE_BEFORE_BOX"


def _rewrite_record_and_receipt(path,record):
    (path/"partial.json").write_text(json.dumps(record,sort_keys=True)+"\n")
    receipt=json.loads((path/"receipt.json").read_text())
    receipt["partial_sha256"]=sha256_file(path/"partial.json")
    receipt["source_provenance_sha256"]=record["source_provenance_sha256"]
    receipt["null_preflight_sha256"]=record["null_preflight_sha256"]
    (path/"receipt.json").write_text(json.dumps(receipt,sort_keys=True)+"\n")


def test_aggregate_rejects_alternate_null_and_incomplete_label_provenance(tmp_path):
    dirs,config,config_sha,code=fixture(tmp_path/"shift")
    record=json.loads((dirs[0]/"partial.json").read_text()); null=record["null_preflight"]
    part=null["semantic_conditional_residual"]["domains"]["articraft"]
    ids=sorted(part["train_mapping"]); part["train_mapping"]={obj:ids[(i+2)%len(ids)] for i,obj in enumerate(ids)}
    part["mapping_sha256"]=mapping_hash(null["semantic_conditional_residual"]["context"],"articraft",part["train_mapping"],part["held_mapping"])
    record["null_preflight_sha256"]=canonical_sha256(null); _rewrite_record_and_receipt(dirs[0],record)
    try: aggregate_partials(dirs,tmp_path/"shift-out",config,config_sha,code)
    except RuntimeError as error: assert "null" in str(error)
    else: raise AssertionError("alternate shift-two null was accepted")
    dirs,config,config_sha,code=fixture(tmp_path/"schema")
    record=json.loads((dirs[0]/"partial.json").read_text()); record["null_preflight"]["schema"]="evil"
    record["null_preflight_sha256"]=canonical_sha256(record["null_preflight"]); _rewrite_record_and_receipt(dirs[0],record)
    try: aggregate_partials(dirs,tmp_path/"schema-out",config,config_sha,code)
    except RuntimeError as error: assert "null" in str(error)
    else: raise AssertionError("evil null schema was accepted")
    dirs,config,config_sha,code=fixture(tmp_path/"labels")
    record=json.loads((dirs[0]/"partial.json").read_text()); record["merge"]["labels_opened"]=False
    record["source_provenance_sha256"]=canonical_sha256(record["merge"]); _rewrite_record_and_receipt(dirs[0],record)
    for path in dirs[1:]:
        other=json.loads((path/"partial.json").read_text()); other["source_provenance_sha256"]=record["source_provenance_sha256"]; _rewrite_record_and_receipt(path,other)
    try: aggregate_partials(dirs,tmp_path/"labels-out",config,config_sha,code)
    except RuntimeError as error: assert "merge" in str(error)
    else: raise AssertionError("unopened label provenance was accepted")
