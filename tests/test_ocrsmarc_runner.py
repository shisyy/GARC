import copy
import hashlib
import json
from pathlib import Path

import torch

from scripts.run_smarc_source_gate import aggregate_partials, canonical_sha256
from splart.frozen_visual import sha256_file
from splart.smarc_source import hash_ids


def mapping_hash(context,domain,train,held):
    payload=f"context={context}\ndomain={domain}\n"+"\n".join(f"train:{key}->{train[key]}" for key in sorted(train))+"\n"+"\n".join(f"held:{key}->{held[key]}" for key in sorted(held))+"\n"
    return hashlib.sha256(payload.encode()).hexdigest()


def null_receipt(context):
    result={"context":context,"domains":{}}
    for domain in ("articraft","njc"):
        ids=[f"{domain}-train-{i}" for i in range(5)]; held_ids=[f"{domain}-held-{i}" for i in range(2)]
        train={obj:ids[(i+1)%5] for i,obj in enumerate(ids)}; held={obj:ids[i] for i,obj in enumerate(held_ids)}
        result["domains"][domain]={"train_objects":5,"held_objects":2,"train_object_hash":"x","held_object_hash":"y","z_sha256":"z","ols_beta_sha256":"b",
            "train_mapping":train,"held_mapping":held,"train_coverage":1.,"held_coverage":1.,"train_self_rate":0.,"train_donor_marginal_exact":True,
            "held_effective_donors":2,"held_effective_donors_per_object":1.,"held_max_donor_load":1,"held_load_bound":1,
            "reconstruction_max_error":0.,"normalized_residual_mean_max":0.,"normalized_residual_z_correlation_max":0.,
            "residual_energy_ratio":1.,"shuffle_rms_over_original_sd":1.,"shuffled_norm_p99_ratio":1.,
            "crossfit":{"fold_counts":[1]*5,"residual_norm_z_absolute_spearman":0.,"residual_z_distance_correlation":0.},
            "mapping_sha256":mapping_hash(context,domain,train,held)}
    return result


def make_bundle():
    semantic=null_receipt("semantic_conditional_residual/final-source")
    mechanical=null_receipt("mechanical_conditional_residual/final-source")
    mechanical["recipient_displacement_bitwise_unchanged"]=True
    result={"schema":"splart-ocrsmarc-null-preflight/v1","semantic_conditional_residual":semantic,
            "mechanical_conditional_residual":mechanical,"passes":{"semantic_conditional_residual":True,"mechanical_conditional_residual":True},"all_pass":True}
    return result,canonical_sha256(result)


def write_partial(root,task,record,config_sha,code):
    root.mkdir(parents=True); torch.save({},root/"models.pt")
    (root/"partial.json").write_text(json.dumps(record,sort_keys=True)+"\n")
    receipt={"task":task,"config_sha256":config_sha,"steps":1200,"code_sha256":code,
             "source_provenance_sha256":"source","null_preflight_sha256":record["null_preflight_sha256"],
             "partial_sha256":sha256_file(root/"partial.json"),"models_sha256":sha256_file(root/"models.pt"),
             "box_labels_read":[],"protected_splits_read":[]}
    (root/"receipt.json").write_text(json.dumps(receipt,sort_keys=True)+"\n")


def fixture(tmp_path):
    config=json.loads(Path("configs/smarc_source_v1.json").read_text()); config_sha="synthetic-config"; code={"runner":"code"}
    config["data"]["articraft"].update(train_objects=5,validation_objects=2)
    config["data"]["njc"].update(train_objects=5,validation_objects=2)
    null,null_sha=make_bundle(); analytics={"stronger_lower_mare":{key:{"articraft":.2,"njc":.2} for key in ("global","displacement","category")}}
    swap={key:{"range_max":0.,"projection_max":0.,"endpoint_max":0.} for key in ("full","mechanical_only","semantic_conditional_residual","mechanical_conditional_residual")}
    final={"schema":"splart-ocrsmarc-source-partial/v1","task":"final","config_sha256":config_sha,"steps":1200,"source_provenance_sha256":"source",
           "null_preflight":null,"null_preflight_sha256":null_sha,"metrics":{key:{"articraft":value,"njc":value} for key,value in (("full",.1),("mechanical_only",.2),("semantic_conditional_residual",.13),("mechanical_conditional_residual",.13))},
           "analytic_controls":analytics,"swap":swap,"box_labels_read":[],"protected_splits_read":[]}
    records=[final]
    counts=(37,51,21)
    keys=("lofo_window_test37","lofo_sewing_test51","lofo_usb_test21_small")
    for index,(count,key) in enumerate(zip(counts,keys)):
        ids=[f"held-{index}-{j}" for j in range(count)]; config["data"]["object_list_hashes"][key]=hash_ids(ids)
        records.append({"schema":"splart-ocrsmarc-source-partial/v1","task":f"lofo_{index}","config_sha256":config_sha,"steps":1200,
            "source_provenance_sha256":"source","null_preflight":null,"null_preflight_sha256":null_sha,"held_object_ids":ids,
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
