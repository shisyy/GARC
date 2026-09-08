#!/usr/bin/env python3
"""Run the preregistered, source-only Phase-A SMARC gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from splart.frozen_visual import encode_state_pair, load_dino_vitb16, sha256_file
from splart.smarc import (extensions_to_endpoints, opaque_row_key, project_extensions_to_range,
                          swap_invariant_pair)
from splart.smarc_source import (analytic_linear_baseline, apply_object_donors,
                                 deterministic_object_donors, fit_preprocessor, hash_ids,
                                 object_domain_weights, object_macro_mare, predict, swap_audit, train_model)


def load_cache(path: Path, domain: str, config: dict) -> tuple[list[dict], dict[str, float], dict]:
    rows=torch.load(path/"inputs.pt",map_location="cpu",weights_only=False)
    labels=torch.load(path/"labels.pt",map_location="cpu",weights_only=False)
    manifest=json.loads((path/"manifest.json").read_text())
    expected=("splart-smarc-articraft-dino-zbuffer-shard/v1" if domain=="articraft" else
              "splart-smarc-njc-dino-zbuffer/v1")
    if manifest.get("schema")!=expected or manifest.get("encoder_sha256")!="bf34ad0f424b9029b593e8dc3ed553bf26e88bcba0d32bf3e62a6209cb64c85e":
        raise RuntimeError("cache schema or frozen DINO hash mismatch")
    if manifest.get("box_labels_read")!=[] or manifest.get("protected_splits_read")!=[]:
        raise RuntimeError("source cache reports protected reads")
    if "bin_size=0" not in manifest.get("renderer",""):
        raise RuntimeError("cache did not use the overflow-safe formal renderer")
    coverage=manifest.get("coverage",[])
    if not coverage or any(item.get("coverage")!=1.0 or item.get("static_visuals",0)+item.get("moving_visuals",0)!=item.get("visual_total") for item in coverage):
        raise RuntimeError("cache visual assignment coverage mismatch")
    for item in coverage:
        foreground=[value for key,value in item.items() if "foreground_min" in key]
        if not foreground or any(not torch.isfinite(torch.tensor(value)) or value<=0 for value in foreground):
            raise RuntimeError("cache foreground coverage invalid")
    render_bank=torch.load(path/"neutral_render_bank.pt",map_location="cpu",weights_only=False)
    if set(render_bank)!={row["joint_id"] for row in rows} or any(tuple(value.shape)!=(2,config["render"]["views_per_state"],3,config["render"]["resolution"],config["render"]["resolution"]) for value in render_bank.values()):
        raise RuntimeError("cache view count/resolution contract mismatch")
    if domain=="articraft" and (manifest.get("empty_views")!=0 or manifest.get("render_repeat_bit_identical") is not True or "full-child-subtree/static-complement" not in manifest["renderer"]):
        raise RuntimeError("Articraft topology/determinism manifest mismatch")
    if len(config["render"]["elevation_azimuth_degrees"])!=config["render"]["views_per_state"]:
        raise RuntimeError("frozen camera contract mismatch")
    for row in rows:
        row["domain"]=domain
    return rows,{key:float(value) for key,value in labels.items()},manifest


F0=(.15,.25,.35); F1=(.65,.75,.85); ORIENTATION=(1,-1); ORDER=("forward","reverse")


def njc_key(split: str, name: str, f0: float, f1: float, orientation: int, order: str) -> str:
    base=f"{split}:{name}:{f0:.2f}:{f1:.2f}:axis{orientation:+d}"
    return hashlib.sha256(f"splart-pilc-v1:{base}:{order}".encode()).hexdigest()


def expand_gauges(canonical: list[dict], art_pilc: Path, njc_pilc: Path) -> tuple[list[dict], Tensor]:
    """Join all 32 public gauges while storing visual semantics only once per joint."""
    canonical_by_joint={row["joint_id"]:row for row in canonical}; expanded=[]; truth=[]; seen=set()
    for domain,path in (("articraft",art_pilc),("njc",njc_pilc)):
        manifest=json.loads((path/"manifest.json").read_text())
        source={row["key"]:row for row in torch.load(path/"inputs.pt",map_location="cpu",weights_only=False)}
        labels={row["key"]:row for row in torch.load(path/"labels.pt",map_location="cpu",weights_only=False)}
        if set(source)!=set(labels): raise RuntimeError(f"{domain} sidecar key mismatch")
        if domain=="articraft":
            object_splits=[(split,name) for split in ("endpoint_pretrain","endpoint_validation") for name in manifest["accepted"][split]]
            for split,name in object_splits:
                joint=0
                while True:
                    canonical_key=opaque_row_key(name,joint,.25,.75,1,"forward")
                    jid=hashlib.sha256(("splart-smarc-joint-v1:"+canonical_key).encode()).hexdigest()
                    if jid not in canonical_by_joint: break
                    base=canonical_by_joint[jid]
                    for f0 in F0:
                        for f1 in F1:
                            if f1-f0<.35: continue
                            for orientation in ORIENTATION:
                                for order in ORDER:
                                    key=opaque_row_key(name,joint,f0,f1,orientation,order)
                                    if key not in source or source[key]["split"]!=split: raise RuntimeError("Articraft gauge join failed")
                                    x,y=source[key],labels[key]
                                    pair_raw=f"{name}:joint{joint}:{f0}:{f1}:{orientation}"
                                    row=dict(base); row["mechanical"]=torch.cat((swap_invariant_pair(x["state0_features"],x["state1_features"]),torch.tensor([abs(float(x["observed_displacement"]))]))); row["observed_displacement"]=abs(float(x["observed_displacement"])); row["gauge_id"]=key; row["swap_pair_id"]=hashlib.sha256(("splart-smarc-swap-v1:"+pair_raw).encode()).hexdigest(); row["order"]=order; row["base_extension"]=torch.tensor([float(y["extension0"]),float(y["extension1"])],dtype=torch.float64)
                                    expanded.append(row); truth.append(float(y["physical_range"])); seen.add(key)
                    joint+=1
        else:
            for split in ("train","calibration"):
                for name in manifest["materialized"][split]:
                    ck=njc_key(split,name,.25,.75,1,"forward"); jid=hashlib.sha256(("splart-smarc-njc-joint-v1:"+ck).encode()).hexdigest()
                    if jid not in canonical_by_joint: raise RuntimeError("NJC canonical join failed")
                    base=canonical_by_joint[jid]
                    for f0 in F0:
                        for f1 in F1:
                            if f1-f0<.35: continue
                            for orientation in ORIENTATION:
                                for order in ORDER:
                                    key=njc_key(split,name,f0,f1,orientation,order)
                                    if key not in source or source[key]["split"]!=split: raise RuntimeError("NJC gauge join failed")
                                    x,y=source[key],labels[key]
                                    pair_raw=f"{split}:{name}:{f0:.2f}:{f1:.2f}:axis{orientation:+d}"
                                    row=dict(base); row["mechanical"]=torch.cat((swap_invariant_pair(x["state0_features"],x["state1_features"]),torch.tensor([abs(float(x["observed_displacement"]))]))); row["observed_displacement"]=abs(float(x["observed_displacement"])); row["gauge_id"]=key; row["swap_pair_id"]=hashlib.sha256(("splart-smarc-swap-v1:"+pair_raw).encode()).hexdigest(); row["order"]=order; row["base_extension"]=torch.tensor([float(y["extension0"]),float(y["extension1"])],dtype=torch.float64)
                                    expanded.append(row); truth.append(float(y["physical_range"])); seen.add(key)
    counts={jid:0 for jid in canonical_by_joint}
    for row in expanded: counts[row["joint_id"]]+=1
    if set(counts.values())!={32} or len(expanded)!=5888 or len(seen)!=5888:
        raise RuntimeError("expected exactly 32 unique gauges per 184 joint")
    pairs={}
    for row in expanded: pairs.setdefault(row["swap_pair_id"],[]).append(row)
    if set(len(value) for value in pairs.values())!={2}: raise RuntimeError("swap pair coverage mismatch")
    for value in pairs.values():
        value=sorted(value,key=lambda row:row["order"])
        if {row["order"] for row in value}!={"forward","reverse"} or not torch.equal(value[0]["base_extension"],value[1]["base_extension"].flip(0)):
            raise RuntimeError("source swap extension contract mismatch")
    return expanded,torch.tensor(truth,dtype=torch.float64)


def merge(args, config):
    rows=[]; labels={}; manifests=[]; source_files=[]
    for path in args.articraft_shards:
        new,target,manifest=load_cache(path,"articraft",config); rows.extend(new); labels.update(target); manifests.append(manifest)
        source_files.extend((str(path/name),sha256_file(path/name)) for name in ("inputs.pt","labels.pt","manifest.json","neutral_render_bank.pt"))
    new,target,manifest=load_cache(args.njc_cache,"njc",config); rows.extend(new); labels.update(target); manifests.append(manifest)
    source_files.extend((str(args.njc_cache/name),sha256_file(args.njc_cache/name)) for name in ("inputs.pt","labels.pt","manifest.json","neutral_render_bank.pt"))
    joint_ids=[row["joint_id"] for row in rows]
    if len(joint_ids)!=184 or len(set(joint_ids))!=184 or set(labels)!=set(joint_ids):
        raise RuntimeError("expected 184 unique joined source joints")
    art=[row for row in rows if row["domain"]=="articraft"]
    njc=[row for row in rows if row["domain"]=="njc"]
    if len(art)!=169 or len(njc)!=15: raise RuntimeError("source domain joint count mismatch")
    counts={split:len({r["object_group_id"] for r in rows if r["split"]==split}) for split in
            ("endpoint_pretrain","endpoint_validation","njc_train","njc_validation")}
    joint_counts={split:sum(r["split"]==split for r in rows) for split in counts}
    if counts!={"endpoint_pretrain":97,"endpoint_validation":12,"njc_train":11,"njc_validation":4}:
        raise RuntimeError(f"object split mismatch {counts}")
    if joint_counts!={"endpoint_pretrain":146,"endpoint_validation":23,"njc_train":11,"njc_validation":4}:
        raise RuntimeError(f"joint split mismatch {joint_counts}")
    hashes=config["data"]["object_list_hashes"]
    groups=lambda split: sorted({r["object_group_id"] for r in rows if r["split"]==split})
    observed={"final_articraft97":hash_ids(groups("endpoint_pretrain")),
              "final_njc11":hash_ids(groups("njc_train")),
              "final_joint108":hash_ids(groups("endpoint_pretrain")+groups("njc_train")),
              "articraft_validation12":hash_ids(groups("endpoint_validation")),
              "njc_validation4":hash_ids(groups("njc_validation"))}
    for key,value in observed.items():
        if value!=hashes[key]: raise RuntimeError(f"frozen object-list hash mismatch {key}")
    rows,target=expand_gauges(rows,args.articraft_pilc_data,args.njc_pilc_data)
    if not torch.isfinite(target).all() or torch.any(target<=0): raise RuntimeError("invalid source truth")
    source_files.extend((str(path/name),sha256_file(path/name)) for path in (args.articraft_pilc_data,args.njc_pilc_data) for name in ("inputs.pt","labels.pt","manifest.json"))
    return rows,target,{"source_file_sha256":dict(source_files),"manifests":manifests,
                        "object_counts":counts,"joint_counts":joint_counts,"object_list_hashes":observed}


def indices_for(rows, splits): return [i for i,row in enumerate(rows) if row["split"] in splits]


def domain_metrics(rows, targets, indices, prediction):
    result={}
    for domain in ("articraft","njc"):
        loc=[k for k,i in enumerate(indices) if rows[i]["domain"]==domain]
        if loc:
            subset=[indices[k] for k in loc]
            result[domain]=object_macro_mare(rows,subset,prediction[loc],targets)
    return result


def analytic_baselines(rows,targets,train,held):
    result={}
    for domain in ("articraft","njc"):
        tr=[i for i in train if rows[i]["domain"]==domain]; va=[i for i in held if rows[i]["domain"]==domain]
        if not tr or not va: continue
        weights=object_domain_weights(rows,tr)
        row_weights=torch.full((len(tr),),1/len(tr),dtype=torch.float64)
        truth=targets[tr]
        global_prediction=torch.full((len(va),),float((row_weights*truth).sum()),dtype=torch.float64)
        d=torch.tensor([rows[i]["observed_displacement"] for i in tr],dtype=torch.float64)
        dv=torch.tensor([rows[i]["observed_displacement"] for i in va],dtype=torch.float64)
        linear=analytic_linear_baseline(d,truth,row_weights,dv)
        result.setdefault("global",{})[domain]=object_macro_mare(rows,va,global_prediction,targets)
        result.setdefault("displacement",{})[domain]=object_macro_mare(rows,va,linear,targets)
        if domain=="articraft":
            means={}
            for family in sorted({rows[i]["family_audit_id"] for i in tr}):
                fi=[i for i in tr if rows[i]["family_audit_id"]==family]
                fw=torch.full((len(fi),),1/len(fi),dtype=torch.float64); means[family]=float((fw*targets[fi]).sum())
            category=torch.tensor([means.get(rows[i]["family_audit_id"],float((row_weights*truth).sum())) for i in va])
        else:
            category=global_prediction
        result.setdefault("category",{})[domain]=object_macro_mare(rows,va,category,targets)
    return result


def object_scalar(rows,indices,kind,reference_indices=None):
    objects=sorted({rows[i]["object_group_id"] for i in indices}); result={}
    if kind=="displacement":
        for obj in objects:
            values=[abs(float(rows[i]["observed_displacement"])) for i in indices if rows[i]["object_group_id"]==obj]
            result[obj]=sum(values)/len(values)
    elif kind=="semantic_distance":
        object_mean={obj:torch.stack([rows[i]["semantic"].double() for i in indices if rows[i]["object_group_id"]==obj]).mean(0) for obj in objects}
        reference_indices=indices if reference_indices is None else reference_indices
        for obj,value in object_mean.items():
            domain=next(rows[i]["domain"] for i in indices if rows[i]["object_group_id"]==obj)
            reference_objects=sorted({rows[i]["object_group_id"] for i in reference_indices if rows[i]["domain"]==domain})
            reference_mean=torch.stack([torch.stack([rows[i]["semantic"].double() for i in reference_indices if rows[i]["object_group_id"]==candidate]).mean(0) for candidate in reference_objects]).mean(0)
            result[obj]=float(1-torch.nn.functional.cosine_similarity(value[None],reference_mean[None]).item())
    return result


def overrides(rows,train,held,config,field,bin_kind):
    all_indices=train+held; values=object_scalar(rows,all_indices,bin_kind,train)
    train_donor,train_receipt=deterministic_object_donors(rows,train,train,values)
    held_donor,held_receipt=deterministic_object_donors(rows,train,held,values)
    preserve_last=field=="mechanical"
    return (apply_object_donors(rows,train,field,train_donor,train,preserve_last),
            apply_object_donors(rows,held,field,held_donor,train,preserve_last),
            {"train_mapping":train_donor,"held_mapping":held_donor,"train":train_receipt,"held":held_receipt})


def train_variant(rows,targets,train,held,prep,variant,config,steps,device,black_raw=None):
    train_sem=held_sem=train_mech=held_mech=None; donor_receipt=None
    if variant=="mechanical_only":
        train_sem=prep.semantic_mean[None].repeat(len(train),1); held_sem=prep.semantic_mean[None].repeat(len(held),1)
    elif variant=="black_image":
        train_sem=black_raw[None].repeat(len(train),1); held_sem=black_raw[None].repeat(len(held),1)
    elif variant=="semantic_shuffle":
        train_sem,held_sem,donor_receipt=overrides(rows,train,held,config,"semantic","displacement")
    elif variant=="mechanical_shuffle":
        train_mech,held_mech,donor_receipt=overrides(rows,train,held,config,"mechanical","semantic_distance")
    model=train_model(rows,targets,train,prep,train_sem,train_mech,steps,device)
    prediction=predict(model,prep,rows,held,held_sem,held_mech)
    return model,prediction,donor_receipt


def matching_preflight(rows,train,held,config):
    receipts={}; available=True
    for name,field,kind in (("semantic_shuffle","semantic","displacement"),("mechanical_shuffle","mechanical","semantic_distance")):
        _,_,receipt=overrides(rows,train,held,config,field,kind); receipts[name]=receipt
        caliper=config["shuffle_contract"]["matching_caliper"]
        train_stats=receipt["train"]["train_distance"]; held_stats=receipt["held"]["held_distance"]
        passed=(train_stats["p90"]<=caliper["distance_p90_at_most"] and train_stats["max"]<=caliper["distance_max_at_most"]
                and held_stats["p90"]<=caliper["distance_p90_at_most"] and held_stats["max"]<=caliper["distance_max_at_most"]
                and receipt["train"]["nearest_other_regret_p90"]<=caliper["nearest_other_regret_p90_at_most"])
        receipt["caliper_pass"]=passed; available &= passed
    return receipts,available


def run_lofo(rows,targets,config,steps,device):
    art=[i for i,row in enumerate(rows) if row["domain"]=="articraft"]
    families=sorted({rows[i]["family_audit_id"] for i in art})
    output={}
    for family in families:
        held=[i for i in art if rows[i]["family_audit_id"]==family]; train=[i for i in art if rows[i]["family_audit_id"]!=family]
        prep=fit_preprocessor(rows,targets,train,16)
        full,pred,_=train_variant(rows,targets,train,held,prep,"full",config,steps,device)
        mech,mpred,_=train_variant(rows,targets,train,held,prep,"mechanical_only",config,steps,device)
        baselines=analytic_baselines(rows,targets,train,held)
        name=f"family_{len({rows[i]['object_group_id'] for i in held})}obj_{len(held)}joint"
        output[name]={"family_audit_id":family,"train_object_hash":hash_ids(sorted({rows[i]["object_group_id"] for i in train})),
                      "held_object_hash":hash_ids(sorted({rows[i]["object_group_id"] for i in held})),
                      "full":object_macro_mare(rows,held,pred,targets),
                      "mechanical_only":object_macro_mare(rows,held,mpred,targets),"analytic":baselines}
    return output


def main():
    p=argparse.ArgumentParser(); p.add_argument("--articraft-shards",type=Path,nargs=3,required=True); p.add_argument("--njc-cache",type=Path,required=True); p.add_argument("--articraft-pilc-data",type=Path,required=True); p.add_argument("--njc-pilc-data",type=Path,required=True); p.add_argument("--render-logs",type=Path,nargs=4,required=True); p.add_argument("--config",type=Path,required=True); p.add_argument("--dino-checkpoint",type=Path,required=True); p.add_argument("--output",type=Path,required=True); p.add_argument("--device",default="cuda"); p.add_argument("--smoke",action="store_true"); p.add_argument("--preflight",action="store_true"); args=p.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    config=json.loads(args.config.read_text()); steps=2 if args.smoke else int(config["optimization"]["steps"])
    if not args.smoke and steps!=1200: raise RuntimeError("formal steps must equal frozen config 1200")
    rows,targets,merge_receipt=merge(args,config)
    overflow={str(path):path.read_text(errors="replace").lower().count("overflow") for path in args.render_logs}
    if any(overflow.values()): raise RuntimeError(f"renderer overflow log evidence {overflow}")
    merge_receipt["render_log_overflow_count"]=overflow
    train=indices_for(rows,{"endpoint_pretrain","njc_train"}); held=indices_for(rows,{"endpoint_validation","njc_validation"})
    analytic=analytic_baselines(rows,targets,train,held)
    match_receipts,matching_available=matching_preflight(rows,train,held,config)
    if args.preflight:
        args.output.mkdir(parents=True)
        result={"schema":"splart-smarc-source-preflight/v1","config_sha256":sha256_file(args.config),"merge":merge_receipt,
                "analytic_baselines":analytic,"matching_receipts":match_receipts,"matched_null_available":matching_available,
                "box_labels_read":[],"protected_splits_read":[]}
        (args.output/"preflight.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
        print(json.dumps({"output":str(args.output),"analytic":analytic,"matched_null_available":matching_available})); return
    if not matching_available: raise RuntimeError("matched null caliper failed; Phase A cannot advance")
    prep=fit_preprocessor(rows,targets,train,16)
    encoder=load_dino_vitb16(args.dino_checkpoint,torch.device(args.device)); black=torch.zeros(2,6,3,224,224,dtype=torch.uint8); black_raw=encode_state_pair(encoder,black).cpu(); del encoder
    variants={}; checkpoints={}; donors={}; predictions={}
    for variant in ("full","mechanical_only","black_image","semantic_shuffle","mechanical_shuffle"):
        model,pred,receipt=train_variant(rows,targets,train,held,prep,variant,config,steps,args.device,black_raw)
        variants[variant]=domain_metrics(rows,targets,held,pred); checkpoints[variant]=model.state_dict(); donors[variant]=receipt; predictions[variant]=pred
    lofo=run_lofo(rows,targets,config,steps,args.device)
    thresholds=config["frozen_baselines"]; gates={}; pass_all=True
    for domain in ("articraft","njc"):
        full=variants["full"][domain]; required=thresholds[domain]["required_below"]
        comparisons={k:(full<=.85*value) for k,value in (("global",analytic["global"][domain]),("displacement",analytic["displacement"][domain]),("category",analytic["category"][domain]),("mechanical_only",variants["mechanical_only"][domain]))}
        shuffles={k:(variants[k][domain]>=1.2*full) for k in ("semantic_shuffle","mechanical_shuffle")}
        gates[domain]={"full":full,"required_below_frozen":required,"below_frozen_15pct_line":full<required,"comparisons":comparisons,"shuffle_worsening_20pct":shuffles,"pass":full<required and all(comparisons.values()) and all(shuffles.values())}
        pass_all &= gates[domain]["pass"]
    lofo_pass=[]
    for fold in lofo.values():
        full=fold["full"]; comparisons=[full<=.85*fold["mechanical_only"]]
        comparisons += [full<=.85*fold["analytic"][kind]["articraft"] for kind in ("global","displacement","category")]
        fold["pass_15pct_all"]=all(comparisons); lofo_pass.append(fold["pass_15pct_all"])
    total_objects=sum(int(name.split("obj_")[0].split("_")[-1]) for name in lofo)
    lofo_aggregate={metric:sum(int(name.split("obj_")[0].split("_")[-1])*fold[metric] for name,fold in lofo.items())/total_objects for metric in ("full","mechanical_only")}
    for metric in ("global","displacement","category"):
        lofo_aggregate[metric]=sum(int(name.split("obj_")[0].split("_")[-1])*fold["analytic"][metric]["articraft"] for name,fold in lofo.items())/total_objects
    gates["articraft_lofo_object_weighted_aggregate"]=lofo_aggregate
    aggregate_pass=all(lofo_aggregate["full"]<=.85*lofo_aggregate[k] for k in ("global","displacement","category","mechanical_only"))
    gates["articraft_lofo_aggregate_pass"]=aggregate_pass
    gates["articraft_lofo_each_fold_pass_strict_diagnostic"]=all(lofo_pass); pass_all &= aggregate_pass
    swap_metrics={}
    for variant,state in checkpoints.items():
        for key,value in state.items():
            if not torch.isfinite(value).all(): raise RuntimeError(f"nonfinite checkpoint {variant}:{key}")
        swap_metrics[variant]=swap_audit(rows,held,predictions[variant])
    swap_max=max(value for metric in swap_metrics.values() for value in metric.values())
    gates["state_swap_metrics"]=swap_metrics; gates["state_swap_max_error"]=swap_max; gates["state_swap_pass"]=swap_max<=float(config["gate"]["state_swap_max_error"]); pass_all &= gates["state_swap_pass"]
    gates["phase_a_pass"]=pass_all; gates["decision"]="ADVANCE_TO_BOX_A_D" if pass_all else "PRUNE_BEFORE_BOX"
    args.output.mkdir(parents=True); torch.save({"checkpoints":checkpoints,"preprocessor":prep,"black_raw":black_raw},args.output/"models.pt")
    result={"schema":"splart-smarc-source-gate/v1","config_sha256":sha256_file(args.config),"checkpoint_sha256":sha256_file(args.dino_checkpoint),"steps":steps,"smoke":args.smoke,"merge":merge_receipt,"metrics":variants,"analytic_baselines":analytic,"lofo":lofo,"gates":gates,"donor_receipts":donors,"box_labels_read":[],"protected_splits_read":[]}
    (args.output/"metrics.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    (args.output/"receipt.json").write_text(json.dumps({"schema":"splart-smarc-source-receipt/v1","metrics_sha256":sha256_file(args.output/"metrics.json"),"models_sha256":sha256_file(args.output/"models.pt"),"decision":gates["decision"],"box_labels_read":[],"protected_splits_read":[]},indent=2,sort_keys=True)+"\n")
    print(json.dumps({"output":str(args.output),"decision":gates["decision"],"metrics":variants}))


if __name__=="__main__": main()
