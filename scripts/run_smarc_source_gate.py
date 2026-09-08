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
                                 object_domain_weights, object_macro_mare, predict, swap_audit, train_model,
                                 validate_raw_swap_pair)
from scripts.build_smarc_articraft_shard import VIEWS


FROZEN_CONFIG_SHA256="c46852b019012d68d6413c2fe15f6938948dac34e6a1ea71360cec84560ea1bc"


def canonical_sha256(value) -> str:
    return hashlib.sha256((json.dumps(value,sort_keys=True,separators=(",",":"))+"\n").encode()).hexdigest()


def code_sha256() -> dict[str,str]:
    root=Path(__file__).resolve().parents[1]
    paths=(Path(__file__).resolve(),root/"src"/"splart"/"smarc_source.py",root/"src"/"splart"/"smarc.py",root/"src"/"splart"/"frozen_visual.py")
    return {str(path):sha256_file(path) for path in paths}


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
    configured_views=tuple(tuple(float(x) for x in pair) for pair in config["render"]["elevation_azimuth_degrees"])
    if configured_views!=VIEWS or len(configured_views)!=config["render"]["views_per_state"]:
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
        expected_schema="splart-pilc-articraft-features-v1" if domain=="articraft" else "splart-pilc-njc-features-v1"
        if manifest.get("schema")!=expected_schema: raise RuntimeError("PILC source manifest schema mismatch")
        for key in ("box_extra_scores_read","protected_splits_read","box_labels_read"):
            if manifest.get(key,[]) not in (None,[]): raise RuntimeError(f"PILC source reports protected read: {key}")
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
                                    row=dict(base); row["mechanical"]=torch.cat((swap_invariant_pair(x["state0_features"],x["state1_features"]),torch.tensor([abs(float(x["observed_displacement"]))]))); row["observed_displacement"]=abs(float(x["observed_displacement"])); row["signed_observed_displacement"]=float(x["observed_displacement"]); row["raw_state0"]=x["state0_features"]; row["raw_state1"]=x["state1_features"]; row["gauge_id"]=key; row["swap_pair_id"]=hashlib.sha256(("splart-smarc-swap-v1:"+pair_raw).encode()).hexdigest(); row["order"]=order; row["base_extension"]=torch.tensor([float(y["extension0"]),float(y["extension1"])],dtype=torch.float64); row["physical_range"]=float(y["physical_range"])
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
                                    row=dict(base); row["mechanical"]=torch.cat((swap_invariant_pair(x["state0_features"],x["state1_features"]),torch.tensor([abs(float(x["observed_displacement"]))]))); row["observed_displacement"]=abs(float(x["observed_displacement"])); row["signed_observed_displacement"]=float(x["observed_displacement"]); row["raw_state0"]=x["state0_features"]; row["raw_state1"]=x["state1_features"]; row["gauge_id"]=key; row["swap_pair_id"]=hashlib.sha256(("splart-smarc-swap-v1:"+pair_raw).encode()).hexdigest(); row["order"]=order; row["base_extension"]=torch.tensor([float(y["extension0"]),float(y["extension1"])],dtype=torch.float64); row["physical_range"]=float(y["physical_range"])
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
        forward=next(row for row in value if row["order"]=="forward"); reverse=next(row for row in value if row["order"]=="reverse")
        if {row["order"] for row in value}!={"forward","reverse"}: raise RuntimeError("source swap pair order mismatch")
        validate_raw_swap_pair(forward,reverse)
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
    expected_sources=config["data"]["frozen_source_sha256"]
    checks={"articraft_inputs":args.articraft_pilc_data/"inputs.pt","articraft_labels":args.articraft_pilc_data/"labels.pt","articraft_manifest":args.articraft_pilc_data/"manifest.json",
            "njc_inputs":args.njc_pilc_data/"inputs.pt","njc_labels":args.njc_pilc_data/"labels.pt","njc_manifest":args.njc_pilc_data/"manifest.json"}
    if any(sha256_file(path)!=expected_sources[key] for key,path in checks.items()): raise RuntimeError("frozen PILC source hash mismatch")
    actual_render=[]
    for path in list(args.articraft_shards)+[args.njc_cache]:
        actual_render.append({"inputs":sha256_file(path/"inputs.pt"),"labels":sha256_file(path/"labels.pt"),"manifest":sha256_file(path/"manifest.json"),"render_bank":sha256_file(path/"neutral_render_bank.pt")})
    if actual_render!=config["data"]["frozen_render_payload_sha256"]: raise RuntimeError("frozen render payload hash mismatch")
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


def analytic_baselines(rows,targets,train,held,weight_mode):
    result={}
    for domain in ("articraft","njc"):
        tr=[i for i in train if rows[i]["domain"]==domain]; va=[i for i in held if rows[i]["domain"]==domain]
        if not tr or not va: continue
        weights=(torch.full((len(tr),),1/len(tr),dtype=torch.float64) if weight_mode=="row_uniform"
                 else object_domain_weights(rows,tr))
        truth=targets[tr]
        global_prediction=torch.full((len(va),),float((weights*truth).sum()),dtype=torch.float64)
        d=torch.tensor([rows[i]["observed_displacement"] for i in tr],dtype=torch.float64)
        dv=torch.tensor([rows[i]["observed_displacement"] for i in va],dtype=torch.float64)
        linear=analytic_linear_baseline(d,truth,weights,dv)
        result.setdefault("global",{})[domain]=object_macro_mare(rows,va,global_prediction,targets)
        result.setdefault("displacement",{})[domain]=object_macro_mare(rows,va,linear,targets)
        if domain=="articraft":
            means={}
            for family in sorted({rows[i]["family_audit_id"] for i in tr}):
                fi=[i for i in tr if rows[i]["family_audit_id"]==family]
                fw=(torch.full((len(fi),),1/len(fi),dtype=torch.float64) if weight_mode=="row_uniform"
                    else object_domain_weights(rows,fi)); means[family]=float((fw*targets[fi]).sum())
            category=torch.tensor([means.get(rows[i]["family_audit_id"],float((weights*truth).sum())) for i in va])
        else:
            category=global_prediction
        result.setdefault("category",{})[domain]=object_macro_mare(rows,va,category,targets)
    return result


def analytic_control_set(rows,targets,train,held):
    historical=analytic_baselines(rows,targets,train,held,"row_uniform")
    hierarchical=analytic_baselines(rows,targets,train,held,"hierarchical")
    stronger={kind:{domain:min(historical[kind][domain],hierarchical[kind][domain])
                    for domain in historical[kind]} for kind in historical}
    return {"historical_row_uniform":historical,"hierarchical":hierarchical,"stronger_lower_mare":stronger}


def assert_historical_controls(controls,config,tolerance=7e-7):
    observed=controls["historical_row_uniform"]; frozen=config["frozen_baselines"]
    expected={"articraft":{"global":frozen["articraft"]["global"],"displacement":frozen["articraft"]["displacement"],"category":frozen["articraft"]["category_frequency_audit_only"]},
              "njc":{"global":frozen["njc"]["global"],"displacement":frozen["njc"]["displacement"],"category":frozen["njc"]["unseen_name_category_fallback"]}}
    for domain in expected:
        for kind,value in expected[domain].items():
            if abs(observed[kind][domain]-value)>tolerance: raise RuntimeError(f"historical {domain}/{kind} baseline mismatch")


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
    caliper=float(config["shuffle_contract"]["matching_caliper"])
    train_donor,train_receipt=deterministic_object_donors(rows,train,train,values,caliper)
    held_donor,held_receipt=deterministic_object_donors(rows,train,held,values,caliper)
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
        minimum=float(config["shuffle_contract"]["minimum_perturbed_object_coverage_each_domain"])
        passed=True
        for split in ("train","held"):
            for domain,part in receipt[split]["domains"].items():
                part["coverage_pass"]=part["coverage"]>=minimum
                passed &= part["coverage_pass"] and part["nonself_distance"]["max"]<=float(config["shuffle_contract"]["matching_caliper"])+1e-12
                if split=="held":
                    diversity=config["shuffle_contract"]["held_donor_diversity"]
                    ratio=part["held_effective_donor_count"]/max(part["perturbed_objects"],1)
                    part["effective_donor_ratio"]=ratio
                    part["diversity_pass"]=ratio>=diversity["effective_donors_per_perturbed_at_least"] and part["maximum_held_donor_load"]<=diversity["maximum_donor_load"]
                    passed &= part["diversity_pass"]
        receipt["caliper_pass"]=passed; available &= passed
    return receipts,available


def lofo_partitions(rows,config):
    art=[i for i,row in enumerate(rows) if row["domain"]=="articraft"]
    families=sorted({rows[i]["family_audit_id"] for i in art})
    if len(families)!=3: raise RuntimeError("expected exactly three Articraft families")
    partitions=[]; held_union=set()
    for family in families:
        held=[i for i in art if rows[i]["family_audit_id"]==family]; train=[i for i in art if rows[i]["family_audit_id"]!=family]
        train_objects=sorted({rows[i]["object_group_id"] for i in train}); held_objects=sorted({rows[i]["object_group_id"] for i in held})
        count=len(held_objects); names={37:("lofo_window_train72","lofo_window_test37"),51:("lofo_sewing_train58","lofo_sewing_test51"),21:("lofo_usb_train88","lofo_usb_test21_small")}
        if count not in names: raise RuntimeError("unexpected Articraft family size")
        train_name,held_name=names[count]; train_hash,held_hash=hash_ids(train_objects),hash_ids(held_objects)
        frozen=config["data"]["object_list_hashes"]
        if train_hash!=frozen[train_name] or held_hash!=frozen[held_name]: raise RuntimeError("frozen LOFO object-list hash mismatch")
        held_set=set(held_objects)
        if held_union & held_set: raise RuntimeError("LOFO held folds overlap")
        held_union |= held_set
        partitions.append({"family_audit_id":family,"train":train,"held":held,"train_object_hash":train_hash,"held_object_hash":held_hash,"held_objects":count})
    if held_union!={rows[i]["object_group_id"] for i in art}: raise RuntimeError("LOFO folds do not partition Articraft")
    return partitions


def run_lofo_fold(rows,targets,config,steps,device,fold_index):
    fold=lofo_partitions(rows,config)[fold_index]; train,held=fold["train"],fold["held"]
    prep=fit_preprocessor(rows,targets,train,16)
    full,pred,_=train_variant(rows,targets,train,held,prep,"full",config,steps,device)
    mech,mpred,_=train_variant(rows,targets,train,held,prep,"mechanical_only",config,steps,device)
    return {"family_audit_id":fold["family_audit_id"],"train_object_hash":fold["train_object_hash"],
            "held_object_hash":fold["held_object_hash"],"held_objects":fold["held_objects"],
            "full":object_macro_mare(rows,held,pred,targets),"mechanical_only":object_macro_mare(rows,held,mpred,targets),
            "analytic_controls":analytic_control_set(rows,targets,train,held),"swap":{"full":swap_audit(rows,held,pred),"mechanical_only":swap_audit(rows,held,mpred)},
            "checkpoint":{"full":full.state_dict(),"mechanical_only":mech.state_dict()},"preprocessor":prep}


def gate_results(final,folds,config):
    gates={}; passed=True; analytic=final["analytic_controls"]["stronger_lower_mare"]
    for domain in ("articraft","njc"):
        full=final["metrics"]["full"][domain]
        comparisons={kind:full<=.85*analytic[kind][domain] for kind in ("global","displacement","category")}
        comparisons["mechanical_only"]=full<=.85*final["metrics"]["mechanical_only"][domain]
        shuffle={kind:final["metrics"][kind][domain]>=1.2*full for kind in ("semantic_shuffle","mechanical_shuffle")}
        coverage={kind:all(final["matching_receipts"][kind][split]["domains"][domain]["coverage"]>=.9 for split in ("train","held")) for kind in ("semantic_shuffle","mechanical_shuffle")}
        gates[domain]={"full":full,"comparisons_15pct":comparisons,"shuffle_worsening_20pct":shuffle,"shuffle_coverage":coverage,"pass":all(comparisons.values()) and all(shuffle.values()) and all(coverage.values())}
        passed &= gates[domain]["pass"]
    total=sum(fold["held_objects"] for fold in folds)
    aggregate={key:sum(fold["held_objects"]*fold[key] for fold in folds)/total for key in ("full","mechanical_only")}
    for kind in ("global","displacement","category"):
        aggregate[kind]=sum(fold["held_objects"]*fold["analytic_controls"]["stronger_lower_mare"][kind]["articraft"] for fold in folds)/total
    lofo_pass=all(aggregate["full"]<=.85*aggregate[key] for key in ("global","displacement","category","mechanical_only"))
    gates["lofo_object_weighted_aggregate"]=aggregate; gates["lofo_pass_15pct"]=lofo_pass; passed &= lofo_pass
    swap_max=max(value for part in [final]+folds for metric in part["swap"].values() for value in metric.values())
    gates["state_swap_max_error"]=swap_max; gates["state_swap_pass"]=swap_max<=float(config["gate"]["state_swap_max_error"]); passed &= gates["state_swap_pass"]
    gates["phase_a_pass"]=passed; gates["decision"]="ADVANCE_TO_BOX_A_D" if passed else "PRUNE_BEFORE_BOX"
    return gates


def main():
    p=argparse.ArgumentParser(); p.add_argument("--articraft-shards",type=Path,nargs=3); p.add_argument("--njc-cache",type=Path); p.add_argument("--articraft-pilc-data",type=Path); p.add_argument("--njc-pilc-data",type=Path); p.add_argument("--render-logs",type=Path,nargs=4); p.add_argument("--config",type=Path,required=True); p.add_argument("--dino-checkpoint",type=Path); p.add_argument("--output",type=Path,required=True); p.add_argument("--device",default="cuda"); p.add_argument("--task",choices=("preflight","smoke","final","lofo_0","lofo_1","lofo_2","aggregate"),required=True); p.add_argument("--partials",type=Path,nargs="*"); args=p.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    config=json.loads(args.config.read_text()); config_sha=sha256_file(args.config)
    if config_sha!=FROZEN_CONFIG_SHA256: raise RuntimeError("formal v1.3 config SHA mismatch")
    if args.task=="aggregate":
        if args.partials is None or len(args.partials)!=4: raise ValueError("aggregate requires exactly four partial directories")
        records=[]; task_receipts=[]; code_receipts=[]
        for path in args.partials:
            record=json.loads((path/"partial.json").read_text()); receipt=json.loads((path/"receipt.json").read_text())
            if receipt["partial_sha256"]!=sha256_file(path/"partial.json") or receipt["models_sha256"]!=sha256_file(path/"models.pt") or receipt["config_sha256"]!=config_sha:
                raise RuntimeError("partial receipt hash mismatch")
            if record.get("box_labels_read")!=[] or record.get("protected_splits_read")!=[]: raise RuntimeError("partial reports protected read")
            records.append(record); task_receipts.append((record["task"],sha256_file(path/"receipt.json"))); code_receipts.append(receipt["code_sha256"])
        if {record["task"] for record in records}!={"final","lofo_0","lofo_1","lofo_2"} or any(record["config_sha256"]!=config_sha or record["steps"]!=1200 for record in records): raise RuntimeError("partial task/config/step mismatch")
        if any(value!=code_receipts[0] for value in code_receipts) or len({record["source_provenance_sha256"] for record in records})!=1: raise RuntimeError("partial code/source provenance mismatch")
        final=next(record for record in records if record["task"]=="final"); folds=[next(record for record in records if record["task"]==f"lofo_{i}") for i in range(3)]
        gates=gate_results(final,folds,config); args.output.mkdir(parents=True)
        result={"schema":"splart-csmarc-source-gate/v1","config_sha256":config_sha,"task_receipt_sha256":[value for _,value in sorted(task_receipts)],"code_sha256":code_receipts[0],"source_provenance_sha256":records[0]["source_provenance_sha256"],"final":final,"lofo":folds,"gates":gates,"box_labels_read":[],"protected_splits_read":[]}
        args.output.mkdir(parents=True); (args.output/"metrics.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n"); aggregate_receipt={"schema":"splart-csmarc-aggregate-receipt/v1","metrics_sha256":sha256_file(args.output/"metrics.json"),"task_receipt_sha256":result["task_receipt_sha256"],"decision":gates["decision"],"box_labels_read":[],"protected_splits_read":[]}; (args.output/"receipt.json").write_text(json.dumps(aggregate_receipt,indent=2,sort_keys=True)+"\n"); print(json.dumps({"output":str(args.output),"decision":gates["decision"],"metrics":final["metrics"]})); return
    if any(value is None for value in (args.articraft_shards,args.njc_cache,args.articraft_pilc_data,args.njc_pilc_data,args.render_logs,args.dino_checkpoint)): raise ValueError("source task missing required inputs")
    steps=2 if args.task=="smoke" else int(config["optimization"]["steps"])
    if args.task not in ("preflight","smoke") and steps!=1200: raise RuntimeError("formal steps must equal frozen config 1200")
    rows,targets,merge_receipt=merge(args,config)
    overflow={str(path):path.read_text(errors="replace").lower().count("overflow") for path in args.render_logs}
    if any(overflow.values()): raise RuntimeError(f"renderer overflow log evidence {overflow}")
    merge_receipt["render_log_overflow_count"]=overflow
    train=indices_for(rows,{"endpoint_pretrain","njc_train"}); held=indices_for(rows,{"endpoint_validation","njc_validation"})
    train_objects={rows[i]["object_group_id"] for i in train}; held_objects={rows[i]["object_group_id"] for i in held}
    if train_objects & held_objects: raise RuntimeError("train/held object leakage")
    analytic=analytic_control_set(rows,targets,train,held)
    assert_historical_controls(analytic,config)
    source_provenance_sha=canonical_sha256(merge_receipt); code_hash=code_sha256()
    match_receipts,matching_available=matching_preflight(rows,train,held,config)
    lofo=lofo_partitions(rows,config)
    if args.task=="preflight":
        args.output.mkdir(parents=True)
        result={"schema":"splart-csmarc-source-preflight/v1","config_sha256":config_sha,"merge":merge_receipt,
                "analytic_baselines":analytic,"matching_receipts":match_receipts,"matched_null_available":matching_available,"source_provenance_sha256":source_provenance_sha,"code_sha256":code_hash,
                "lofo_partitions":[{k:v for k,v in fold.items() if k not in ("train","held")} for fold in lofo],
                "box_labels_read":[],"protected_splits_read":[]}
        (args.output/"preflight.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n"); (args.output/"receipt.json").write_text(json.dumps({"schema":"splart-csmarc-preflight-receipt/v1","config_sha256":config_sha,"preflight_sha256":sha256_file(args.output/"preflight.json"),"code_sha256":code_hash,"box_labels_read":[],"protected_splits_read":[]},indent=2,sort_keys=True)+"\n")
        print(json.dumps({"output":str(args.output),"analytic":analytic,"matched_null_available":matching_available})); return
    if not matching_available: raise RuntimeError("matched null caliper failed; Phase A cannot advance")
    prep=fit_preprocessor(rows,targets,train,16)
    if args.task.startswith("lofo_"):
        index=int(args.task[-1]); fold=run_lofo_fold(rows,targets,config,steps,args.device,index); checkpoint=fold.pop("checkpoint"); preprocessor=fold.pop("preprocessor")
        args.output.mkdir(parents=True); torch.save({"checkpoint":checkpoint,"preprocessor":preprocessor},args.output/"models.pt")
        partial={"schema":"splart-csmarc-source-partial/v1","task":args.task,"config_sha256":config_sha,"steps":steps,"source_provenance_sha256":source_provenance_sha,**fold,"box_labels_read":[],"protected_splits_read":[]}
        (args.output/"partial.json").write_text(json.dumps(partial,indent=2,sort_keys=True)+"\n"); receipt={"task":args.task,"config_sha256":config_sha,"steps":steps,"code_sha256":code_hash,"source_provenance_sha256":source_provenance_sha,"partial_sha256":sha256_file(args.output/"partial.json"),"models_sha256":sha256_file(args.output/"models.pt")}; (args.output/"receipt.json").write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n"); print(json.dumps({"output":str(args.output),"task":args.task,"full":fold["full"]})); return
    encoder=load_dino_vitb16(args.dino_checkpoint,torch.device(args.device)); black=torch.zeros(2,6,3,224,224,dtype=torch.uint8); black_raw=encode_state_pair(encoder,black).cpu(); del encoder
    variants={}; checkpoints={}; donors={}; predictions={}
    for variant in ("full","mechanical_only","black_image","semantic_shuffle","mechanical_shuffle"):
        model,pred,receipt=train_variant(rows,targets,train,held,prep,variant,config,steps,args.device,black_raw)
        variants[variant]=domain_metrics(rows,targets,held,pred); checkpoints[variant]=model.state_dict(); donors[variant]=receipt; predictions[variant]=pred
    swap_metrics={}
    for variant,state in checkpoints.items():
        for key,value in state.items():
            if not torch.isfinite(value).all(): raise RuntimeError(f"nonfinite checkpoint {variant}:{key}")
        swap_metrics[variant]=swap_audit(rows,held,predictions[variant])
    final={"schema":"splart-csmarc-source-partial/v1","task":"final","config_sha256":config_sha,"steps":steps,"source_provenance_sha256":source_provenance_sha,"merge":merge_receipt,"metrics":variants,"analytic_controls":analytic,"matching_receipts":match_receipts,"swap":swap_metrics,"box_labels_read":[],"protected_splits_read":[]}
    if args.task=="smoke":
        folds=[]
        for index in range(3):
            fold=run_lofo_fold(rows,targets,config,steps,args.device,index); fold.pop("checkpoint"); fold.pop("preprocessor"); folds.append(fold)
        gates=gate_results(final,folds,config); args.output.mkdir(parents=True); (args.output/"metrics.json").write_text(json.dumps({"schema":"splart-csmarc-smoke/v1","final":final,"lofo":folds,"gates":gates,"smoke":True,"box_labels_read":[],"protected_splits_read":[]},indent=2,sort_keys=True)+"\n"); print(json.dumps({"output":str(args.output),"task":"smoke","metrics":variants})); return
    args.output.mkdir(parents=True); torch.save({"checkpoints":checkpoints,"preprocessor":prep,"black_raw":black_raw},args.output/"models.pt")
    (args.output/"partial.json").write_text(json.dumps(final,indent=2,sort_keys=True)+"\n"); receipt={"task":"final","config_sha256":config_sha,"steps":steps,"code_sha256":code_hash,"source_provenance_sha256":source_provenance_sha,"partial_sha256":sha256_file(args.output/"partial.json"),"models_sha256":sha256_file(args.output/"models.pt")}; (args.output/"receipt.json").write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n"); print(json.dumps({"output":str(args.output),"task":"final","metrics":variants}))


if __name__=="__main__": main()
