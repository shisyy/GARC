#!/usr/bin/env python3
"""Run the preregistered, source-only Phase-A SMARC gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from splart.conditional_residual import conditional_residual_ablation, receipt_passes
from splart.frozen_visual import sha256_file
from splart.smarc import (extensions_to_endpoints, opaque_row_key, project_extensions_to_range,
                          swap_invariant_pair)
from splart.smarc_source import (analytic_linear_baseline, fit_feature_preprocessor, fit_preprocessor, hash_ids,
                                 object_domain_weights, object_macro_mare, predict, swap_audit, train_model,
                                 transform, validate_raw_swap_pair)
from scripts.build_smarc_articraft_shard import VIEWS


FROZEN_CONFIG_SHA256="454f4e16af01e4986b9e4534168b6be7bc96eb4617a63bcc988dfdb7841aaca8"
FROZEN_RENDER_LOG_SHA256=(
    "f55625bcc3cd64bb44ad1c8068af6823e5e1cea2059f2c907667133cf04cc7f2",
    "2efab422d86f0507bbea97517b8acb47ffa68fc0eefc93b2c472b5f3f4e04e69",
    "559bce38db7e52bb09c396c5d318a0087b5ba9489bc7d2fef5d2f608937464a8",
    "be5487ba82b68ab72551338ca78139ece3d1799c99635c2c62ac86fbadd38367",
)
NULL_FIELD_DEFINITIONS={
    "semantic":"train-only PCA16 of raw 1536D swap-invariant DINO pair; z=object-mean absolute observed displacement",
    "mechanical":"joint-training-set domain-balanced standardized retained geometry excluding final displacement; z=cosine distance of raw 1536D swap-invariant DINO pair to train-domain raw mean; no PCA or cross-domain normalization",
}
NULL_CONTEXTS={"semantic_conditional_residual":"semantic_conditional_residual/final-source",
               "mechanical_conditional_residual":"mechanical_conditional_residual/final-source"}


def canonical_sha256(value) -> str:
    return hashlib.sha256((json.dumps(value,sort_keys=True,separators=(",",":"))+"\n").encode()).hexdigest()


def frozen_text_sha256(path: Path) -> str:
    """Hash repository text bytes canonically so Windows checkout newlines cannot change the lock."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n",b"\n")).hexdigest()


def code_sha256() -> dict[str,str]:
    root=Path(__file__).resolve().parents[1]
    paths=(Path(__file__).resolve(),root/"src"/"splart"/"smarc_source.py",
           root/"src"/"splart"/"conditional_residual.py",root/"src"/"splart"/"smarc.py",
           root/"src"/"splart"/"frozen_visual.py")
    return {path.relative_to(root).as_posix():sha256_file(path) for path in paths}


def load_cache(path: Path, domain: str, config: dict) -> tuple[list[dict], dict]:
    rows=torch.load(path/"inputs.pt",map_location="cpu",weights_only=False)
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
    return rows,manifest


F0=(.15,.25,.35); F1=(.65,.75,.85); ORIENTATION=(1,-1); ORDER=("forward","reverse")


def njc_key(split: str, name: str, f0: float, f1: float, orientation: int, order: str) -> str:
    base=f"{split}:{name}:{f0:.2f}:{f1:.2f}:axis{orientation:+d}"
    return hashlib.sha256(f"splart-pilc-v1:{base}:{order}".encode()).hexdigest()


def expand_gauges_features(canonical: list[dict], art_pilc: Path, njc_pilc: Path) -> list[dict]:
    """Join all 32 public gauges without opening any source label payload."""
    canonical_by_joint={row["joint_id"]:row for row in canonical}; expanded=[]; seen=set()
    for domain,path in (("articraft",art_pilc),("njc",njc_pilc)):
        manifest=json.loads((path/"manifest.json").read_text())
        expected_schema="splart-pilc-articraft-features-v1" if domain=="articraft" else "splart-pilc-njc-features-v1"
        if manifest.get("schema")!=expected_schema: raise RuntimeError("PILC source manifest schema mismatch")
        for key in ("box_extra_scores_read","protected_splits_read","box_labels_read"):
            if manifest.get(key,[]) not in (None,[]): raise RuntimeError(f"PILC source reports protected read: {key}")
        source={row["key"]:row for row in torch.load(path/"inputs.pt",map_location="cpu",weights_only=False)}
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
                                    x=source[key]
                                    pair_raw=f"{name}:joint{joint}:{f0}:{f1}:{orientation}"
                                    row=dict(base); row["mechanical"]=torch.cat((swap_invariant_pair(x["state0_features"],x["state1_features"]),torch.tensor([abs(float(x["observed_displacement"]))]))); row["observed_displacement"]=abs(float(x["observed_displacement"])); row["signed_observed_displacement"]=float(x["observed_displacement"]); row["raw_state0"]=x["state0_features"]; row["raw_state1"]=x["state1_features"]; row["gauge_id"]=key; row["swap_pair_id"]=hashlib.sha256(("splart-smarc-swap-v1:"+pair_raw).encode()).hexdigest(); row["order"]=order
                                    expanded.append(row); seen.add(key)
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
                                    x=source[key]
                                    pair_raw=f"{split}:{name}:{f0:.2f}:{f1:.2f}:axis{orientation:+d}"
                                    row=dict(base); row["mechanical"]=torch.cat((swap_invariant_pair(x["state0_features"],x["state1_features"]),torch.tensor([abs(float(x["observed_displacement"]))]))); row["observed_displacement"]=abs(float(x["observed_displacement"])); row["signed_observed_displacement"]=float(x["observed_displacement"]); row["raw_state0"]=x["state0_features"]; row["raw_state1"]=x["state1_features"]; row["gauge_id"]=key; row["swap_pair_id"]=hashlib.sha256(("splart-smarc-swap-v1:"+pair_raw).encode()).hexdigest(); row["order"]=order
                                    expanded.append(row); seen.add(key)
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
        if (not torch.equal(forward["raw_state0"],reverse["raw_state1"]) or
                not torch.equal(forward["raw_state1"],reverse["raw_state0"]) or
                forward["signed_observed_displacement"]!=-reverse["signed_observed_displacement"]):
            raise RuntimeError("target-free raw state swap mismatch")
    return expanded


def attach_source_truth(rows: list[dict],args,config: dict) -> tuple[Tensor,dict]:
    """Open source label sidecars only after both target-free null gates pass."""
    labels={}
    label_files={}; joint_labels={}
    for index,path in enumerate(list(args.articraft_shards)+[args.njc_cache]):
        expected=config["data"]["frozen_render_payload_sha256"][index]["labels"]
        actual=sha256_file(path/"labels.pt")
        if actual!=expected: raise RuntimeError("frozen render label hash mismatch")
        payload=torch.load(path/"labels.pt",map_location="cpu",weights_only=False)
        if set(payload)&set(joint_labels): raise RuntimeError("duplicate rendered source label joint")
        joint_labels.update(payload); label_files[str(path/"labels.pt")]=actual
    if set(joint_labels)!={row["joint_id"] for row in rows}: raise RuntimeError("render feature/label joint mismatch")
    for domain,path in (("articraft",args.articraft_pilc_data),("njc",args.njc_pilc_data)):
        expected=config["data"]["frozen_source_sha256"][f"{domain}_labels"]
        actual=sha256_file(path/"labels.pt")
        if actual!=expected: raise RuntimeError("frozen PILC label hash mismatch")
        payload=torch.load(path/"labels.pt",map_location="cpu",weights_only=False)
        for item in payload:
            if item["key"] in labels: raise RuntimeError("duplicate source truth key")
            labels[item["key"]]=item
        label_files[str(path/"labels.pt")]=actual
    gauge_ids={row["gauge_id"] for row in rows}
    if gauge_ids!=set(labels): raise RuntimeError("source feature/truth key mismatch")
    truth=[]
    for row in rows:
        item=labels[row["gauge_id"]]
        row["base_extension"]=torch.tensor([float(item["extension0"]),float(item["extension1"])],dtype=torch.float64)
        row["physical_range"]=float(item["physical_range"]); truth.append(row["physical_range"])
    pairs={}
    for row in rows: pairs.setdefault(row["swap_pair_id"],[]).append(row)
    for pair in pairs.values():
        forward=next(row for row in pair if row["order"]=="forward"); reverse=next(row for row in pair if row["order"]=="reverse")
        validate_raw_swap_pair(forward,reverse)
    result=torch.tensor(truth,dtype=torch.float64)
    if not torch.isfinite(result).all() or torch.any(result<=0): raise RuntimeError("invalid source truth")
    return result,{"label_file_sha256":label_files,"labels_opened":True,"target_payloads_read":sorted(label_files)}


def merge_features(args, config):
    rows=[]; manifests=[]; source_files=[]
    for path in args.articraft_shards:
        new,manifest=load_cache(path,"articraft",config); rows.extend(new); manifests.append(manifest)
        source_files.extend((str(path/name),sha256_file(path/name)) for name in ("inputs.pt","manifest.json","neutral_render_bank.pt"))
    new,manifest=load_cache(args.njc_cache,"njc",config); rows.extend(new); manifests.append(manifest)
    source_files.extend((str(args.njc_cache/name),sha256_file(args.njc_cache/name)) for name in ("inputs.pt","manifest.json","neutral_render_bank.pt"))
    joint_ids=[row["joint_id"] for row in rows]
    if len(joint_ids)!=184 or len(set(joint_ids))!=184:
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
    rows=expand_gauges_features(rows,args.articraft_pilc_data,args.njc_pilc_data)
    source_files.extend((str(path/name),sha256_file(path/name)) for path in (args.articraft_pilc_data,args.njc_pilc_data) for name in ("inputs.pt","manifest.json"))
    expected_sources=config["data"]["frozen_source_sha256"]
    checks={"articraft_inputs":args.articraft_pilc_data/"inputs.pt","articraft_manifest":args.articraft_pilc_data/"manifest.json",
            "njc_inputs":args.njc_pilc_data/"inputs.pt","njc_manifest":args.njc_pilc_data/"manifest.json"}
    if any(sha256_file(path)!=expected_sources[key] for key,path in checks.items()): raise RuntimeError("frozen PILC source hash mismatch")
    actual_render=[]
    for path in list(args.articraft_shards)+[args.njc_cache]:
        actual_render.append({"inputs":sha256_file(path/"inputs.pt"),"manifest":sha256_file(path/"manifest.json"),"render_bank":sha256_file(path/"neutral_render_bank.pt")})
    expected_render=[{key:value for key,value in item.items() if key!="labels"} for item in config["data"]["frozen_render_payload_sha256"]]
    if actual_render!=expected_render: raise RuntimeError("frozen render feature payload hash mismatch")
    return rows,{"source_file_sha256":dict(source_files),"manifests":manifests,
                 "object_counts":counts,"joint_counts":joint_counts,"object_list_hashes":observed,
                 "labels_opened":False,"target_payloads_read":[]}


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


def conditional_overrides(rows,train,held,prep,config):
    """Construct both preregistered nulls wholly from train covariates."""
    train_mech,train_sem,_=transform(prep,rows,train); held_mech,held_sem,_=transform(prep,rows,held)
    train_disp=object_scalar(rows,train,"displacement")
    held_disp=object_scalar(rows,held,"displacement")
    semantic_train,semantic_held,semantic_receipt=conditional_residual_ablation(
        rows,train,held,train_sem,held_sem,train_disp,held_disp,
        folds=int(config["conditional_residual_contract"]["crossfit"]["folds"]),
        context=NULL_CONTEXTS["semantic_conditional_residual"])
    if not bool(prep.mechanical_keep[-1]) or int(torch.nonzero(prep.mechanical_keep)[-1])!=len(prep.mechanical_keep)-1:
        raise RuntimeError("recipient displacement was not retained as the last processed mechanical column")
    semantic_distance_train=object_scalar(rows,train,"semantic_distance",train)
    semantic_distance_held=object_scalar(rows,held,"semantic_distance",train)
    geometry_train,geometry_held,mechanical_receipt=conditional_residual_ablation(
        rows,train,held,train_mech[:,:-1],held_mech[:,:-1],semantic_distance_train,semantic_distance_held,
        folds=int(config["conditional_residual_contract"]["crossfit"]["folds"]),
        context=NULL_CONTEXTS["mechanical_conditional_residual"])
    mechanical_train=torch.cat((geometry_train,train_mech[:,-1:]),-1)
    mechanical_held=torch.cat((geometry_held,held_mech[:,-1:]),-1)
    d_bitwise=torch.equal(mechanical_train[:,-1],train_mech[:,-1]) and torch.equal(mechanical_held[:,-1],held_mech[:,-1])
    mechanical_receipt["recipient_displacement_bitwise_unchanged"]=d_bitwise
    contract=config["conditional_residual_contract"]
    expected_domains={"articraft","njc"}
    passes={"semantic_conditional_residual":receipt_passes(semantic_receipt,contract,expected_domains),
            "mechanical_conditional_residual":receipt_passes(mechanical_receipt,contract,expected_domains) and d_bitwise}
    receipt={"schema":"splart-ocrsmarc-null-preflight/v1","semantic_conditional_residual":semantic_receipt,
             "mechanical_conditional_residual":mechanical_receipt,
             "field_definitions":NULL_FIELD_DEFINITIONS,
             "passes":passes,"all_pass":all(passes.values())}
    receipt["all_pass"]=_null_preflight_passes(receipt,config)
    return {"semantic_conditional_residual":(semantic_train,semantic_held),
            "mechanical_conditional_residual":(mechanical_train,mechanical_held)},receipt


def _null_preflight_passes(null: dict,config: dict) -> bool:
    contract=config["conditional_residual_contract"]; domains={"articraft","njc"}
    expected={"articraft":(config["data"]["articraft"]["train_objects"],config["data"]["articraft"]["validation_objects"]),
              "njc":(config["data"]["njc"]["train_objects"],config["data"]["njc"]["validation_objects"])}
    variants=("semantic_conditional_residual","mechanical_conditional_residual")
    if (null.get("schema")!="splart-ocrsmarc-null-preflight/v1" or null.get("field_definitions")!=NULL_FIELD_DEFINITIONS or
            set(null.get("passes",{}))!=set(variants)): return False
    for variant in variants:
        receipt=null.get(variant,{})
        if receipt.get("context")!=NULL_CONTEXTS[variant]: return False
        if not receipt_passes(receipt,contract,domains): return False
        for domain,(train_count,held_count) in expected.items():
            part=receipt["domains"][domain]
            if (part["train_objects"],part["held_objects"])!=(train_count,held_count): return False
    if null["mechanical_conditional_residual"].get("recipient_displacement_bitwise_unchanged") is not True: return False
    return all(null["passes"].values())


def train_variant(rows,targets,train,held,prep,variant,config,steps,device,null_overrides=None):
    train_sem=held_sem=train_mech=held_mech=None
    if variant=="mechanical_only":
        _,raw_sem,_=transform(prep,rows,train); train_sem=torch.zeros_like(raw_sem)
        _,raw_sem,_=transform(prep,rows,held); held_sem=torch.zeros_like(raw_sem)
    elif variant=="semantic_conditional_residual":
        if null_overrides is None: raise ValueError("semantic conditional override missing")
        train_sem,held_sem=null_overrides[variant]
    elif variant=="mechanical_conditional_residual":
        if null_overrides is None: raise ValueError("mechanical conditional override missing")
        train_mech,held_mech=null_overrides[variant]
    model=train_model(rows,targets,train,prep,steps=steps,device=device,
                      semantic_processed_override=train_sem,mechanical_processed_override=train_mech)
    prediction=predict(model,prep,rows,held,semantic_processed_override=held_sem,
                       mechanical_processed_override=held_mech)
    return model,prediction


def lofo_partitions(rows,config):
    art=[i for i,row in enumerate(rows) if row["domain"]=="articraft"]
    families=sorted({rows[i]["family_audit_id"] for i in art})
    if len(families)!=3: raise RuntimeError("expected exactly three Articraft families")
    contracts={item["family_audit_id"]:item for item in config["lofo_fold_contract"]}
    if set(families)!=set(contracts): raise RuntimeError("frozen LOFO family IDs mismatch")
    partitions=[]; held_union=set()
    for family in families:
        held=[i for i in art if rows[i]["family_audit_id"]==family]; train=[i for i in art if rows[i]["family_audit_id"]!=family]
        train_objects=sorted({rows[i]["object_group_id"] for i in train}); held_objects=sorted({rows[i]["object_group_id"] for i in held})
        contract=contracts[family]; count=len(held_objects)
        if count!=contract["held_objects"]: raise RuntimeError("unexpected Articraft family size")
        train_name,held_name=contract["train_hash_key"],contract["held_hash_key"]; train_hash,held_hash=hash_ids(train_objects),hash_ids(held_objects)
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
    full,pred=train_variant(rows,targets,train,held,prep,"full",config,steps,device)
    mech,mpred=train_variant(rows,targets,train,held,prep,"mechanical_only",config,steps,device)
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
        shuffle={kind:final["metrics"][kind][domain]>=1.2*full for kind in ("semantic_conditional_residual","mechanical_conditional_residual")}
        gates[domain]={"full":full,"comparisons_15pct":comparisons,"shuffle_worsening_20pct":shuffle,"null_preflight_pass":final["null_preflight"]["all_pass"],"pass":all(comparisons.values()) and all(shuffle.values()) and final["null_preflight"]["all_pass"]}
        passed &= gates[domain]["pass"]
    total=sum(fold["held_objects"] for fold in folds)
    aggregate={key:sum(fold["held_objects"]*fold[key] for fold in folds)/total for key in ("full","mechanical_only")}
    for kind in ("global","displacement","category"):
        aggregate[kind]=sum(fold["held_objects"]*fold["analytic_controls"]["stronger_lower_mare"][kind]["articraft"] for fold in folds)/total
    lofo_pass=all(aggregate["full"]<=.85*aggregate[key] for key in ("global","displacement","category","mechanical_only"))
    fold_gates=[]
    for fold in folds:
        controls={key:fold["analytic_controls"]["stronger_lower_mare"][key]["articraft"] for key in ("global","displacement","category")}
        controls["mechanical_only"]=fold["mechanical_only"]
        comparisons={key:fold["full"]<=.85*value for key,value in controls.items()}
        fold_gates.append({"family_audit_id":fold["family_audit_id"],"full":fold["full"],"controls":controls,"comparisons_15pct":comparisons,"pass":all(comparisons.values())})
    each_fold_pass=all(item["pass"] for item in fold_gates)
    gates["lofo_object_weighted_aggregate"]=aggregate; gates["lofo_aggregate_pass_15pct"]=lofo_pass
    gates["lofo_each_fold"]=fold_gates; gates["lofo_each_fold_pass_15pct"]=each_fold_pass
    passed &= lofo_pass and each_fold_pass
    swap_max=max(value for part in [final]+folds for metric in part["swap"].values() for value in metric.values())
    gates["state_swap_max_error"]=swap_max; gates["state_swap_pass"]=swap_max<=float(config["gate"]["state_swap_max_error"]); passed &= gates["state_swap_pass"]
    gates["phase_a_pass"]=passed; gates["decision"]="ADVANCE_TO_BOX_A_D" if passed else "PRUNE_BEFORE_BOX"
    return gates


def _atomic_json(path: Path,value: dict) -> None:
    temporary=path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n")
    temporary.replace(path)


def _begin_atomic_directory(output: Path) -> Path:
    if output.exists(): raise FileExistsError(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    temporary=output.parent/(output.name+".atomic-tmp")
    if temporary.exists(): raise FileExistsError(temporary)
    temporary.mkdir()
    return temporary


def _finish_atomic_directory(temporary: Path,output: Path) -> None:
    if output.exists() or not temporary.is_dir(): raise FileExistsError(output)
    temporary.replace(output)


def aggregate_partials(partial_dirs: list[Path],output: Path,config: dict,config_sha: str,
                       current_code: dict[str,str]) -> dict:
    """Validate and deterministically aggregate four immutable task directories."""
    if len(partial_dirs)!=4: raise ValueError("aggregate requires exactly four partial directories")
    records={}; receipt_hashes={}; source_sha=None; null_sha=None
    for path in partial_dirs:
        record=json.loads((path/"partial.json").read_text()); receipt=json.loads((path/"receipt.json").read_text())
        task=record.get("task")
        if task in records: raise RuntimeError("duplicate partial task")
        if (receipt.get("partial_sha256")!=sha256_file(path/"partial.json") or
                receipt.get("models_sha256")!=sha256_file(path/"models.pt") or
                receipt.get("config_sha256")!=config_sha or receipt.get("task")!=task):
            raise RuntimeError("partial receipt hash mismatch")
        if (receipt.get("steps")!=record.get("steps") or
                receipt.get("source_provenance_sha256")!=record.get("source_provenance_sha256") or
                receipt.get("null_preflight_sha256")!=record.get("null_preflight_sha256")):
            raise RuntimeError("partial receipt is not bound to record provenance")
        if receipt.get("code_sha256")!=current_code: raise RuntimeError("partial code is stale or mutated")
        if any(item.get("box_labels_read")!=[] or item.get("protected_splits_read")!=[] for item in (record,receipt)):
            raise RuntimeError("partial reports protected read")
        if record.get("config_sha256")!=config_sha or record.get("steps")!=int(config["optimization"]["steps"]):
            raise RuntimeError("partial config/step mismatch")
        if record.get("null_preflight_sha256")!=canonical_sha256(record.get("null_preflight")):
            raise RuntimeError("conditional null receipt hash mismatch")
        null=record["null_preflight"]
        if not null.get("all_pass") or not _null_preflight_passes(null,config):
            raise RuntimeError("conditional null gates weakened or failed")
        source_sha=record["source_provenance_sha256"] if source_sha is None else source_sha
        null_sha=record["null_preflight_sha256"] if null_sha is None else null_sha
        if record["source_provenance_sha256"]!=source_sha or record["null_preflight_sha256"]!=null_sha:
            raise RuntimeError("partial source/null provenance mismatch")
        records[task]=record; receipt_hashes[task]=sha256_file(path/"receipt.json")
    expected_tasks={"final","lofo_0","lofo_1","lofo_2"}
    if set(records)!=expected_tasks: raise RuntimeError("partial task set mismatch")
    final_record=records["final"]; merge=final_record.get("merge")
    if not isinstance(merge,dict) or canonical_sha256(merge)!=source_sha or merge.get("labels_opened") is not True:
        raise RuntimeError("final merge is absent or not bound to source provenance")
    label_files=merge.get("label_file_sha256",{}); target_reads=merge.get("target_payloads_read")
    expected_label_hashes=sorted([item["labels"] for item in config["data"]["frozen_render_payload_sha256"]]+[
        config["data"]["frozen_source_sha256"]["articraft_labels"],config["data"]["frozen_source_sha256"]["njc_labels"]])
    if (not isinstance(label_files,dict) or len(label_files)!=6 or sorted(label_files.values())!=expected_label_hashes or
            target_reads!=sorted(label_files)):
        raise RuntimeError("formal source label provenance is incomplete or mutated")
    folds=[records[f"lofo_{i}"] for i in range(3)]; frozen=config["data"]["object_list_hashes"]
    contracts=sorted(config["lofo_fold_contract"],key=lambda item:item["family_audit_id"])
    if any(fold.get("family_audit_id")!=contract["family_audit_id"] or
           fold.get("train_object_hash")!=frozen[contract["train_hash_key"]] or
           fold.get("held_object_hash")!=frozen[contract["held_hash_key"]] or
           fold.get("held_objects")!=contract["held_objects"] for fold,contract in zip(folds,contracts)):
        raise RuntimeError("LOFO frozen family/train/held contract mismatch")
    held_sets=[set(fold.get("held_object_ids",[])) for fold in folds]
    if any(not value for value in held_sets) or sum(map(len,held_sets))!=109 or any(held_sets[i]&held_sets[j] for i in range(3) for j in range(i)):
        raise RuntimeError("LOFO held sets are incomplete or overlapping")
    for fold,ids in zip(folds,held_sets):
        if hash_ids(sorted(ids))!=fold["held_object_hash"] or len(ids)!=fold["held_objects"]: raise RuntimeError("LOFO held object receipt mismatch")
    gates=gate_results(final_record,folds,config)
    result={"schema":"splart-ocrsmarc-source-gate/v1","config_sha256":config_sha,
            "task_receipt_sha256":[receipt_hashes[key] for key in sorted(receipt_hashes)],
            "code_sha256":current_code,"source_provenance_sha256":source_sha,"null_preflight_sha256":null_sha,
            "final":final_record,"lofo":folds,"gates":gates,"box_labels_read":[],"protected_splits_read":[]}
    temporary=_begin_atomic_directory(output)
    _atomic_json(temporary/"metrics.json",result)
    aggregate_receipt={"schema":"splart-ocrsmarc-aggregate-receipt/v1","metrics_sha256":sha256_file(temporary/"metrics.json"),
                       "task_receipt_sha256":result["task_receipt_sha256"],"decision":gates["decision"],
                       "config_sha256":config_sha,"code_sha256":current_code,"source_provenance_sha256":source_sha,
                       "box_labels_read":[],"protected_splits_read":[]}
    _atomic_json(temporary/"receipt.json",aggregate_receipt)
    _finish_atomic_directory(temporary,output)
    return result


def main():
    p=argparse.ArgumentParser(); p.add_argument("--articraft-shards",type=Path,nargs=3); p.add_argument("--njc-cache",type=Path); p.add_argument("--articraft-pilc-data",type=Path); p.add_argument("--njc-pilc-data",type=Path); p.add_argument("--render-logs",type=Path,nargs=4); p.add_argument("--config",type=Path,required=True); p.add_argument("--dino-checkpoint",type=Path); p.add_argument("--output",type=Path,required=True); p.add_argument("--device",default="cuda"); p.add_argument("--task",choices=("preflight","smoke","final","lofo_0","lofo_1","lofo_2","aggregate"),required=True); p.add_argument("--partials",type=Path,nargs="*"); args=p.parse_args()
    if args.output.exists() or (args.output.parent/(args.output.name+".atomic-tmp")).exists(): raise FileExistsError(args.output)
    config=json.loads(args.config.read_text()); config_sha=frozen_text_sha256(args.config)
    if config_sha!=FROZEN_CONFIG_SHA256: raise RuntimeError("formal v1.3 config SHA mismatch")
    if args.task=="aggregate":
        result=aggregate_partials(args.partials or [],args.output,config,config_sha,code_sha256())
        print(json.dumps({"output":str(args.output),"decision":result["gates"]["decision"],"metrics":result["final"]["metrics"]})); return
    if any(value is None for value in (args.articraft_shards,args.njc_cache,args.articraft_pilc_data,args.njc_pilc_data,args.render_logs,args.dino_checkpoint)): raise ValueError("source task missing required inputs")
    steps=2 if args.task=="smoke" else int(config["optimization"]["steps"])
    if args.task not in ("preflight","smoke") and steps!=1200: raise RuntimeError("formal steps must equal frozen config 1200")
    rows,merge_receipt=merge_features(args,config)
    log_hashes=[sha256_file(path) for path in args.render_logs]
    if tuple(log_hashes)!=FROZEN_RENDER_LOG_SHA256: raise RuntimeError("frozen renderer log hash mismatch")
    overflow={str(path):path.read_text(errors="replace").lower().count("overflow") for path in args.render_logs}
    if any(overflow.values()): raise RuntimeError(f"renderer overflow log evidence {overflow}")
    merge_receipt["render_log_overflow_count"]=overflow; merge_receipt["render_log_sha256"]=log_hashes
    if sha256_file(args.dino_checkpoint)!=config["encoder"]["checkpoint_sha256"]: raise RuntimeError("frozen DINO checkpoint hash mismatch")
    train=indices_for(rows,{"endpoint_pretrain","njc_train"}); held=indices_for(rows,{"endpoint_validation","njc_validation"})
    train_objects={rows[i]["object_group_id"] for i in train}; held_objects={rows[i]["object_group_id"] for i in held}
    if train_objects & held_objects: raise RuntimeError("train/held object leakage")
    feature_provenance_sha=canonical_sha256(merge_receipt); code_hash=code_sha256()
    feature_prep=fit_feature_preprocessor(rows,train,16)
    null_overrides,null_preflight=conditional_overrides(rows,train,held,feature_prep,config)
    null_preflight_sha=canonical_sha256(null_preflight)
    lofo=lofo_partitions(rows,config)
    if not null_preflight["all_pass"]:
        if args.task!="preflight": raise RuntimeError("conditional-residual null preflight failed; source labels remain unopened")
        temporary=_begin_atomic_directory(args.output)
        result={"schema":"splart-ocrsmarc-source-preflight/v1","config_sha256":config_sha,"merge":merge_receipt,
                "null_preflight":null_preflight,"null_preflight_sha256":null_preflight_sha,"feature_provenance_sha256":feature_provenance_sha,
                "code_sha256":code_hash,"source_labels_opened":False,"source_scores_computed":False,
                "lofo_partitions":[{k:v for k,v in fold.items() if k not in ("train","held")} for fold in lofo],
                "box_labels_read":[],"protected_splits_read":[]}
        _atomic_json(temporary/"preflight.json",result)
        _atomic_json(temporary/"receipt.json",{"schema":"splart-ocrsmarc-preflight-receipt/v1","config_sha256":config_sha,
            "preflight_sha256":sha256_file(temporary/"preflight.json"),"code_sha256":code_hash,"decision":"PRUNE_NULL_PREFLIGHT",
            "box_labels_read":[],"protected_splits_read":[]})
        _finish_atomic_directory(temporary,args.output)
        print(json.dumps({"output":str(args.output),"conditional_null_available":False,"source_labels_opened":False})); return
    targets,label_provenance=attach_source_truth(rows,args,config)
    merge_receipt.update(label_provenance)
    source_provenance_sha=canonical_sha256(merge_receipt)
    analytic=analytic_control_set(rows,targets,train,held)
    assert_historical_controls(analytic,config)
    prep=fit_preprocessor(rows,targets,train,16)
    for name in ("mechanical_mean","mechanical_scale","mechanical_keep","semantic_mean","semantic_components"):
        if not torch.equal(getattr(prep,name),getattr(feature_prep,name)): raise RuntimeError("target-free/full preprocessor feature transform mismatch")
    if args.task=="preflight":
        temporary=_begin_atomic_directory(args.output)
        result={"schema":"splart-ocrsmarc-source-preflight/v1","config_sha256":config_sha,"merge":merge_receipt,
                "analytic_baselines":analytic,"null_preflight":null_preflight,"null_preflight_sha256":null_preflight_sha,"source_provenance_sha256":source_provenance_sha,"code_sha256":code_hash,
                "lofo_partitions":[{k:v for k,v in fold.items() if k not in ("train","held")} for fold in lofo],
                "box_labels_read":[],"protected_splits_read":[]}
        _atomic_json(temporary/"preflight.json",result); _atomic_json(temporary/"receipt.json",{"schema":"splart-ocrsmarc-preflight-receipt/v1","config_sha256":config_sha,"preflight_sha256":sha256_file(temporary/"preflight.json"),"code_sha256":code_hash,"box_labels_read":[],"protected_splits_read":[]})
        _finish_atomic_directory(temporary,args.output)
        print(json.dumps({"output":str(args.output),"analytic":analytic,"conditional_null_available":null_preflight["all_pass"]})); return
    if args.task.startswith("lofo_"):
        index=int(args.task[-1]); fold=run_lofo_fold(rows,targets,config,steps,args.device,index); checkpoint=fold.pop("checkpoint"); preprocessor=fold.pop("preprocessor")
        temporary=_begin_atomic_directory(args.output); torch.save({"checkpoint":checkpoint,"preprocessor":preprocessor},temporary/"models.pt")
        partial={"schema":"splart-ocrsmarc-source-partial/v1","task":args.task,"config_sha256":config_sha,"steps":steps,"source_provenance_sha256":source_provenance_sha,"null_preflight":null_preflight,"null_preflight_sha256":null_preflight_sha,"held_object_ids":sorted({rows[i]["object_group_id"] for i in lofo[index]["held"]}),**fold,"box_labels_read":[],"protected_splits_read":[]}
        _atomic_json(temporary/"partial.json",partial); receipt={"task":args.task,"config_sha256":config_sha,"steps":steps,"code_sha256":code_hash,"source_provenance_sha256":source_provenance_sha,"null_preflight_sha256":null_preflight_sha,"partial_sha256":sha256_file(temporary/"partial.json"),"models_sha256":sha256_file(temporary/"models.pt"),"box_labels_read":[],"protected_splits_read":[]}; _atomic_json(temporary/"receipt.json",receipt); _finish_atomic_directory(temporary,args.output); print(json.dumps({"output":str(args.output),"task":args.task,"full":fold["full"]})); return
    variants={}; checkpoints={}; predictions={}
    for variant in ("full","mechanical_only","semantic_conditional_residual","mechanical_conditional_residual"):
        model,pred=train_variant(rows,targets,train,held,prep,variant,config,steps,args.device,null_overrides)
        variants[variant]=domain_metrics(rows,targets,held,pred); checkpoints[variant]=model.state_dict(); predictions[variant]=pred
    swap_metrics={}
    for variant,state in checkpoints.items():
        for key,value in state.items():
            if not torch.isfinite(value).all(): raise RuntimeError(f"nonfinite checkpoint {variant}:{key}")
        swap_metrics[variant]=swap_audit(rows,held,predictions[variant])
    final={"schema":"splart-ocrsmarc-source-partial/v1","task":"final","config_sha256":config_sha,"steps":steps,"source_provenance_sha256":source_provenance_sha,"null_preflight":null_preflight,"null_preflight_sha256":null_preflight_sha,"merge":merge_receipt,"metrics":variants,"analytic_controls":analytic,"swap":swap_metrics,"box_labels_read":[],"protected_splits_read":[]}
    if args.task=="smoke":
        folds=[]
        for index in range(3):
            fold=run_lofo_fold(rows,targets,config,steps,args.device,index); fold.pop("checkpoint"); fold.pop("preprocessor"); folds.append(fold)
        gates=gate_results(final,folds,config); temporary=_begin_atomic_directory(args.output); _atomic_json(temporary/"metrics.json",{"schema":"splart-ocrsmarc-smoke/v1","final":final,"lofo":folds,"gates":gates,"smoke":True,"box_labels_read":[],"protected_splits_read":[]}); _finish_atomic_directory(temporary,args.output); print(json.dumps({"output":str(args.output),"task":"smoke","metrics":variants})); return
    temporary=_begin_atomic_directory(args.output); torch.save({"checkpoints":checkpoints,"preprocessor":prep},temporary/"models.pt")
    _atomic_json(temporary/"partial.json",final); receipt={"task":"final","config_sha256":config_sha,"steps":steps,"code_sha256":code_hash,"source_provenance_sha256":source_provenance_sha,"null_preflight_sha256":null_preflight_sha,"partial_sha256":sha256_file(temporary/"partial.json"),"models_sha256":sha256_file(temporary/"models.pt"),"box_labels_read":[],"protected_splits_read":[]}; _atomic_json(temporary/"receipt.json",receipt); _finish_atomic_directory(temporary,args.output); print(json.dumps({"output":str(args.output),"task":"final","metrics":variants}))


if __name__=="__main__": main()
