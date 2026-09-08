#!/usr/bin/env python3
"""Run the preregistered, source-only Phase-A SMARC gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from splart.frozen_visual import encode_state_pair, load_dino_vitb16, sha256_file
from splart.smarc_source import (analytic_linear_baseline, apply_object_donors,
                                 deterministic_object_donors, fit_preprocessor, hash_ids,
                                 object_domain_weights, object_macro_mare, predict, train_model)


def load_cache(path: Path, domain: str) -> tuple[list[dict], dict[str, float], dict]:
    rows=torch.load(path/"inputs.pt",map_location="cpu",weights_only=False)
    labels=torch.load(path/"labels.pt",map_location="cpu",weights_only=False)
    manifest=json.loads((path/"manifest.json").read_text())
    for row in rows:
        row["domain"]=domain
    return rows,{key:float(value) for key,value in labels.items()},manifest


def merge(args, config):
    rows=[]; labels={}; manifests=[]; source_files=[]
    for path in args.articraft_shards:
        new,target,manifest=load_cache(path,"articraft"); rows.extend(new); labels.update(target); manifests.append(manifest)
        source_files.extend((str(path/name),sha256_file(path/name)) for name in ("inputs.pt","labels.pt","manifest.json"))
    new,target,manifest=load_cache(args.njc_cache,"njc"); rows.extend(new); labels.update(target); manifests.append(manifest)
    source_files.extend((str(args.njc_cache/name),sha256_file(args.njc_cache/name)) for name in ("inputs.pt","labels.pt","manifest.json"))
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
    target=torch.tensor([labels[jid] for jid in joint_ids],dtype=torch.float64)
    if not torch.isfinite(target).all() or torch.any(target<=0): raise RuntimeError("invalid source truth")
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
                fw=object_domain_weights(rows,fi); means[family]=float((fw*targets[fi]).sum())
            category=torch.tensor([means.get(rows[i]["family_audit_id"],float((weights*truth).sum())) for i in va])
        else:
            category=global_prediction
        result.setdefault("category",{})[domain]=object_macro_mare(rows,va,category,targets)
    return result


def object_scalar(rows,indices,kind):
    objects=sorted({rows[i]["object_group_id"] for i in indices}); result={}
    if kind=="displacement":
        for obj in objects:
            values=[abs(float(rows[i]["observed_displacement"])) for i in indices if rows[i]["object_group_id"]==obj]
            result[obj]=sum(values)/len(values)
    elif kind=="semantic_distance":
        object_mean={obj:torch.stack([rows[i]["semantic"].double() for i in indices if rows[i]["object_group_id"]==obj]).mean(0) for obj in objects}
        global_mean=torch.stack(list(object_mean.values())).mean(0)
        for obj,value in object_mean.items():
            result[obj]=float(1-torch.nn.functional.cosine_similarity(value[None],global_mean[None]).item())
    return result


def overrides(rows,train,held,config,field,bin_kind):
    all_indices=train+held; values=object_scalar(rows,all_indices,bin_kind)
    edges=(config["shuffle_contract"]["semantic_shuffle_mechanical_bin_edges_abs_displacement_radians"]
           if bin_kind=="displacement" else config["shuffle_contract"]["mechanical_shuffle_semantic_bin_edges_cosine_distance_to_train_mean"])
    train_donor=deterministic_object_donors(rows,train,train,values,edges)
    held_donor=deterministic_object_donors(rows,train,held,values,edges)
    return (apply_object_donors(rows,train,field,train_donor,train),
            apply_object_donors(rows,held,field,held_donor,train),{"train":train_donor,"held":held_donor})


def train_variant(rows,targets,train,held,prep,variant,config,steps,device,black_raw=None):
    train_sem=held_sem=train_mech=held_mech=None; donor_receipt=None
    if variant=="mechanical_only":
        train_sem=torch.zeros(len(train),rows[0]["semantic"].numel()); held_sem=torch.zeros(len(held),rows[0]["semantic"].numel())
    elif variant=="black_image":
        train_sem=black_raw[None].repeat(len(train),1); held_sem=black_raw[None].repeat(len(held),1)
    elif variant=="semantic_shuffle":
        train_sem,held_sem,donor_receipt=overrides(rows,train,held,config,"semantic","displacement")
    elif variant=="mechanical_shuffle":
        train_mech,held_mech,donor_receipt=overrides(rows,train,held,config,"mechanical","semantic_distance")
    model=train_model(rows,targets,train,prep,train_sem,train_mech,steps,device)
    prediction=predict(model,prep,rows,held,held_sem,held_mech)
    return model,prediction,donor_receipt


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
    p=argparse.ArgumentParser(); p.add_argument("--articraft-shards",type=Path,nargs=3,required=True); p.add_argument("--njc-cache",type=Path,required=True); p.add_argument("--config",type=Path,required=True); p.add_argument("--dino-checkpoint",type=Path,required=True); p.add_argument("--output",type=Path,required=True); p.add_argument("--device",default="cuda"); p.add_argument("--steps",type=int,default=1200); args=p.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    config=json.loads(args.config.read_text()); rows,targets,merge_receipt=merge(args,config)
    train=indices_for(rows,{"endpoint_pretrain","njc_train"}); held=indices_for(rows,{"endpoint_validation","njc_validation"})
    prep=fit_preprocessor(rows,targets,train,16)
    encoder=load_dino_vitb16(args.dino_checkpoint,torch.device(args.device)); black=torch.zeros(2,6,3,224,224,dtype=torch.uint8); black_raw=encode_state_pair(encoder,black).cpu(); del encoder
    variants={}; checkpoints={}; donors={}
    for variant in ("full","mechanical_only","black_image","semantic_shuffle","mechanical_shuffle"):
        model,pred,receipt=train_variant(rows,targets,train,held,prep,variant,config,args.steps,args.device,black_raw)
        variants[variant]=domain_metrics(rows,targets,held,pred); checkpoints[variant]=model.state_dict(); donors[variant]=receipt
    analytic=analytic_baselines(rows,targets,train,held)
    lofo=run_lofo(rows,targets,config,args.steps,args.device)
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
    gates["articraft_lofo_all_folds_pass"]=all(lofo_pass); pass_all &= all(lofo_pass)
    gates["phase_a_pass"]=pass_all; gates["decision"]="ADVANCE_TO_BOX_A_D" if pass_all else "PRUNE_BEFORE_BOX"
    args.output.mkdir(parents=True); torch.save({"checkpoints":checkpoints,"preprocessor":prep,"black_raw":black_raw},args.output/"models.pt")
    result={"schema":"splart-smarc-source-gate/v1","config_sha256":sha256_file(args.config),"checkpoint_sha256":sha256_file(args.dino_checkpoint),"steps":args.steps,"merge":merge_receipt,"metrics":variants,"analytic_baselines":analytic,"lofo":lofo,"gates":gates,"donor_receipts":donors,"box_labels_read":[],"protected_splits_read":[]}
    (args.output/"metrics.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    (args.output/"receipt.json").write_text(json.dumps({"schema":"splart-smarc-source-receipt/v1","metrics_sha256":sha256_file(args.output/"metrics.json"),"models_sha256":sha256_file(args.output/"models.pt"),"decision":gates["decision"],"box_labels_read":[],"protected_splits_read":[]},indent=2,sort_keys=True)+"\n")
    print(json.dumps({"output":str(args.output),"decision":gates["decision"],"metrics":variants}))


if __name__=="__main__": main()
