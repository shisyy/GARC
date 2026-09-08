#!/usr/bin/env python3
"""Build the formal fifteen-object NJC neutral-render frozen-DINO cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch

from splart.frozen_visual import DINO_VITB16_SHA256, encode_state_pair, load_dino_vitb16, sha256_file
from splart.smarc import swap_invariant_pair
from scripts.build_smarc_articraft_shard import parse_formal, render_states


def njc_group(name: str) -> str:
    return hashlib.sha256(("splart-smarc-njc-object-v1:" + name).encode()).hexdigest()


def njc_key(split: str, name: str) -> str:
    base = f"{split}:{name}:0.25:0.75:axis+1"
    return hashlib.sha256(f"splart-pilc-v1:{base}:forward".encode()).hexdigest()


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--asset-root",type=Path,required=True); parser.add_argument("--pilc-data",type=Path,required=True); parser.add_argument("--dino-checkpoint",type=Path,required=True); parser.add_argument("--output",type=Path,required=True); parser.add_argument("--device",default="cuda"); args=parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    manifest=json.loads((args.pilc_data/"manifest.json").read_text()); materialized=manifest["materialized"]
    inputs={row["key"]:row for row in torch.load(args.pilc_data/"inputs.pt",map_location="cpu",weights_only=False)}; labels={row["key"]:row for row in torch.load(args.pilc_data/"labels.pt",map_location="cpu",weights_only=False)}
    device=torch.device(args.device); encoder=load_dino_vitb16(args.dino_checkpoint,device); rows=[]; targets={}; renders={}; assets=[]; coverage=[]
    for split in ("train","calibration"):
        for name in materialized[split]:
            root=args.asset_root/"assets"/name; key=njc_key(split,name)
            if key not in inputs or key not in labels or inputs[key]["split"]!=split: raise ValueError("NJC key/split join mismatch")
            joints=parse_formal(root)
            if len(joints)!=1: raise ValueError("NJC must contain exactly one formal revolute joint")
            static,mobile,axis,pivot,lo,hi,total,ns,nm=joints[0]; d=abs(float(inputs[key]["observed_displacement"])); target=float(labels[key]["physical_range"])
            if abs(target-(hi-lo))>1e-5 or not d<=target<=2*math.pi+1e-6: raise ValueError("NJC range mismatch")
            images=render_states(static,mobile,axis,pivot,[lo+.25*target,lo+.75*target],device); foreground=(images.float().mean(2)>20).float().mean((-2,-1))
            if torch.any(foreground<=0) or not torch.isfinite(foreground).all(): raise RuntimeError("empty NJC view")
            semantic=encode_state_pair(encoder,images); mechanical=torch.cat((swap_invariant_pair(inputs[key]["state0_features"],inputs[key]["state1_features"]),torch.tensor([d]))); group=njc_group(name); jid=hashlib.sha256(("splart-smarc-njc-joint-v1:"+key).encode()).hexdigest(); rows.append({"joint_id":jid,"object_group_id":group,"split":"njc_train" if split=="train" else "njc_validation","joint_type":"revolute","mechanical":mechanical,"semantic":semantic,"observed_displacement":d}); targets[jid]=target; renders[jid]=images
            asset_hashes={p.name:sha256_file(p) for p in (root/"object.urdf",root/"base_final.obj",root/"lid_final.obj")}; assets.append({"object_group_id":group,"source_sha256":asset_hashes,"split":rows[-1]["split"]}); coverage.append({"joint_id":jid,"visual_total":total,"static_visuals":ns,"moving_visuals":nm,"coverage":1.0,"state0_foreground_min":float(foreground[0].min()),"state1_foreground_min":float(foreground[1].min())})
    if len(rows)!=15 or len({r["object_group_id"] for r in rows})!=15: raise RuntimeError("NJC object coverage mismatch")
    args.output.mkdir(parents=True); torch.save(rows,args.output/"inputs.pt"); torch.save(targets,args.output/"labels.pt"); torch.save(renders,args.output/"neutral_render_bank.pt"); out={"schema":"splart-smarc-njc-dino-zbuffer/v1","objects":15,"train_objects":11,"validation_objects":4,"assets":assets,"coverage":coverage,"encoder":"DINO ViT-B/16","encoder_sha256":DINO_VITB16_SHA256,"embedding_dim":int(rows[0]["semantic"].numel()),"mechanical_dim_raw":int(rows[0]["mechanical"].numel()),"renderer":"PyTorch3D faces z-buffer neutral gray RGB bin_size=0","family_audit":"NOT_APPLICABLE_BY_NO_TAXONOMY","box_labels_read":[],"protected_splits_read":[]}; (args.output/"manifest.json").write_text(json.dumps(out,indent=2,sort_keys=True)+"\n"); print(json.dumps({"output":str(args.output),"objects":15}))


if __name__=="__main__": main()
