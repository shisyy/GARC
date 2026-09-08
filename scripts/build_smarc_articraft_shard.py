#!/usr/bin/env python3
"""Formal faces/z-buffer Articraft render and frozen-DINO cache shard."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import torch

from splart.frozen_visual import DINO_VITB16_SHA256, encode_state_pair, load_dino_vitb16, sha256_file
from splart.smarc import opaque_object_group, opaque_row_key, swap_invariant_pair
from scripts.build_smarc_smoke import origin_transform, resolve_mesh, safe_extract


VIEWS = ((0., 0.), (0., 90.), (0., 180.), (0., 270.), (35., 45.), (-35., 225.))


def transform_points(vertices: torch.Tensor, rotation: torch.Tensor, translation: torch.Tensor) -> torch.Tensor:
    return vertices @ rotation.T + translation


def obj_mesh(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    vertices, faces = [], []
    with path.open("r", errors="replace") as handle:
        for line in handle:
            if line.startswith("v "):
                vertices.append(tuple(float(x) for x in line.split()[1:4]))
            elif line.startswith("f "):
                indices = [int(token.split("/")[0]) for token in line.split()[1:]]
                indices = [(index - 1 if index > 0 else len(vertices) + index) for index in indices]
                faces.extend((indices[0], indices[i], indices[i+1]) for i in range(1, len(indices)-1))
    if len(vertices) < 3 or not faces:
        raise ValueError("visual OBJ has no triangle faces")
    result_v, result_f = torch.tensor(vertices, dtype=torch.float32), torch.tensor(faces, dtype=torch.int64)
    if result_f.min() < 0 or result_f.max() >= len(result_v):
        raise ValueError("OBJ face index outside vertices")
    return result_v, result_f


def primitive_mesh(kind: str, node: ET.Element) -> tuple[torch.Tensor, torch.Tensor]:
    if kind == "box":
        sx, sy, sz = (float(v)/2 for v in node.attrib["size"].split())
        vertices = torch.tensor([[x,y,z] for x in (-sx,sx) for y in (-sy,sy) for z in (-sz,sz)])
        faces = torch.tensor([[0,1,3],[0,3,2],[4,6,7],[4,7,5],[0,4,5],[0,5,1],
                              [2,3,7],[2,7,6],[0,2,6],[0,6,4],[1,5,7],[1,7,3]])
        return vertices, faces
    if kind == "cylinder":
        n, radius, length = 32, float(node.attrib["radius"]), float(node.attrib["length"])
        vertices = [(radius*math.cos(2*math.pi*i/n), radius*math.sin(2*math.pi*i/n), z)
                    for z in (-length/2,length/2) for i in range(n)] + [(0,0,-length/2),(0,0,length/2)]
        faces = []
        for i in range(n):
            j=(i+1)%n; faces.extend(((i,j,n+j),(i,n+j,n+i),(2*n,j,i),(2*n+1,n+i,n+j)))
        return torch.tensor(vertices,dtype=torch.float32),torch.tensor(faces,dtype=torch.int64)
    if kind == "sphere":
        nlat,nlon,radius=16,32,float(node.attrib["radius"]); vertices=[]
        for lat in range(nlat+1):
            a=-math.pi/2+math.pi*lat/nlat
            for lon in range(nlon):
                b=2*math.pi*lon/nlon; vertices.append((radius*math.cos(a)*math.cos(b),radius*math.cos(a)*math.sin(b),radius*math.sin(a)))
        faces=[]
        for lat in range(nlat):
            for lon in range(nlon):
                j=(lon+1)%nlon; a=lat*nlon+lon; b=lat*nlon+j; c=(lat+1)*nlon+lon; d=(lat+1)*nlon+j
                faces.extend(((a,b,d),(a,d,c)))
        return torch.tensor(vertices),torch.tensor(faces)
    raise ValueError("unsupported primitive")


def visual_meshes(link: ET.Element, root: Path) -> list[tuple[torch.Tensor, torch.Tensor]]:
    result=[]
    for visual in link.findall("visual"):
        geometry=visual.find("geometry")
        if geometry is None: raise ValueError("visual missing geometry")
        mesh=None
        for kind in ("box","cylinder","sphere"):
            node=geometry.find(kind)
            if node is not None: mesh=primitive_mesh(kind,node); break
        mesh_node=geometry.find("mesh")
        if mesh is None and mesh_node is not None:
            mesh=obj_mesh(resolve_mesh(root,mesh_node.attrib["filename"])); scale=torch.tensor([float(v) for v in mesh_node.attrib.get("scale","1 1 1").split()]); mesh=(mesh[0]*scale,mesh[1])
        if mesh is None: raise ValueError("unsupported visual geometry")
        rotation,translation=origin_transform(visual.find("origin")); result.append((transform_points(mesh[0],rotation,translation),mesh[1]))
    return result


def combine(meshes: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
    if not meshes: raise ValueError("empty formal mesh part")
    vertices=[]; faces=[]; offset=0
    for v,f in meshes: vertices.append(v); faces.append(f+offset); offset+=len(v)
    return torch.cat(vertices),torch.cat(faces)


def parse_formal(root: Path):
    urdfs=list(root.rglob("model.urdf"))
    if len(urdfs)!=1: raise ValueError("archive must contain one model.urdf")
    robot=ET.parse(urdfs[0]).getroot(); links={x.attrib["name"]:x for x in robot.findall("link")}; joints=robot.findall("joint")
    records=[]; children={name:[] for name in links}; child_names=set()
    for joint in joints:
        parent=joint.find("parent").attrib["link"]; child=joint.find("child").attrib["link"]; R,t=origin_transform(joint.find("origin")); records.append((joint,parent,child,R,t)); children[parent].append(child); child_names.add(child)
    roots=set(links)-child_names
    if len(roots)!=1: raise ValueError("URDF must have one root link")
    global_pose={next(iter(roots)):(torch.eye(3),torch.zeros(3))}; pending=list(records)
    while pending:
        progressed=False
        for record in pending[:]:
            _,parent,child,R,t=record
            if parent in global_pose:
                pR,pt=global_pose[parent]; global_pose[child]=(pR@R,transform_points(t[None],pR,pt)[0]); pending.remove(record); progressed=True
        if not progressed: raise ValueError("disconnected/cyclic URDF")
    link_mesh={}
    visual_total=0
    for name,link in links.items():
        local=visual_meshes(link,urdfs[0].parent); visual_total+=len(local); R,t=global_pose[name]; link_mesh[name]=[(transform_points(v,R,t),f) for v,f in local]
    samples=[]
    for joint,parent,child,jR,jt in records:
        if joint.attrib.get("type")!="revolute": continue
        limit=joint.find("limit")
        if limit is None: continue
        lower,upper=float(limit.attrib["lower"]),float(limit.attrib["upper"])
        if not math.isfinite(lower+upper) or upper<=lower: continue
        subtree=set(); stack=[child]
        while stack:
            node=stack.pop(); subtree.add(node); stack.extend(children[node])
        mobile_meshes=[mesh for name in subtree for mesh in link_mesh[name]]; static_meshes=[mesh for name in links if name not in subtree for mesh in link_mesh[name]]
        if len(mobile_meshes)+len(static_meshes)!=visual_total: raise RuntimeError("visual assignment coverage failed")
        pR,pt=global_pose[parent]; axis_node=joint.find("axis"); axis_local=torch.tensor([float(v) for v in ("1 0 0" if axis_node is None else axis_node.attrib.get("xyz","1 0 0")).split()]); axis=pR@jR@axis_local; axis=axis/axis.norm(); pivot=transform_points(jt[None],pR,pt)[0]
        samples.append((combine(static_meshes),combine(mobile_meshes),axis,pivot,lower,upper,visual_total,len(static_meshes),len(mobile_meshes)))
    if not samples: raise ValueError("no formal finite revolute sample")
    return samples


def rotate(vertices,axis,pivot,angle):
    centered=vertices-pivot; a=torch.tensor(angle); return centered*torch.cos(a)+torch.cross(axis.expand_as(centered),centered,dim=-1)*torch.sin(a)+axis*(centered@axis)[:,None]*(1-torch.cos(a))+pivot


def render_states(static_mesh,mobile_mesh,axis,pivot,angles,device):
    from pytorch3d.renderer import (AmbientLights, BlendParams, HardFlatShader, MeshRasterizer, MeshRenderer,
                                    OrthographicCameras, RasterizationSettings, TexturesVertex, look_at_view_transform)
    from pytorch3d.structures import Meshes
    sv,sf=static_mesh; mv,mf=mobile_mesh; all_states=[torch.cat((sv,rotate(mv,axis,pivot,a))) for a in angles]; faces=torch.cat((sf,mf+len(sv)))
    all_vertices=torch.cat(all_states); center=(all_vertices.amin(0)+all_vertices.amax(0))*.5; scale=(all_vertices-center).abs().max().clamp_min(1e-6)
    colors=torch.cat((torch.full((len(sv),3),.45),torch.full((len(mv),3),.88))).to(device); faces=faces.to(device); images=[]
    # Naive rasterization avoids PyTorch3D's finite coarse-bin face capacity;
    # an overflow warning would silently drop faces and invalidate semantics.
    raster=RasterizationSettings(image_size=224,blur_radius=0.,faces_per_pixel=1,
                                 cull_backfaces=False,bin_size=0)
    lights=AmbientLights(device=device,ambient_color=((1.,1.,1.),))
    for vertices in all_states:
        mesh=Meshes(verts=[((vertices-center)/scale).to(device)],faces=[faces],textures=TexturesVertex(verts_features=[colors]))
        views=[]
        for elev,azim in VIEWS:
            R,T=look_at_view_transform(dist=3.,elev=elev,azim=azim,device=device); camera=OrthographicCameras(device=device,R=R,T=T)
            renderer=MeshRenderer(MeshRasterizer(camera,raster),HardFlatShader(device=device,cameras=camera,lights=lights,
                                                                                blend_params=BlendParams(background_color=(.05,.05,.05))))
            rgba=renderer(mesh)[0]; rgb=(rgba[...,:3].clamp(0,1)*255).to(torch.uint8).permute(2,0,1).cpu(); views.append(rgb)
        images.append(torch.stack(views))
    return torch.stack(images)


def family_hash(name):
    prefixes=("rec_hinged_window_or_hatch_","rec_sewing_box_with_simple_hinged_lid_","rec_usb_drive_with_swivel_cover_"); matches=[p for p in prefixes if name.startswith(p)]
    if len(matches)!=1: raise ValueError("unknown preregistered family")
    return hashlib.sha256(("splart-smarc-family-v1:"+matches[0]).encode()).hexdigest()


def main():
    p=argparse.ArgumentParser(); p.add_argument("--archive-root",type=Path,required=True); p.add_argument("--pilc-data",type=Path,required=True); p.add_argument("--dino-checkpoint",type=Path,required=True); p.add_argument("--output",type=Path,required=True); p.add_argument("--shard-index",type=int,required=True); p.add_argument("--shard-count",type=int,default=3); p.add_argument("--device",default="cuda"); args=p.parse_args()
    if args.output.exists() or not 0<=args.shard_index<args.shard_count: raise ValueError("existing output or bad shard")
    manifest=json.loads((args.pilc_data/"manifest.json").read_text()); accepted=manifest["accepted"]; objects=[(s,n) for s in ("endpoint_pretrain","endpoint_validation") for n in accepted[s]]; selected=[(s,n) for s,n in objects if int(opaque_object_group(n),16)%args.shard_count==args.shard_index]
    source=torch.load(args.pilc_data/"inputs.pt",map_location="cpu",weights_only=False); labels=torch.load(args.pilc_data/"labels.pt",map_location="cpu",weights_only=False); source={r["key"]:r for r in source}; labels={r["key"]:r for r in labels}; device=torch.device(args.device); encoder=load_dino_vitb16(args.dino_checkpoint,device)
    rows=[]; targets={}; renders={}; assets=[]; coverage=[]
    for split,name in selected:
        archive=args.archive_root/name; group=opaque_object_group(name); assets.append({"object_group_id":group,"archive_sha256":sha256_file(archive),"split":split})
        with tempfile.TemporaryDirectory(prefix="smarc-formal-") as td: safe_extract(archive,Path(td)); joints=parse_formal(Path(td))
        for j,(static,mobile,axis,pivot,lo,hi,total,ns,nm) in enumerate(joints):
            key=opaque_row_key(name,j,.25,.75,1,"forward")
            if key not in source or key not in labels or source[key]["split"]!=split: raise ValueError("join/split mismatch")
            d=abs(float(source[key]["observed_displacement"])); target=float(labels[key]["physical_range"])
            if abs(target-(hi-lo))>1e-5 or not d<=target<=2*math.pi+1e-6: raise ValueError("invalid range")
            images=render_states(static,mobile,axis,pivot,[lo+.25*target,lo+.75*target],device)
            foreground=(images.float().mean(2)>20).float().mean(dim=(-2,-1))
            if not torch.isfinite(foreground).all() or torch.any(foreground<=0): raise RuntimeError("empty/nonfinite formal render view")
            if not rows:
                repeated_render=render_states(static,mobile,axis,pivot,[lo+.25*target,lo+.75*target],device)
                if not torch.equal(images,repeated_render): raise RuntimeError("formal render is not bit-identical")
            semantic=encode_state_pair(encoder,images); semantic2=encode_state_pair(encoder,images)
            if not torch.equal(semantic,semantic2): raise RuntimeError("DINO non-deterministic")
            mech=torch.cat((swap_invariant_pair(source[key]["state0_features"],source[key]["state1_features"]),torch.tensor([d]))); jid=hashlib.sha256(("splart-smarc-joint-v1:"+key).encode()).hexdigest(); rows.append({"joint_id":jid,"object_group_id":group,"family_audit_id":family_hash(name),"split":split,"joint_type":"revolute","mechanical":mech,"semantic":semantic,"observed_displacement":d}); targets[jid]=target; renders[jid]=images; coverage.append({"joint_id":jid,"visual_total":total,"static_visuals":ns,"moving_visuals":nm,"coverage":1.0,"state0_foreground_min":float(foreground[0].min()),"state0_foreground_mean":float(foreground[0].mean()),"state0_foreground_max":float(foreground[0].max()),"state1_foreground_min":float(foreground[1].min()),"state1_foreground_mean":float(foreground[1].mean()),"state1_foreground_max":float(foreground[1].max())})
    if len({a["object_group_id"] for a in assets})!=len(selected) or not rows: raise RuntimeError("object coverage/collision")
    args.output.mkdir(parents=True); torch.save(rows,args.output/"inputs.pt"); torch.save(targets,args.output/"labels.pt"); torch.save(renders,args.output/"neutral_render_bank.pt"); out={"schema":"splart-smarc-articraft-dino-zbuffer-shard/v1","shard_index":args.shard_index,"shard_count":args.shard_count,"objects":len(selected),"joints":len(rows),"assets":assets,"coverage":coverage,"encoder":"DINO ViT-B/16","encoder_sha256":DINO_VITB16_SHA256,"embedding_dim":int(rows[0]["semantic"].numel()),"mechanical_dim_raw":int(rows[0]["mechanical"].numel()),"renderer":"PyTorch3D faces z-buffer full-child-subtree/static-complement bin_size=0","render_repeat_bit_identical":True,"empty_views":0,"box_labels_read":[],"protected_splits_read":[]}; (args.output/"manifest.json").write_text(json.dumps(out,indent=2,sort_keys=True)+"\n"); print(json.dumps({"output":str(args.output),"objects":len(selected),"joints":len(rows)}))


if __name__=="__main__": main()
