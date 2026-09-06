"""Frozen node2.2 profile-head runtime; contains no dataset or evaluator paths."""
from __future__ import annotations
from dataclasses import asdict, dataclass
from math import ceil
from typing import Any, Iterable
import hashlib, json, random
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from .gauge_energy_profile import canonical_outward_coordinates, swap_profiles

PRIVATE_KEYS={"target","targets","presentation","presentation_bit","canonical_side","joint_limits","fractions","split_membership","b_test","full22"}
VARIANTS=("shared","shared_distance_only","coordinate_only","scalar_only_mlp","pooled_summary_mlp","fixed_permutation","unshared_head")

@dataclass(frozen=True)
class TrainConfig:
    seed:int=2202; steps:int=4000; learning_rate:float=.001; betas:tuple[float,float]=(.9,.999)
    weight_decay:float=.0001; gradient_clip_norm:float=1.; epsilon:float=1e-6

FROZEN_CONFIG=TrainConfig()

def reject_private(value:Any,path:str="root") -> None:
    if isinstance(value,dict):
        for k,v in value.items():
            if str(k).lower() in PRIVATE_KEYS: raise ValueError(f"private field: {path}.{k}")
            reject_private(v,f"{path}.{k}")
    elif isinstance(value,list):
        for i,v in enumerate(value): reject_private(v,f"{path}[{i}]")
    elif isinstance(value,str) and any(x in value.lower() for x in ("b_test","full22")):
        raise ValueError(f"private path marker: {path}")

class ProfileNormalizer(nn.Module):
    def __init__(self,features:Tensor,scalars:Tensor,eps:float=1e-6):
        super().__init__(); out=torch.log1p(canonical_outward_coordinates(scalars))
        self.register_buffer("feature_mean",features.mean((0,1,2,3))); self.register_buffer("feature_std",torch.sqrt(features.var((0,1,2,3),unbiased=False)+eps))
        self.register_buffer("scalar_mean",out.mean()); self.register_buffer("scalar_std",torch.sqrt(out.var(unbiased=False)+eps))
    def forward(self,f:Tensor,s:Tensor)->tuple[Tensor,Tensor]:
        return (f-self.feature_mean)/self.feature_std,(torch.log1p(canonical_outward_coordinates(s))-self.scalar_mean)/self.scalar_std
    def digest(self)->str:
        h=hashlib.sha256()
        for x in (self.feature_mean,self.feature_std,self.scalar_mean,self.scalar_std): h.update(x.detach().cpu().numpy().tobytes())
        return h.hexdigest()

class SideEncoder(nn.Module):
    def __init__(self,in_channels:int):
        super().__init__(); self.point=nn.Sequential(nn.Linear(in_channels,64),nn.SiLU(),nn.Linear(64,64),nn.SiLU()); self.seq=nn.Conv1d(64,64,5,padding=2); self.read=nn.Sequential(nn.Linear(128,64),nn.SiLU(),nn.Linear(64,2))
    def forward(self,x:Tensor)->Tensor:
        b,r,s,c=x.shape; y=self.point(x.reshape(b*r,s,c)); y=F.silu(self.seq(y.transpose(1,2))).transpose(1,2); y=torch.cat((y.mean(1),y.amax(1)),1).reshape(b,r,128).mean(1); return self.read(y)

class Node22Head(nn.Module):
    def __init__(self,variant:str="shared"):
        super().__init__();
        if variant not in VARIANTS: raise ValueError("unknown frozen variant")
        self.variant=variant; inc=1 if variant=="scalar_only_mlp" else (20 if variant=="pooled_summary_mlp" else 10)
        self.shared=SideEncoder(inc) if variant!="unshared_head" else None
        self.sides=nn.ModuleList([SideEncoder(10),SideEncoder(10)]) if variant=="unshared_head" else None
    def _side_input(self,f:Tensor,x:Tensor)->Tensor:
        if self.variant=="coordinate_only": f=torch.zeros_like(f)
        if self.variant=="fixed_permutation": f=f.flip(-2)
        if self.variant=="scalar_only_mlp": return x[...,None]
        if self.variant=="pooled_summary_mlp":
            z=torch.cat((f.mean(-2),f.amax(-2),x.mean(-1,keepdim=True),x.amax(-1,keepdim=True)),2); return z.unsqueeze(-2)
        return torch.cat((f,x[...,None]),3)
    def forward(self,f:Tensor,x:Tensor)->tuple[Tensor,Tensor]:
        if f.ndim!=5 or f.shape[1]!=2 or f.shape[-1]!=9 or x.shape!=f.shape[:-1]: raise ValueError("profile shape")
        outs=[(self.sides[i] if self.sides else self.shared)(self._side_input(f[:,i],x[:,i])) for i in range(2)]
        raw=torch.stack(outs,1); scale=torch.ones_like(raw[...,1]) if self.variant=="shared_distance_only" else F.softplus(raw[...,1])+1e-4; return F.softplus(raw[...,0]),scale

def initialize_frozen(module:nn.Module,seed:int=2202)->None:
    random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    for m in module.modules():
        if isinstance(m,(nn.Linear,nn.Conv1d)): nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)

class RunGuard:
    def __init__(self): self.optimizer_initialized=False; self.target_read=False; self.retry_count=0
    def may_retry(self)->bool: return not self.optimizer_initialized and not self.target_read and self.retry_count<1
    def record_retry(self)->None:
        if not self.may_retry(): raise RuntimeError("rerun forbidden after optimizer initialization or target read")
        self.retry_count+=1

def train_once(features:Tensor,scalars:Tensor,targets:Tensor,object_ids:Iterable[str],variant:str="shared",config:TrainConfig=FROZEN_CONFIG)->tuple[Node22Head,ProfileNormalizer,dict]:
    ids=tuple(object_ids)
    if len(ids)!=18 or len(set(ids))!=18 or ids!=tuple(sorted(ids)): raise ValueError("exactly 18 unique lexicographically ordered train objects required")
    if config!=FROZEN_CONFIG: raise ValueError("training config is frozen")
    torch.use_deterministic_algorithms(True); normalizer=ProfileNormalizer(features,scalars,config.epsilon); nf,nx=normalizer(features,scalars)
    model=Node22Head(variant).to(features.device); initialize_frozen(model,config.seed); opt=torch.optim.AdamW(model.parameters(),lr=config.learning_rate,betas=config.betas,weight_decay=config.weight_decay)
    for _ in range(config.steps):
        opt.zero_grad(set_to_none=True); pred,scale=model(nf,nx); loss=(torch.max((pred-targets).abs()/scale,1).values+.05*torch.log(scale).mean(1)).mean(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),config.gradient_clip_norm); opt.step()
    return model,normalizer,{"schema":"splart-node2.2-training-receipt/v1","objects":18,"steps":4000,"variant":variant,"normalizer_sha256":normalizer.digest(),"checkpoint_policy":"single-final-only","config":asdict(config)}

@dataclass(frozen=True)
class JointConformal:
    q:float; coverage:float; kind:str
    def interval(self,pred:Tensor,scale:Tensor|None=None)->tuple[Tensor,Tensor]:
        radius=torch.full_like(pred,self.q) if self.kind=="constant" else scale*self.q
        return (pred-radius).clamp_min(0),pred+radius

def fit_joint_conformal(pred:Tensor,scale:Tensor,target:Tensor,object_ids:Iterable[str],coverage:float=.9,kind:str="scaled")->JointConformal:
    ids=tuple(object_ids)
    if len(ids)!=9 or len(set(ids))!=9 or pred.shape!=(9,2) or scale.shape!=(9,2) or target.shape!=(9,2): raise ValueError("exactly nine unique calibration objects required")
    scores=((pred-target).abs()/scale).amax(1) if kind=="scaled" else (pred-target).abs().amax(1); rank=ceil((len(ids)+1)*coverage)
    if rank>len(ids): raise ValueError("finite conformal rank unavailable")
    return JointConformal(float(scores.sort().values[rank-1]),coverage,kind)

def aggregate_confirmatory(rows:list[dict])->dict:
    if len(rows)!=9 or len({r["object_id"] for r in rows})!=9: raise ValueError("nine unique confirmatory rows required")
    keys=("endpoint_nmae","joint_covered","mean_joint_width")
    return {"schema":"splart-node2.2-confirmatory-aggregate/v1","unit":"object","objects":9,"metrics":{k:sum(float(r[k]) for r in rows)/9 for k in keys},"per_object_targets_emitted":False,"protected_splits_read":[]}

def exact_swap_error(model:Node22Head,f:Tensor,x:Tensor)->float:
    if model.variant=="unshared_head": raise ValueError("unshared comparator is not structurally equivariant")
    p,s=model(f,x); q,t=model(f.flip(1),x.flip(1)); return float(torch.max(torch.stack(((q-p.flip(1)).abs().max(),(t-s.flip(1)).abs().max()))))

__all__=["FROZEN_CONFIG","JointConformal","Node22Head","ProfileNormalizer","RunGuard","VARIANTS","aggregate_confirmatory","exact_swap_error","fit_joint_conformal","initialize_frozen","reject_private","train_once"]
