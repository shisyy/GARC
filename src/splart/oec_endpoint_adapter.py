"""Differentiable OEC endpoint-only optimizer."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import torch
from torch import Tensor, nn
from torch.nn.functional import normalize
from .oec_certificate import GEOMETRY_WEIGHT_THRESHOLD, _axis_angle_matrix, _quat_to_matrix
from .pcfg_certificate import COUNTERFACTUAL_DELTA, CONTACT_TOLERANCE_M, MIN_CONTACT_GAIN, RADIUS_MULTIPLIERS, deterministic_sample_indices

BROAD_PHASE_K = 8
SOFT_INDICATOR_TEMPERATURE_M = 0.002

@dataclass(frozen=True)
class OECFit:
    lower: Tensor
    upper: Tensor
    loss: Tensor

class _Scalars(nn.Module):
    def __init__(self, device, dtype):
        super().__init__(); initial=torch.log(torch.expm1(torch.tensor(.48,device=device,dtype=dtype)))
        self.lower_raw=nn.Parameter(initial.clone()); self.upper_raw=nn.Parameter(initial.clone())
    def values(self):
        return -.02-torch.nn.functional.softplus(self.lower_raw), 1.02+torch.nn.functional.softplus(self.upper_raw)

def _support_gap(m, mr, ms, s, sr, ss, multiplier=1.):
    d=s-m; dist=d.norm(dim=-1).clamp_min(torch.finfo(d.dtype).eps); u=d/dist[...,None]
    lm=torch.einsum('...ji,...j->...i',mr,u); ls=torch.einsum('...ji,...j->...i',sr,u)
    return dist-multiplier*((lm/ms).square().sum(-1).rsqrt()+(ls/ss).square().sum(-1).rsqrt())

def _soft_coverage(pair_gaps: Tensor) -> Tensor:
    # Smooth counterpart of: each mobile point contacts iff its nearest
    # candidate surface satisfies |gap| <= 5 mm.
    pair_contact=torch.sigmoid((CONTACT_TOLERANCE_M-pair_gaps.abs())/SOFT_INDICATOR_TEMPERATURE_M)
    return (1-torch.prod(1-pair_contact,dim=-1)).mean()

def fit_oec_endpoints(model: Any, ns_scale: float, iterations: int=300, lr: float=3e-2) -> OECFit:
    """Optimize exactly two scalars; every scene tensor is detached."""
    means=model.means.detach(); scales=model.scales.detach().exp(); rots=_quat_to_matrix(model.quats.detach())
    states=model.states.detach().squeeze(-1); rawmob=model.mobilities.detach().squeeze(-1)
    op=model.opacities.detach().sigmoid().squeeze(-1); mob=rawmob.sigmoid(); target=~rawmob.isnan()
    axis=normalize(model.articulation_params.axis.detach(),dim=0); kind=int(model.articulation_params.articulation_type.item())
    pivot=model.articulation_params.pivot.detach(); angle=model.articulation_params.angle.detach(); distp=model.articulation_params.dist.detach()
    groups=[]
    available_states=tuple(int(x) for x in torch.unique(states[target]).tolist())
    if not available_states: raise ValueError("no target states")
    for state in available_states:
        base=target&(states==state); smask=base&(op*(1-mob)>=GEOMETRY_WEIGHT_THRESHOLD); mmask=base&(op*mob>=GEOMETRY_WEIGHT_THRESHOLD)
        si=deterministic_sample_indices(int(smask.sum()),1024).to(means.device); mi=deterministic_sample_indices(int(mmask.sum()),1024).to(means.device)
        sm,ss,sr=means[smask][si],scales[smask][si],rots[smask][si]; mm,ms,mr=means[mmask][mi],scales[mmask][mi],rots[mmask][mi]
        # Fixed broad-phase candidates are geometry-derived, never label-derived.
        k=min(BROAD_PHASE_K,len(sm)); nearest=torch.cdist(mm,sm).topk(k,largest=False).indices
        groups.append((state,mm,ms,mr,sm[nearest],ss[nearest],sr[nearest]))
    p=_Scalars(means.device,means.dtype); opt=torch.optim.Adam(p.parameters(),lr=lr); final=means.new_zeros(())
    def gaps(group,q):
        state,mm,ms,mr,sm,ss,sr=group; factor=q-float(state); moved=mm; movedr=mr
        if kind in {1,3}:
            motion=_axis_angle_matrix(axis,angle*factor); moved=(motion@(mm-pivot).T).T+pivot; movedr=motion@mr
        if kind in {2,3}: moved=moved+axis*(distp*factor)
        k=sm.shape[1]
        return _support_gap(moved[:,None].expand(-1,k,-1),movedr[:,None].expand(-1,k,-1,-1),
            ms[:,None].expand(-1,k,-1),sm,sr,ss)/float(ns_scale)
    for _ in range(iterations):
        opt.zero_grad(set_to_none=True); terms=[]
        lower,upper=p.values()
        for q,direction in ((lower,-1.),(upper,1.)):
            anchor=0. if direction<0 else 1.
            for group in groups:
                for multiplier in RADIUS_MULTIPLIERS:
                    ge=gaps(group,q)*1.0; gi=gaps(group,q-direction*COUNTERFACTUAL_DELTA); go=gaps(group,q+direction*COUNTERFACTUAL_DELTA)
                    # Radius scaling changes supports rather than distances.
                    if multiplier!=1.:
                        # Re-evaluate by scaling the support component: d-gap is support.
                        # This remains exact and avoids rebuilding the broad phase.
                        def scaled(xq):
                            g1=gaps(group,xq); state,mm,ms,mr,sm,ss,sr=group; factor=xq-float(state); moved=mm; movedr=mr
                            if kind in {1,3}:
                                motion=_axis_angle_matrix(axis,angle*factor); moved=(motion@(mm-pivot).T).T+pivot; movedr=motion@mr
                            if kind in {2,3}: moved=moved+axis*(distp*factor)
                            k=sm.shape[1]; return _support_gap(moved[:,None].expand(-1,k,-1),movedr[:,None].expand(-1,k,-1,-1),ms[:,None].expand(-1,k,-1),sm,sr,ss,multiplier)/float(ns_scale)
                        ge,gi,go=scaled(q),scaled(q-direction*COUNTERFACTUAL_DELTA),scaled(q+direction*COUNTERFACTUAL_DELTA)
                    ga=gaps(group,torch.tensor(anchor,dtype=means.dtype,device=means.device)) if multiplier==1. else scaled(torch.tensor(anchor,dtype=means.dtype,device=means.device))
                    ce,ci,co,ca=map(_soft_coverage,(ge,gi,go,ga))
                    penetration=torch.nn.functional.softplus(-gi/SOFT_INDICATOR_TEMPERATURE_M).mean()*SOFT_INDICATOR_TEMPERATURE_M
                    terms += [torch.relu(ca+MIN_CONTACT_GAIN-ce).square(),
                              torch.relu(MIN_CONTACT_GAIN+ci-co).square(),16*penetration.square()]
        final=torch.stack(terms).mean(); final.backward(); opt.step()
    lower,upper=p.values(); return OECFit(lower.detach(),upper.detach(),final.detach())

__all__=['OECFit','fit_oec_endpoints']
