"""Differentiable OEC endpoint-only optimizer."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import torch
from torch import Tensor, nn
from torch.nn.functional import normalize
from .oec_certificate import GEOMETRY_WEIGHT_THRESHOLD, _axis_angle_matrix, _quat_to_matrix
from .pcfg_certificate import COUNTERFACTUAL_DELTA, CONTACT_TOLERANCE_M, MIN_CONTACT_GAIN, deterministic_sample_indices

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

def _support_gap(m, mr, ms, s, sr, ss):
    d=s-m; dist=d.norm(dim=-1).clamp_min(torch.finfo(d.dtype).eps); u=d/dist[:,None]
    lm=torch.einsum('nji,nj->ni',mr,u); ls=torch.einsum('nji,nj->ni',sr,u)
    return dist-(lm/ms).square().sum(-1).rsqrt()-(ls/ss).square().sum(-1).rsqrt()

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
        # Fixed nearest-center association is geometry-derived and keeps the 300-step field compact.
        nearest=torch.cdist(mm,sm).argmin(-1); groups.append((state,mm,ms,mr,sm[nearest],ss[nearest],sr[nearest]))
    p=_Scalars(means.device,means.dtype); opt=torch.optim.Adam(p.parameters(),lr=lr); final=means.new_zeros(())
    def gaps(group,q):
        state,mm,ms,mr,sm,ss,sr=group; factor=q-float(state); moved=mm; movedr=mr
        if kind in {1,3}:
            motion=_axis_angle_matrix(axis,angle*factor); moved=(motion@(mm-pivot).T).T+pivot; movedr=motion@mr
        if kind in {2,3}: moved=moved+axis*(distp*factor)
        return _support_gap(moved,movedr,ms,sm,sr,ss)/float(ns_scale)
    for _ in range(iterations):
        opt.zero_grad(set_to_none=True); terms=[]
        lower,upper=p.values()
        for q,direction in ((lower,-1.),(upper,1.)):
            for group in groups:
                ge=gaps(group,q); gi=gaps(group,q-direction*COUNTERFACTUAL_DELTA); go=gaps(group,q+direction*COUNTERFACTUAL_DELTA)
                softgap=-CONTACT_TOLERANCE_M*torch.logsumexp(-ge.abs()/CONTACT_TOLERANCE_M,0)+CONTACT_TOLERANCE_M*torch.log(torch.tensor(float(ge.numel()),device=ge.device))
                overlap_i=torch.nn.functional.softplus(-gi/CONTACT_TOLERANCE_M).mean(); overlap_o=torch.nn.functional.softplus(-go/CONTACT_TOLERANCE_M).mean()
                terms += [softgap.square(), torch.relu(MIN_CONTACT_GAIN+overlap_i-overlap_o).square(), 16*torch.relu(-gi).square().mean()]
        final=torch.stack(terms).mean(); final.backward(); opt.step()
    lower,upper=p.values(); return OECFit(lower.detach(),upper.detach(),final.detach())

__all__=['OECFit','fit_oec_endpoints']
