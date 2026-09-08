"""Semantic-Gated Mechanical Authored-Range Completion primitives."""

from __future__ import annotations

import hashlib

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def opaque_row_key(archive_name: str, joint_index: int, f0: float, f1: float,
                   orientation: int, order: str) -> str:
    raw = f"{archive_name}:joint{joint_index}:{f0}:{f1}:{orientation}:{order}"
    return hashlib.sha256(("splart-pilc-articraft-row-v1:" + raw).encode()).hexdigest()


def opaque_object_group(archive_name: str) -> str:
    return hashlib.sha256(("splart-smarc-object-v1:" + archive_name).encode()).hexdigest()


def swap_invariant_pair(state0: Tensor, state1: Tensor) -> Tensor:
    if state0.shape != state1.shape:
        raise ValueError("state feature shapes differ")
    return torch.cat(((state0 + state1) * .5, (state0 - state1).abs()), dim=-1)


def model_tensors(rows: list[dict]) -> dict[str, Tensor]:
    """Create the exact model whitelist while retaining IDs only in sidecars."""
    result = {"mechanical": torch.stack([row["mechanical"] for row in rows]),
              "semantic": torch.stack([row["semantic"] for row in rows]),
              "observed_displacement": torch.tensor([row["observed_displacement"] for row in rows])}
    if set(result) != {"mechanical", "semantic", "observed_displacement"}:
        raise RuntimeError("model tensor whitelist changed")
    return result


class SMARC(nn.Module):
    """Semantics gates K bias-free mechanical residual experts.

    There is deliberately no semantic additive path and no bias in the
    mechanical experts. Zero semantic features yield the uniform-gate
    mechanical-only control exactly.
    """

    def __init__(self, mechanical_dim: int, semantic_dim: int, global_prior_logit: float,
                 experts: int = 4, joint_kind: str = "revolute") -> None:
        super().__init__()
        if mechanical_dim < 1 or semantic_dim < 1 or experts != 4:
            raise ValueError("positive dimensions and exactly four experts required")
        self.expert_weight = nn.Parameter(torch.empty(experts, mechanical_dim))
        self.semantic_gate = nn.Linear(semantic_dim, experts, bias=False)
        if joint_kind not in {"revolute", "prismatic"}:
            raise ValueError("joint kind must be revolute or prismatic")
        self.joint_kind = joint_kind
        self.register_buffer("global_prior_logit", torch.tensor(float(global_prior_logit)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            row = torch.arange(self.expert_weight.shape[0], dtype=self.expert_weight.dtype)[:, None]
            col = torch.arange(self.expert_weight.shape[1], dtype=self.expert_weight.dtype)[None, :]
            self.expert_weight.copy_(torch.sin((row + 1) * (col + 1) * .017) /
                                     self.expert_weight.shape[1] ** .5)
            grow = torch.arange(self.semantic_gate.weight.shape[0], dtype=self.semantic_gate.weight.dtype)[:, None]
            gcol = torch.arange(self.semantic_gate.weight.shape[1], dtype=self.semantic_gate.weight.dtype)[None, :]
            self.semantic_gate.weight.copy_(torch.sin((grow + 1) * (gcol + 1) * .013) /
                                            self.semantic_gate.weight.shape[1] ** .5)

    def forward(self, mechanical: Tensor, semantic: Tensor, observed_displacement: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        if mechanical.shape[:-1] != semantic.shape[:-1] or mechanical.shape[-1] != self.expert_weight.shape[1]:
            raise ValueError("mechanical/semantic shape mismatch")
        if semantic.shape[-1] != self.semantic_gate.in_features:
            raise ValueError("semantic dimension mismatch")
        if observed_displacement.shape != mechanical.shape[:-1]:
            raise ValueError("displacement shape mismatch")
        gate = torch.softmax(self.semantic_gate(semantic), dim=-1)
        expert_residual = F.linear(mechanical, self.expert_weight)
        residual = (gate * expert_residual).sum(-1)
        displacement = observed_displacement.abs()
        if self.joint_kind == "revolute":
            if torch.any(displacement >= 2 * torch.pi):
                raise ValueError("revolute displacement must be below 2pi")
            fraction = torch.sigmoid(self.global_prior_logit + residual)
            physical_range = displacement + (2 * torch.pi - displacement) * fraction
        else:
            fraction = None
            physical_range = displacement + F.softplus(self.global_prior_logit + residual)
        return physical_range, {"gate": gate, "expert_residual": expert_residual,
                                "completion_fraction": fraction,
                                "global_prior_logit": self.global_prior_logit}


def project_extensions_to_range(base_extension: Tensor, physical_range: Tensor,
                                observed_displacement: Tensor) -> Tensor:
    """Euclidean project two extensions onto the nonnegative fixed-sum simplex."""
    if base_extension.shape[-1] != 2 or physical_range.shape != base_extension.shape[:-1]:
        raise ValueError("invalid extension/range shapes")
    displacement = observed_displacement.abs()
    if torch.any(displacement <= 0) or torch.any(physical_range < displacement):
        raise ValueError("physical range must cover the observed displacement")
    total = physical_range / displacement - 1.0
    first = ((total + base_extension[..., 0] - base_extension[..., 1]) * .5).clamp_min(0)
    first = torch.minimum(first, total)
    return torch.stack((first, total - first), -1)


def extensions_to_endpoints(extension: Tensor) -> Tensor:
    if extension.shape[-1] != 2 or torch.any(extension < 0):
        raise ValueError("two nonnegative extensions required")
    return torch.stack((-extension[..., 0], 1.0 + extension[..., 1]), -1)


__all__ = ["SMARC", "extensions_to_endpoints", "model_tensors", "opaque_object_group", "opaque_row_key",
           "project_extensions_to_range", "swap_invariant_pair"]
