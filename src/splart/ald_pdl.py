"""Anchor-relative log-distance primal-dual loop for endpoint refinement.

ALD-PDL preserves the tied, input-recalled Transformer structure of GLPDT but
changes the primal geometry.  The primal state is an accumulated log distance
ratio, so every positive distance is reachable without a correction cap.  A
learned log-energy read from the shared dual token can only *increase* the
dual penalty; decreases never suppress a primal update.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


ALD_PDL_LOOPS = 4
ALD_PDL_WIDTH = 64
ALD_PDL_HEADS = 4
ALD_PDL_RESIDUAL_SCALE = 0.25
ALD_PDL_EPSILON = 1e-6


@dataclass(frozen=True)
class ALDPDLOutput:
    """Predictions plus the complete four-loop primal-dual trajectory."""

    distance: Tensor
    terminal_violation: Tensor
    loop_distances: Tensor
    loop_log_distance_ratios: Tensor
    loop_log_energy: Tensor
    loop_dual_penalty: Tensor
    loop_primal_updates: Tensor
    loop_prediction_residuals: Tensor
    loop_state_residual_l2: Tensor


class _PreNormRecalledBlock(nn.Module):
    """One pre-norm Transformer step, reused exactly four times."""

    def __init__(self) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(ALD_PDL_WIDTH)
        self.attention = nn.MultiheadAttention(
            ALD_PDL_WIDTH, ALD_PDL_HEADS, dropout=0.0, batch_first=True
        )
        self.feed_forward_norm = nn.LayerNorm(ALD_PDL_WIDTH)
        self.feed_forward = nn.Sequential(
            nn.Linear(ALD_PDL_WIDTH, 4 * ALD_PDL_WIDTH),
            nn.GELU(),
            nn.Linear(4 * ALD_PDL_WIDTH, ALD_PDL_WIDTH),
        )

    def forward(self, state: Tensor, original_input: Tensor) -> Tensor:
        state = state + ALD_PDL_RESIDUAL_SCALE * original_input
        normalized = self.attention_norm(state)
        attended = self.attention(normalized, normalized, normalized, need_weights=False)[0]
        state = state + ALD_PDL_RESIDUAL_SCALE * attended
        state = state + ALD_PDL_RESIDUAL_SCALE * self.feed_forward(
            self.feed_forward_norm(state)
        )
        return state


def dual_penalty_update(
    previous_log_energy: Tensor | None, current_log_energy: Tensor, penalty: Tensor
) -> Tensor:
    """Accumulate only positive log-energy changes, without a threshold."""

    if previous_log_energy is None:
        return penalty
    return penalty + F.relu(current_log_energy - previous_log_energy)


class _SharedSideALDPDL(nn.Module):
    """One exactly shared side function for lower and upper endpoint profiles."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.profile_tokenizer = nn.Sequential(
            nn.Linear(channels + 2, ALD_PDL_WIDTH),
            nn.GELU(),
            nn.Linear(ALD_PDL_WIDTH, ALD_PDL_WIDTH),
        )
        self.anchor_tokenizer = nn.Sequential(
            nn.Linear(1, ALD_PDL_WIDTH),
            nn.GELU(),
            nn.Linear(ALD_PDL_WIDTH, ALD_PDL_WIDTH),
        )
        self.dual_token = nn.Parameter(torch.zeros(1, 1, ALD_PDL_WIDTH))
        self.loop_block = _PreNormRecalledBlock()
        self.readout_norm = nn.LayerNorm(ALD_PDL_WIDTH)
        self.primal_readout = nn.Linear(ALD_PDL_WIDTH, 1)
        self.log_energy_readout = nn.Linear(ALD_PDL_WIDTH, 1)

        # Start from the frozen D2 anchor.  This is an initialization, not an
        # action-space bound: training can learn any finite log ratio.
        nn.init.zeros_(self.primal_readout.weight)
        nn.init.zeros_(self.primal_readout.bias)

    def _original_tokens(self, features: Tensor, coordinates: Tensor, anchor: Tensor) -> Tensor:
        batch, radii, samples, _ = features.shape
        if radii == 1:
            radius_coordinate = torch.zeros(1, device=features.device, dtype=features.dtype)
        else:
            radius_coordinate = torch.linspace(
                -1.0, 1.0, radii, device=features.device, dtype=features.dtype
            )
        radius_coordinate = radius_coordinate.view(1, radii, 1).expand(batch, radii, samples)
        payload = torch.cat(
            (features, coordinates[..., None], radius_coordinate[..., None]), dim=-1
        )
        profile_tokens = self.profile_tokenizer(payload.reshape(batch, radii * samples, -1))
        positive_anchor = anchor.clamp_min(ALD_PDL_EPSILON)
        anchor_token = self.anchor_tokenizer(torch.log(positive_anchor).view(batch, 1, 1))
        dual_token = self.dual_token.expand(batch, -1, -1)
        return torch.cat((anchor_token, dual_token, profile_tokens), dim=1)

    def forward(self, features: Tensor, coordinates: Tensor, anchor: Tensor) -> tuple[Tensor, ...]:
        original = self._original_tokens(features, coordinates, anchor)
        state = original
        positive_anchor = anchor.clamp_min(ALD_PDL_EPSILON)
        log_anchor = torch.log(positive_anchor)
        log_ratio = torch.zeros_like(anchor)
        dual_penalty = torch.zeros_like(anchor)
        previous_log_energy: Tensor | None = None
        previous_distance = positive_anchor

        distances, log_ratios, log_energies, penalties = [], [], [], []
        primal_updates, prediction_residuals, state_residuals = [], [], []
        for _ in range(ALD_PDL_LOOPS):
            previous_state = state
            state = self.loop_block(state, original)
            primal = self.readout_norm(state[:, 0])
            dual = self.readout_norm(state[:, 1])

            log_energy = F.softplus(self.log_energy_readout(dual).squeeze(-1))
            dual_penalty = dual_penalty_update(previous_log_energy, log_energy, dual_penalty)
            raw_update = self.primal_readout(primal).squeeze(-1)
            primal_update = raw_update * torch.exp(-dual_penalty)
            log_ratio = log_ratio + primal_update
            distance = torch.exp(log_anchor + log_ratio)

            distances.append(distance)
            log_ratios.append(log_ratio)
            log_energies.append(log_energy)
            penalties.append(dual_penalty)
            primal_updates.append(primal_update)
            prediction_residuals.append((distance - previous_distance).abs())
            state_residuals.append((state - previous_state).square().mean(dim=(1, 2)).sqrt())
            previous_log_energy = log_energy
            previous_distance = distance

        return tuple(
            torch.stack(values, dim=-1)
            for values in (
                distances,
                log_ratios,
                log_energies,
                penalties,
                primal_updates,
                prediction_residuals,
                state_residuals,
            )
        )


class ALDPDLHead(nn.Module):
    """Four-loop anchor-relative log-distance primal-dual endpoint head."""

    def __init__(self, channels: int = 9) -> None:
        super().__init__()
        if channels < 1:
            raise ValueError("channels must be positive")
        self.channels = channels
        self.shared_side = _SharedSideALDPDL(channels)

    def forward(self, features: Tensor, coordinates: Tensor, anchor_distance: Tensor) -> ALDPDLOutput:
        if features.ndim != 5 or features.shape[1] != 2 or features.shape[-1] != self.channels:
            raise ValueError("features must be [B,2,R,S,C]")
        if coordinates.shape != features.shape[:-1]:
            raise ValueError("coordinates must be [B,2,R,S]")
        if anchor_distance.shape != features.shape[:2]:
            raise ValueError("anchor_distance must be [B,2]")
        if not (torch.isfinite(features).all() and torch.isfinite(coordinates).all()):
            raise ValueError("profiles must be finite")
        if not torch.isfinite(anchor_distance).all() or (anchor_distance < 0).any():
            raise ValueError("anchor_distance must be finite and non-negative")

        sides = [
            self.shared_side(features[:, side], coordinates[:, side], anchor_distance[:, side])
            for side in range(2)
        ]
        stacked = [torch.stack((sides[0][index], sides[1][index]), dim=1) for index in range(7)]
        loop_distances, loop_log_ratios, loop_log_energy, loop_penalty = stacked[:4]
        loop_updates, loop_prediction_residuals, loop_state_residuals = stacked[4:]
        return ALDPDLOutput(
            distance=loop_distances[..., -1],
            terminal_violation=1.0 - torch.exp(-loop_log_energy[..., -1]),
            loop_distances=loop_distances,
            loop_log_distance_ratios=loop_log_ratios,
            loop_log_energy=loop_log_energy,
            loop_dual_penalty=loop_penalty,
            loop_primal_updates=loop_updates,
            loop_prediction_residuals=loop_prediction_residuals,
            loop_state_residual_l2=loop_state_residuals,
        )


def ald_pdl_deep_supervision_loss(
    output: ALDPDLOutput,
    target_distance: Tensor,
    terminal_violation_target: Tensor | None = None,
) -> Tensor:
    """Train every loop in log-distance and calibrate its learned log-energy."""

    if target_distance.shape != output.distance.shape:
        raise ValueError("target_distance must match [B,2] output")
    if not torch.isfinite(target_distance).all() or (target_distance <= 0).any():
        raise ValueError("target_distance must be finite and positive")
    target_log = torch.log(target_distance).unsqueeze(-1).expand_as(output.loop_distances)
    predicted_log = torch.log(output.loop_distances)
    log_error = (predicted_log - target_log).abs()
    loss = F.smooth_l1_loss(predicted_log, target_log)
    loss = loss + F.smooth_l1_loss(output.loop_log_energy, log_error.detach())
    if terminal_violation_target is not None:
        if terminal_violation_target.shape != output.terminal_violation.shape:
            raise ValueError("terminal_violation_target must match [B,2] output")
        dual_target = terminal_violation_target.unsqueeze(-1).expand_as(output.loop_log_energy)
        violation = 1.0 - torch.exp(-output.loop_log_energy)
        loss = loss + F.binary_cross_entropy(violation, dual_target)
    return loss


def exact_ald_pdl_swap_error(
    model: ALDPDLHead, features: Tensor, coordinates: Tensor, anchor_distance: Tensor
) -> float:
    """Return maximum side-swap discrepancy over outputs and diagnostics."""

    output = model(features, coordinates, anchor_distance)
    swapped = model(features.flip(1), coordinates.flip(1), anchor_distance.flip(1))
    fields = (
        "distance",
        "terminal_violation",
        "loop_distances",
        "loop_log_distance_ratios",
        "loop_log_energy",
        "loop_dual_penalty",
        "loop_primal_updates",
        "loop_prediction_residuals",
        "loop_state_residual_l2",
    )
    return float(
        max(
            (getattr(swapped, field) - getattr(output, field).flip(1)).abs().max().item()
            for field in fields
        )
    )


__all__ = [
    "ALD_PDL_EPSILON",
    "ALD_PDL_HEADS",
    "ALD_PDL_LOOPS",
    "ALD_PDL_RESIDUAL_SCALE",
    "ALD_PDL_WIDTH",
    "ALDPDLHead",
    "ALDPDLOutput",
    "ald_pdl_deep_supervision_loss",
    "dual_penalty_update",
    "exact_ald_pdl_swap_error",
]
