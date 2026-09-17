"""Counterfactual re-query fixed-point loop for endpoint refinement.

CR-FPL keeps ALD-PDL's positive anchor-relative log-distance state, but makes
the recurrent evidence state dependent.  Before every tied Transformer step,
the current distance continuously re-queries the ordered multi-radius profile.
The sampled geometry, its change from the anchor query, and the current
log-ratio form a dynamic feedback token.  The primal update is a learned
convex interpolation toward a fixed-point proposal rather than an accumulated
free residual.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from splart.ald_pdl import dual_penalty_update


CR_FPL_LOOPS = 4
CR_FPL_WIDTH = 64
CR_FPL_HEADS = 4
CR_FPL_RESIDUAL_SCALE = 0.25
CR_FPL_EPSILON = 1e-6


@dataclass(frozen=True)
class CRFPLOutput:
    """Predictions and the full state-dependent fixed-point trajectory."""

    distance: Tensor
    terminal_violation: Tensor
    loop_distances: Tensor
    loop_log_distance_ratios: Tensor
    loop_log_energy: Tensor
    loop_dual_penalty: Tensor
    loop_fixed_point_proposals: Tensor
    loop_convex_weights: Tensor
    loop_fixed_point_residuals: Tensor
    loop_query_residual_l2: Tensor
    loop_prediction_residuals: Tensor
    loop_state_residual_l2: Tensor


def differentiable_profile_query(
    features: Tensor, coordinates: Tensor, query_distance: Tensor
) -> Tensor:
    """Piecewise-linearly sample an ordered profile at a positive distance.

    The function accepts ascending or descending profile storage.  Segment
    selection is discrete, as in ordinary linear interpolation, while the
    interpolation weight and sampled geometry remain differentiable with
    respect to ``query_distance``.  Boundary segments extrapolate rather than
    clamp so a state outside cached support still receives a nonconstant
    counterfactual response.  No energy threshold or crossing search is used.
    """

    if features.ndim != 4:
        raise ValueError("features must be [B,R,S,C]")
    if coordinates.shape != features.shape[:-1]:
        raise ValueError("coordinates must be [B,R,S]")
    if query_distance.shape != features.shape[:1]:
        raise ValueError("query_distance must be [B]")
    if features.shape[2] < 2:
        raise ValueError("at least two profile samples are required")
    if not (
        torch.isfinite(features).all()
        and torch.isfinite(coordinates).all()
        and torch.isfinite(query_distance).all()
    ):
        raise ValueError("query inputs must be finite")

    sorted_coordinates, order = coordinates.sort(dim=-1)
    if not torch.all(sorted_coordinates[..., 1:] > sorted_coordinates[..., :-1]):
        raise ValueError("coordinates must be strictly ordered within each radius")
    sorted_features = features.gather(
        2, order[..., None].expand(*order.shape, features.shape[-1])
    )
    batch, radii, samples = sorted_coordinates.shape
    query = query_distance[:, None, None].expand(batch, radii, 1).contiguous()
    upper = torch.searchsorted(sorted_coordinates.contiguous(), query, right=False)
    upper = upper.clamp(1, samples - 1)
    lower = upper - 1
    lower_coordinate = sorted_coordinates.gather(2, lower)
    upper_coordinate = sorted_coordinates.gather(2, upper)
    weight = (query - lower_coordinate) / (upper_coordinate - lower_coordinate)
    gather_shape = (batch, radii, 1, features.shape[-1])
    left = sorted_features.gather(2, lower[..., None].expand(gather_shape)).squeeze(2)
    right = sorted_features.gather(2, upper[..., None].expand(gather_shape)).squeeze(2)
    return left + weight.squeeze(-1)[..., None] * (right - left)


class _PreNormDynamicRecallBlock(nn.Module):
    """One pre-norm Transformer step reused by all four re-query loops."""

    def __init__(self) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(CR_FPL_WIDTH)
        self.attention = nn.MultiheadAttention(
            CR_FPL_WIDTH, CR_FPL_HEADS, dropout=0.0, batch_first=True
        )
        self.feed_forward_norm = nn.LayerNorm(CR_FPL_WIDTH)
        self.feed_forward = nn.Sequential(
            nn.Linear(CR_FPL_WIDTH, 4 * CR_FPL_WIDTH),
            nn.GELU(),
            nn.Linear(4 * CR_FPL_WIDTH, CR_FPL_WIDTH),
        )

    def forward(self, state: Tensor, recalled_input: Tensor) -> Tensor:
        state = state + CR_FPL_RESIDUAL_SCALE * recalled_input
        normalized = self.attention_norm(state)
        attended = self.attention(normalized, normalized, normalized, need_weights=False)[0]
        state = state + CR_FPL_RESIDUAL_SCALE * attended
        return state + CR_FPL_RESIDUAL_SCALE * self.feed_forward(
            self.feed_forward_norm(state)
        )


class _SharedSideCRFPL(nn.Module):
    """The single side function shared exactly across both endpoint sides."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.profile_tokenizer = nn.Sequential(
            nn.Linear(channels + 2, CR_FPL_WIDTH),
            nn.GELU(),
            nn.Linear(CR_FPL_WIDTH, CR_FPL_WIDTH),
        )
        self.anchor_tokenizer = nn.Sequential(
            nn.Linear(1, CR_FPL_WIDTH),
            nn.GELU(),
            nn.Linear(CR_FPL_WIDTH, CR_FPL_WIDTH),
        )
        # sampled geometry + anchor-query geometry residual + current log ratio
        self.feedback_tokenizer = nn.Sequential(
            nn.Linear(2 * channels + 1, CR_FPL_WIDTH),
            nn.GELU(),
            nn.Linear(CR_FPL_WIDTH, CR_FPL_WIDTH),
        )
        self.dual_token = nn.Parameter(torch.zeros(1, 1, CR_FPL_WIDTH))
        self.feedback_seed = nn.Parameter(torch.zeros(1, 1, CR_FPL_WIDTH))
        self.loop_block = _PreNormDynamicRecallBlock()
        self.readout_norm = nn.LayerNorm(CR_FPL_WIDTH)
        self.fixed_point_readout = nn.Linear(CR_FPL_WIDTH, 1)
        self.convex_weight_readout = nn.Linear(CR_FPL_WIDTH, 1)
        self.log_energy_readout = nn.Linear(CR_FPL_WIDTH, 1)

        # The untrained model is the frozen D2 anchor at every loop.
        nn.init.zeros_(self.fixed_point_readout.weight)
        nn.init.zeros_(self.fixed_point_readout.bias)

    def _static_tokens(
        self, features: Tensor, coordinates: Tensor, anchor: Tensor
    ) -> tuple[Tensor, int]:
        batch, radii, samples, _ = features.shape
        if radii == 1:
            radius_coordinate = torch.zeros(
                1, device=features.device, dtype=features.dtype
            )
        else:
            radius_coordinate = torch.linspace(
                -1.0, 1.0, radii, device=features.device, dtype=features.dtype
            )
        radius_coordinate = radius_coordinate.view(1, radii, 1).expand(
            batch, radii, samples
        )
        payload = torch.cat(
            (features, coordinates[..., None], radius_coordinate[..., None]), dim=-1
        )
        profile_tokens = self.profile_tokenizer(
            payload.reshape(batch, radii * samples, -1)
        )
        positive_anchor = anchor.clamp_min(CR_FPL_EPSILON)
        anchor_token = self.anchor_tokenizer(torch.log(positive_anchor).view(batch, 1, 1))
        return (
            torch.cat(
                (
                    anchor_token,
                    self.dual_token.expand(batch, -1, -1),
                    self.feedback_seed.expand(batch, -1, -1),
                    profile_tokens,
                ),
                dim=1,
            ),
            2,
        )

    def forward(self, features: Tensor, coordinates: Tensor, anchor: Tensor) -> tuple[Tensor, ...]:
        static_input, feedback_index = self._static_tokens(features, coordinates, anchor)
        state = static_input
        positive_anchor = anchor.clamp_min(CR_FPL_EPSILON)
        log_anchor = torch.log(positive_anchor)
        log_ratio = torch.zeros_like(anchor)
        dual_penalty = torch.zeros_like(anchor)
        previous_log_energy: Tensor | None = None
        previous_distance = positive_anchor
        anchor_sample = differentiable_profile_query(features, coordinates, positive_anchor)

        collections: list[list[Tensor]] = [[] for _ in range(10)]
        for _ in range(CR_FPL_LOOPS):
            query_distance = torch.exp(log_anchor + log_ratio)
            sampled = differentiable_profile_query(features, coordinates, query_distance)
            query_residual = sampled - anchor_sample
            feedback_payload = torch.cat(
                (
                    sampled.mean(dim=1),
                    query_residual.mean(dim=1),
                    log_ratio[:, None],
                ),
                dim=-1,
            )
            feedback = self.feedback_tokenizer(feedback_payload)
            recalled_input = static_input.clone()
            recalled_input[:, feedback_index] = feedback

            previous_state = state
            state = self.loop_block(state, recalled_input)
            primal = self.readout_norm(state[:, 0])
            dual = self.readout_norm(state[:, 1])
            log_energy = F.softplus(self.log_energy_readout(dual).squeeze(-1))
            dual_penalty = dual_penalty_update(
                previous_log_energy, log_energy, dual_penalty
            )
            fixed_point = self.fixed_point_readout(primal).squeeze(-1) * torch.exp(
                -dual_penalty
            )
            convex_weight = torch.sigmoid(
                self.convex_weight_readout(primal).squeeze(-1)
            )
            fixed_point_residual = (fixed_point - log_ratio).abs()
            log_ratio = torch.lerp(log_ratio, fixed_point, convex_weight)
            distance = torch.exp(log_anchor + log_ratio)

            values = (
                distance,
                log_ratio,
                log_energy,
                dual_penalty,
                fixed_point,
                convex_weight,
                fixed_point_residual,
                query_residual.square().mean(dim=(1, 2)).sqrt(),
                (distance - previous_distance).abs(),
                (state - previous_state).square().mean(dim=(1, 2)).sqrt(),
            )
            for collection, value in zip(collections, values):
                collection.append(value)
            previous_log_energy = log_energy
            previous_distance = distance

        return tuple(torch.stack(values, dim=-1) for values in collections)


class CRFPLHead(nn.Module):
    """Four-loop counterfactual re-query fixed-point endpoint head."""

    def __init__(self, channels: int = 9) -> None:
        super().__init__()
        if channels < 1:
            raise ValueError("channels must be positive")
        self.channels = channels
        self.shared_side = _SharedSideCRFPL(channels)

    def forward(
        self, features: Tensor, coordinates: Tensor, anchor_distance: Tensor
    ) -> CRFPLOutput:
        if (
            features.ndim != 5
            or features.shape[1] != 2
            or features.shape[-1] != self.channels
        ):
            raise ValueError("features must be [B,2,R,S,C]")
        if coordinates.shape != features.shape[:-1]:
            raise ValueError("coordinates must be [B,2,R,S]")
        if anchor_distance.shape != features.shape[:2]:
            raise ValueError("anchor_distance must be [B,2]")
        if not (torch.isfinite(features).all() and torch.isfinite(coordinates).all()):
            raise ValueError("profiles must be finite")
        if (coordinates < 0).any():
            raise ValueError("coordinates must be non-negative outward distances")
        if not torch.isfinite(anchor_distance).all() or (anchor_distance < 0).any():
            raise ValueError("anchor_distance must be finite and non-negative")

        sides = [
            self.shared_side(
                features[:, side], coordinates[:, side], anchor_distance[:, side]
            )
            for side in range(2)
        ]
        stacked = [
            torch.stack((sides[0][index], sides[1][index]), dim=1)
            for index in range(10)
        ]
        return CRFPLOutput(
            distance=stacked[0][..., -1],
            terminal_violation=1.0 - torch.exp(-stacked[2][..., -1]),
            loop_distances=stacked[0],
            loop_log_distance_ratios=stacked[1],
            loop_log_energy=stacked[2],
            loop_dual_penalty=stacked[3],
            loop_fixed_point_proposals=stacked[4],
            loop_convex_weights=stacked[5],
            loop_fixed_point_residuals=stacked[6],
            loop_query_residual_l2=stacked[7],
            loop_prediction_residuals=stacked[8],
            loop_state_residual_l2=stacked[9],
        )


def cr_fpl_deep_supervision_loss(
    output: CRFPLOutput,
    target_distance: Tensor,
    terminal_violation_target: Tensor | None = None,
) -> Tensor:
    """Supervise every iterate and proposal, with monotone residual pressure."""

    if target_distance.shape != output.distance.shape:
        raise ValueError("target_distance must match [B,2] output")
    if not torch.isfinite(target_distance).all() or (target_distance <= 0).any():
        raise ValueError("target_distance must be finite and positive")
    target_log_ratio = (
        torch.log(target_distance).unsqueeze(-1)
        - torch.log(output.loop_distances[..., :1])
        + output.loop_log_distance_ratios[..., :1]
    )
    target_log_ratio = target_log_ratio.expand_as(output.loop_log_distance_ratios)
    predicted_log = torch.log(output.loop_distances)
    target_log = torch.log(target_distance).unsqueeze(-1).expand_as(predicted_log)
    log_error = (predicted_log - target_log).abs()
    loss = F.smooth_l1_loss(predicted_log, target_log)
    loss = loss + F.smooth_l1_loss(
        output.loop_fixed_point_proposals, target_log_ratio
    )
    loss = loss + F.smooth_l1_loss(output.loop_log_energy, log_error.detach())
    residual_increase = F.relu(
        output.loop_fixed_point_residuals[..., 1:]
        - output.loop_fixed_point_residuals[..., :-1]
    )
    loss = loss + residual_increase.mean()
    if terminal_violation_target is not None:
        if terminal_violation_target.shape != output.terminal_violation.shape:
            raise ValueError("terminal_violation_target must match [B,2] output")
        dual_target = terminal_violation_target.unsqueeze(-1).expand_as(
            output.loop_log_energy
        )
        violation = 1.0 - torch.exp(-output.loop_log_energy)
        loss = loss + F.binary_cross_entropy(violation, dual_target)
    return loss


def exact_cr_fpl_swap_error(
    model: CRFPLHead,
    features: Tensor,
    coordinates: Tensor,
    anchor_distance: Tensor,
) -> float:
    """Maximum discrepancy under an exact exchange of endpoint sides."""

    output = model(features, coordinates, anchor_distance)
    swapped = model(features.flip(1), coordinates.flip(1), anchor_distance.flip(1))
    fields = tuple(CRFPLOutput.__dataclass_fields__)
    return float(
        max(
            (getattr(swapped, field) - getattr(output, field).flip(1))
            .abs()
            .max()
            .item()
            for field in fields
        )
    )


__all__ = [
    "CR_FPL_EPSILON",
    "CR_FPL_HEADS",
    "CR_FPL_LOOPS",
    "CR_FPL_RESIDUAL_SCALE",
    "CR_FPL_WIDTH",
    "CRFPLHead",
    "CRFPLOutput",
    "cr_fpl_deep_supervision_loss",
    "differentiable_profile_query",
    "exact_cr_fpl_swap_error",
]
