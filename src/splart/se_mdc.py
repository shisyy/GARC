"""Swap-equivariant minimax dual coupling for looped endpoint refinement.

SE-MDC retains CR-FPL's positive anchor-relative state, differentiable profile
re-query, and four tied fixed-point iterations.  Its only structural change is
a permutation-equivariant two-side dual coupler.  At every loop, the two
shared side encoders exchange their slack tokens through self-attention.  A
smooth relative-risk distribution then allocates a shared update budget across
the two convex fixed-point steps without side IDs, thresholds, or quantiles.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from splart.ald_pdl import dual_penalty_update
from splart.cr_fpl import differentiable_profile_query


SE_MDC_LOOPS = 4
SE_MDC_WIDTH = 64
SE_MDC_HEADS = 4
SE_MDC_RESIDUAL_SCALE = 0.25
SE_MDC_EPSILON = 1e-6


@dataclass(frozen=True)
class SEMDCOutput:
    """Predictions and complete coupled fixed-point diagnostics."""

    distance: Tensor
    terminal_violation: Tensor
    loop_distances: Tensor
    loop_log_distance_ratios: Tensor
    loop_log_energy: Tensor
    loop_dual_penalty: Tensor
    loop_fixed_point_proposals: Tensor
    loop_convex_weights: Tensor
    loop_dual_allocation: Tensor
    loop_update_budget: Tensor
    loop_fixed_point_residuals: Tensor
    loop_query_residual_l2: Tensor
    loop_prediction_residuals: Tensor
    loop_state_residual_l2: Tensor
    loop_coupled_slack_residual_l2: Tensor


class _PreNormDynamicRecallBlock(nn.Module):
    """One width-64 pre-norm block tied across all four loops and both sides."""

    def __init__(self) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(SE_MDC_WIDTH)
        self.attention = nn.MultiheadAttention(
            SE_MDC_WIDTH, SE_MDC_HEADS, dropout=0.0, batch_first=True
        )
        self.feed_forward_norm = nn.LayerNorm(SE_MDC_WIDTH)
        self.feed_forward = nn.Sequential(
            nn.Linear(SE_MDC_WIDTH, 4 * SE_MDC_WIDTH),
            nn.GELU(),
            nn.Linear(4 * SE_MDC_WIDTH, SE_MDC_WIDTH),
        )

    def forward(self, state: Tensor, recalled_input: Tensor) -> Tensor:
        state = state + SE_MDC_RESIDUAL_SCALE * recalled_input
        normalized = self.attention_norm(state)
        attended = self.attention(normalized, normalized, normalized, need_weights=False)[0]
        state = state + SE_MDC_RESIDUAL_SCALE * attended
        return state + SE_MDC_RESIDUAL_SCALE * self.feed_forward(
            self.feed_forward_norm(state)
        )


class SwapEquivariantDualCoupler(nn.Module):
    """Exchange two anonymous slack tokens and allocate a smooth update budget.

    Self-attention has no positional or side embedding, so exchanging the two
    inputs exchanges the two outputs exactly.  The shared scalar risk readout
    defines an entropic dual distribution.  A permutation-invariant budget is
    divided by that distribution and mapped smoothly into ``(0, 1)``, making
    every resulting primal step a convex interpolation.
    """

    def __init__(self) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(SE_MDC_WIDTH)
        self.attention = nn.MultiheadAttention(
            SE_MDC_WIDTH, SE_MDC_HEADS, dropout=0.0, batch_first=True
        )
        self.feed_forward_norm = nn.LayerNorm(SE_MDC_WIDTH)
        self.feed_forward = nn.Sequential(
            nn.Linear(SE_MDC_WIDTH, 2 * SE_MDC_WIDTH),
            nn.GELU(),
            nn.Linear(2 * SE_MDC_WIDTH, SE_MDC_WIDTH),
        )
        self.risk_readout = nn.Linear(SE_MDC_WIDTH, 1)
        self.budget_readout = nn.Linear(SE_MDC_WIDTH, 1)

    def forward(
        self, slack: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        if slack.ndim != 3 or slack.shape[1:] != (2, SE_MDC_WIDTH):
            raise ValueError("slack must be [B,2,64]")
        previous = slack
        normalized = self.attention_norm(slack)
        exchanged = self.attention(normalized, normalized, normalized, need_weights=False)[0]
        slack = slack + SE_MDC_RESIDUAL_SCALE * exchanged
        slack = slack + SE_MDC_RESIDUAL_SCALE * self.feed_forward(
            self.feed_forward_norm(slack)
        )
        risk_logits = self.risk_readout(slack).squeeze(-1)
        dual_allocation = torch.softmax(risk_logits, dim=1)
        # Total smooth budget is in (0, 2).  The exponential map keeps each
        # side's allocated interpolation weight strictly inside (0, 1).
        update_budget = 2.0 * torch.sigmoid(
            self.budget_readout(slack.mean(dim=1)).squeeze(-1)
        )
        convex_weights = 1.0 - torch.exp(
            -update_budget[:, None] * dual_allocation
        )
        residual = (slack - previous).square().mean(dim=-1).sqrt()
        return slack, risk_logits, dual_allocation, update_budget, convex_weights, residual


class SEMDCHead(nn.Module):
    """Four-loop CR-FPL head with symmetric minimax dual coupling."""

    def __init__(self, channels: int = 9, coupling_mode: str = "coupled") -> None:
        super().__init__()
        if channels < 1:
            raise ValueError("channels must be positive")
        if coupling_mode not in {"coupled", "independent"}:
            raise ValueError("coupling_mode must be coupled or independent")
        self.channels = channels
        self.coupling_mode = coupling_mode
        self.profile_tokenizer = nn.Sequential(
            nn.Linear(channels + 2, SE_MDC_WIDTH),
            nn.GELU(),
            nn.Linear(SE_MDC_WIDTH, SE_MDC_WIDTH),
        )
        self.anchor_tokenizer = nn.Sequential(
            nn.Linear(1, SE_MDC_WIDTH),
            nn.GELU(),
            nn.Linear(SE_MDC_WIDTH, SE_MDC_WIDTH),
        )
        self.feedback_tokenizer = nn.Sequential(
            nn.Linear(2 * channels + 1, SE_MDC_WIDTH),
            nn.GELU(),
            nn.Linear(SE_MDC_WIDTH, SE_MDC_WIDTH),
        )
        self.dual_token = nn.Parameter(torch.zeros(1, 1, SE_MDC_WIDTH))
        self.feedback_seed = nn.Parameter(torch.zeros(1, 1, SE_MDC_WIDTH))
        self.loop_block = _PreNormDynamicRecallBlock()
        self.dual_coupler = SwapEquivariantDualCoupler()
        self.readout_norm = nn.LayerNorm(SE_MDC_WIDTH)
        self.fixed_point_readout = nn.Linear(SE_MDC_WIDTH, 1)

        # Initialization is the frozen D2 anchor at every loop, as in CR-FPL.
        nn.init.zeros_(self.fixed_point_readout.weight)
        nn.init.zeros_(self.fixed_point_readout.bias)

    def _static_tokens(
        self, features: Tensor, coordinates: Tensor, anchor: Tensor
    ) -> tuple[Tensor, int]:
        batch, sides, radii, samples, _ = features.shape
        if radii == 1:
            radius_coordinate = torch.zeros(
                1, device=features.device, dtype=features.dtype
            )
        else:
            radius_coordinate = torch.linspace(
                -1.0, 1.0, radii, device=features.device, dtype=features.dtype
            )
        radius_coordinate = radius_coordinate.view(1, 1, radii, 1).expand(
            batch, sides, radii, samples
        )
        payload = torch.cat(
            (features, coordinates[..., None], radius_coordinate[..., None]), dim=-1
        )
        profile_tokens = self.profile_tokenizer(
            payload.reshape(batch, sides, radii * samples, -1)
        )
        positive_anchor = anchor.clamp_min(SE_MDC_EPSILON)
        anchor_token = self.anchor_tokenizer(torch.log(positive_anchor)[..., None, None])
        dual_token = self.dual_token.view(1, 1, 1, -1).expand(batch, sides, -1, -1)
        feedback_seed = self.feedback_seed.view(1, 1, 1, -1).expand(
            batch, sides, -1, -1
        )
        return (
            torch.cat((anchor_token, dual_token, feedback_seed, profile_tokens), dim=2),
            2,
        )

    def _validate(
        self, features: Tensor, coordinates: Tensor, anchor_distance: Tensor
    ) -> None:
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

    def forward(
        self, features: Tensor, coordinates: Tensor, anchor_distance: Tensor
    ) -> SEMDCOutput:
        self._validate(features, coordinates, anchor_distance)
        batch, sides, radii, _, channels = features.shape
        static_input, feedback_index = self._static_tokens(
            features, coordinates, anchor_distance
        )
        state = static_input
        positive_anchor = anchor_distance.clamp_min(SE_MDC_EPSILON)
        log_anchor = torch.log(positive_anchor)
        log_ratio = torch.zeros_like(anchor_distance)
        dual_penalty = torch.zeros_like(anchor_distance)
        previous_log_energy: Tensor | None = None
        previous_distance = positive_anchor
        flat_features = features.reshape(batch * sides, *features.shape[2:])
        flat_coordinates = coordinates.reshape(batch * sides, *coordinates.shape[2:])
        anchor_sample = differentiable_profile_query(
            flat_features, flat_coordinates, positive_anchor.reshape(-1)
        ).reshape(batch, sides, radii, channels)

        collections: list[list[Tensor]] = [[] for _ in range(13)]
        for _ in range(SE_MDC_LOOPS):
            query_distance = torch.exp(log_anchor + log_ratio)
            sampled = differentiable_profile_query(
                flat_features, flat_coordinates, query_distance.reshape(-1)
            ).reshape(batch, sides, radii, channels)
            query_residual = sampled - anchor_sample
            feedback_payload = torch.cat(
                (
                    sampled.mean(dim=2),
                    query_residual.mean(dim=2),
                    log_ratio[..., None],
                ),
                dim=-1,
            )
            feedback = self.feedback_tokenizer(feedback_payload)
            recalled_input = static_input.clone()
            recalled_input[:, :, feedback_index] = feedback

            previous_state = state
            token_count = state.shape[2]
            state = self.loop_block(
                state.reshape(batch * sides, token_count, SE_MDC_WIDTH),
                recalled_input.reshape(batch * sides, token_count, SE_MDC_WIDTH),
            ).reshape(batch, sides, token_count, SE_MDC_WIDTH)
            local_slack = state[:, :, 1]
            if self.coupling_mode == "coupled":
                (
                    coupled_slack,
                    risk_logits,
                    dual_allocation,
                    update_budget,
                    convex_weight,
                    coupled_slack_residual,
                ) = self.dual_coupler(local_slack)
            else:
                # Parameter-matched independent-side control: execute the same
                # coupler on two identical copies of each side, so no token can
                # carry evidence from the opposite endpoint.
                independent_input = local_slack.reshape(batch * sides, 1, -1).expand(
                    -1, 2, -1
                )
                (
                    independent_slack,
                    independent_risk,
                    _,
                    independent_budget,
                    independent_weight,
                    independent_residual,
                ) = self.dual_coupler(independent_input)
                coupled_slack = independent_slack[:, 0].reshape(batch, sides, -1)
                risk_logits = independent_risk[:, 0].reshape(batch, sides)
                # The true two-side diagnostic is uniform because the control
                # makes no relative comparison.  Update weights/budgets remain
                # learned independently with exactly the same parameters.
                dual_allocation = torch.full_like(risk_logits, 0.5)
                update_budget = independent_budget.reshape(batch, sides).mean(dim=1)
                convex_weight = independent_weight[:, 0].reshape(batch, sides)
                coupled_slack_residual = independent_residual[:, 0].reshape(
                    batch, sides
                )
            # Feed the exchanged slack back into the recurrent state so later
            # re-queries depend on the other anonymous endpoint as well.
            state = torch.cat(
                (state[:, :, :1], coupled_slack[:, :, None], state[:, :, 2:]), dim=2
            )
            primal = self.readout_norm(state[:, :, 0])
            log_energy = F.softplus(risk_logits)
            dual_penalty = dual_penalty_update(
                previous_log_energy, log_energy, dual_penalty
            )
            fixed_point = self.fixed_point_readout(primal).squeeze(-1) * torch.exp(
                -dual_penalty
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
                dual_allocation,
                update_budget[:, None].expand(-1, sides),
                fixed_point_residual,
                query_residual.square().mean(dim=(2, 3)).sqrt(),
                (distance - previous_distance).abs(),
                (state - previous_state).square().mean(dim=(2, 3)).sqrt(),
                coupled_slack_residual,
            )
            for collection, value in zip(collections, values):
                collection.append(value)
            previous_log_energy = log_energy
            previous_distance = distance

        stacked = [torch.stack(values, dim=-1) for values in collections]
        return SEMDCOutput(
            distance=stacked[0][..., -1],
            terminal_violation=1.0 - torch.exp(-stacked[2][..., -1]),
            loop_distances=stacked[0],
            loop_log_distance_ratios=stacked[1],
            loop_log_energy=stacked[2],
            loop_dual_penalty=stacked[3],
            loop_fixed_point_proposals=stacked[4],
            loop_convex_weights=stacked[5],
            loop_dual_allocation=stacked[6],
            loop_update_budget=stacked[7],
            loop_fixed_point_residuals=stacked[8],
            loop_query_residual_l2=stacked[9],
            loop_prediction_residuals=stacked[10],
            loop_state_residual_l2=stacked[11],
            loop_coupled_slack_residual_l2=stacked[12],
        )


def _object_side_mean(values: Tensor, object_ids: Iterable[str] | None) -> Tensor:
    """Aggregate ``[B,2,L]`` values to object-level side risks."""

    if object_ids is None:
        return values
    identifiers = tuple(object_ids)
    if len(identifiers) != values.shape[0]:
        raise ValueError("object_ids must align with the batch")
    groups = []
    for object_id in sorted(set(identifiers)):
        indices = [index for index, value in enumerate(identifiers) if value == object_id]
        groups.append(values[indices].mean(dim=0))
    return torch.stack(groups)


def fixed_point_loss_terms(
    output: Any,
    target_distance: Tensor,
    terminal_violation_target: Tensor | None = None,
    object_ids: Iterable[str] | None = None,
    minimax: bool = True,
) -> dict[str, Tensor]:
    """Return each unweighted deep-supervision/minimax objective term.

    ``logsumexp`` is the closed-form inner maximization of a two-side adversary
    with entropy regularization.  Matching the predicted dual allocation to
    that detached adversary makes the architectural slack meaningful while the
    primal model minimizes the resulting smooth worst-side risk.
    """

    if target_distance.shape != output.distance.shape:
        raise ValueError("target_distance must match [B,2] output")
    if not torch.isfinite(target_distance).all() or (target_distance <= 0).any():
        raise ValueError("target_distance must be finite and positive")
    target_log = torch.log(target_distance).unsqueeze(-1).expand_as(
        output.loop_distances
    )
    predicted_log = torch.log(output.loop_distances)
    log_error = (predicted_log - target_log).abs()
    anchor_log = predicted_log[..., :1] - output.loop_log_distance_ratios[..., :1]
    target_log_ratio = (target_log - anchor_log).expand_as(
        output.loop_fixed_point_proposals
    )

    terms = {
        "log_distance": F.smooth_l1_loss(predicted_log, target_log),
        "fixed_point_proposal": F.smooth_l1_loss(
            output.loop_fixed_point_proposals, target_log_ratio
        ),
        "log_energy": F.smooth_l1_loss(output.loop_log_energy, log_error.detach()),
    }
    residual_increase = F.relu(
        output.loop_fixed_point_residuals[..., 1:]
        - output.loop_fixed_point_residuals[..., :-1]
    )
    terms["residual_monotonicity"] = residual_increase.mean()

    if minimax:
        object_risk = _object_side_mean(log_error, object_ids)
        smooth_worst = torch.logsumexp(object_risk, dim=1) - math.log(2.0)
        terms["object_smooth_worst_side"] = smooth_worst.mean()
        if hasattr(output, "loop_dual_allocation"):
            adversarial_dual = torch.softmax(object_risk.detach(), dim=1)
            predicted_dual = _object_side_mean(
                output.loop_dual_allocation, object_ids
            )
            terms["dual_alignment"] = F.kl_div(
                predicted_dual.clamp_min(SE_MDC_EPSILON).log(),
                adversarial_dual,
                reduction="batchmean",
            )

    if terminal_violation_target is not None:
        if terminal_violation_target.shape != output.terminal_violation.shape:
            raise ValueError("terminal_violation_target must match [B,2] output")
        dual_target = terminal_violation_target.unsqueeze(-1).expand_as(
            output.loop_log_energy
        )
        violation = 1.0 - torch.exp(-output.loop_log_energy)
        terms["terminal_violation"] = F.binary_cross_entropy(violation, dual_target)
    return terms


def se_mdc_deep_supervision_loss(
    output: SEMDCOutput,
    target_distance: Tensor,
    terminal_violation_target: Tensor | None = None,
    object_ids: Iterable[str] | None = None,
) -> Tensor:
    """Sum the unweighted entropic minimax deep-supervision terms."""

    terms = fixed_point_loss_terms(
        output,
        target_distance,
        terminal_violation_target,
        object_ids,
        minimax=True,
    )
    return torch.stack(tuple(terms.values())).sum()


def exact_se_mdc_swap_error(
    model: SEMDCHead,
    features: Tensor,
    coordinates: Tensor,
    anchor_distance: Tensor,
) -> float:
    """Maximum output/diagnostic discrepancy after exchanging endpoint sides."""

    output = model(features, coordinates, anchor_distance)
    swapped = model(features.flip(1), coordinates.flip(1), anchor_distance.flip(1))
    return float(
        max(
            (getattr(swapped, field) - getattr(output, field).flip(1))
            .abs()
            .max()
            .item()
            for field in SEMDCOutput.__dataclass_fields__
        )
    )


__all__ = [
    "SE_MDC_EPSILON",
    "SE_MDC_HEADS",
    "SE_MDC_LOOPS",
    "SE_MDC_RESIDUAL_SCALE",
    "SE_MDC_WIDTH",
    "SEMDCHead",
    "SEMDCOutput",
    "SwapEquivariantDualCoupler",
    "exact_se_mdc_swap_error",
    "fixed_point_loss_terms",
    "se_mdc_deep_supervision_loss",
]
