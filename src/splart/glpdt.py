"""Gauge-equivariant looped primal-dual Transformer for endpoint refinement.

The module is deliberately dataset agnostic.  It consumes normalized, ordered
multi-radius D2 profiles and a frozen D2 distance anchor.  Both sides are run
through the *same* side function, so observation-order equivariance is a
structural property rather than a learned augmentation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


GLPDT_LOOPS = 4
GLPDT_WIDTH = 64
GLPDT_HEADS = 4
GLPDT_RESIDUAL_SCALE = 0.25
GLPDT_CORRECTION_FRACTION = 0.5
GLPDT_EPSILON = 1e-6


@dataclass(frozen=True)
class GLPDTOutput:
    """Final predictions and the complete deep-supervision trajectory.

    All loop tensors are shaped ``[B, 2, 4]``.  The side dimension is kept
    explicit so the same swap action applies to final and diagnostic outputs.
    """

    distance: Tensor
    terminal_violation: Tensor
    loop_distances: Tensor
    loop_terminal_violations: Tensor
    loop_prediction_residuals: Tensor
    loop_state_residual_l2: Tensor


class _PreNormRecalledBlock(nn.Module):
    """One tied pre-norm Transformer step with fixed input recall."""

    def __init__(self) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(GLPDT_WIDTH)
        self.attention = nn.MultiheadAttention(
            GLPDT_WIDTH, GLPDT_HEADS, dropout=0.0, batch_first=True
        )
        self.feed_forward_norm = nn.LayerNorm(GLPDT_WIDTH)
        self.feed_forward = nn.Sequential(
            nn.Linear(GLPDT_WIDTH, 4 * GLPDT_WIDTH),
            nn.GELU(),
            nn.Linear(4 * GLPDT_WIDTH, GLPDT_WIDTH),
        )

    def forward(self, state: Tensor, original_input: Tensor) -> Tensor:
        # Recall is applied on every recurrence, including the two state tokens.
        state = state + GLPDT_RESIDUAL_SCALE * original_input
        normalized = self.attention_norm(state)
        attended = self.attention(normalized, normalized, normalized, need_weights=False)[0]
        state = state + GLPDT_RESIDUAL_SCALE * attended
        state = state + GLPDT_RESIDUAL_SCALE * self.feed_forward(self.feed_forward_norm(state))
        return state


class _SharedSideGLPDT(nn.Module):
    """The single side function shared exactly by lower and upper profiles."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        # profile channels + outward coordinate + normalized radius index
        self.profile_tokenizer = nn.Sequential(
            nn.Linear(channels + 2, GLPDT_WIDTH),
            nn.GELU(),
            nn.Linear(GLPDT_WIDTH, GLPDT_WIDTH),
        )
        self.anchor_tokenizer = nn.Sequential(
            nn.Linear(1, GLPDT_WIDTH), nn.GELU(), nn.Linear(GLPDT_WIDTH, GLPDT_WIDTH)
        )
        self.dual_token = nn.Parameter(torch.zeros(1, 1, GLPDT_WIDTH))
        # This is the only recurrent block.  Calling it four times ties weights.
        self.loop_block = _PreNormRecalledBlock()
        self.readout_norm = nn.LayerNorm(GLPDT_WIDTH)
        self.correction_readout = nn.Linear(GLPDT_WIDTH, 1)
        self.violation_readout = nn.Linear(GLPDT_WIDTH, 1)

    @staticmethod
    def _bounded_distance(anchor: Tensor, raw_correction: Tensor) -> Tensor:
        """Apply a signed bounded correction while preserving positivity.

        Negative movement is at most half the anchor.  Positive movement is at
        most half ``anchor + 1`` (the observed interval is normalized to one).
        This asymmetry avoids an arbitrary global scale and cannot cross zero.
        """

        signed = torch.tanh(raw_correction)
        negative_bound = GLPDT_CORRECTION_FRACTION * anchor
        positive_bound = GLPDT_CORRECTION_FRACTION * (anchor + 1.0)
        correction = torch.where(signed < 0.0, signed * negative_bound, signed * positive_bound)
        return (anchor + correction).clamp_min(GLPDT_EPSILON)

    def _original_tokens(self, features: Tensor, coordinates: Tensor, anchor: Tensor) -> Tensor:
        batch, radii, samples, _ = features.shape
        if radii == 1:
            radius_coordinate = torch.zeros(1, device=features.device, dtype=features.dtype)
        else:
            radius_coordinate = torch.linspace(-1.0, 1.0, radii, device=features.device, dtype=features.dtype)
        radius_coordinate = radius_coordinate.view(1, radii, 1).expand(batch, radii, samples)
        payload = torch.cat((features, coordinates[..., None], radius_coordinate[..., None]), dim=-1)
        profile_tokens = self.profile_tokenizer(payload.reshape(batch, radii * samples, -1))
        anchor_token = self.anchor_tokenizer(torch.log1p(anchor).view(batch, 1, 1))
        dual_token = self.dual_token.expand(batch, -1, -1)
        return torch.cat((anchor_token, dual_token, profile_tokens), dim=1)

    def forward(self, features: Tensor, coordinates: Tensor, anchor: Tensor) -> tuple[Tensor, ...]:
        original = self._original_tokens(features, coordinates, anchor)
        state = original
        previous_distance = anchor
        distances, violations, prediction_residuals, state_residuals = [], [], [], []
        for _ in range(GLPDT_LOOPS):
            previous_state = state
            state = self.loop_block(state, original)
            primal = self.readout_norm(state[:, 0])
            dual = self.readout_norm(state[:, 1])
            distance = self._bounded_distance(anchor, self.correction_readout(primal).squeeze(-1))
            violation = torch.sigmoid(self.violation_readout(dual).squeeze(-1))
            distances.append(distance)
            violations.append(violation)
            prediction_residuals.append((distance - previous_distance).abs())
            state_residuals.append((state - previous_state).square().mean(dim=(1, 2)).sqrt())
            previous_distance = distance
        return (
            torch.stack(distances, dim=-1),
            torch.stack(violations, dim=-1),
            torch.stack(prediction_residuals, dim=-1),
            torch.stack(state_residuals, dim=-1),
        )


class GLPDTHead(nn.Module):
    """Four-loop gauge-equivariant D2-anchor refinement head.

    ``features`` and ``coordinates`` are expected to have already received the
    same train-split normalization as Node22.  ``anchor_distance`` is the
    non-negative frozen D2 distance in the normalized endpoint gauge.
    """

    def __init__(self, channels: int = 9) -> None:
        super().__init__()
        if channels < 1:
            raise ValueError("channels must be positive")
        self.channels = channels
        self.shared_side = _SharedSideGLPDT(channels)

    def forward(self, features: Tensor, coordinates: Tensor, anchor_distance: Tensor) -> GLPDTOutput:
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

        # Calling one module per side, rather than learning side embeddings or
        # mixing the sides, makes swapping an exact permutation of outputs.
        sides = [
            self.shared_side(features[:, side], coordinates[:, side], anchor_distance[:, side])
            for side in range(2)
        ]
        loop_distances = torch.stack((sides[0][0], sides[1][0]), dim=1)
        loop_violations = torch.stack((sides[0][1], sides[1][1]), dim=1)
        loop_prediction_residuals = torch.stack((sides[0][2], sides[1][2]), dim=1)
        loop_state_residuals = torch.stack((sides[0][3], sides[1][3]), dim=1)
        return GLPDTOutput(
            distance=loop_distances[..., -1],
            terminal_violation=loop_violations[..., -1],
            loop_distances=loop_distances,
            loop_terminal_violations=loop_violations,
            loop_prediction_residuals=loop_prediction_residuals,
            loop_state_residual_l2=loop_state_residuals,
        )


def glpdt_deep_supervision_loss(
    output: GLPDTOutput, target_distance: Tensor, terminal_violation_target: Tensor | None = None
) -> Tensor:
    """Equal-weight supervision at all four loops, with optional dual targets."""

    if target_distance.shape != output.distance.shape:
        raise ValueError("target_distance must match [B,2] output")
    target = target_distance.unsqueeze(-1).expand_as(output.loop_distances)
    loss = F.smooth_l1_loss(output.loop_distances, target)
    if terminal_violation_target is not None:
        if terminal_violation_target.shape != output.terminal_violation.shape:
            raise ValueError("terminal_violation_target must match [B,2] output")
        dual_target = terminal_violation_target.unsqueeze(-1).expand_as(output.loop_terminal_violations)
        loss = loss + F.binary_cross_entropy(output.loop_terminal_violations, dual_target)
    return loss


def exact_glpdt_swap_error(
    model: GLPDTHead, features: Tensor, coordinates: Tensor, anchor_distance: Tensor
) -> float:
    """Return the maximum swap discrepancy over predictions and diagnostics."""

    output = model(features, coordinates, anchor_distance)
    swapped = model(features.flip(1), coordinates.flip(1), anchor_distance.flip(1))
    pairs = (
        (output.distance, swapped.distance),
        (output.terminal_violation, swapped.terminal_violation),
        (output.loop_distances, swapped.loop_distances),
        (output.loop_terminal_violations, swapped.loop_terminal_violations),
        (output.loop_prediction_residuals, swapped.loop_prediction_residuals),
        (output.loop_state_residual_l2, swapped.loop_state_residual_l2),
    )
    return float(max((right - left.flip(1)).abs().max().item() for left, right in pairs))


__all__ = [
    "GLPDT_CORRECTION_FRACTION",
    "GLPDT_EPSILON",
    "GLPDT_HEADS",
    "GLPDT_LOOPS",
    "GLPDT_RESIDUAL_SCALE",
    "GLPDT_WIDTH",
    "GLPDTHead",
    "GLPDTOutput",
    "exact_glpdt_swap_error",
    "glpdt_deep_supervision_loss",
]
