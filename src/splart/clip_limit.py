"""Counterfactual semantic tangent residuals for the CR-FPL LOOP transformer.

The frozen visual-language adapter turns multi-view renders along each
outward trajectory into a signed ``mechanical stop - free to move`` margin.
The trainable head then re-queries that trajectory at every tied CR-FPL loop.
No object/category embedding is exposed to the endpoint predictor.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Callable, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from splart.ald_pdl import dual_penalty_update
from splart.cr_fpl import (
    CR_FPL_EPSILON,
    CR_FPL_LOOPS,
    CR_FPL_WIDTH,
    CRFPLOutput,
    _SharedSideCRFPL,
    cr_fpl_deep_supervision_loss,
    differentiable_profile_query,
)


STOP_PROMPTS = (
    "a movable object exactly at its mechanical stop",
    "an articulated part fully closed against its hard limit",
    "an articulated joint at the end of its allowed motion",
    "a moving part in contact with its physical motion boundary",
)
FREE_PROMPTS = (
    "a movable object with room to continue moving",
    "an articulated part away from its mechanical stop",
    "an articulated joint in the interior of its allowed motion",
    "a moving part free from its physical motion boundary",
)
CLIP_IMAGE_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_IMAGE_STD = (0.26862954, 0.26130258, 0.27577711)
OPENAI_VIT_B32_SHA256 = "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af"
OPENAI_VIT_B32_BYTES = 353_976_522
OPEN_CLIP_TORCH_VERSION = "3.3.0"
CSTR_TANGENT_SAMPLES = 17
CSTR_ENERGY_SAMPLES = 9
CSTR_CHANNELS = 3


def counterfactual_semantic_tangent_residual(evidence: Tensor, coordinates: Tensor) -> Tensor:
    """Convert a signed CLIP trajectory to the fixed sign-invariant CSTR basis.

    The affine tangent is fit independently for every object/side on the
    preregistered first 17 samples (outward distance <= 0.25 on the fixed
    cache grid).  The three returned channels are point residual magnitude,
    tangent-slope deviation, and a centered nine-sample residual RMS.  Thus
    every affine trajectory maps to zero, while prompt sign reversal leaves
    the representation unchanged.
    """

    if evidence.ndim != 4 or evidence.shape[-1] != 1:
        raise ValueError("raw CLIP evidence must be [B,2,S,1]")
    if coordinates.shape != evidence.shape[:-1]:
        raise ValueError("CSTR coordinates must be [B,2,S]")
    if evidence.shape[-2] < CSTR_TANGENT_SAMPLES:
        raise ValueError(f"CSTR requires at least {CSTR_TANGENT_SAMPLES} trajectory samples")
    if not evidence.is_floating_point() or not coordinates.is_floating_point():
        raise ValueError("CSTR inputs must be floating point")
    if not torch.isfinite(evidence).all() or not torch.isfinite(coordinates).all():
        raise ValueError("CSTR inputs must be finite")
    delta = coordinates[..., 1:] - coordinates[..., :-1]
    if not torch.all(delta > 0):
        raise ValueError("CSTR coordinates must be strictly increasing")

    values = evidence[..., 0]
    tangent_x = coordinates[..., :CSTR_TANGENT_SAMPLES]
    tangent_y = values[..., :CSTR_TANGENT_SAMPLES]
    mean_x = tangent_x.mean(dim=-1, keepdim=True)
    mean_y = tangent_y.mean(dim=-1, keepdim=True)
    centered_x = tangent_x - mean_x
    centered_y = tangent_y - mean_y
    denominator = centered_x.square().sum(dim=-1, keepdim=True)
    if torch.any(denominator <= 0):
        raise ValueError("CSTR tangent coordinates are degenerate")
    tangent_slope = (centered_x * centered_y).sum(dim=-1, keepdim=True) / denominator
    tangent = mean_y + tangent_slope * (coordinates - mean_x)
    signed_residual = values - tangent
    # Float32 samples of an analytically affine trajectory can differ from
    # their refit by a few ulps; remove only that representation noise before
    # taking finite differences, whose division by grid spacing amplifies it.
    tolerance = 32.0 * torch.finfo(values.dtype).eps * values.abs().amax(dim=-1, keepdim=True).clamp_min(1.0)
    signed_residual = torch.where(
        signed_residual.abs() <= tolerance, torch.zeros_like(signed_residual), signed_residual
    )

    residual_magnitude = signed_residual.abs()
    segment_slope = (signed_residual[..., 1:] - signed_residual[..., :-1]) / delta
    slope_deviation = torch.empty_like(signed_residual)
    slope_deviation[..., 0] = segment_slope[..., 0].abs()
    slope_deviation[..., -1] = segment_slope[..., -1].abs()
    slope_deviation[..., 1:-1] = (0.5 * (segment_slope[..., :-1] + segment_slope[..., 1:])).abs()

    leading = signed_residual.shape[:-1]
    squared = signed_residual.square().reshape(-1, 1, signed_residual.shape[-1])
    residual_energy = (
        F.avg_pool1d(
            squared,
            kernel_size=CSTR_ENERGY_SAMPLES,
            stride=1,
            padding=CSTR_ENERGY_SAMPLES // 2,
            count_include_pad=False,
        )
        .sqrt()
        .reshape(*leading, signed_residual.shape[-1])
    )
    result = torch.stack((residual_magnitude, slope_deviation, residual_energy), dim=-1)
    if not torch.isfinite(result).all():
        raise RuntimeError("nonfinite CSTR representation")
    return result


def prompt_ensemble_sha256(
    stop_prompts: Sequence[str] = STOP_PROMPTS, free_prompts: Sequence[str] = FREE_PROMPTS
) -> str:
    """Return the canonical binding for the two ordered prompt ensembles."""

    payload = json.dumps(
        {"free": list(free_prompts), "stop": list(stop_prompts)}, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class FrozenCLIPLimitDiscriminator(nn.Module):
    """Frozen CLIP stop/free discriminator with prompt and view ensembles.

    ``clip_model`` only needs ``encode_image`` and ``encode_text`` methods,
    which keeps unit tests independent of OpenCLIP and network access.  The
    production loader below accepts an explicit local checkpoint and hash.
    """

    def __init__(
        self,
        clip_model: nn.Module,
        tokenizer: Callable[[Sequence[str]], Tensor],
        stop_prompts: Sequence[str] = STOP_PROMPTS,
        free_prompts: Sequence[str] = FREE_PROMPTS,
    ) -> None:
        super().__init__()
        if not stop_prompts or not free_prompts:
            raise ValueError("both CLIP prompt ensembles must be non-empty")
        self.clip_model = clip_model.eval()
        for parameter in self.clip_model.parameters():
            parameter.requires_grad_(False)
        try:
            device = next(self.clip_model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
        tokens = tokenizer(tuple(stop_prompts) + tuple(free_prompts)).to(device)
        with torch.no_grad():
            text = F.normalize(self.clip_model.encode_text(tokens).float(), dim=-1)
        if text.ndim != 2 or text.shape[0] != len(stop_prompts) + len(free_prompts):
            raise ValueError("CLIP text encoder returned an unexpected shape")
        stop = F.normalize(text[: len(stop_prompts)].mean(dim=0), dim=0)
        free = F.normalize(text[len(stop_prompts) :].mean(dim=0), dim=0)
        self.register_buffer("stop_text", stop)
        self.register_buffer("free_text", free)
        self.prompt_sha256 = prompt_ensemble_sha256(stop_prompts, free_prompts)

    @staticmethod
    def preprocess(images: Tensor) -> Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("CLIP images must be [N,3,H,W]")
        if images.dtype == torch.uint8:
            images = images.float().div(255.0)
        elif not images.is_floating_point():
            raise ValueError("CLIP images must be uint8 or floating point")
        images = F.interpolate(images, size=(224, 224), mode="bicubic", align_corners=False)
        mean = images.new_tensor(CLIP_IMAGE_MEAN).view(1, 3, 1, 1)
        std = images.new_tensor(CLIP_IMAGE_STD).view(1, 3, 1, 1)
        return (images - mean) / std

    def forward(self, images: Tensor, *, prompt_swap: bool = False) -> Tensor:
        """Score ``[...,V,3,H,W]`` renders and average the view margins."""

        if images.ndim < 5 or images.shape[-3] != 3 or images.shape[-4] < 1:
            raise ValueError("renders must be [...,V,3,H,W] with at least one view")
        leading = images.shape[:-4]
        views = images.shape[-4]
        flat = images.reshape(-1, *images.shape[-3:]).to(self.stop_text.device)
        with torch.no_grad():
            encoded = F.normalize(self.clip_model.encode_image(self.preprocess(flat)).float(), dim=-1)
            stop, free = (self.free_text, self.stop_text) if prompt_swap else (self.stop_text, self.free_text)
            margin = encoded @ (stop - free)
        result = margin.reshape(*leading, views).mean(dim=-1)
        if not torch.isfinite(result).all():
            raise RuntimeError("nonfinite CLIP limit evidence")
        return result


def load_frozen_open_clip_vit_b32(
    checkpoint: Path, checkpoint_sha256: str, device: torch.device
) -> FrozenCLIPLimitDiscriminator:
    """Load OpenCLIP ViT-B/32 strictly from a hash-bound local checkpoint."""

    if checkpoint.is_symlink():
        raise ValueError("CLIP checkpoint must be a regular local file")
    checkpoint = checkpoint.resolve()
    if not checkpoint.is_file():
        raise ValueError("CLIP checkpoint must be a regular local file")
    if checkpoint_sha256 != OPENAI_VIT_B32_SHA256:
        raise ValueError("only the pinned official OpenAI ViT-B/32 checkpoint is accepted")
    if checkpoint.stat().st_size != OPENAI_VIT_B32_BYTES or _sha256_file(checkpoint) != checkpoint_sha256:
        raise ValueError("CLIP checkpoint digest mismatch")
    try:
        import open_clip
    except ImportError as exc:  # pragma: no cover - exercised in deployment
        raise RuntimeError("open_clip_torch is required to build a semantic cache") from exc
    # weights_only=False is limited to the exact official TorchScript archive
    # verified above; arbitrary checkpoints never reach deserialization.
    model = (
        open_clip.create_model("ViT-B-32", pretrained=str(checkpoint), force_quick_gelu=True, weights_only=False)
        .eval()
        .to(device)
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    return FrozenCLIPLimitDiscriminator(model, tokenizer).to(device).eval()


class BiasFreeSemanticResidual(nn.Module):
    """A zero-preserving gated residual; both learned maps have no bias."""

    def __init__(self, channels: int, width: int = CR_FPL_WIDTH) -> None:
        super().__init__()
        if channels < 1:
            raise ValueError("semantic channels must be positive")
        self.value = nn.Linear(channels, width, bias=False)
        self.gate = nn.Linear(channels, width, bias=False)

    def forward(self, evidence: Tensor) -> Tensor:
        return torch.tanh(self.value(evidence)) * torch.sigmoid(self.gate(evidence.abs()))


@dataclass(frozen=True)
class CCLIPLDOutput(CRFPLOutput):
    """CR-FPL trajectory augmented with queried semantic diagnostics."""

    loop_semantic_evidence: Tensor
    loop_semantic_residual_l2: Tensor


class _SharedSideCCLIPLD(_SharedSideCRFPL):
    def __init__(self, channels: int, semantic_channels: int) -> None:
        super().__init__(channels)
        self.semantic_residual = BiasFreeSemanticResidual(semantic_channels)

    def forward(
        self,
        features: Tensor,
        coordinates: Tensor,
        anchor: Tensor,
        semantic_evidence: Tensor,
        semantic_coordinates: Tensor,
    ) -> tuple[Tensor, ...]:
        static_input, feedback_index = self._static_tokens(features, coordinates, anchor)
        state = static_input
        positive_anchor = anchor.clamp_min(CR_FPL_EPSILON)
        log_anchor = torch.log(positive_anchor)
        log_ratio = torch.zeros_like(anchor)
        dual_penalty = torch.zeros_like(anchor)
        previous_log_energy: Tensor | None = None
        previous_distance = positive_anchor
        anchor_sample = differentiable_profile_query(features, coordinates, positive_anchor)

        collections: list[list[Tensor]] = [[] for _ in range(12)]
        for _ in range(CR_FPL_LOOPS):
            query_distance = torch.exp(log_anchor + log_ratio)
            sampled = differentiable_profile_query(features, coordinates, query_distance)
            query_residual = sampled - anchor_sample
            feedback_payload = torch.cat((sampled.mean(dim=1), query_residual.mean(dim=1), log_ratio[:, None]), dim=-1)
            feedback = self.feedback_tokenizer(feedback_payload)
            queried_semantic = differentiable_profile_query(
                semantic_evidence[:, None], semantic_coordinates[:, None], query_distance
            ).squeeze(1)
            semantic_residual = self.semantic_residual(queried_semantic)
            recalled_input = static_input.clone()
            recalled_input[:, feedback_index] = feedback + semantic_residual

            previous_state = state
            state = self.loop_block(state, recalled_input)
            primal = self.readout_norm(state[:, 0])
            dual = self.readout_norm(state[:, 1])
            log_energy = F.softplus(self.log_energy_readout(dual).squeeze(-1))
            dual_penalty = dual_penalty_update(previous_log_energy, log_energy, dual_penalty)
            fixed_point = self.fixed_point_readout(primal).squeeze(-1) * torch.exp(-dual_penalty)
            convex_weight = torch.sigmoid(self.convex_weight_readout(primal).squeeze(-1))
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
                queried_semantic,
                semantic_residual.square().mean(dim=-1).sqrt(),
            )
            for collection, value in zip(collections, values):
                collection.append(value)
            previous_log_energy = log_energy
            previous_distance = distance
        return tuple(torch.stack(values, dim=-1) for values in collections)


class CCLIPLDHead(nn.Module):
    """Four-loop CR-FPL head with position-conditioned CLIP limit evidence."""

    def __init__(self, channels: int = 9, semantic_channels: int = 1) -> None:
        super().__init__()
        if channels < 1 or semantic_channels < 1:
            raise ValueError("geometry and semantic channels must be positive")
        self.channels = channels
        self.semantic_channels = semantic_channels
        self.shared_side = _SharedSideCCLIPLD(channels, semantic_channels)

    def forward(
        self,
        features: Tensor,
        coordinates: Tensor,
        anchor_distance: Tensor,
        semantic_evidence: Tensor,
        semantic_coordinates: Tensor,
    ) -> CCLIPLDOutput:
        if features.ndim != 5 or features.shape[1] != 2 or features.shape[-1] != self.channels:
            raise ValueError("features must be [B,2,R,S,C]")
        if coordinates.shape != features.shape[:-1]:
            raise ValueError("coordinates must be [B,2,R,S]")
        if anchor_distance.shape != features.shape[:2]:
            raise ValueError("anchor_distance must be [B,2]")
        if (
            semantic_evidence.ndim != 4
            or semantic_evidence.shape[:2] != features.shape[:2]
            or semantic_evidence.shape[-1] != self.semantic_channels
        ):
            raise ValueError("semantic_evidence must be [B,2,S,K]")
        if semantic_coordinates.shape != semantic_evidence.shape[:-1]:
            raise ValueError("semantic_coordinates must be [B,2,S]")
        tensors = (features, coordinates, anchor_distance, semantic_evidence, semantic_coordinates)
        if not all(torch.isfinite(value).all() for value in tensors):
            raise ValueError("all C-CLIP-LD inputs must be finite")
        if (coordinates < 0).any() or (semantic_coordinates < 0).any() or (anchor_distance < 0).any():
            raise ValueError("distances and coordinates must be non-negative")

        sides = [
            self.shared_side(
                features[:, side],
                coordinates[:, side],
                anchor_distance[:, side],
                semantic_evidence[:, side],
                semantic_coordinates[:, side],
            )
            for side in range(2)
        ]
        stacked = [torch.stack((sides[0][index], sides[1][index]), dim=1) for index in range(12)]
        return CCLIPLDOutput(
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
            loop_semantic_evidence=stacked[10],
            loop_semantic_residual_l2=stacked[11],
        )


class CSTRHead(CCLIPLDHead):
    """CR-FPL with the preregistered three-channel CSTR representation."""

    def __init__(self, channels: int = 9) -> None:
        super().__init__(channels=channels, semantic_channels=CSTR_CHANNELS)


def c_clip_ld_deep_supervision_loss(
    output: CCLIPLDOutput, target_distance: Tensor, terminal_violation_target: Tensor | None = None
) -> Tensor:
    return cr_fpl_deep_supervision_loss(output, target_distance, terminal_violation_target)


def exact_c_clip_ld_swap_error(
    model: CCLIPLDHead,
    features: Tensor,
    coordinates: Tensor,
    anchor_distance: Tensor,
    semantic_evidence: Tensor,
    semantic_coordinates: Tensor,
) -> float:
    output = model(features, coordinates, anchor_distance, semantic_evidence, semantic_coordinates)
    swapped = model(
        features.flip(1),
        coordinates.flip(1),
        anchor_distance.flip(1),
        semantic_evidence.flip(1),
        semantic_coordinates.flip(1),
    )
    return float(
        max(
            (getattr(swapped, field) - getattr(output, field).flip(1)).abs().max().item()
            for field in CCLIPLDOutput.__dataclass_fields__
        )
    )


def exact_cstr_swap_error(
    model: CSTRHead,
    features: Tensor,
    coordinates: Tensor,
    anchor_distance: Tensor,
    semantic_evidence: Tensor,
    semantic_coordinates: Tensor,
) -> float:
    return exact_c_clip_ld_swap_error(
        model, features, coordinates, anchor_distance, semantic_evidence, semantic_coordinates
    )


__all__ = [
    "BiasFreeSemanticResidual",
    "CCLIPLDHead",
    "CCLIPLDOutput",
    "CSTR_CHANNELS",
    "CSTR_ENERGY_SAMPLES",
    "CSTR_TANGENT_SAMPLES",
    "CSTRHead",
    "CLIP_IMAGE_MEAN",
    "CLIP_IMAGE_STD",
    "FREE_PROMPTS",
    "FrozenCLIPLimitDiscriminator",
    "OPENAI_VIT_B32_BYTES",
    "OPENAI_VIT_B32_SHA256",
    "OPEN_CLIP_TORCH_VERSION",
    "STOP_PROMPTS",
    "c_clip_ld_deep_supervision_loss",
    "counterfactual_semantic_tangent_residual",
    "exact_c_clip_ld_swap_error",
    "exact_cstr_swap_error",
    "load_frozen_open_clip_vit_b32",
    "prompt_ensemble_sha256",
]
