"""Strict frozen visual encoder utilities for SMARC."""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from torch import Tensor, nn

from splart.smarc import swap_invariant_pair


DINO_VITB16_SHA256 = "bf34ad0f424b9029b593e8dc3ed553bf26e88bcba0d32bf3e62a6209cb64c85e"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_dino_vitb16(checkpoint: Path, device: torch.device) -> nn.Module:
    if sha256_file(checkpoint) != DINO_VITB16_SHA256:
        raise ValueError("official DINO ViT-B/16 checkpoint hash mismatch")
    import timm
    model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=0)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or len(state) != 150:
        raise ValueError("unexpected DINO state schema")
    model.load_state_dict(state, strict=True)
    model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def encode_state_pair(model: nn.Module, images: Tensor) -> Tensor:
    """Encode [state,view,3,224,224] and return a swap-invariant pair."""
    if images.ndim != 5 or images.shape[0] != 2 or images.shape[2:] != (3, 224, 224):
        raise ValueError("expected [2,V,3,224,224]")
    device = next(model.parameters()).device
    batch = images.float().to(device) / 255.0 if images.dtype == torch.uint8 else images.to(device)
    mean = torch.tensor([.485, .456, .406], device=device)[None, :, None, None]
    std = torch.tensor([.229, .224, .225], device=device)[None, :, None, None]
    states = []
    with torch.no_grad():
        for index in range(2):
            views = torch.nn.functional.normalize(model((batch[index]-mean)/std), dim=-1)
            states.append(torch.nn.functional.normalize(views.mean(0), dim=0))
    result = swap_invariant_pair(states[0], states[1])
    if not torch.isfinite(result).all():
        raise RuntimeError("nonfinite DINO pair")
    return result.cpu()


__all__ = ["DINO_VITB16_SHA256", "encode_state_pair", "load_dino_vitb16", "sha256_file"]
