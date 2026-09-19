import pytest
import torch

from splart.clip_limit import (
    CCLIPLDHead,
    CCLIPLDOutput,
    FrozenCLIPLimitDiscriminator,
    load_frozen_open_clip_vit_b32,
    c_clip_ld_deep_supervision_loss,
    exact_c_clip_ld_swap_error,
)
from splart.cr_fpl import CRFPLHead, CRFPLOutput


class TinyCLIP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def encode_text(self, tokens):
        return tokens.float()

    def encode_image(self, images):
        mean = images.mean(dim=(-2, -1))
        return torch.stack((mean[:, 0], mean[:, 2]), dim=-1)


def tokenizer(prompts):
    rows = len(prompts)
    split = rows // 2
    result = torch.zeros(rows, 2)
    result[:split, 0] = 1.0
    result[split:, 1] = 1.0
    return result


def profiles(batch=3):
    generator = torch.Generator().manual_seed(818)
    features = torch.randn(batch, 2, 3, 11, 9, generator=generator)
    coordinates = torch.linspace(0.05, 2.05, 11).view(1, 1, 1, 11).expand(batch, 2, 3, -1).clone()
    anchor = torch.rand(batch, 2, generator=generator) + 0.2
    semantic_coordinates = torch.linspace(0.0, 2.0, 9).view(1, 1, 9).expand(batch, 2, -1).clone()
    semantic_evidence = semantic_coordinates[..., None] - 0.75
    return features, coordinates, anchor, semantic_evidence, semantic_coordinates


def test_frozen_clip_prompt_and_view_ensembles_have_exact_prompt_swap():
    encoder = FrozenCLIPLimitDiscriminator(TinyCLIP(), tokenizer)
    assert not any(parameter.requires_grad for parameter in encoder.parameters())
    images = torch.zeros(2, 5, 3, 32, 32, dtype=torch.uint8)
    images[0, :, 0] = 255
    images[1, :, 2] = 255
    margin = encoder(images)
    swapped = encoder(images, prompt_swap=True)
    assert margin.shape == (2,)
    assert margin[0] > margin[1]
    assert torch.equal(swapped, -margin)


def test_zero_semantic_exactly_recovers_same_weight_cr_fpl():
    torch.manual_seed(818)
    geometry = CRFPLHead()
    model = CCLIPLDHead()
    incompatible = model.load_state_dict(geometry.state_dict(), strict=False)
    assert incompatible.unexpected_keys == []
    assert set(incompatible.missing_keys) == {
        "shared_side.semantic_residual.value.weight",
        "shared_side.semantic_residual.gate.weight",
    }
    features, coordinates, anchor, evidence, semantic_coordinates = profiles()
    baseline = geometry(features, coordinates, anchor)
    candidate = model(features, coordinates, anchor, torch.zeros_like(evidence), semantic_coordinates)
    for field in CRFPLOutput.__dataclass_fields__:
        assert torch.equal(getattr(candidate, field), getattr(baseline, field))
    assert torch.equal(candidate.loop_semantic_residual_l2, torch.zeros_like(candidate.loop_semantic_residual_l2))


def test_semantic_trajectory_is_requeried_each_loop_and_swap_equivariant():
    model = CCLIPLDHead()
    with torch.no_grad():
        model.shared_side.fixed_point_readout.bias.fill_(0.8)
    inputs = profiles()
    output = model(*inputs)
    assert output.loop_semantic_evidence.shape == (3, 2, 1, 4)
    assert torch.all(output.loop_semantic_evidence[..., 1:] != output.loop_semantic_evidence[..., :1])
    assert exact_c_clip_ld_swap_error(model.eval(), *inputs) == 0.0


def test_c_clip_ld_deep_supervision_reaches_bias_free_semantic_branch():
    torch.manual_seed(818)
    model = CCLIPLDHead()
    output = model(*profiles())
    assert isinstance(output, CCLIPLDOutput)
    target = torch.full_like(output.distance, 1.2)
    loss = c_clip_ld_deep_supervision_loss(output, target)
    loss.backward()
    for parameter in (
        model.shared_side.semantic_residual.value.weight,
        model.shared_side.semantic_residual.gate.weight,
    ):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()


def test_loader_rejects_unpinned_checkpoint_before_deserialization(tmp_path):
    checkpoint = tmp_path / "arbitrary.pt"
    checkpoint.write_bytes(b"not a trusted checkpoint")
    with pytest.raises(ValueError, match="pinned official"):
        load_frozen_open_clip_vit_b32(checkpoint, "0" * 64, torch.device("cpu"))
