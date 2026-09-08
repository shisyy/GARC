import torch

from splart.smarc import SMARC, opaque_object_group, opaque_row_key, project_extensions_to_range, swap_invariant_pair
from splart.frozen_visual import encode_state_pair


def test_node4_join_key_is_exact_and_group_is_opaque():
    archive = "rec_example_0001.tar.gz"
    key = opaque_row_key(archive, 2, .25, .75, 1, "forward")
    assert key == opaque_row_key(archive, 2, .25, .75, 1, "forward")
    assert len(key) == len(opaque_object_group(archive)) == 64
    assert archive not in opaque_object_group(archive)


def test_semantic_gate_has_no_bias_and_zero_image_equals_uniform_mechanical():
    model = SMARC(5, 7, global_prior_logit=-1.0)
    assert model.semantic_gate.bias is None
    mechanical = torch.tensor([[.2, .1, .4, .3, .5]])
    semantic = torch.zeros(1, 7)
    prediction, details = model(mechanical, semantic, torch.tensor([.5]))
    expected_residual = details["expert_residual"].mean(-1)
    expected = .5 + (2 * torch.pi - .5) * torch.sigmoid(torch.tensor(-1.) + expected_residual)
    assert torch.equal(prediction, expected)
    assert torch.equal(details["gate"], torch.full((1, 4), .25))


def test_swap_invariant_pair_and_exact_range_projection():
    a = torch.tensor([[1., 2., 3.]])
    b = torch.tensor([[3., 0., 4.]])
    assert torch.equal(swap_invariant_pair(a, b), swap_invariant_pair(b, a))
    base = torch.tensor([[.2, .8], [.9, .1]], dtype=torch.float64)
    ranges = torch.tensor([2., 1.5], dtype=torch.float64)
    displacement = torch.ones(2, dtype=torch.float64)
    projected = project_extensions_to_range(base, ranges, displacement)
    assert torch.all(projected >= 0)
    assert torch.allclose(projected.sum(-1), ranges / displacement - 1, atol=1e-12, rtol=0)
    swapped = project_extensions_to_range(base.flip(-1), ranges, displacement)
    assert torch.equal(swapped, projected.flip(-1))


def test_frozen_pair_encoder_is_state_swap_invariant():
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_parameter("anchor", torch.nn.Parameter(torch.zeros(()), requires_grad=False))
        def forward(self, x):
            return x.mean(dim=(-2, -1))
    images = torch.arange(2*2*3*224*224, dtype=torch.int64).remainder(256).to(torch.uint8).reshape(2,2,3,224,224)
    encoder = Tiny()
    assert torch.equal(encode_state_pair(encoder, images), encode_state_pair(encoder, images.flip(0)))
