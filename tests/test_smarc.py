import torch

from splart.smarc import SMARC, opaque_object_group, opaque_row_key, project_extensions_to_range, swap_invariant_pair
from splart.frozen_visual import encode_state_pair
from scripts.build_smarc_articraft_shard import obj_mesh, parse_formal


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


def test_formal_parser_triangulates_faces_and_assigns_full_subtree(tmp_path):
    mesh = tmp_path / "quad.obj"
    mesh.write_text("v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf 1 2 3 4\n")
    vertices, faces = obj_mesh(mesh)
    assert vertices.shape == (4, 3) and faces.shape == (2, 3)
    urdf = tmp_path / "model.urdf"
    urdf.write_text("""<robot name='x'>
      <link name='root'><visual><geometry><box size='1 1 1'/></geometry></visual></link>
      <link name='moving'><visual><geometry><mesh filename='quad.obj'/></geometry></visual></link>
      <link name='tip'><visual><geometry><box size='.2 .2 .2'/></geometry></visual></link>
      <joint name='hinge' type='revolute'><parent link='root'/><child link='moving'/><axis xyz='0 0 1'/><limit lower='0' upper='1.5'/></joint>
      <joint name='fixed' type='fixed'><parent link='moving'/><child link='tip'/></joint>
    </robot>""")
    samples = parse_formal(tmp_path)
    assert len(samples) == 1
    _, _, _, _, _, _, visual_total, static_visuals, moving_visuals = samples[0]
    assert visual_total == 3 and static_visuals == 1 and moving_visuals == 2
