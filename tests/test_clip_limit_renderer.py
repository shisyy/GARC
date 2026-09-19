import copy
import json
from pathlib import Path
import tarfile

import pytest
import torch

import build_clip_limit_cache as cache_builder
from renderer.renderer_clip_stream import VIEWS, _bundle_digest, create_renderer


BOX_OBJ = """v -0.5 -0.5 -0.5
v -0.5 -0.5 0.5
v -0.5 0.5 -0.5
v -0.5 0.5 0.5
v 0.5 -0.5 -0.5
v 0.5 -0.5 0.5
v 0.5 0.5 -0.5
v 0.5 0.5 0.5
f 1 2 4 3
f 5 7 8 6
f 1 5 6 2
f 3 4 8 7
f 1 3 7 5
f 2 6 8 4
"""


def plan(domain: str):
    return {
        "schema": "splart-c-clip-ld-render-plan/v2",
        "domain": domain,
        "target_labels_used": False,
        "views": [list(value) for value in VIEWS],
        "canonical_q": {"side0": [0.0], "side1": [1.0]},
        "required_render_shape": [2, 1, 6, 3, 224, 224],
    }


def njc_row():
    return {
        "split": "source_train",
        "object_id": "object-a",
        "episode_id": "episode-a",
        "state0_fraction": 0.25,
        "state1_fraction": 0.75,
        "axis_orientation": 1,
        "state_order": "forward",
        "joint_selector": {"kind": "exact_name", "name": "lid_hinge"},
        "asset": {
            "kind": "njc_directory",
            "relative_path": "object-a",
            "required_files": ["object.urdf", "base_final.obj", "lid_final.obj", "qc.json"],
        },
    }


def make_njc_assets(root: Path):
    directory = root / "object-a"
    directory.mkdir(parents=True)
    (directory / "base_final.obj").write_text(BOX_OBJ)
    (directory / "lid_final.obj").write_text(BOX_OBJ)
    (directory / "qc.json").write_text(json.dumps({"source": "synthetic"}))
    (directory / "object.urdf").write_text("""<robot name="synthetic">
      <link name="base"><visual><geometry><mesh filename="base_final.obj" scale="1 0.8 0.4"/></geometry></visual></link>
      <link name="lid"><visual><origin xyz="0.4 0 0"/><geometry><mesh filename="lid_final.obj" scale="0.8 0.2 0.2"/></geometry></visual></link>
      <joint name="lid_hinge" type="revolute"><parent link="base"/><child link="lid"/>
        <origin xyz="0.4 0 0"/><axis xyz="0 1 0"/><limit lower="0" upper="1.2"/>
      </joint>
    </robot>""")


def make_articraft_assets(root: Path):
    source = root / "archive-source"
    source.mkdir(parents=True)
    # The first bounded revolute is deliberately followed by another one; the
    # renderer must use document order rather than selecting a convenient joint.
    (source / "model.urdf").write_text("""<robot name="synthetic">
      <link name="base"><visual><geometry><box size="1 0.8 0.4"/></geometry></visual></link>
      <link name="door"><visual><origin xyz="0.4 0 0"/><geometry><box size="0.8 0.2 0.2"/></geometry></visual></link>
      <link name="handle"><visual><geometry><box size="0.2 0.2 0.2"/></geometry></visual></link>
      <joint name="first" type="revolute"><parent link="base"/><child link="door"/>
        <origin xyz="0.4 0 0"/><axis xyz="0 1 0"/><limit lower="-0.2" upper="1.0"/>
      </joint>
      <joint name="second" type="revolute"><parent link="door"/><child link="handle"/>
        <origin xyz="0.8 0 0"/><axis xyz="1 0 0"/><limit lower="0" upper="0.5"/>
      </joint>
    </robot>""")
    archive = root / "object-a.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        handle.add(source / "model.urdf", arcname="nested/model.urdf")
    return {
        "split": "source_train",
        "object_id": "object-a.tar",
        "episode_id": "object-a.tar",
        "state0_fraction": 0.25,
        "state1_fraction": 0.75,
        "axis_orientation": 1,
        "state_order": "forward",
        "joint_selector": {"kind": "first_bounded_revolute_document_order"},
        "asset": {"kind": "articraft_archive", "relative_path": archive.name},
    }


def render_one(renderer, row, side=0):
    return next(renderer.iter_candidate_batches(
        row=row,
        side=side,
        canonical_q=tuple(renderer._canonical_q[f"side{side}"]),
        views=VIEWS,
        batch_size=1,
    ))["images"]


def test_njc_stream_is_uint8_nonempty_repeat_identical_and_hash_bound(tmp_path):
    make_njc_assets(tmp_path)
    renderer = create_renderer(tmp_path, "njc", plan("njc"), device="cpu")
    row = njc_row()
    first = render_one(renderer, row)
    repeated = render_one(renderer, row)
    assert first.shape == (1, 6, 3, 224, 224) and first.dtype == torch.uint8
    assert torch.equal(first, repeated)
    assert ((first != 255).any(dim=2).flatten(2).any(-1)).all()
    assert len(renderer.asset_bundle_sha256(row)) == len(renderer.render_config_sha256) == 64
    assert renderer.audit_receipt()["repeat_bit_identical"] is True


def test_articraft_archive_mapping_and_first_bounded_revolute_render(tmp_path):
    row = make_articraft_assets(tmp_path)
    renderer = create_renderer(tmp_path, "articraft", plan("articraft"), device="cpu")
    images = render_one(renderer, row)
    assert images.shape == (1, 6, 3, 224, 224)
    assert ((images != 255).any(dim=2).flatten(2).any(-1)).all()


def test_renderer_rejects_true_endpoints_even_when_assets_are_valid(tmp_path):
    make_njc_assets(tmp_path)
    renderer = create_renderer(tmp_path, "njc", plan("njc"), device="cpu")
    leaked = copy.deepcopy(njc_row())
    leaked["true_endpoints"] = [0.0, 1.0]
    try:
        renderer.asset_bundle_sha256(leaked)
    except ValueError as error:
        assert "target-only" in str(error) or "endpoint" in str(error)
    else:
        raise AssertionError("true_endpoints must never reach the renderer")


def test_bundled_plugin_loads_through_the_hash_bound_dynamic_interface(tmp_path):
    script = Path(__file__).parents[1] / "renderer" / "renderer_clip_stream.py"
    renderer, config_hash = cache_builder._load_renderer_plugin(
        script,
        cache_builder.sha256_file(script),
        tmp_path,
        "njc",
        plan("njc"),
        "cpu",
    )
    assert renderer.backend == "software-face-zbuffer"
    assert config_hash == renderer.render_config_sha256


def test_plugin_independently_rejects_nonfixed_view_or_shape_plan(tmp_path):
    wrong_view = plan("njc")
    wrong_view["views"][0] = [10, 0]
    with pytest.raises(ValueError, match="six-view"):
        create_renderer(tmp_path, "njc", wrong_view, device="cpu")
    wrong_shape = plan("njc")
    wrong_shape["required_render_shape"][-1] = 256
    with pytest.raises(ValueError, match="224"):
        create_renderer(tmp_path, "njc", wrong_shape, device="cpu")


def test_njc_rejects_unhashed_extra_visual_mesh_that_could_change_pixels_same_hash(tmp_path):
    make_njc_assets(tmp_path)
    directory = tmp_path / "object-a"
    urdf = directory / "object.urdf"
    extra = directory / "extra.obj"
    extra.write_text(BOX_OBJ)
    urdf.write_text(urdf.read_text().replace("lid_final.obj", "extra.obj"))
    required = [(name, directory / name) for name in ("object.urdf", "base_final.obj", "lid_final.obj", "qc.json")]
    old_unhardened_hash = _bundle_digest(required)
    extra.write_text(BOX_OBJ.replace("v 0.5 0.5 0.5", "v 2.5 2.5 2.5"))
    assert _bundle_digest(required) == old_unhardened_hash

    renderer = create_renderer(tmp_path, "njc", plan("njc"), device="cpu")
    with pytest.raises(ValueError, match="unauthorized mesh"):
        renderer.asset_bundle_sha256(njc_row())


def test_njc_requires_both_hash_bound_visual_meshes_to_be_used(tmp_path):
    make_njc_assets(tmp_path)
    urdf = tmp_path / "object-a" / "object.urdf"
    urdf.write_text(urdf.read_text().replace("lid_final.obj", "base_final.obj"))
    renderer = create_renderer(tmp_path, "njc", plan("njc"), device="cpu")
    with pytest.raises(ValueError, match="must use both"):
        renderer.asset_bundle_sha256(njc_row())
