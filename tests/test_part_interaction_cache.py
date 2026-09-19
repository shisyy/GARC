import copy
import json

import pytest
import torch
from torch.nn import functional as F

import build_part_interaction_cache as builder
from splart import part_interaction_cache as cache


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def image():
    value = torch.full((6, 3, 4, 5), 255, dtype=torch.uint8)
    value[:, :, 0, :3] = torch.tensor(cache.STATIC_RGB, dtype=torch.uint8)[None, :, None]
    value[:, :, 1, :4] = torch.tensor(cache.MOBILE_RGB, dtype=torch.uint8)[None, :, None]
    return value


def test_masks_deletions_and_exact_area_null():
    value = image()
    static, mobile = cache.visible_masks(value)
    assert static.sum().item() == 18 and mobile.sum().item() == 24
    a, b = cache.matched_partition(static, mobile)
    assert torch.equal(a.sum((1, 2)), static.sum((1, 2)))
    assert torch.equal(b.sum((1, 2)), mobile.sum((1, 2)))
    expected = torch.tensor([False, False, True, False, True, False, True])
    assert torch.equal(a[0][(static | mobile)[0]], expected)
    for (key, deleted), mask in zip(cache.deletion_images(value), (static, mobile, a, b)):
        assert key in cache.FEATURE_KEYS
        assert torch.equal(deleted.permute(0, 2, 3, 1)[mask], value.permute(0, 2, 3, 1)[mask])
        assert (deleted.permute(0, 2, 3, 1)[~mask] == 255).all()


def test_palette_all_depth_shades_unknown_and_empty():
    shades = torch.linspace(0.72, 1.0, 10001)
    values = [(torch.tensor(rgb)[None] * shades[:, None]).round().to(torch.uint8) for rgb in (cache.STATIC_RGB, cache.MOBILE_RGB)]
    pixels = torch.cat(values).T[None, :, None]
    static, mobile = cache.visible_masks(pixels)
    assert static.sum() == 10001 and mobile.sum() == 10001
    value = image()
    value[0, :, 0, 0] = torch.tensor([148, 147, 148])
    with pytest.raises(ValueError, match="unknown"):
        cache.visible_masks(value)
    with pytest.raises(ValueError, match="empty"):
        cache.visible_masks(torch.full((6, 3, 2, 2), 255, dtype=torch.uint8))


@pytest.mark.parametrize("static_count", [0, 1, 7, 20])
def test_partition_extreme_areas(static_count):
    static = (torch.arange(20).reshape(1, 4, 5) < static_count)
    a, b = cache.matched_partition(static, ~static)
    assert a.sum() == static_count and b.sum() == 20 - static_count
    assert not (a & b).any() and (a | b).all()


def artifact():
    return {
        "schema": cache.ARTIFACT_SCHEMA, "object_id": "a" * 64, "split": "source_train",
        "coordinates": builder.base.fixed_candidate_grid().expand(2, -1).clone(),
        **{key: F.normalize(torch.ones(2, 129, 6, 512), dim=-1) for key in cache.FEATURE_KEYS},
        "blank_embedding": F.normalize(torch.ones(512), dim=0),
        "provenance": {key: "b" * 64 for key in cache.HASH_KEYS | {"asset_bundle_sha256"}},
    }


def test_artifact_strict_schema_unit_and_grid():
    value = artifact()
    cache.validate_part_artifact(value)
    value["target"] = 0
    with pytest.raises(ValueError, match="schema"):
        cache.validate_part_artifact(value)
    del value["target"]
    value["static_embeddings"][0, 0] = 0
    with pytest.raises(ValueError, match="unit"):
        cache.validate_part_artifact(value)
    value = artifact()
    value["coordinates"][0, 0] = 0.1
    with pytest.raises(ValueError, match="grid"):
        cache.validate_part_artifact(value)


def test_index_strict_binding():
    value = {"schema": cache.INDEX_SCHEMA, "rows": [{"object_id": "a" * 64, "split": "source_train", "artifact": f"artifacts/{'a' * 64}.pt", "artifact_sha256": "c" * 64}], "provenance": {key: "b" * 64 for key in cache.HASH_KEYS}, "split_counts": {"source_train": 1}}
    cache.validate_part_index(value)
    changed = copy.deepcopy(value)
    changed["rows"][0]["artifact"] = "../labels.pt"
    with pytest.raises(ValueError, match="path"):
        cache.validate_part_index(changed)
    changed = copy.deepcopy(value)
    changed["provenance"]["target"] = "d" * 64
    with pytest.raises(ValueError, match="provenance"):
        cache.validate_part_index(changed)


def test_streamed_joint_recomputation_and_six_view_batches(monkeypatch):
    pixels = torch.full((1, 6, 3, 224, 224), 255, dtype=torch.uint8)
    pixels[:, :, :, 0, 0] = torch.tensor(cache.STATIC_RGB, dtype=torch.uint8)
    pixels[:, :, :, 0, 1] = torch.tensor(cache.MOBILE_RGB, dtype=torch.uint8)
    class Renderer:
        def iter_candidate_batches(self, **kwargs):
            assert kwargs["batch_size"] == 1
            for i in range(129):
                yield {"start": i, "images": pixels}
    class Model:
        def encode_image(self, value):
            assert not torch.is_grad_enabled() and len(value) == 6
            result = torch.ones(6, 512)
            result[:, :3] = value[:, :, 0, 0]
            return result
    monkeypatch.setattr(builder.base.FrozenCLIPLimitDiscriminator, "preprocess", lambda value: value.float())
    model, device = Model(), torch.device("cpu")
    joint = builder.encode_views(model, pixels[0], device)
    baseline = {"image_embeddings": joint.expand(2, 129, 6, 512).clone()}
    plan = {"canonical_q": {"side0": [], "side1": []}, "views": []}
    output = builder.encode_object(Renderer(), model, {"object_id": "synthetic"}, plan, baseline, device)
    assert set(output) == set(cache.FEATURE_KEYS)
    assert all(value.shape == (2, 129, 6, 512) for value in output.values())
    assert torch.equal(output["static_embeddings"][0, 0], joint)
    baseline["image_embeddings"][0, 0] *= -1
    with pytest.raises(ValueError, match="pinned baseline"):
        builder.encode_object(Renderer(), model, {"object_id": "synthetic"}, plan, baseline, device)
