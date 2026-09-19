import argparse
import json

import pytest
import torch
from torch.nn import functional as F

import build_shared_rgb_part_cache as shared


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def unit(value=0):
    result = torch.zeros(6, 512)
    result[:, value] = 1
    return result


def test_original_rgb_shard_roundtrip_hash_and_immutability(tmp_path):
    object_id = "a" * 64
    (tmp_path / "rgb" / object_id).mkdir(parents=True)
    pixels = torch.arange(6 * 3 * 224 * 224).remainder(256).to(torch.uint8).reshape(6, 3, 224, 224)
    row = shared.save_rgb_shard(tmp_path, object_id, 1, 128, pixels)
    saved = torch.load(tmp_path / row["path"], weights_only=True)
    assert torch.equal(saved, pixels)
    assert row["sha256"] == shared.base.sha256_file(tmp_path / row["path"])
    assert row["side"] == 1 and row["candidate_index"] == 128
    with pytest.raises(FileExistsError):
        shared.save_rgb_shard(tmp_path, object_id, 1, 128, pixels)


def test_single_acquisition_shared_by_all_arms_without_historical_selection(monkeypatch, tmp_path):
    calls, saved, encoded = [], [], []
    pixels = torch.full((1, 6, 3, 224, 224), 7, dtype=torch.uint8)
    class Renderer:
        def iter_candidate_batches(self, **kwargs):
            calls.append(kwargs["side"])
            assert kwargs["batch_size"] == 1
            for index in range(129):
                yield {"start": index, "images": pixels}
    def save(root, object_id, side, candidate, image):
        assert torch.equal(image, pixels[0])
        saved.append((side, candidate))
        return {"path": "mock", "sha256": "a" * 64, "side": side, "candidate_index": candidate}
    def delete(image):
        assert image.data_ptr() == pixels.data_ptr()
        for offset, key in enumerate(shared.rules.FEATURE_KEYS, 1):
            yield key, image + offset
    def encode(model, image, device):
        value = int(image[0, 0, 0, 0])
        encoded.append(value)
        return unit(value)
    monkeypatch.setattr(shared, "save_rgb_shard", save)
    monkeypatch.setattr(shared.rules, "deletion_images", delete)
    monkeypatch.setattr(shared.previous, "encode_views", encode)
    joint, auxiliary, shards = shared.acquire_object(Renderer(), None, {"object_id": "synthetic"}, {"canonical_q": {"side0": [], "side1": []}, "views": []}, None, tmp_path)
    assert calls == [0, 1]
    assert saved == [(side, candidate) for side in range(2) for candidate in range(129)]
    assert len(shards) == 258 and encoded == [7, 8, 9, 10, 11] * 258
    assert torch.equal(joint[1, 128], unit(7))
    assert all(torch.equal(auxiliary[key][1, 128], unit(offset)) for offset, key in enumerate(shared.rules.FEATURE_KEYS, 8))


def test_compatible_indices_fresh_binding_and_completion_receipt(monkeypatch, tmp_path):
    old_root = tmp_path / "historical"
    (old_root / "artifacts").mkdir(parents=True)
    object_id = shared.base.opaque_object_id("synthetic")
    common = {key: "a" * 64 for key in ("render_plan_sha256", "renderer_sha256", "render_config_sha256", "encoder_checkpoint_sha256", "open_clip_wheel_sha256")}
    provenance = common | {"prompt_sha256": shared.base._digest({"open": shared.base.OPEN_PROMPTS, "closed": shared.base.CLOSED_PROMPTS}), "builder_sha256": "b" * 64}
    old = {"schema": shared.base.ARTIFACT_SCHEMA, "object_id": object_id, "split": "source_train", "coordinates": shared.base.fixed_candidate_grid().expand(2, -1).clone(), "image_embeddings": unit(1).expand(2, 129, 6, 512).clone(), "observed_embeddings": unit(1).expand(2, 6, 512).clone(), "text_direction": F.normalize(torch.arange(512).float(), dim=0), "provenance": provenance | {"asset_bundle_sha256": "c" * 64}}
    old_row = shared.save_artifact(old_root, old)
    old_index = {"schema": shared.base.INDEX_SCHEMA, "rows": [old_row], "provenance": provenance, "split_counts": {"source_train": 1}}
    shared.publish_json(old_root / "index.json", old_index)
    row = {"object_id": "synthetic", "split": "source_train"}
    class Renderer:
        def asset_bundle_sha256(self, row):
            return "c" * 64
        def audit_receipt(self):
            return {"no_empty_views_checked": True, "repeat_bit_identical": True, "assets_repeat_audited": 1, "views_checked": 2 * 129 * 6, "render_config_sha256": "a" * 64}
    class Model:
        def eval(self):
            return self
        def parameters(self):
            return []
    monkeypatch.setattr(shared.base, "load_render_plan", lambda *args: {"rows": [row]})
    monkeypatch.setattr(shared.base, "_verify_open_clip_install", lambda *args: None)
    monkeypatch.setattr(shared.base, "_load_renderer_plugin", lambda *args: (Renderer(), "a" * 64))
    monkeypatch.setattr(shared.base, "load_frozen_open_clip_vit_b32", lambda *args: argparse.Namespace(clip_model=Model()))
    monkeypatch.setattr(shared.previous, "encode_views", lambda *args: unit(3))
    fresh = unit(2).expand(2, 129, 6, 512).clone()
    monkeypatch.setattr(shared, "acquire_object", lambda *args: (fresh, {key: fresh.clone() for key in shared.rules.FEATURE_KEYS}, []))
    args = argparse.Namespace(output=tmp_path / "fresh", batch_size=1, render_plan=tmp_path / "plan", render_plan_sha256="a" * 64, max_objects=1, base_semantic_index=old_root / "index.json", device="cpu", open_clip_wheel=tmp_path / "wheel", open_clip_wheel_sha256="a" * 64, renderer_script=tmp_path / "renderer", renderer_sha256="a" * 64, asset_root=tmp_path, clip_checkpoint=tmp_path / "checkpoint", clip_checkpoint_sha256="a" * 64)
    receipt = shared.build_cache(args)
    relative_index = json.loads((args.output / "relative/index.json").read_text())
    part_index = json.loads((args.output / "part/index.json").read_text())
    shared.base.validate_relative_index(relative_index)
    shared.rules.validate_part_index(part_index)
    assert receipt["relative_index_sha256"] == shared.base.sha256_file(args.output / "relative/index.json")
    assert part_index["provenance"]["base_semantic_index_sha256"] == receipt["relative_index_sha256"]
    assert receipt["original_base_semantic_index_sha256"] == shared.base.sha256_file(old_root / "index.json")
    assert receipt["rows"][0]["historical_semantic_artifact_sha256"] == old_row["artifact_sha256"]
    new_raw = torch.load(args.output / "relative" / relative_index["rows"][0]["artifact"], weights_only=True)
    new_part = torch.load(args.output / "part" / part_index["rows"][0]["artifact"], weights_only=True)
    shared.base.validate_relative_artifact(new_raw)
    shared.rules.validate_part_artifact(new_part)
    assert torch.equal(new_raw["image_embeddings"], fresh)
    assert not torch.equal(new_raw["image_embeddings"], old["image_embeddings"])
    assert torch.equal(new_raw["text_direction"], old["text_direction"])
    assert json.loads((args.output / "acquisition_audit.json").read_text()) == receipt
    with pytest.raises(ValueError, match="immutable"):
        shared.build_cache(args)
