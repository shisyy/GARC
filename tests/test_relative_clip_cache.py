import copy
import json

import pytest
import torch
from torch.nn import functional as F

import build_relative_clip_cache as cache
from prepare_clip_limit_render_plan import reconstruct_articraft_rows


def artifact():
    features = F.normalize(torch.ones(2, 129, 6, 512), dim=-1)
    return {
        "schema": cache.ARTIFACT_SCHEMA, "object_id": cache.opaque_object_id("opaque01"), "split": "source_train",
        "coordinates": cache.fixed_candidate_grid().expand(2, -1).clone(),
        "image_embeddings": features, "observed_embeddings": features[:, 0].clone(),
        "text_direction": F.normalize(torch.arange(512).float(), dim=0),
        "provenance": {key: "a" * 64 for key in cache.HASH_KEYS | {"asset_bundle_sha256"}},
    }


def test_artifact_retains_views_and_exact_observed_anchors():
    value = artifact()
    cache.validate_relative_artifact(value)
    value["observed_embeddings"][0, 0] *= -1
    with pytest.raises(ValueError, match="anchors"):
        cache.validate_relative_artifact(value)


@pytest.mark.parametrize("key", ["target_endpoints", "state0_fraction", "urdf_path", "limits", "source_index"])
def test_rejects_metadata_leakage(key):
    value = artifact()
    value[key] = 0
    with pytest.raises(ValueError, match="schema"):
        cache.validate_relative_artifact(value)
    value.pop(key)
    value["provenance"][key] = "a" * 64
    with pytest.raises(ValueError, match="provenance"):
        cache.validate_relative_artifact(value)


def test_rejects_nonunit_and_changed_gauge():
    value = artifact()
    value["image_embeddings"][0, 1, 0] = 0
    with pytest.raises(ValueError, match="unit"):
        cache.validate_relative_artifact(value)
    value = artifact()
    value["coordinates"][0, 0] = 0.1
    with pytest.raises(ValueError, match="grid"):
        cache.validate_relative_artifact(value)


def test_index_allowlist_and_canonical_paths():
    object_id = cache.opaque_object_id("opaque01")
    value = {
        "schema": cache.INDEX_SCHEMA,
        "rows": [{"object_id": object_id, "split": "source_train", "artifact": f"artifacts/{object_id}.pt", "artifact_sha256": "b" * 64}],
        "provenance": {key: "a" * 64 for key in cache.HASH_KEYS},
        "split_counts": {"source_train": 1},
    }
    cache.validate_relative_index(value)
    unsafe = copy.deepcopy(value)
    unsafe["rows"][0]["artifact"] = "../labels.pt"
    with pytest.raises(ValueError, match="path"):
        cache.validate_relative_index(unsafe)
    value["rows"][0]["target"] = 1
    with pytest.raises(ValueError, match="unapproved"):
        cache.validate_relative_index(value)


def test_category_bearing_ids_are_anonymized():
    import hashlib
    raw = "rec_hinged_window_or_hatch_0003.tar"
    assert cache.opaque_object_id(raw) == hashlib.sha256(("node91-object:" + raw).encode()).hexdigest()
    value = artifact()
    value["object_id"] = raw
    with pytest.raises(ValueError, match="opaque SHA256"):
        cache.validate_relative_artifact(value)


def test_plan_train_only_selection_never_opens_source_index(tmp_path):
    rows = reconstruct_articraft_rows({"selection": {
        "endpoint_pretrain": [f"opaque{i:02}.gz" for i in reversed(range(13))],
        "endpoint_validation": [f"opaque{i:02}.gz" for i in range(13, 19)],
    }})
    grid = cache.fixed_candidate_grid().tolist()
    plan = {
        "schema": cache.PLAN_SCHEMA, "domain": "articraft", "target_labels_used": False,
        "candidate_coordinates": grid, "canonical_q": {"side0": [-v for v in grid], "side1": [1 + v for v in grid]},
        "views": [list(v) for v in cache.AZIMUTH_ELEVATION_DEGREES],
        "required_render_shape": [2, 129, 6, 3, 224, 224], "rows": rows,
        "source_index": str(tmp_path / "DOES_NOT_EXIST.json"),
    }
    path = tmp_path / "render_plan.json"
    path.write_text(json.dumps(plan))
    result = cache.load_render_plan(path, cache.sha256_file(path), 2)
    assert [row["object_id"] for row in result["rows"]] == ["opaque00", "opaque01"]
    assert all(row["split"] == "source_train" for row in result["rows"])
    assert "source_index" not in result
    assert len(cache.load_render_plan(path, cache.sha256_file(path), None)["rows"]) == 19
    with pytest.raises(ValueError, match="digest"):
        cache.load_render_plan(path, "0" * 64, 2)


def test_text_direction_is_frozen_closed_minus_open():
    class Model:
        def encode_text(self, tokens):
            assert not torch.is_grad_enabled()
            features = torch.zeros(6, 512)
            features[:3, 0] = 1
            features[3:, 1] = 1
            return features

    def tokenize(prompts):
        assert prompts == cache.OPEN_PROMPTS + cache.CLOSED_PROMPTS
        return torch.zeros(6, 77, dtype=torch.long)

    direction = cache.text_direction(Model(), tokenize, torch.device("cpu"))
    expected = torch.zeros(512)
    expected[0], expected[1] = -1, 1
    assert torch.equal(direction, F.normalize(expected, dim=0))


def test_encode_preserves_view_and_candidate_identity(monkeypatch):
    class Renderer:
        def iter_candidate_batches(self, *, side, **kwargs):
            for i in range(129):
                pixels = torch.zeros(1, 6, 3, 224, 224, dtype=torch.uint8)
                for view in range(6):
                    pixels[:, view, 0] = i
                    pixels[:, view, 1] = view
                    pixels[:, view, 2] = side
                yield {"start": i, "images": pixels}

    class Model:
        def encode_image(self, pixels):
            assert not torch.is_grad_enabled()
            features = torch.ones(len(pixels), 512)
            features[:, :3] = pixels[:, :, 0, 0]
            return features

    monkeypatch.setattr(cache.FrozenCLIPLimitDiscriminator, "preprocess", lambda value: value.float())
    # Single-threaded tiny synthetic workloads avoid oversubscribed CPU kernels.
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        output = cache.encode_object(Renderer(), Model(), {}, {"canonical_q": {"side0": [], "side1": []}, "views": []}, 1, torch.device("cpu"))
    finally:
        torch.set_num_threads(previous)
    assert output.shape == (2, 129, 6, 512)
    for side, candidate, view in [(0, 0, 0), (1, 128, 5), (0, 17, 3)]:
        expected = torch.ones(512)
        expected[:3] = torch.tensor([candidate, view, side])
        assert torch.equal(output[side, candidate, view], F.normalize(expected, dim=0))
