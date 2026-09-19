import json

import pytest
import torch

import build_part_interaction_cache as cache


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def pixels(value):
    return torch.full((6, 3, 224, 224), value, dtype=torch.uint8)


def feature(value):
    result = torch.zeros(6, 512)
    result[:, value] = 1
    return result


@pytest.mark.parametrize("attempt_values,expected_attempts", [([], 1), ([2], 2), ([1, 2], 3), ([1, 1], None)])
def test_fixed_three_attempts_only_same_candidate(monkeypatch, attempt_values, expected_attempts):
    plan = {"canonical_q": {"side0": list(range(129)), "side1": list(range(129))}, "views": [[0, 0]] * 6}
    row = {"object_id": "synthetic"}
    calls = []
    class Renderer:
        def iter_candidate_batches(self, **kwargs):
            assert kwargs["row"] is row
            assert kwargs["side"] == 1
            assert kwargs["canonical_q"] is plan["canonical_q"]["side1"]
            assert kwargs["views"] is plan["views"] and kwargs["batch_size"] == 1
            value = attempt_values[len(calls)]
            calls.append(kwargs)
            for index in range(129):
                yield {"start": index, "images": pixels(value)[None]}
    monkeypatch.setattr(cache, "encode_views", lambda model, image, device: feature(int(image[0, 0, 0, 0])))
    baseline = {"image_embeddings": feature(2).expand(2, 129, 6, 512)}
    initial = pixels(2 if expected_attempts == 1 else 1)
    audit = cache.reproduction_audit()
    if expected_attempts is None:
        with pytest.raises(ValueError, match="after 3 fixed render attempts"):
            cache.matched_joint_images(Renderer(), None, row, plan, baseline, None, 1, 7, initial, audit)
        assert audit["summary"]["failed_attempts"] == 3
        assert audit["summary"]["unrecovered_candidates"] == 1
        assert audit["events"][0]["attempt_count"] == 3
    else:
        accepted = cache.matched_joint_images(Renderer(), None, row, plan, baseline, None, 1, 7, initial, audit)
        assert torch.equal(accepted, pixels(2))
        assert len(calls) == expected_attempts - 1
        if expected_attempts == 1:
            assert accepted is initial and not audit["events"]
        else:
            event = audit["events"][0]
            assert event["side"] == 1 and event["candidate_index"] == 7
            assert event["attempt_count"] == expected_attempts
            assert event["failed_attempts"] == expected_attempts - 1
            assert event["accepted_pixel_sha256"] == cache._tensor_hash(accepted)
            assert event["accepted_feature_sha256"] == cache._tensor_hash(feature(2))
    assert len(calls) <= 2


def test_deletions_only_use_accepted_pixels(monkeypatch):
    first = pixels(1)
    replacement = pixels(2)
    calls = []
    class Renderer:
        def iter_candidate_batches(self, **kwargs):
            call = len(calls)
            calls.append(call)
            for index in range(129):
                yield {"start": index, "images": (replacement if call == 1 else first)[None]}
    seen = []
    def delete(images):
        seen.append(int(images[0, 0, 0, 0]))
        for key in cache.FEATURE_KEYS:
            yield key, images
    monkeypatch.setattr(cache.rules, "deletion_images", delete)
    monkeypatch.setattr(cache, "encode_views", lambda model, image, device: feature(int(image[0, 0, 0, 0])))
    baseline = {"image_embeddings": feature(1).expand(2, 129, 6, 512).clone()}
    baseline["image_embeddings"][0, 0] = feature(2)
    audit = cache.reproduction_audit()
    result = cache.encode_object(Renderer(), None, {"object_id": "synthetic"}, {"canonical_q": {"side0": [], "side1": []}, "views": []}, baseline, None, audit)
    assert seen == [2] + [1] * 257
    assert all(torch.equal(value[0, 0], feature(2)) for value in result.values())
    assert audit["summary"]["candidates_checked"] == 258
    assert audit["summary"]["direct_matches"] == 257
    assert audit["summary"]["additional_renders"] == 1


def test_audit_is_separate_immutable_and_saves_failure(tmp_path):
    audit = cache.reproduction_audit()
    cache.write_reproduction_audit(tmp_path, audit, "failed")
    assert json.loads((tmp_path / "render_reproduction_audit.json").read_text())["status"] == "failed"
    with pytest.raises(FileExistsError):
        cache.write_reproduction_audit(tmp_path, audit, "complete")
