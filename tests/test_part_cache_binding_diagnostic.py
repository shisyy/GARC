import torch

import diagnose_part_cache_binding as diagnostic


def test_feature_diff_preserves_pinned_tolerance_and_views():
    value = torch.ones(6, 512)
    assert diagnostic.difference(value, value)["bit_identical"]
    changed = value.clone()
    changed[3, 7] += 0.01
    report = diagnostic.difference(changed, value)
    assert not report["within_pinned_tolerance"]
    assert report["per_view_max_abs"][3] > 0.009
    assert report["per_view_max_abs"][0] == 0


def test_first_mismatch_repeats_same_pixels_and_renderer(monkeypatch):
    pixels = torch.zeros(1, 6, 3, 224, 224, dtype=torch.uint8)
    class Renderer:
        def iter_candidate_batches(self, **kwargs):
            assert kwargs["batch_size"] == 1
            for index in range(129):
                yield {"start": index, "images": pixels.clone()}
    def encode(model, images, device):
        return torch.ones(6, 512)
    monkeypatch.setattr(diagnostic.cache, "encode_views", encode)
    baseline = {"image_embeddings": torch.full((2, 129, 6, 512), -1.0)}
    report = diagnostic.scan(Renderer(), None, {}, {"canonical_q": {"side0": [], "side1": []}, "views": []}, baseline, torch.device("cpu"))
    assert report["status"] == "first_mismatch"
    assert report["candidate_index"] == 0 and report["side"] == 0
    assert report["candidates_scanned"] == 1
    assert report["pixels_repeat_bit_identical"]
    assert len(set(report["pixel_sha256"])) == 1
    assert report["same_pixels_repeat_vs_first"]["bit_identical"]
    assert not report["rerender_vs_baseline"]["within_pinned_tolerance"]
