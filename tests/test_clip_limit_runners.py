import pytest
import torch

import build_clip_limit_cache as cache_builder
import run_clip_limit_njc_gate as njc_gate
import run_clip_limit_source_gate as source_gate
from run_glpdt_source_gate import CONFIG
from splart.clip_limit import CSTRHead
from splart.clip_limit_gate import CONTROLS, summarize_cr_fpl_control


def test_both_runners_share_frozen_training_and_control_contract():
    assert source_gate.CONFIG is CONFIG
    assert njc_gate.CONFIG is CONFIG
    assert (CONFIG.seed, CONFIG.steps) == (2202, 4000)
    assert CONTROLS == ("cstr", "zero", "object_trajectory_shuffle", "coordinate_only", "sign_flip")
    model = CSTRHead()
    assert model.shared_side.semantic_residual.value.bias is None
    assert model.shared_side.semantic_residual.gate.bias is None


def test_frozen_cr_fpl_control_uses_max_side_tail_metric():
    class Output:
        distance = torch.tensor([[1.0, 2.0], [1.5, 4.0]])
        loop_distances = torch.stack((distance + 1.0, distance), dim=-1)

    target = torch.ones(2, 2)
    metrics = summarize_cr_fpl_control(Output(), target)
    assert metrics["cr_fpl_control_target_relative_p99_max_side"] >= max(
        metrics["cr_fpl_control_target_relative_p99_side"]
    )
    assert metrics["cr_fpl_control_worst_side_nmae"] >= metrics["cr_fpl_control_mean_side_nmae"]


def test_streaming_builder_rejects_observed_state_rgb_bank():
    class ObservedOnlyRenderer:
        def iter_candidate_batches(self, **_):
            yield {"start": 0, "images": torch.zeros(2, 6, 3, 224, 224, dtype=torch.uint8)}

    class Discriminator:
        def __call__(self, images):
            return torch.zeros(images.shape[0])

    with pytest.raises(ValueError, match="every fixed candidate"):
        cache_builder._encode_row(
            ObservedOnlyRenderer(),
            Discriminator(),
            {"split": "source_train", "object_id": "object", "episode_id": "episode"},
            {
                "side0": [-float(value) for value in torch.linspace(0, 2, 129)],
                "side1": [1 + float(value) for value in torch.linspace(0, 2, 129)],
            },
            [[0, 0], [90, 0], [180, 0], [270, 0], [45, 35], [225, -35]],
            8,
        )


def test_streaming_builder_encodes_all_candidates_without_rgb_output():
    class Renderer:
        def iter_candidate_batches(self, batch_size, **_):
            for start in range(0, 129, batch_size):
                count = min(batch_size, 129 - start)
                yield {"start": start, "images": torch.full((count, 6, 3, 224, 224), start % 255, dtype=torch.uint8)}

    class Discriminator:
        def __call__(self, images):
            return images[:, 0, 0, 0, 0].float()

    evidence, audits = cache_builder._encode_row(
        Renderer(),
        Discriminator(),
        {"split": "source_train", "object_id": "object", "episode_id": "episode"},
        {
            "side0": [-float(value) for value in torch.linspace(0, 2, 129)],
            "side1": [1 + float(value) for value in torch.linspace(0, 2, 129)],
        },
        [[0, 0], [90, 0], [180, 0], [270, 0], [45, 35], [225, -35]],
        8,
    )
    assert evidence.shape == (2, 129, 1)
    assert set(audits) == {f"side{side}:{index:03d}" for side in range(2) for index in (0, 64, 128)}


def test_open_clip_install_is_version_and_wheel_hash_bound(tmp_path, monkeypatch):
    wheel = tmp_path / "open_clip_torch.whl"
    wheel.write_bytes(b"pinned-wheel")
    digest = cache_builder.sha256_file(wheel)
    monkeypatch.setattr(cache_builder.metadata, "version", lambda _: "3.3.0")
    cache_builder._verify_open_clip_install(wheel, digest, "3.3.0")
    with pytest.raises(ValueError, match="digest"):
        cache_builder._verify_open_clip_install(wheel, "0" * 64, "3.3.0")
    with pytest.raises(ValueError, match="pinned"):
        cache_builder._verify_open_clip_install(wheel, digest, "3.2.0")
