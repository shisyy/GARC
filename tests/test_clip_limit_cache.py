import json
from pathlib import Path

import pytest
import torch

from splart.clip_limit import prompt_ensemble_sha256
from splart.clip_limit_cache import (
    CACHE_ARTIFACT_SCHEMA,
    CACHE_INDEX_SCHEMA,
    fixed_candidate_grid,
    load_semantic_cache,
    sha256_file,
    validate_cache_index,
)


def make_cache(tmp_path: Path):
    source_hash = "a" * 64
    checkpoint_hash = "b" * 64
    renderer_hash = "c" * 64
    render_config_hash = "d" * 64
    metadata_hash = "1" * 64
    wheel_hash = "2" * 64
    rows = []
    identities = []
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    for split, object_id in (("source_train", "train-a"), ("source_validation", "val-a")):
        episode_id = object_id
        coordinates = fixed_candidate_grid().expand(2, -1).clone()
        evidence = (coordinates - 1.0)[..., None]
        artifact = {
            "schema": CACHE_ARTIFACT_SCHEMA,
            "split": split,
            "object_id": object_id,
            "episode_id": episode_id,
            "coordinates": coordinates,
            "signed_limit_evidence": evidence,
            "view_count": 6,
            "prompt_ensemble_sha256": prompt_ensemble_sha256(),
            "encoder_checkpoint_sha256": checkpoint_hash,
            "open_clip_version": "3.3.0",
            "open_clip_wheel_sha256": wheel_hash,
            "renderer_sha256": renderer_hash,
            "render_config_sha256": render_config_hash,
            "asset_bundle_sha256": "e" * 64,
            "audit_thumbnail_sha256": {
                f"side{side}:{index:03d}": "f" * 64 for side in range(2) for index in (0, 64, 128)
            },
            "render_metadata_sha256": metadata_hash,
        }
        path = artifacts / f"{episode_id}.pt"
        torch.save(artifact, path)
        rows.append(
            {
                "artifact": str(path.relative_to(tmp_path)),
                "artifact_sha256": sha256_file(path),
                "episode_id": episode_id,
                "object_id": object_id,
                "render_metadata_sha256": metadata_hash,
                "shape": [2, 129, 1],
                "split": split,
            }
        )
        identities.append((split, object_id, episode_id))
    index = {
        "schema": CACHE_INDEX_SCHEMA,
        "encoder": "open_clip:ViT-B-32",
        "encoder_checkpoint_sha256": checkpoint_hash,
        "prompt_ensemble_sha256": prompt_ensemble_sha256(),
        "source_index_sha256": source_hash,
        "input_bindings": {"selection": {"path": "/public/selection.json", "sha256": "3" * 64}},
        "open_clip_version": "3.3.0",
        "open_clip_wheel_sha256": wheel_hash,
        "renderer_sha256": renderer_hash,
        "render_config_sha256": render_config_hash,
        "streaming_rgb_persisted": False,
        "renderer_audit": {
            "schema": "splart-c-clip-ld-render-audit/v1",
            "no_empty_views_checked": True,
            "repeat_bit_identical": True,
            "assets_repeat_audited": 2,
            "views_checked": 2 * 2 * 129 * 6,
            "render_config_sha256": render_config_hash,
            "runtime": {
                "device_type": "cuda",
                "torch_version": "2.1.2+cu121",
                "torch_cuda_version": "12.1",
                "pytorch3d_version": "0.7.8",
                "pytorch3d_distribution_sha256": "4" * 64,
            },
        },
        "rows": rows,
        "protected_splits_read": [],
        "box_labels_read": False,
    }
    index_path = tmp_path / "index.json"
    index_path.write_text(json.dumps(index))
    return index_path, source_hash, identities


def test_cache_is_hash_prompt_grid_and_identity_bound(tmp_path):
    index, source_hash, identities = make_cache(tmp_path)
    result = load_semantic_cache(
        index,
        source_index_sha256=source_hash,
        expected_split_counts={"source_train": 1, "source_validation": 1},
        expected_identities=identities,
    )
    assert result["source_train"][0].shape == (1, 2, 129, 1)
    assert result["source_validation"][1].shape == (1, 2, 129)


def test_cache_fails_closed_on_digest_and_legacy_observed_state_bank(tmp_path):
    index, source_hash, identities = make_cache(tmp_path)
    payload = json.loads(index.read_text())
    payload["rows"][0]["artifact_sha256"] = "0" * 64
    index.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="digest"):
        validate_cache_index(
            index,
            source_index_sha256=source_hash,
            expected_split_counts={"source_train": 1, "source_validation": 1},
            expected_identities=identities,
        )
    payload["schema"] = "splart-smarc-observed-state-bank/v1"
    index.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="schema"):
        validate_cache_index(
            index, source_index_sha256=source_hash, expected_split_counts={"source_train": 1, "source_validation": 1}
        )


def test_cache_rejects_target_adaptive_candidate_coordinates(tmp_path):
    index, source_hash, identities = make_cache(tmp_path)
    payload = json.loads(index.read_text())
    artifact_path = tmp_path / payload["rows"][0]["artifact"]
    artifact = torch.load(artifact_path, weights_only=True)
    artifact["coordinates"] = artifact["coordinates"] + 0.01
    torch.save(artifact, artifact_path)
    payload["rows"][0]["artifact_sha256"] = sha256_file(artifact_path)
    index.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="fixed label-independent"):
        validate_cache_index(
            index,
            source_index_sha256=source_hash,
            expected_split_counts={"source_train": 1, "source_validation": 1},
            expected_identities=identities,
        )


def test_cache_fails_closed_on_tampered_render_metadata_binding(tmp_path):
    index, source_hash, identities = make_cache(tmp_path)
    payload = json.loads(index.read_text())
    payload["rows"][0]["render_metadata_sha256"] = "9" * 64
    index.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="identity/binding"):
        validate_cache_index(
            index,
            source_index_sha256=source_hash,
            expected_split_counts={"source_train": 1, "source_validation": 1},
            expected_identities=identities,
        )
