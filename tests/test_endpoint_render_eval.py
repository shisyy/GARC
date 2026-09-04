import json
import subprocess
from pathlib import Path

import pytest

from endpoint_render_eval import expected_asset_hash, scene_record, verify_candidate
from splart.endpoint_baselines import make_prediction


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    scene = tmp_path / "100247-Box"
    scene.mkdir()
    (scene / "transforms.json").write_text(
        json.dumps(
            {
                "articulation": {"type": 1},
                "frames": [
                    {"file_path": "color/train/0.png", "state": 0},
                    {"file_path": "color/train/1.png", "state": 1},
                ],
            }
        ),
        encoding="utf-8",
    )
    (scene / "endpoint_queries.json").write_text(
        json.dumps(
            {
                "queries": [
                    {"query_id": "outside_state_0", "local_direction": -1},
                    {"query_id": "outside_state_1", "local_direction": 1},
                ]
            }
        ),
        encoding="utf-8",
    )
    for modality in ("color", "depth", "part-seg"):
        for split in ("train", "val"):
            (scene / modality / split).mkdir(parents=True)

    checkpoint_dir = tmp_path / "model" / "nerfstudio_models"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "step-000024999.ckpt").write_bytes(b"frozen candidate")
    (checkpoint_dir.parent / "config.yml").write_text("model: splart\n", encoding="utf-8")
    (checkpoint_dir.parent / "dataparser_transforms.json").write_text("{}\n", encoding="utf-8")

    source = tmp_path / "source"
    source.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=source, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=source, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=source, check=True)
    (source / "source.py").write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "source.py"], cwd=source, check=True)
    subprocess.run(["git", "commit", "-qm", "source"], cwd=source, check=True)
    return scene, checkpoint_dir, source


def test_candidate_provenance_is_bound_before_renderer_load(tmp_path: Path) -> None:
    scene, checkpoint_dir, source = _fixture(tmp_path)
    prediction = make_prediction(scene, "splart-middle", checkpoint_dir, source)
    paths = verify_candidate(prediction, scene)
    assert paths["checkpoint"].name == "step-000024999.ckpt"
    (checkpoint_dir / "step-000024999.ckpt").write_bytes(b"mutated")
    with pytest.raises(ValueError, match="sha256 mismatch"):
        verify_candidate(prediction, scene)


def test_sealed_episode_and_asset_hash_must_be_explicit() -> None:
    manifest = {"episodes": {"100247-Box": {"target": {}}}}
    assert scene_record(manifest, "100247-Box") == {"target": {}}
    view = {"source_sha256": {"color/val/0000.png": "a" * 64}}
    assert expected_asset_hash(view, "color", "color/val/0000.png") == "a" * 64
    with pytest.raises(ValueError, match="sha256"):
        expected_asset_hash(view, "depth", "depth/val/0000.png")
