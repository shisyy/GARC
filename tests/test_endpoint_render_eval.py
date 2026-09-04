import hashlib
import json
import subprocess
from pathlib import Path

import pytest

import endpoint_render_eval
from endpoint_render_eval import (
    deterministic_sample_indices,
    expected_asset_hash,
    load_postbuild_seal,
    scene_record,
    verify_candidate,
)
from splart.endpoint_baselines import audit_public_tree, content_sha256, make_prediction, sha256_file


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
    (scene / "PROXY_RECEIPT.json").write_text('{"binding_id": "binding"}\n', encoding="utf-8")
    (scene / "COMPLETE.json").write_text('{"binding_id": "binding"}\n', encoding="utf-8")
    for modality in ("color", "depth", "part-seg"):
        for split in ("train", "val"):
            (scene / modality / split).mkdir(parents=True)
        for index in range(220):
            split = "train" if index < 200 else "val"
            (scene / modality / split / f"{index:04d}.png").write_bytes(b"payload")

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
    assert expected_asset_hash({"part_seg_sha256": "b" * 64}, "part_seg", "/private/mask.png") == "b" * 64
    with pytest.raises(ValueError, match="sha256"):
        expected_asset_hash(view, "depth", "depth/val/0000.png")


def test_physical_sampling_indices_are_exact_and_deterministic() -> None:
    assert deterministic_sample_indices(0, 4) == ()
    assert deterministic_sample_indices(4, 4) == (0, 1, 2, 3)
    assert deterministic_sample_indices(10, 4) == (0, 2, 5, 7)
    assert deterministic_sample_indices(10, 4) == deterministic_sample_indices(10, 4)
    with pytest.raises(ValueError):
        deterministic_sample_indices(10, 0)


def test_authoritative_postbuild_binding_and_mutation_gate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    public_scene, _, _ = _fixture(tmp_path)
    source_root = tmp_path / "source-dataset"
    source_scene = source_root / public_scene.name
    source_scene.mkdir(parents=True)
    (source_scene / "transforms.json").write_text("{}\n", encoding="utf-8")
    source_hashes = {}
    for index in range(400):
        path = source_scene / f"source-{index:04d}.png"
        path.write_bytes(f"source-{index}".encode())
        source_hashes[path.name] = sha256_file(path)
    continuous_hashes = {}
    for index in range(18):
        path = source_scene / f"continuous-{index:04d}.png"
        path.write_bytes(f"continuous-{index}".encode())
        continuous_hashes[path.name] = sha256_file(path)
    part_seg_hashes = {}
    for index in range(20):
        path = tmp_path / f"private-seg-{index:04d}.png"
        path.write_bytes(f"seg-{index}".encode())
        part_seg_hashes[str(path.resolve())] = sha256_file(path)
    builder = tmp_path / "endpoint_proxy.py"
    builder.write_text("# frozen builder\n", encoding="utf-8")
    plan = {"schema": "sealed-plan", "source_root": str(source_root.resolve())}
    plan_path = tmp_path / "manifest.json"
    plan_path.write_text(json.dumps(plan, sort_keys=True) + "\n", encoding="utf-8")

    payload = {
        key: value
        for key, value in audit_public_tree(public_scene).items()
        if key not in {"PROXY_RECEIPT.json", "COMPLETE.json"}
    }
    plan_content_sha = content_sha256(plan)
    payload_tree_sha = content_sha256(payload)
    binding = hashlib.sha256(f"{public_scene.name}:{plan_content_sha}:{payload_tree_sha}".encode()).hexdigest()
    for filename in ("PROXY_RECEIPT.json", "COMPLETE.json"):
        (public_scene / filename).write_text(json.dumps({"binding_id": binding}) + "\n", encoding="utf-8")
    complete = audit_public_tree(public_scene)
    endpoint_views = {
        query_id: {"views": [{"part_seg_status": "builder_geometry_proxy_complete"} for _ in range(10)]}
        for query_id in ("outside_state_0", "outside_state_1")
    }
    seal = {
        "schema": "splart-endpoint-extrapolation-postbuild-seal-v1",
        "scene_id": public_scene.name,
        "binding_id": binding,
        "pass": True,
        "sealed_plan_file_sha256": sha256_file(plan_path),
        "sealed_plan_content_sha256": plan_content_sha,
        "builder_sha256": sha256_file(builder),
        "public_payload_tree_sha256": content_sha256(payload),
        "public_payload_sha256": payload,
        "public_complete_tree_sha256": content_sha256(complete),
        "public_complete_sha256": complete,
        "source_synthesis_sha256": source_hashes,
        "source_transforms_sha256": sha256_file(source_scene / "transforms.json"),
        "continuous_validation_asset_sha256": continuous_hashes,
        "sealed_endpoint_part_seg_sha256": part_seg_hashes,
        "evaluator_truth": {
            "scene_id": public_scene.name,
            "target": {
                "local_scalars": {"outside_state_0": -1.0, "outside_state_1": 2.0},
                "closed_query": "outside_state_0",
            },
            "endpoint_metrics": endpoint_views,
        },
        "quality_gate": {"pass": True},
        "geometry_certificate": {
            "alignment_pass": True,
            "closure": {"identifiable": True, "closed_query": "outside_state_0"},
        },
    }
    monkeypatch.setattr(endpoint_render_eval, "V4_PLAN_FILE_SHA256", sha256_file(plan_path))
    monkeypatch.setattr(endpoint_render_eval, "V4_BUILDER_SHA256", sha256_file(builder))
    seal_path = tmp_path / "postbuild.json"
    seal_path.write_text(json.dumps(seal, sort_keys=True) + "\n", encoding="utf-8")
    loaded, loaded_source = load_postbuild_seal(seal_path, public_scene, builder, plan_path)
    assert loaded["binding_id"] == binding
    assert loaded_source == source_root.resolve()
    (public_scene / "color" / "train" / "0000.png").write_bytes(b"mutated")
    with pytest.raises(ValueError, match="public"):
        load_postbuild_seal(seal_path, public_scene, builder, plan_path)
