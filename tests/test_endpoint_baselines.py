import json
import subprocess
from pathlib import Path

import pytest

from endpoint_eval import aggregate_scenes, evaluate_scene, validate_complete_measurement
from splart.endpoint_baselines import (
    QUERY_IDS,
    BaselinePolicy,
    assert_model_process_boundary,
    build_splart_training_argv,
    make_prediction,
    validate_public_scene,
    validate_query_scalar,
)


def _public_scene(tmp_path: Path) -> Path:
    scene = tmp_path / "100247-Box"
    scene.mkdir()
    transforms = {
        "articulation": {"type": "revolute"},
        "frames": [
            {"file_path": "color/train/0000.png", "state": 0, "local_state": 0},
            {"file_path": "color/train/0100.png", "state": 1, "local_state": 1},
        ],
    }
    queries = {
        "queries": [
            {"query_id": "outside_state_0", "local_direction": -1},
            {"query_id": "outside_state_1", "local_direction": 1},
        ]
    }
    (scene / "transforms.json").write_text(json.dumps(transforms), encoding="utf-8")
    (scene / "endpoint_queries.json").write_text(json.dumps(queries), encoding="utf-8")
    (scene / "PROXY_RECEIPT.json").write_text('{"public": true}\n', encoding="utf-8")
    (scene / "COMPLETE.json").write_text('{"complete": true}\n', encoding="utf-8")
    for modality in ("color", "depth", "part-seg"):
        for split in ("train", "val"):
            (scene / modality / split).mkdir(parents=True)
        for index in range(220):
            split = "train" if index < 200 else "val"
            (scene / modality / split / f"{index:04d}.png").write_bytes(b"public-image-payload")
    return scene


def test_three_policy_scalars_are_frozen() -> None:
    assert [x["predicted_local_scalar"] for x in BaselinePolicy.from_name("observed-span").predictions()] == [0, 1]
    assert [x["predicted_local_scalar"] for x in BaselinePolicy.from_name("symmetric-linear").predictions()] == [
        -0.5,
        1.5,
    ]
    assert [x["predicted_local_scalar"] for x in BaselinePolicy.from_name("splart-middle").predictions()] == [0, 1]


def test_public_scene_and_fresh_training_request(tmp_path: Path) -> None:
    scene = _public_scene(tmp_path)
    summary = validate_public_scene(scene)
    assert summary["state_counts"] == {"0": 1, "1": 1}
    argv = build_splart_training_argv(scene, tmp_path / "models", "100247-Box/run-1")
    assert "--load-dir" not in argv
    assert str(scene.resolve()) in argv
    checkpoint_dir = tmp_path / "run" / "nerfstudio_models"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "step-000024999.ckpt").write_bytes(b"candidate")
    (checkpoint_dir.parent / "config.yml").write_text("candidate: true\n", encoding="utf-8")
    (checkpoint_dir.parent / "dataparser_transforms.json").write_text("{}\n", encoding="utf-8")
    source = tmp_path / "source"
    source.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=source, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=source, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=source, check=True)
    (source / "tracked.txt").write_text("source\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=source, check=True)
    subprocess.run(["git", "commit", "-qm", "source"], cwd=source, check=True)
    prediction = make_prediction(scene, "splart-middle", checkpoint_dir, source)
    assert prediction["training"]["endpoint_physics_enabled"] is False
    assert prediction["training"]["from_scratch"] is True
    assert prediction["closed_prediction"] == {"status": "unknown", "query_id": None}
    assert all(row["predicted_closed"] is None for row in prediction["endpoint_predictions"])
    assert prediction["candidate"]["checkpoint"]["step"] == 24999
    assert len(prediction["candidate"]["checkpoint"]["sha256"]) == 64


@pytest.mark.parametrize(
    "leak_key",
    ["physical_fractions", "local_gt_scalars", "closed_query", "true_joint_limits", "axis", "pivot", "angle", "dist"],
)
def test_public_scene_rejects_evaluator_leakage(tmp_path: Path, leak_key: str) -> None:
    scene = _public_scene(tmp_path)
    transforms = json.loads((scene / "transforms.json").read_text(encoding="utf-8"))
    transforms[leak_key] = [0.25, 0.75]
    (scene / "transforms.json").write_text(json.dumps(transforms), encoding="utf-8")
    with pytest.raises(ValueError, match="evaluator-only"):
        validate_public_scene(scene)


def test_public_scene_rejects_unexpected_sidecar(tmp_path: Path) -> None:
    scene = _public_scene(tmp_path)
    (scene / "true_limits.json").write_text('{"limits": [0, 1]}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected file"):
        validate_public_scene(scene)


def test_public_scene_rejects_endpoint_asset_directory(tmp_path: Path) -> None:
    scene = _public_scene(tmp_path)
    (scene / "endpoint-targets").mkdir()
    with pytest.raises(ValueError, match="unexpected directory"):
        validate_public_scene(scene)


def test_public_scene_requires_exact_payload_count(tmp_path: Path) -> None:
    scene = _public_scene(tmp_path)
    (scene / "color" / "train" / "0000.png").unlink()
    with pytest.raises(ValueError, match="660"):
        validate_public_scene(scene)


def test_launch_boundary_rejects_sealed_path() -> None:
    with pytest.raises(ValueError, match="evaluator-only"):
        assert_model_process_boundary(["ns-train", "--data", "/tmp/sealed-evaluator/scene"])


def test_query_scalars_are_not_clamped() -> None:
    assert validate_query_scalar(-4.25) == -4.25
    assert validate_query_scalar(3.5) == 3.5
    with pytest.raises(ValueError):
        validate_query_scalar(float("nan"))


def test_evaluator_aggregates_limit_render_physics_and_articulation(tmp_path: Path) -> None:
    scene = _public_scene(tmp_path)
    prediction = make_prediction(scene, "observed-span")
    sealed = {
        "episodes": {
            "100247-Box": {
                "scene_id": "100247-Box",
                "target": {
                    "local_scalars": {"outside_state_0": -1.0, "outside_state_1": 2.0},
                    "closed_query": "outside_state_1",
                },
                # These are descriptors/GT parameters, not numeric predictions.
                "endpoint_metrics": {
                    "outside_state_0": {"views": [{"assets": {"color": "private.png"}}]},
                    "outside_state_1": {"views": [{"assets": {"color": "private.png"}}]},
                },
                "articulation": {"type": "revolute", "angle": 1.2},
            }
        }
    }
    measurements = {
        "scene_id": "100247-Box",
        "endpoint_metrics": {
            "outside_state_0": {
                "views": [
                    {"psnr": 30.0, "ssim": 0.9, "lpips": 0.1, "depth_mae": 0.05, "part_seg_ious": [0.8, 0.7, 0.9]}
                ]
            },
            "outside_state_1": {
                "views": [
                    {
                        "psnr": 34.0,
                        "ssim": 0.94,
                        "lpips": 0.06,
                        "depth_mae": 0.03,
                        "static_iou": 0.9,
                        "mobile_iou": 0.8,
                        "background_iou": 1.0,
                    }
                ]
            },
        },
        "physics": {"terminal_contact_valid": True, "penetration_depth": 0.002},
        "articulation": {"valid": True, "axis_error": 0.1},
    }
    result = evaluate_scene(prediction, sealed, measurements)
    metrics = result["metrics"]
    assert metrics["normalized_limit_error_lower"] == pytest.approx(1 / 3)
    assert metrics["normalized_limit_error_upper"] == pytest.approx(1 / 3)
    assert metrics["normalized_limit_error"] == pytest.approx(1 / 3)
    assert metrics["closed_endpoint_accuracy"] == 0
    assert metrics["closed_endpoint_coverage"] == 1
    assert metrics["psnr"] == 32
    assert metrics["static_iou"] == pytest.approx(0.85)
    assert metrics["mobile_iou"] == pytest.approx(0.75)
    assert metrics["background_iou"] == pytest.approx(0.95)
    assert metrics["miou"] == pytest.approx((0.8 + 0.9) / 2)
    assert metrics["terminal_contact_valid"] == 1
    assert metrics["articulation_valid"] == 1
    aggregate = aggregate_scenes([result])
    assert aggregate["macro"]["psnr"] == 32
    assert aggregate["coverage"]["psnr"] == 1


def test_sealed_asset_descriptors_are_not_misread_as_metrics(tmp_path: Path) -> None:
    scene = _public_scene(tmp_path)
    prediction = make_prediction(scene, "observed-span")
    sealed = {
        "episodes": {
            "100247-Box": {
                "target": {
                    "local_scalars": {"outside_state_0": -0.5, "outside_state_1": 1.5},
                    "closed_query": "outside_state_0",
                },
                "endpoint_metrics": {
                    query_id: {"views": [{"assets": {"color": f"{query_id}.png"}}]}
                    for query_id in ("outside_state_0", "outside_state_1")
                },
                "articulation": {"type": "revolute", "angle": 1.2},
            }
        }
    }
    result = evaluate_scene(prediction, sealed)
    assert result["metrics"]["psnr"] is None
    assert "articulation_angle" not in result["metrics"]


def test_splart_middle_closed_side_abstains(tmp_path: Path) -> None:
    scene = _public_scene(tmp_path)
    checkpoint_dir = tmp_path / "run" / "nerfstudio_models"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "step-000024999.ckpt").write_bytes(b"candidate")
    (checkpoint_dir.parent / "config.yml").write_text("candidate: true\n", encoding="utf-8")
    (checkpoint_dir.parent / "dataparser_transforms.json").write_text("{}\n", encoding="utf-8")
    source = tmp_path / "source-abstain"
    source.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=source, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=source, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=source, check=True)
    (source / "tracked.txt").write_text("source\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=source, check=True)
    subprocess.run(["git", "commit", "-qm", "source"], cwd=source, check=True)
    prediction = make_prediction(scene, "splart-middle", checkpoint_dir, source)
    sealed = {
        "scene_id": scene.name,
        "target": {
            "local_scalars": {"outside_state_0": -1.0, "outside_state_1": 2.0},
            "closed_query": "outside_state_0",
        },
    }
    metrics = evaluate_scene(prediction, sealed)["metrics"]
    assert metrics["closed_endpoint_accuracy"] is None
    assert metrics["closed_endpoint_coverage"] == 0


def _complete_measurement() -> dict:
    view = {
        "psnr": 30.0,
        "ssim": 0.9,
        "lpips": 0.1,
        "depth_mae": 0.02,
        "part_seg_ious": [0.8, 0.7, 0.9],
        "candidate_artifacts": {"color.png": "a" * 64, "depth.npy": "b" * 64, "seg.png": "c" * 64},
    }
    certificate = {
        "valid": True,
        "terminal_contact_valid": False,
        "contact_fraction": 0.01,
        "anchor_contact_fraction": 0.01,
        "contact_fraction_gain": 0.0,
        "penetration_fraction": 0.0,
        "penetration_q99_m": 0.0,
        "penetration_depth": 0.0,
        "static_count": 5000,
        "mobile_count": 100,
        "static_sample_count": 4096,
        "mobile_sample_count": 100,
        "sample_cap_per_part": 4096,
        "sampling_rule": "original-index-even-stride-v1",
    }
    return {
        "endpoint_metrics": {query_id: {"views": [dict(view) for _ in range(10)]} for query_id in QUERY_IDS},
        "physics": {"queries": {query_id: dict(certificate) for query_id in QUERY_IDS}},
        "articulation": {"valid": True, "type_correct": True, "axis": 1.0, "pivot": 0.01, "r": 2.0},
    }


def test_complete_measurement_gate_rejects_partial_coverage() -> None:
    measurement = _complete_measurement()
    validate_complete_measurement(measurement, "revolute")
    measurement["endpoint_metrics"]["outside_state_1"]["views"].pop()
    with pytest.raises(ValueError, match="10 views"):
        validate_complete_measurement(measurement, "revolute")
