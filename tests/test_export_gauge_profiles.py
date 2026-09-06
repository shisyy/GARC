import json
from pathlib import Path

import pytest

from export_gauge_profiles import build_handoff, load_handoff, preflight_public_materialization


def _write(tmp_path: Path, value: dict) -> Path:
    path = tmp_path / "handoff.json"
    path.write_text(json.dumps(value))
    return path


def test_public_handoff_requires_unlocked_object_disjoint_train_val(tmp_path: Path) -> None:
    value = {"schema": "splart-gauge-energy-public-episodes/v1", "training_handoff": "UNLOCKED", "episodes": [
        {"object_id": "a", "split": "train"}, {"object_id": "b", "split": "val"}]}
    assert len(load_handoff(_write(tmp_path, value))["episodes"]) == 2


def test_target_free_incremental_handoff_can_defer_split(tmp_path: Path) -> None:
    value = {"schema": "splart-gauge-energy-public-episodes/v1", "training_handoff": "UNLOCKED", "episodes": [
        {"object_id": "a", "split": "unassigned"}]}
    assert load_handoff(_write(tmp_path, value))["episodes"][0]["split"] == "unassigned"


@pytest.mark.parametrize("key", ["joint_limits", "closed_side", "normalized_input_states", "ground_truth"])
def test_model_facing_handoff_rejects_target_leakage(tmp_path: Path, key: str) -> None:
    value = {"schema": "splart-gauge-energy-public-episodes/v1", "training_handoff": "UNLOCKED", "episodes": [
        {"object_id": "a", "split": "train", key: [0, 1]}]}
    with pytest.raises(ValueError, match="forbidden model-facing key"):
        load_handoff(_write(tmp_path, value))


def test_model_facing_handoff_rejects_evaluator_paths(tmp_path: Path) -> None:
    value = {"schema": "splart-gauge-energy-public-episodes/v1", "training_handoff": "UNLOCKED", "episodes": [
        {"object_id": "a", "split": "train", "path": "/x/B_test/y"}]}
    with pytest.raises(ValueError, match="evaluator-only path"):
        load_handoff(_write(tmp_path, value))


def test_real_public_manifest_and_materialization_receipt_join_training_outputs(tmp_path: Path) -> None:
    import hashlib
    public_rows, receipt_rows, trained_rows = [], [], []
    for index in range(12):
        object_id, episode_id = f"obj-{index}", f"ep-{index}"
        episode_dir = tmp_path / "episodes" / episode_id
        episode_dir.mkdir(parents=True)
        (episode_dir / "rgb0.png").write_bytes(b"rgb0")
        (episode_dir / "rgb1.png").write_bytes(b"rgb1")
        transform = episode_dir / "transforms.json"
        transform.write_text(json.dumps({"frames": [
            {"state": 0, "file_path": "rgb0.png"}, {"state": 1, "file_path": "rgb1.png"}]}))
        transform_sha = hashlib.sha256(transform.read_bytes()).hexdigest()
        public_rows.append({"object_id": object_id, "episode_id": episode_id})
        receipt_rows.append({"object_id": object_id, "episode_id": episode_id, "status": "materialized", "transforms_sha256": transform_sha})
        trained_rows.append({"object_id": object_id, "split": "val" if index >= 10 else "train", "checkpoint": {}, "config": {}, "dataparser_transforms": {}})
    public = {"schema": "splart-endpoint-free-order-public/v1", "episodes": public_rows}
    public_path = tmp_path / "public.json"; public_path.write_text(json.dumps(public))
    public_sha = hashlib.sha256(public_path.read_bytes()).hexdigest()
    receipt = {"schema": "splart-order-materialization-receipt/v1", "ready": 12, "blocked": 0,
        "public_manifest_sha256": public_sha, "objects": receipt_rows}
    receipt_path = tmp_path / "receipt.json"; receipt_path.write_text(json.dumps(receipt))
    trained = {"schema": "splart-frozen-d2-training-outputs/v1", "status": "COMPLETE", "objects": trained_rows}
    trained_path = tmp_path / "trained.json"; trained_path.write_text(json.dumps(trained))
    handoff = build_handoff(public_path, receipt_path, trained_path)
    assert len(handoff["episodes"]) == 12
    assert preflight_public_materialization(public_path, receipt_path)["objects"][0]["files"] == 3

    trained["objects"] = trained["objects"][:6]
    trained_path.write_text(json.dumps(trained))
    assert len(build_handoff(public_path, receipt_path, trained_path)["episodes"]) == 6

    public["schema"] = "splart-node7.2-target-free-public-profiles/v1"
    public["profiles"] = public.pop("episodes") + [{"object_id": "not-materialized", "episode_id": "ep72-extra"}]
    public_path.write_text(json.dumps(public))
    receipt["public_manifest_sha256"] = hashlib.sha256(public_path.read_bytes()).hexdigest()
    receipt_path.write_text(json.dumps(receipt))
    assert len(build_handoff(public_path, receipt_path, trained_path)["episodes"]) == 6
