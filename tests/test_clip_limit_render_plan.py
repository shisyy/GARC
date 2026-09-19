import copy
import hashlib
import json

import pytest

import prepare_clip_limit_render_plan as planner


def njc_preregister():
    return {
        "schema": "splart-pilc-preregister-v1",
        "object_split": {
            "train": [f"train-{index:02d}" for index in range(11)],
            "calibration": [f"validation-{index:02d}" for index in range(4)],
            "target_evaluation_only": [],
        },
        "episode_grid": {
            "state0_fraction": [0.15, 0.25, 0.35],
            "state1_fraction": [0.65, 0.75, 0.85],
            "minimum_span": 0.35,
            "axis_orientation_augmentation": [1, -1],
            "state_order_augmentation": ["forward", "reverse"],
        },
    }


def test_njc_reconstructs_exact_480_public_pose_rows():
    rows = planner.reconstruct_njc_rows(njc_preregister())
    assert len(rows) == len({row["episode_id"] for row in rows}) == 480
    assert sum(row["split"] == "source_train" for row in rows) == 352
    assert sum(row["split"] == "source_validation" for row in rows) == 128
    first = rows[0]
    expected = hashlib.sha256(b"splart-pilc-v1:train:train-00:0.15:0.65:axis+1:forward").hexdigest()
    assert first["episode_id"] == expected
    assert first["state0_fraction"] == 0.15 and first["state1_fraction"] == 0.65
    reverse = rows[1]
    assert reverse["state_order"] == "reverse"
    assert reverse["state0_fraction"] == 0.65 and reverse["state1_fraction"] == 0.15
    assert reverse["joint_selector"] == {"kind": "exact_name", "name": "lid_hinge"}


def test_plan_fails_closed_on_missing_extra_or_tampered_pose_metadata(tmp_path, monkeypatch):
    preregister_path = tmp_path / "preregister.json"
    preregister_path.write_text(json.dumps(njc_preregister()))
    source_index = tmp_path / "index.json"
    source_index.write_text("source-index")
    public_rows = planner.reconstruct_njc_rows(njc_preregister())
    identities = {(row["split"], row["object_id"], row["episode_id"]) for row in public_rows}
    monkeypatch.setattr(planner, "_geometry_identities", lambda *_: identities)
    monkeypatch.setattr(planner, "NJC_PREREGISTER_SHA256", planner.sha256_file(preregister_path))
    output = tmp_path / "plan.json"
    plan = planner.prepare(source_index, "njc", output, preregister_path=preregister_path)
    assert plan["rows"] == sorted(public_rows, key=lambda row: (row["split"], row["object_id"], row["episode_id"]))

    missing = copy.deepcopy(plan)
    missing["rows"][0].pop("state0_fraction")
    with pytest.raises(ValueError, match="missing or extra"):
        planner.validate_plan(missing, source_index, "njc", preregister_path=preregister_path)

    extra = copy.deepcopy(plan)
    extra["rows"][0]["physical_range"] = 1.0
    with pytest.raises(ValueError, match="missing or extra"):
        planner.validate_plan(extra, source_index, "njc", preregister_path=preregister_path)

    tampered = copy.deepcopy(plan)
    tampered["rows"][0]["state0_fraction"] += 0.01
    with pytest.raises(ValueError, match="differs from public reconstruction"):
        planner.validate_plan(tampered, source_index, "njc", preregister_path=preregister_path)


def test_articraft_plan_reproduces_object_hash_episode_and_joint_rule(tmp_path, monkeypatch):
    archives = [f"rec_object_{index:02d}.tar.gz" for index in range(19)]
    selection = {"selection": {"endpoint_pretrain": archives[:13], "endpoint_validation": archives[13:]}}
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(json.dumps(selection))
    source_index = tmp_path / "index.json"
    source_index.write_text("source-index")
    public_rows = planner.reconstruct_articraft_rows(selection)
    identities = {(row["split"], row["object_id"], row["episode_id"]) for row in public_rows}
    monkeypatch.setattr(planner, "_geometry_identities", lambda *_: identities)
    monkeypatch.setattr(planner, "ART_SELECTION_SHA256", planner.sha256_file(selection_path))
    plan = planner.prepare(source_index, "articraft", tmp_path / "plan.json", selection_path=selection_path)
    row = plan["rows"][0]
    assert (row["state0_fraction"], row["state1_fraction"]) == planner.articraft_fractions(row["object_id"])
    assert row["joint_selector"] == {"kind": "first_bounded_revolute_document_order"}
    assert row["episode_id"] == row["object_id"]


def test_plan_independently_rejects_nonfixed_views_and_render_shape(tmp_path, monkeypatch):
    preregister_path = tmp_path / "preregister.json"
    preregister_path.write_text(json.dumps(njc_preregister()))
    source_index = tmp_path / "index.json"
    source_index.write_text("source-index")
    rows = planner.reconstruct_njc_rows(njc_preregister())
    identities = {(row["split"], row["object_id"], row["episode_id"]) for row in rows}
    monkeypatch.setattr(planner, "_geometry_identities", lambda *_: identities)
    monkeypatch.setattr(planner, "NJC_PREREGISTER_SHA256", planner.sha256_file(preregister_path))
    frozen = planner.prepare(source_index, "njc", tmp_path / "plan.json", preregister_path=preregister_path)

    wrong_view = copy.deepcopy(frozen)
    wrong_view["views"][0] = [1, 0]
    with pytest.raises(ValueError, match="six-view"):
        planner.validate_plan(wrong_view, source_index, "njc", preregister_path=preregister_path)

    wrong_shape = copy.deepcopy(frozen)
    wrong_shape["required_render_shape"][-1] = 256
    with pytest.raises(ValueError, match="224"):
        planner.validate_plan(wrong_shape, source_index, "njc", preregister_path=preregister_path)
