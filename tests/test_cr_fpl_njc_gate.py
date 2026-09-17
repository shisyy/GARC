import hashlib
import json
from pathlib import Path

import pytest
import torch

import build_njc_full_trajectory_profiles as builder
import run_cr_fpl_njc_gate as njc_gate
from run_cr_fpl_source_gate import CONFIG as SOURCE_CONFIG
from splart.cr_fpl import CRFPLHead


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_obj(path: Path, offset: float) -> None:
    path.write_text(
        "\n".join((f"v {offset} 0 0", f"v {offset + 1} 0 0", f"v {offset} 1 0", f"v {offset} 0 1")) + "\n",
        encoding="utf-8",
    )


def make_fixture(tmp_path: Path) -> tuple[dict[str, Path], dict[str, str]]:
    train = [f"train-{index:02d}" for index in range(11)]
    validation = [f"validation-{index:02d}" for index in range(4)]
    prereg = {
        "schema": "splart-pilc-preregister-v1",
        "object_split": {"train": train, "calibration": validation, "target_evaluation_only": []},
        "episode_grid": {
            "state0_fraction": [0.15, 0.25, 0.35],
            "state1_fraction": [0.65, 0.75, 0.85],
            "minimum_span": 0.35,
            "axis_orientation_augmentation": [1, -1],
            "state_order_augmentation": ["forward", "reverse"],
        },
    }
    prereg_path = tmp_path / "preregister.json"
    prereg_path.write_text(json.dumps(prereg), encoding="utf-8")
    assets = tmp_path / "assets"
    asset_receipts = []
    ranges = {}
    for index, object_id in enumerate(train + validation):
        root = assets / object_id
        root.mkdir(parents=True)
        write_obj(root / "base_final.obj", 0.0)
        write_obj(root / "lid_final.obj", 0.2)
        physical_range = 1.0 + index * 0.01
        ranges[object_id] = physical_range
        (root / "object.urdf").write_text(
            f"""<robot name="asset">
  <link name="base"><visual><geometry><mesh filename="base_final.obj" scale="1 1 1"/></geometry></visual></link>
  <link name="lid"><visual><geometry><mesh filename="lid_final.obj" scale="1 1 1"/></geometry></visual></link>
  <joint name="hinge" type="revolute"><parent link="base"/><child link="lid"/>
    <origin xyz="0 0 0" rpy="0 0 0"/><axis xyz="0 0 1"/>
    <limit lower="0" upper="{physical_range}"/></joint>
</robot>\n""",
            encoding="utf-8",
        )
        (root / "qc.json").write_text('{"status":"PASS"}\n', encoding="utf-8")
        asset_receipts.append(
            {
                "name": object_id,
                "urdf_sha256": digest(root / "object.urdf"),
                "base_sha256": digest(root / "base_final.obj"),
                "mobile_sha256": digest(root / "lid_final.obj"),
                "qc_sha256": digest(root / "qc.json"),
            }
        )
    inputs, labels = [], []
    for split, objects in (("train", train), ("calibration", validation)):
        for object_id in objects:
            for f0 in prereg["episode_grid"]["state0_fraction"]:
                for f1 in prereg["episode_grid"]["state1_fraction"]:
                    if f1 - f0 < prereg["episode_grid"]["minimum_span"]:
                        continue
                    for orientation in (1, -1):
                        base = f"{split}:{object_id}:{f0:.2f}:{f1:.2f}:axis{orientation:+d}"
                        for order in ("forward", "reverse"):
                            key = hashlib.sha256(f"splart-pilc-v1:{base}:{order}".encode()).hexdigest()
                            forward = order == "forward"
                            extension0 = f0 / (f1 - f0)
                            extension1 = (1.0 - f1) / (f1 - f0)
                            inputs.append(
                                {
                                    "key": key,
                                    "split": split,
                                    "state0_features": torch.zeros(47),
                                    "state1_features": torch.zeros(47),
                                    "observed_displacement": (
                                        (f1 - f0) * ranges[object_id] * orientation * (1 if forward else -1)
                                    ),
                                }
                            )
                            labels.append(
                                {
                                    "key": key,
                                    "extension0": extension0 if forward else extension1,
                                    "extension1": extension1 if forward else extension0,
                                    "closed_index": 0 if forward else 1,
                                    "physical_range": ranges[object_id],
                                }
                            )
    inputs_path, labels_path = tmp_path / "inputs.pt", tmp_path / "labels.pt"
    torch.save(inputs, inputs_path)
    torch.save(labels, labels_path)
    manifest = {
        "schema": "splart-pilc-njc-features-v1",
        "preregister_sha256": digest(prereg_path),
        "materialized": {"train": train, "calibration": validation},
        "missing": [],
        "rows": 480,
        "feature_dim": 47,
        "input_fields": sorted(inputs[0]),
        "label_fields": sorted(labels[0]),
        "separate_sidecars": True,
    }
    manifest["canonical_sha256"] = builder._canonical_sha256(manifest)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    hashes = {
        "preregister": digest(prereg_path),
        "inputs": digest(inputs_path),
        "labels": digest(labels_path),
        "manifest": digest(manifest_path),
    }
    qc = {
        "schema": "splart-pilc-njc-audit-v1",
        "status": "PASS",
        "protected_splits_read": [],
        "box_extra_scores_read": [],
        "assets": asset_receipts,
        **{f"{name}_sha256": value for name, value in hashes.items()},
    }
    qc_path = tmp_path / "qc-receipt.json"
    qc_path.write_text(json.dumps(qc), encoding="utf-8")
    hashes["qc_receipt"] = digest(qc_path)
    paths = {
        "asset_root": assets,
        "prereg_path": prereg_path,
        "inputs_path": inputs_path,
        "labels_path": labels_path,
        "manifest_path": manifest_path,
        "qc_receipt_path": qc_path,
    }
    return paths, hashes


def test_materializes_exact_11_4_episode_contract_and_loads(tmp_path, monkeypatch):
    paths, hashes = make_fixture(tmp_path)
    output = tmp_path / "profiles"
    result = builder.build_dataset(
        **paths, output=output, device_name="cpu", expected_hashes=hashes, samples=17, points=4
    )
    assert result["episodes"] == {"source_train": 352, "source_validation": 128}
    index = json.loads((output / "index.json").read_text())
    assert len(index["rows"]) == 480
    assert set(index["rows"][0]) == njc_gate.ROW_KEYS
    monkeypatch.setattr(njc_gate, "AUTHORIZED_HASHES", hashes)
    data = njc_gate.load_njc(output / "index.json", expected_samples=17)
    assert data["source_train"][0].shape == (352, 2, 3, 129, 9)
    assert data["source_validation"][0].shape == (128, 2, 3, 129, 9)
    assert len(set(data["source_train"][4])) == 11
    assert len(set(data["source_validation"][4])) == 4


def test_njc_index_fails_closed_on_digest(tmp_path, monkeypatch):
    paths, hashes = make_fixture(tmp_path)
    output = tmp_path / "profiles"
    builder.build_dataset(**paths, output=output, device_name="cpu", expected_hashes=hashes, samples=17, points=4)
    monkeypatch.setattr(njc_gate, "AUTHORIZED_HASHES", hashes)
    index_path = output / "index.json"
    payload = json.loads(index_path.read_text())
    payload["rows"][0]["artifact_sha256"] = "0" * 64
    index_path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="digest"):
        njc_gate.validate_njc_index(index_path, expected_samples=17)


def test_njc_gate_reuses_source_config_and_reports_object_macro():
    assert njc_gate.CONFIG is SOURCE_CONFIG
    assert (njc_gate.CONFIG.seed, njc_gate.CONFIG.steps) == (2202, 4000)
    generator = torch.Generator().manual_seed(816)
    features = torch.randn(8, 2, 3, 7, 9, generator=generator)
    coordinates = torch.linspace(0.05, 2.0, 7).view(1, 1, 1, 7).expand(8, 2, 3, -1)
    anchor = torch.rand(8, 2, generator=generator) + 0.1
    target = torch.rand(8, 2, generator=generator) + 0.1
    output = CRFPLHead()(features, coordinates, anchor)
    metrics = njc_gate.summarize_metrics(output, anchor, target)
    njc_gate.add_object_macro_metrics(metrics, output, anchor, target, ("a",) * 4 + ("b",) * 4)
    assert metrics["cr_fpl_object_macro_endpoint_nmae"] >= 0.0
    assert len(metrics["cr_fpl_object_macro_side_nmae"]) == 2
