import hashlib
import json
from pathlib import Path

import pytest
import torch

from run_glpdt_source_gate import PROFILE_SAMPLES, load_source, validate_index, virtual_profile
from splart.gauge_energy_profile import PROFILE_CHANNELS


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_index(tmp_path: Path) -> Path:
    q = torch.linspace(-2.0, 3.0, 513)
    rows = []
    for index in range(19):
        split = "source_train" if index < 13 else "source_validation"
        object_id = f"source-{index:02d}"
        fields = torch.zeros(2, 2, 3, 513, 9)
        fields[..., 7] = (q.view(1, 1, 1, -1) - (-0.3 if index % 2 == 0 else 1.3)).square()
        artifact = {
            "schema": "source-artifact/v1",
            "object_id": object_id,
            "split": split,
            "q": q,
            "fields": fields,
            "true_endpoints": torch.tensor([-0.35, 1.35]),
            "channels": PROFILE_CHANNELS,
        }
        path = tmp_path / f"{object_id}.pt"
        torch.save(artifact, path)
        rows.append(
            {
                "artifact": str(path) if index == 0 else path.name,
                "artifact_sha256": digest(path),
                "object_id": object_id,
                "shape": list(fields.shape),
                "split": split,
            }
        )
    index_path = tmp_path / "index.json"
    index_path.write_text(
        json.dumps(
            {
                "schema": "splart-source-full-trajectory-index/v1",
                "box_labels_read": False,
                "protected_splits_read": [],
                "channels": list(PROFILE_CHANNELS),
                "rejected": [],
                "rows": rows,
            }
        ),
        encoding="utf-8",
    )
    return index_path


def test_source_loader_accepts_relative_and_absolute_verified_artifacts(tmp_path):
    index_path = make_index(tmp_path)
    payload = json.loads(index_path.read_text())
    payload["box_labels_read"] = []
    index_path.write_text(json.dumps(payload))
    data = load_source(index_path)
    assert data["source_train"][0].shape == (13, 2, 3, PROFILE_SAMPLES, 9)
    assert data["source_validation"][0].shape == (6, 2, 3, PROFILE_SAMPLES, 9)
    assert data["source_train"][4] == tuple(f"source-{index:02d}" for index in range(13))
    features, coordinates, anchor, target = data["source_train"][:4]
    assert torch.isfinite(features).all() and torch.isfinite(coordinates).all()
    assert (anchor >= 0).all() and (target >= 0).all()


def test_virtual_profile_recomputes_normalized_posterior(tmp_path):
    index_path = make_index(tmp_path)
    _, artifact_path = validate_index(index_path)[0]
    artifact = torch.load(artifact_path, weights_only=True)
    features, _, _, _ = virtual_profile(artifact)
    assert torch.allclose(features[..., 8].sum(-1), torch.ones(2, 3), atol=1e-6)


def test_source_index_fails_closed_on_protected_access_and_digest(tmp_path):
    index_path = make_index(tmp_path)
    payload = json.loads(index_path.read_text())
    payload["protected_splits_read"] = ["B_test"]
    index_path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="protected"):
        validate_index(index_path)
    payload["protected_splits_read"] = []
    payload["rows"][0]["artifact_sha256"] = "0" * 64
    index_path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="digest"):
        validate_index(index_path)
