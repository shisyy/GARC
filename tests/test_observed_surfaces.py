"""Synthetic tests of the trusted conversion and strict consumer boundary."""
import ast
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from build_observed_surfaces import export_object
from splart.observed_surfaces import (ARTIFACT_HASH_KEYS, COUNTS, HASH_KEYS, INDEX_SCHEMA, SCHEMA,
                                      load_surfaces, mobile_at, rotate_vertices, swap_surface, validate_surface)
from splart.relative_search import file_sha256, opaque_object_id, write_json_new


def fixture(identity="a", split="source_train"):
    vertices = torch.tensor([[1., 0, 0], [2., 0, 0], [1., 1, 0]], dtype=torch.float64)
    axis, pivot = torch.tensor([0., 0, 1.], dtype=torch.float64), torch.zeros(3, dtype=torch.float64)
    return {"schema": SCHEMA, "object_id": opaque_object_id(identity), "split": split,
            "static_vertices": vertices + torch.tensor([0., 0, 2.]), "static_faces": torch.tensor([[0, 1, 2]]),
            "mobile_vertices": torch.stack((vertices, rotate_vertices(vertices, axis, pivot, .4))),
            "mobile_faces": torch.tensor([[0, 1, 2]]), "axis": axis, "pivot": pivot, "angular_increment": .4,
            "coordinates": torch.linspace(0, 2, 129).repeat(2, 1),
            "provenance": {key: "a" * 64 for key in ARTIFACT_HASH_KEYS}}


def test_exact_swap_and_rigid_replay():
    value = fixture()
    validate_surface(value)
    swapped = swap_surface(value)
    validate_surface(swapped)
    assert swapped["angular_increment"] == -value["angular_increment"]
    assert torch.equal(mobile_at(value, 0, -.0), value["mobile_vertices"][0])
    assert torch.allclose(mobile_at(value, 0, -1), value["mobile_vertices"][1], rtol=0, atol=1e-12)
    for side in (0, 1):
        for distance in (0., .125, 1., 2.):
            assert torch.equal(mobile_at(value, side, distance), mobile_at(swapped, 1 - side, distance))


@pytest.mark.parametrize("key", ["true_endpoints", "lower", "upper", "state0_fraction", "category", "mobile_zero", "absolute_angles"])
def test_extra_or_endpoint_metadata_rejected(key):
    value = fixture()
    value[key] = None
    with pytest.raises(ValueError, match="strict observed-only"):
        validate_surface(value)


@pytest.mark.parametrize("error", ["face", "nan", "motion", "axis", "provenance", "grid"])
def test_corrupted_artifact_rejected(error):
    value = fixture()
    if error == "face":
        value["mobile_faces"][0, 0] = 99
    elif error == "nan":
        value["static_vertices"][0, 0] = float("nan")
    elif error == "motion":
        value["mobile_vertices"][1, 0, 0] += .01
    elif error == "axis":
        value["axis"] *= 2
    elif error == "provenance":
        value["provenance"]["category"] = "forbidden"
    else:
        value["coordinates"][0, -1] += .1
    with pytest.raises(ValueError):
        validate_surface(value)


def test_export_discards_canonical_pose_and_authored_metadata():
    value = fixture()
    canonical = value["mobile_vertices"][0]
    asset = SimpleNamespace(static=SimpleNamespace(vertices=value["static_vertices"], faces=value["static_faces"]),
                            mobile_zero=SimpleNamespace(vertices=canonical, faces=value["mobile_faces"]),
                            axis=value["axis"], pivot=value["pivot"], bundle_sha256="b" * 64,
                            lower=-90, upper=90, frame_center=object(), frame_scale=object())
    renderer = SimpleNamespace(_load_asset=lambda row: asset, _angles=lambda row, q, a: [.2, .6])
    row = {"object_id": "a", "split": "source_train", "state0_fraction": .2, "state1_fraction": .6}
    provenance = {key: "a" * 64 for key in HASH_KEYS}
    output = export_object(renderer, row, value, provenance, "c" * 64)
    assert torch.allclose(output["mobile_vertices"][0], rotate_vertices(canonical, value["axis"], value["pivot"], .2))
    assert not torch.equal(output["mobile_vertices"][0], canonical)
    assert output["angular_increment"] == pytest.approx(.4)
    # An arbitrary authored rest-pose convention is not exported as an input.
    asset.mobile_zero.vertices = rotate_vertices(canonical, value["axis"], value["pivot"], 1.)
    asset.lower, asset.upper = -100, 100
    renderer._angles = lambda row, q, a: [-.8, -.4]
    shifted = export_object(renderer, row, value, provenance, "c" * 64)
    assert torch.allclose(output["mobile_vertices"], shifted["mobile_vertices"], rtol=0, atol=1e-12)
    assert output["angular_increment"] == pytest.approx(shifted["angular_increment"])


def make_export(path):
    path.mkdir()
    (path / "artifacts").mkdir()
    rows = []
    for i in range(19):
        value = fixture(str(i), "source_train" if i < 13 else "source_validation")
        target = path / "artifacts" / f"{value['object_id']}.pt"
        torch.save(value, target)
        rows.append({"object_id": value["object_id"], "split": value["split"],
                     "artifact": f"artifacts/{value['object_id']}.pt", "artifact_sha256": file_sha256(target)})
    index = {"schema": INDEX_SCHEMA, "rows": rows, "provenance": {key: "a" * 64 for key in HASH_KEYS}, "split_counts": COUNTS}
    index_path = path / "index.json"
    write_json_new(index_path, index)
    return index_path, index


def test_complete_surface_cache_and_digest_tampering(tmp_path):
    index_path, index = make_export(tmp_path / "surface")
    assert len(load_surfaces(index_path)[1]) == 19
    target = index_path.parent / index["rows"][0]["artifact"]
    with target.open("ab") as stream:
        stream.write(b"bad")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_surfaces(index_path)


@pytest.mark.parametrize("error", ["count", "duplicate", "path", "extra"])
def test_index_contract_rejects_unapproved_roster_or_paths(tmp_path, error):
    index_path, index = make_export(tmp_path / "surface")
    if error == "count":
        index["rows"].pop()
    elif error == "duplicate":
        index["rows"][1] = index["rows"][0]
    elif error == "path":
        index["rows"][0]["artifact"] = "../escaped.pt"
    else:
        index["category"] = "forbidden"
    index_path.write_text(json.dumps(index), encoding="utf-8")
    with pytest.raises(ValueError):
        load_surfaces(index_path)


def test_consumer_does_not_import_trusted_asset_exporter():
    tree = ast.parse((ROOT / "src/splart/observed_surfaces.py").read_text(encoding="utf-8"))
    imports = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    assert imports == ["__future__", "pathlib", "splart.relative_search"]
    strings = [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)]
    assert "true_endpoints" not in strings
    assert "mobile_zero" not in strings


def test_base_hash_join_rejects_valid_but_wrong_surface_identity(tmp_path):
    index_path, index = make_export(tmp_path / "surface")
    base_dir = tmp_path / "base"
    base_dir.mkdir()
    base_rows = []
    for i, row in enumerate(index["rows"]):
        d = torch.linspace(0, 2, 129).repeat(2, 1)
        embeddings = torch.zeros(2, 129, 3, 2)
        embeddings[..., 0] = 1
        base = {"schema": "relative-search-input/v1", "object_id": row["object_id"], "split": row["split"],
                "coordinates": d, "geometry": torch.zeros(1, 2, 1, 129, 8), "image_embeddings": embeddings,
                "observed_embeddings": embeddings[:, 0].clone(), "text_direction": torch.tensor([1., 0.]),
                "provenance": {"source_index_sha256": "0" * 64, "geometry_artifact_sha256": "1" * 64,
                               "semantic_artifact_sha256": "2" * 64}}
        target = base_dir / f"{i}.pt"
        torch.save(base, target)
        base_rows.append({"object_id": row["object_id"], "split": row["split"], "artifact": target.name,
                          "artifact_sha256": file_sha256(target)})
    base_index = base_dir / "index.json"
    write_json_new(base_index, {"schema": "relative-search-input-index/v1", "rows": base_rows,
                                "source_index_sha256": "0" * 64, "diagnostic": "synthetic"})
    index["provenance"]["base_input_index_sha256"] = file_sha256(base_index)
    for row, base_row in zip(index["rows"], base_rows):
        target = index_path.parent / row["artifact"]
        value = torch.load(target, weights_only=True)
        value["provenance"]["base_input_index_sha256"] = file_sha256(base_index)
        value["provenance"]["base_input_artifact_sha256"] = base_row["artifact_sha256"]
        torch.save(value, target)
        row["artifact_sha256"] = file_sha256(target)
    index_path.write_text(json.dumps(index), encoding="utf-8")
    assert len(load_surfaces(index_path, base_index)[1]) == 19
    row = index["rows"][0]
    value = torch.load(index_path.parent / row["artifact"], weights_only=True)
    value["object_id"] = opaque_object_id("wrong-but-valid-identity")
    row["object_id"] = value["object_id"]
    row["artifact"] = f"artifacts/{value['object_id']}.pt"
    torch.save(value, index_path.parent / row["artifact"])
    row["artifact_sha256"] = file_sha256(index_path.parent / row["artifact"])
    index_path.write_text(json.dumps(index), encoding="utf-8")
    with pytest.raises(ValueError, match="surface/base roster"):
        load_surfaces(index_path, base_index)
