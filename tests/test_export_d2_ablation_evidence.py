from pathlib import Path
from dataclasses import dataclass

import pytest
import torch

from export_d2_ablation_evidence import MODES, _exclusive_json, mode_field_configs, profile_sufficiency, validate_public_rows


def test_legacy_profile_is_not_sufficient_for_nonlinear_ablations() -> None:
    report = profile_sufficiency()
    assert report["exactly_reconstructible"] == {"full": True, "single-radius": True}
    assert set(report["not_exactly_reconstructible"]) == {"no-contact", "no-penetration", "no-terminal-support"}
    # Mean then softmax is not softmax then mean: legacy geometry averaging is lossy.
    a, b = torch.tensor([0.0, 8.0]), torch.tensor([4.0, 0.0])
    assert not torch.allclose(torch.softmax((a + b) / -2, 0), (torch.softmax(-a, 0) + torch.softmax(-b, 0)) / 2)


def test_all_preregistered_modes_have_exact_expected_weights() -> None:
    # Deliberately dependency-independent: this contract test does not import D2.
    @dataclass(frozen=True)
    class Config:
        contact_weight: float = 1.0
        penetration_weight: float = 64.0
        inside_weight: float = 64.0
        support_weight: float = 2.0
    base = Config()
    fixed = (Config(), Config(), Config())
    assert tuple(mode_field_configs(base, fixed, "single-radius")) == (fixed[1],)
    assert all(c.contact_weight == 0 for c in mode_field_configs(base, fixed, "no-contact"))
    assert all(c.penetration_weight == c.inside_weight == 0 for c in mode_field_configs(base, fixed, "no-penetration"))
    assert all(c.support_weight == 0 for c in mode_field_configs(base, fixed, "no-terminal-support"))
    assert set(MODES) == {"full", "single-radius", "no-contact", "no-penetration", "no-terminal-support"}


def test_evidence_writer_is_write_once(tmp_path: Path) -> None:
    target = tmp_path / "evidence.json"
    _exclusive_json(target, {"schema": "x"})
    assert target.read_text().endswith("\n")
    with pytest.raises(FileExistsError):
        _exclusive_json(target, {"schema": "y"})


def _rows() -> list[dict]:
    return [{"object_id": f"ep-{index:02d}", "checkpoint": {}, "config": {},
             "dataparser_transforms": {}} for index in range(36)]


def test_input_is_exact36_minimal_and_path_safe() -> None:
    assert len(validate_public_rows({"rows": _rows()})) == 36
    for bad in (_rows()[:35], _rows() + [_rows()[0]]):
        with pytest.raises(ValueError, match="36|exactly"):
            validate_public_rows({"rows": bad})
    rows = _rows(); rows[0]["object_id"] = "../escape"
    with pytest.raises(ValueError, match="unsafe object_id"):
        validate_public_rows({"rows": rows})


def test_input_recursively_rejects_private_fields() -> None:
    rows = _rows(); rows[0]["checkpoint"] = {"nested": {"target": 1}}
    with pytest.raises(ValueError, match="forbidden private key"):
        validate_public_rows({"rows": rows})


def test_exact36_contract_can_be_deterministically_sharded() -> None:
    rows = validate_public_rows({"rows": _rows()})
    shards = [[row for index, row in enumerate(rows) if index % 5 == shard] for shard in range(5)]
    assert [len(value) for value in shards] == [8, 7, 7, 7, 7]
    assert {row["object_id"] for shard in shards for row in shard} == {row["object_id"] for row in rows}
    rows = _rows(); rows[0]["config"] = {"path": "/x/sealed/y"}
    with pytest.raises(ValueError, match="evaluator-only path"):
        validate_public_rows({"rows": rows})
