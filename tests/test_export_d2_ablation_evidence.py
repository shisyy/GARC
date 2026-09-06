from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

from export_d2_ablation_evidence import MODES, _exclusive_json, mode_field_configs, profile_sufficiency
from splart.contact_endpoint_field import EndpointFieldConfig
from splart.endpoint_adapter import fixed_multiradius_counterfactuals


def test_legacy_profile_is_not_sufficient_for_nonlinear_ablations() -> None:
    report = profile_sufficiency()
    assert report["exactly_reconstructible"] == {"full": True, "single-radius": True}
    assert set(report["not_exactly_reconstructible"]) == {"no-contact", "no-penetration", "no-terminal-support"}
    # Mean then softmax is not softmax then mean: legacy geometry averaging is lossy.
    a, b = torch.tensor([0.0, 8.0]), torch.tensor([4.0, 0.0])
    assert not torch.allclose(torch.softmax((a + b) / -2, 0), (torch.softmax(-a, 0) + torch.softmax(-b, 0)) / 2)


def test_all_preregistered_modes_have_exact_expected_weights() -> None:
    base = EndpointFieldConfig()
    fixed = fixed_multiradius_counterfactuals(base)
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
