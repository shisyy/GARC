import json
import subprocess
from pathlib import Path

import build_node72_sealed_truth as builder
import verify_node72_sealed_truth as verifier


ROOT = Path(__file__).resolve().parents[1]


def test_increasing_axis_local_endpoint_and_nmae_formula():
    joint = {"lower": -2.0, "upper": 4.0}
    target = builder.local_target(joint, 0.0, 2.0)
    assert target["local_direction_mapping"] == {
        "local_lower_physical_side": "lower",
        "local_upper_physical_side": "upper",
    }
    assert target["local_endpoint_targets"]["local_lower_scalar"] == -1.0
    assert target["local_endpoint_targets"]["local_upper_scalar"] == 2.0
    assert target["target_nmae_normalization"]["local_scalar_error_multiplier"] == 1.0 / 3.0
    independent = verifier.expected_endpoint(joint, 0.0, 2.0)
    assert independent["direction"] == ("lower", "upper")
    assert independent["scalars"] == (-1.0, 2.0)


def test_decreasing_axis_swaps_physical_direction_without_rewriting_limits():
    joint = {"lower": -2.0, "upper": 4.0}
    target = builder.local_target(joint, 2.0, 0.0)
    assert target["local_direction_mapping"] == {
        "local_lower_physical_side": "upper",
        "local_upper_physical_side": "lower",
    }
    assert target["local_endpoint_targets"]["local_lower_scalar"] == -1.0
    assert target["local_endpoint_targets"]["local_upper_scalar"] == 2.0
    independent = verifier.expected_endpoint(joint, 2.0, 0.0)
    assert independent["direction"] == ("upper", "lower")
    assert independent["scalars"] == (-1.0, 2.0)


def test_authored_upper_fraction_moves_toward_original_lower_limit():
    joint = {"lower": -3.0, "upper": 1.0, "zero_relation": "upper"}
    assert builder.state_from_authored_fraction(joint, 0.25) == 0.0
    assert builder.state_from_authored_fraction(joint, 0.75) == -2.0


def test_committed_receipts_are_aggregate_only_and_manifest_is_absent():
    receipts = [
        ROOT / "node72_sealed_truth_build_receipt.json",
        ROOT / "node72_sealed_truth_verification_receipt.json",
    ]
    forbidden_keys = {
        "episodes", "episode_id", "object_id", "membership", "split_rank",
        "split_order_key", "primary_joint", "joint_limits", "lower", "upper",
        "fraction", "fractions", "presentation_bit", "state0_q", "state1_q",
        "endpoint_truth", "target", "targets",
    }

    def keys(value):
        if isinstance(value, dict):
            return set(value) | {key for child in value.values() for key in keys(child)}
        if isinstance(value, list):
            return {key for child in value for key in keys(child)}
        return set()

    for path in receipts:
        payload = json.loads(path.read_text())
        assert not (keys(payload) & forbidden_keys)
        assert payload["object_count"] == 36
        assert payload["split_counts"] == {"train": 18, "calibration": 9, "confirmatory": 9}

    tracked = subprocess.check_output(
        ["git", "ls-files"], cwd=ROOT, text=True
    ).splitlines()
    assert not any(Path(path).name == "sealed-truth-v1.json" for path in tracked)

