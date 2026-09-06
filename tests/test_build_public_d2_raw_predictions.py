import json
from pathlib import Path

import pytest

from build_public_d2_raw_predictions import canonical_sha, opaque_id, reject_private, scalar_to_distance, write_exclusive


def test_scalar_to_nonnegative_observation_distance() -> None:
    assert scalar_to_distance(-0.25, 1.75) == [0.25, 0.75]
    for values in ((0.1, 1.5), (-0.1, 0.9), (float("nan"), 2.0)):
        with pytest.raises(ValueError, match="invalid"):
            scalar_to_distance(*values)


def test_canonical_config_hash_is_order_independent() -> None:
    assert canonical_sha({"a": 1, "b": 2}) == canonical_sha({"b": 2, "a": 1})


def test_source_names_are_replaced_by_opaque_ids() -> None:
    value = opaque_id("rec_sewing_box_with_simple_hinged_lid_0001")
    assert value.startswith("obj-") and len(value) == 28
    assert "sewing" not in value
    with pytest.raises(ValueError, match="unsafe"):
        opaque_id("../escape")


@pytest.mark.parametrize("key", ["target", "split", "membership", "aggregate", "score"])
def test_recursive_private_key_rejection(key: str) -> None:
    with pytest.raises(ValueError, match="forbidden key"):
        reject_private({"nested": [{key: 1}]})


def test_write_is_exclusive_and_read_only(tmp_path: Path) -> None:
    path = tmp_path / "raw.json"
    write_exclusive(path, {"x": 1})
    assert json.loads(path.read_text()) == {"x": 1}
    with pytest.raises(FileExistsError):
        write_exclusive(path, {"x": 2})
