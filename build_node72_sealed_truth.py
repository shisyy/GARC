#!/usr/bin/env python3
"""Build node7.2 endpoint truth and object-disjoint split in a sealed root.

The command intentionally prints only a sanitized receipt summary. Per-object
limits, state fractions, presentation directions, targets, and split membership
are written exclusively to the mode-0600 sealed manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
import tarfile
import tempfile
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path, PurePosixPath


PREREG_SHA256 = "8d00e654a990e2d96b912164bf4c5aaaefc7159304aeceaf3544e0f4ea6d85cc"
INVENTORY_SHA256 = "83bc4abac1a7c9761e7e1f37d4c723e16c30c3ed6cc5409a2f8795dda42c5880"
PUBLIC_MANIFEST_SHA256 = "79c0f575135fed75d92d6837ec78e7aeb7ebd56b19f827d2880639802f4eb61f"
NODE71_PUBLIC_SHA256 = "b10b61f7022f81a0b6e62c7fc03a7b37f376b40dd6dc67b12baec1d664e41274"
NODE71_PREREG_SHA256 = "6b6e0e2dde74c74a22c85c42d2909248d4e6035e0ac217015d58c97c860246fc"
NODE71_MAPPING_SHA256 = "8ddedecb2c86b19bb443976728dc59a4f0397f33d0e3d9db18c0c7d559424c6a"
NODE71_PRESENTATION_SALT_SHA256 = "1592983de07a9a595730f754154356e724ecd984fdc8e2269c20226aaed629a9"
NODE72_SPLIT_SALT_SHA256 = "7d94832908e12ec238b4abba232b140214e47cda6f1d2074c95a27f919c600c9"
FORMULA_VERSION = "splart-node7.2-local-endpoint-truth/v1"
SELECTION_DOMAIN = "node7.1-object-selection-v1"


class BuildFailure(RuntimeError):
    """A deliberately non-secret-bearing build failure."""


def fail(code: str) -> None:
    raise BuildFailure(code)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def hash_hex(domain: str, source: str, object_id: str) -> str:
    return hashlib.sha256(f"{domain}|{source}|{object_id}".encode()).hexdigest()


def selection_key(row: dict) -> str:
    return hash_hex(SELECTION_DOMAIN, row["source"], row["object_id"])


def eligible(row: dict) -> bool:
    joint = row.get("primary_joint") or {}
    lower, upper = joint.get("lower"), joint.get("upper")
    return (
        row.get("error") is None
        and row.get("visual_geometry_count", 0) > 0
        and row.get("primary_candidate_count") == 1
        and joint.get("type") in {"revolute", "prismatic"}
        and isinstance(lower, (int, float))
        and isinstance(upper, (int, float))
        and math.isfinite(lower)
        and math.isfinite(upper)
        and upper > lower
        and joint.get("zero_relation") in {"lower", "upper"}
    )


def band_fraction(domain: str, source: str, object_id: str, lo: float, hi: float) -> float:
    raw = bytes.fromhex(hash_hex(domain, source, object_id))[:8]
    unit = int.from_bytes(raw, "big") / float((1 << 64) - 1)
    return lo + (hi - lo) * unit


def state_from_authored_fraction(joint: dict, fraction: float) -> float:
    lower, upper = joint["lower"], joint["upper"]
    if joint["zero_relation"] == "lower":
        return lower + fraction * (upper - lower)
    if joint["zero_relation"] == "upper":
        return upper - fraction * (upper - lower)
    fail("invalid_canonical_side")


def parse_primary_joint(payload: bytes) -> dict:
    root = ET.fromstring(payload)
    links = {node.get("name") for node in root.findall("link")}
    children = {
        node.find("child").get("link")
        for node in root.findall("joint")
        if node.find("child") is not None
    }
    root_links = links - children
    joints = []
    for node in root.findall("joint"):
        if node.get("type") not in {"revolute", "prismatic"} or node.find("mimic") is not None:
            continue
        parent = node.find("parent")
        child = node.find("child")
        limit = node.find("limit")
        if parent is None or child is None or limit is None:
            continue
        try:
            lower, upper = float(limit.get("lower")), float(limit.get("upper"))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(lower + upper) or upper <= lower:
            continue
        joints.append({
            "name": node.get("name"),
            "type": node.get("type"),
            "parent": parent.get("link"),
            "child": child.get("link"),
            "lower": lower,
            "upper": upper,
        })
    rooted = sorted((joint for joint in joints if joint["parent"] in root_links), key=lambda x: x["name"])
    candidates = rooted or sorted(joints, key=lambda x: x["name"])
    if not candidates:
        fail("original_urdf_has_no_primary_joint")
    return candidates[0]


def read_original_urdf(row: dict, archive_root: Path, njc_root: Path) -> tuple[bytes, str]:
    if row["source"] == "articraft":
        archive = archive_root / f"{row['object_id']}.tar.gz"
        if not archive.is_file() or sha256(archive) != row.get("archive_sha256"):
            fail("articraft_asset_identity_mismatch")
        with tarfile.open(archive, "r:gz") as handle:
            members = []
            for member in handle.getmembers():
                member_path = PurePosixPath(member.name)
                if member_path.is_absolute() or ".." in member_path.parts or member.issym() or member.islnk():
                    fail("unsafe_articraft_archive")
                if member.isfile() and member.name.lower().endswith(".urdf"):
                    members.append(member)
            if len(members) != 1:
                fail("articraft_urdf_cardinality")
            extracted = handle.extractfile(members[0])
            if extracted is None:
                fail("articraft_urdf_unreadable")
            payload = extracted.read()
        return payload, hashlib.sha256(payload).hexdigest()
    if row["source"] == "njc":
        asset = njc_root / row["object_id"]
        urdfs = sorted(asset.rglob("*.urdf")) if asset.is_dir() else []
        if len(urdfs) != 1 or sha256(urdfs[0]) != row.get("urdf_sha256"):
            fail("njc_asset_identity_mismatch")
        return urdfs[0].read_bytes(), sha256(urdfs[0])
    fail("unsupported_source")


def close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-10)


def assert_original_joint_matches(record: dict, parsed: dict) -> None:
    cached = record["primary_joint"]
    if (
        cached["name"] != parsed["name"]
        or cached["type"] != parsed["type"]
        or cached["parent"] != parsed["parent"]
        or cached["child"] != parsed["child"]
        or not close(cached["lower"], parsed["lower"])
        or not close(cached["upper"], parsed["upper"])
    ):
        fail("inventory_original_urdf_joint_mismatch")
    zero_relation = "lower" if close(parsed["lower"], 0.0) else "upper" if close(parsed["upper"], 0.0) else None
    if cached["zero_relation"] != zero_relation:
        fail("inventory_original_urdf_canonical_mismatch")


def local_target(joint: dict, q0: float, q1: float) -> dict:
    lower, upper = joint["lower"], joint["upper"]
    span = upper - lower
    delta = q1 - q0
    observed = abs(delta)
    if not lower < q0 < upper or not lower < q1 < upper or observed <= 0:
        fail("non_interior_or_degenerate_observations")
    if delta > 0:
        local_lower_q, local_upper_q = lower, upper
        local_lower_side, local_upper_side = "lower", "upper"
    else:
        local_lower_q, local_upper_q = upper, lower
        local_lower_side, local_upper_side = "upper", "lower"
    extension0_units = abs(q0 - local_lower_q)
    extension1_units = abs(local_upper_q - q1)
    extension0_local = extension0_units / observed
    extension1_local = extension1_units / observed
    local_lower_scalar = -extension0_local
    local_upper_scalar = 1.0 + extension1_local
    scale = observed / span
    physical_distances = {
        "lower": {
            "state0": q0 - lower,
            "state1": q1 - lower,
            "nearest_observation": min(q0, q1) - lower,
        },
        "upper": {
            "state0": upper - q0,
            "state1": upper - q1,
            "nearest_observation": upper - max(q0, q1),
        },
    }
    target = {
        "observed_state_q": {"state0": q0, "state1": q1},
        "observed_signed_displacement": delta,
        "observed_displacement_abs": observed,
        "physical_endpoint_distances_joint_units": physical_distances,
        "physical_endpoint_distances_normalized_by_range": {
            side: {key: value / span for key, value in distances.items()}
            for side, distances in physical_distances.items()
        },
        "local_direction_mapping": {
            "local_lower_physical_side": local_lower_side,
            "local_upper_physical_side": local_upper_side,
        },
        "local_endpoint_targets": {
            "extension0_joint_units": extension0_units,
            "extension1_joint_units": extension1_units,
            "extension0_observation_units": extension0_local,
            "extension1_observation_units": extension1_local,
            "local_lower_scalar": local_lower_scalar,
            "local_upper_scalar": local_upper_scalar,
        },
        "target_nmae_normalization": {
            "physical_range": span,
            "local_scalar_error_multiplier": scale,
            "per_endpoint_formula": "abs(predicted_local_scalar-target_local_scalar)*local_scalar_error_multiplier",
            "object_score_formula": "max(lower_endpoint_nmae,upper_endpoint_nmae)",
        },
    }
    # Independent reconstruction in joint units verifies direction and scaling.
    reconstructed_lower = q0 + local_lower_scalar * delta
    reconstructed_upper = q0 + local_upper_scalar * delta
    if not close(reconstructed_lower, local_lower_q) or not close(reconstructed_upper, local_upper_q):
        fail("local_direction_reconstruction_failed")
    if not close(extension0_local * scale, extension0_units / span):
        fail("lower_nmae_normalization_failed")
    if not close(extension1_local * scale, extension1_units / span):
        fail("upper_nmae_normalization_failed")
    return target


def mode(path: Path) -> str:
    return oct(stat.S_IMODE(path.stat().st_mode))[2:]


def write_temp_json(root: Path, value: dict, prefix: str) -> Path:
    descriptor, name = tempfile.mkstemp(prefix=prefix, suffix=".tmp", dir=root)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        return Path(name)
    except Exception:
        os.close(descriptor)
        Path(name).unlink(missing_ok=True)
        raise


def publish_no_overwrite(temp: Path, final: Path) -> None:
    try:
        os.link(temp, final)
    except FileExistsError:
        fail("sealed_output_exists")
    finally:
        temp.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preregister", required=True, type=Path)
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--public-manifest", required=True, type=Path)
    parser.add_argument("--node71-public-manifest", required=True, type=Path)
    parser.add_argument("--node71-preregister", required=True, type=Path)
    parser.add_argument("--node71-sealed-mapping", required=True, type=Path)
    parser.add_argument("--node71-presentation-salt", required=True, type=Path)
    parser.add_argument("--split-salt", required=True, type=Path)
    parser.add_argument("--archive-root", required=True, type=Path)
    parser.add_argument("--njc-root", required=True, type=Path)
    parser.add_argument("--sealed-root", required=True, type=Path)
    parser.add_argument("--manifest-name", default="sealed-truth-v1.json")
    parser.add_argument("--receipt-name", default="sealed-truth-v1-receipt.json")
    args = parser.parse_args()

    args.sealed_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(args.sealed_root, 0o700)
    if mode(args.sealed_root) != "700" or str(args.sealed_root.resolve()).startswith("/data"):
        fail("invalid_sealed_root")
    manifest_path = args.sealed_root / args.manifest_name
    receipt_path = args.sealed_root / args.receipt_name
    if manifest_path.exists() or receipt_path.exists():
        fail("sealed_output_exists")

    inputs = {
        "preregister": sha256(args.preregister),
        "inventory": sha256(args.inventory),
        "public_manifest": sha256(args.public_manifest),
        "node71_public_manifest": sha256(args.node71_public_manifest),
        "node71_preregister": sha256(args.node71_preregister),
        "node71_sealed_mapping": sha256(args.node71_sealed_mapping),
        "node71_presentation_salt": sha256(args.node71_presentation_salt),
        "node72_split_salt": sha256(args.split_salt),
    }
    expected = {
        "preregister": PREREG_SHA256,
        "inventory": INVENTORY_SHA256,
        "public_manifest": PUBLIC_MANIFEST_SHA256,
        "node71_public_manifest": NODE71_PUBLIC_SHA256,
        "node71_preregister": NODE71_PREREG_SHA256,
        "node71_sealed_mapping": NODE71_MAPPING_SHA256,
        "node71_presentation_salt": NODE71_PRESENTATION_SALT_SHA256,
        "node72_split_salt": NODE72_SPLIT_SALT_SHA256,
    }
    if inputs != expected:
        fail("frozen_input_hash_mismatch")

    prereg = json.loads(args.preregister.read_text())
    inventory = json.loads(args.inventory.read_text())
    public = json.loads(args.public_manifest.read_text())
    node71_public = json.loads(args.node71_public_manifest.read_text())
    node71_prereg = json.loads(args.node71_preregister.read_text())
    node71_mapping = json.loads(args.node71_sealed_mapping.read_text())
    split_salt = args.split_salt.read_bytes()
    presentation_salt = args.node71_presentation_salt.read_bytes()
    if len(split_salt) != 32 or len(presentation_salt) != 32:
        fail("sealed_salt_length_mismatch")
    if prereg["sealed_split_salt_sha256"] != inputs["node72_split_salt"]:
        fail("split_salt_commitment_mismatch")

    selected = sorted((row for row in inventory["records"] if eligible(row)), key=selection_key)[:36]
    if len(selected) != 36:
        fail("first36_cardinality")
    selected_pairs = {(row["source"], row["object_id"]) for row in selected}
    profiles = public["profiles"]
    public_pairs = {(row["source"], row["object_id"]) for row in profiles}
    if len(profiles) != 36 or len(public_pairs) != 36 or public_pairs != selected_pairs:
        fail("public_first36_coverage_mismatch")

    old_public_by_pair = {(row["source"], row["object_id"]): row for row in node71_public["episodes"]}
    old_sealed_by_pair = {(row["source"], row["object_id"]): row for row in node71_mapping["episodes"]}
    prefix12 = selected[:12]
    prefix_pairs = {(row["source"], row["object_id"]) for row in prefix12}
    if set(old_public_by_pair) != prefix_pairs or set(old_sealed_by_pair) != prefix_pairs:
        fail("node71_prefix_coverage_mismatch")

    def presentation_key(row: dict) -> str:
        identity = f"|node7.1-presentation-rank-v1|{row['source']}|{row['object_id']}".encode()
        return hashlib.sha256(presentation_salt + identity).hexdigest()

    presentation_rank = {
        (row["source"], row["object_id"]): rank
        for rank, row in enumerate(sorted(prefix12, key=presentation_key))
    }
    split_key = lambda row: hashlib.sha256(
        split_salt + f"|node7.2-split-rank-v1|{row['source']}|{row['object_id']}".encode()
    ).hexdigest()
    split_rank = {
        (row["source"], row["object_id"]): rank
        for rank, row in enumerate(sorted(selected, key=split_key))
    }
    profile_by_pair = {(row["source"], row["object_id"]): row for row in profiles}
    urdf_bundle_rows = []
    sealed_rows = []
    old_mapping_verified = 0
    for record in selected:
        pair = (record["source"], record["object_id"])
        profile = profile_by_pair[pair]
        expected_asset_hash = record.get("archive_sha256") or record.get("urdf_sha256")
        expected_observations = {
            "state0": f"episodes/{profile['episode_id']}/state0",
            "state1": f"episodes/{profile['episode_id']}/state1",
        }
        if profile["asset_sha256"] != expected_asset_hash or profile["observations"] != expected_observations:
            fail("public_profile_identity_or_observation_mapping_mismatch")
        urdf_payload, urdf_hash = read_original_urdf(record, args.archive_root, args.njc_root)
        parsed = parse_primary_joint(urdf_payload)
        assert_original_joint_matches(record, parsed)
        urdf_bundle_rows.append([record["source"], record["object_id"], urdf_hash])
        joint = record["primary_joint"]

        if profile["existing_node71_profile"]:
            rank = presentation_rank[pair]
            near = band_fraction("node7.1-near-v1", *pair, *node71_prereg["physical_states"]["near_band"])
            far = band_fraction("node7.1-far-v1", *pair, *node71_prereg["physical_states"]["far_band"])
            near_q = state_from_authored_fraction(joint, near)
            far_q = state_from_authored_fraction(joint, far)
            q0, q1 = (near_q, far_q) if rank % 2 == 0 else (far_q, near_q)
            old_public_row = old_public_by_pair[pair]
            old_sealed_row = old_sealed_by_pair[pair]
            if (
                profile["episode_id"] != old_public_row["episode_id"]
                or profile["observations"] != old_public_row["observations"]
                or old_sealed_row["presentation_rank"] != rank
                or old_sealed_row["presentation_order_key"] != presentation_key(record)
                or not close(old_sealed_row["state0_q"], q0)
                or not close(old_sealed_row["state1_q"], q1)
                or old_sealed_row["presentation_bit"] != (0 if rank % 2 == 0 else 1)
            ):
                fail("node71_presentation_recomputation_mismatch")
            state_rule = "node7.1-frozen-hash-bands-and-sealed-presentation-rank"
            old_mapping_verified += 1
        else:
            span = joint["upper"] - joint["lower"]
            q0 = joint["lower"] + span / 3.0
            q1 = joint["lower"] + 2.0 * span / 3.0
            state_rule = "node7.2-lower-to-upper-one-third-two-thirds"

        rank = split_rank[pair]
        membership = "train" if rank < 18 else "calibration" if rank < 27 else "confirmatory"
        target = local_target(joint, q0, q1)
        local_closed = (
            "local_lower"
            if target["local_direction_mapping"]["local_lower_physical_side"] == joint["zero_relation"]
            else "local_upper"
        )
        sealed_rows.append({
            "episode_id": profile["episode_id"],
            "source": record["source"],
            "object_id": record["object_id"],
            "asset_sha256": profile["asset_sha256"],
            "observations": profile["observations"],
            "state_rule": state_rule,
            "split": membership,
            "split_rank": rank,
            "split_order_key": split_key(record),
            "primary_joint": {
                "name": joint["name"],
                "type": joint["type"],
                "lower": joint["lower"],
                "upper": joint["upper"],
                "range": joint["upper"] - joint["lower"],
                "authored_closed_physical_side": joint["zero_relation"],
                "authored_closed_local_side": local_closed,
                "original_urdf_sha256": urdf_hash,
            },
            "endpoint_truth": target,
        })

    counts = Counter(row["split"] for row in sealed_rows)
    if counts != Counter({"train": 18, "calibration": 9, "confirmatory": 9}):
        fail("split_count_mismatch")
    if len({row["episode_id"] for row in sealed_rows}) != 36 or old_mapping_verified != 12:
        fail("sealed_coverage_mismatch")
    if any(not math.isfinite(value) for row in sealed_rows for value in (
        row["primary_joint"]["lower"], row["primary_joint"]["upper"],
        row["endpoint_truth"]["observed_state_q"]["state0"],
        row["endpoint_truth"]["observed_state_q"]["state1"],
    )):
        fail("nonfinite_truth")

    manifest = {
        "schema": "splart-node7.2-sealed-endpoint-truth/v1",
        "formula_version": FORMULA_VERSION,
        "preregister_sha256": inputs["preregister"],
        "public_manifest_sha256": inputs["public_manifest"],
        "source_inventory_sha256": inputs["inventory"],
        "split_salt_sha256": inputs["node72_split_salt"],
        "object_count": 36,
        "split_counts": dict(counts),
        "episodes": sorted(sealed_rows, key=lambda row: row["episode_id"]),
    }
    # Validate the serialized form before publishing it.
    manifest_temp = write_temp_json(args.sealed_root, manifest, ".sealed-truth-v1.")
    roundtrip = json.loads(manifest_temp.read_text())
    roundtrip_counts = Counter(row["split"] for row in roundtrip["episodes"])
    if len(roundtrip["episodes"]) != 36 or roundtrip_counts != counts or mode(manifest_temp) != "600":
        manifest_temp.unlink(missing_ok=True)
        fail("sealed_roundtrip_validation_failed")

    manifest_hash = sha256(manifest_temp)
    public_unchanged = sha256(args.public_manifest) == PUBLIC_MANIFEST_SHA256
    inventory_unchanged = sha256(args.inventory) == INVENTORY_SHA256
    receipt = {
        "schema": "splart-node7.2-sealed-endpoint-truth-build-receipt/v1",
        "status": "PASS",
        "formula_version": FORMULA_VERSION,
        "object_count": 36,
        "split_counts": {"train": 18, "calibration": 9, "confirmatory": 9},
        "input_hashes": inputs,
        "combined_input_sha256": canonical_sha256(inputs),
        "original_urdf_bundle_sha256": canonical_sha256(sorted(urdf_bundle_rows)),
        "original_urdf_verified_count": 36,
        "node71_presentation_recomputed_count": old_mapping_verified,
        "sealed_manifest_sha256": manifest_hash,
        "permissions": {"directory": "0700", "manifest": "0600", "receipt": "0600"},
        "independent_validation": {
            "object_coverage": True,
            "direction_mapping_reconstruction": True,
            "joint_unit_and_range_checks": True,
            "nmae_normalization_checks": True,
            "split_hash_binding": True,
        },
        "no_public_leak_checks": {
            "sealed_root_outside_public_data_roots": True,
            "public_manifest_hash_unchanged": public_unchanged,
            "source_inventory_hash_unchanged": inventory_unchanged,
            "no_public_output_written": True,
            "receipt_contains_no_membership_or_target_values": True,
        },
        "score_files_read": [],
        "protected_benchmark_files_read": [],
    }
    if not public_unchanged or not inventory_unchanged:
        manifest_temp.unlink(missing_ok=True)
        fail("public_input_changed_during_build")
    receipt_temp = write_temp_json(args.sealed_root, receipt, ".sealed-truth-v1-receipt.")
    try:
        publish_no_overwrite(manifest_temp, manifest_path)
        publish_no_overwrite(receipt_temp, receipt_path)
    except Exception:
        # Remove only this run's newly linked manifest if the receipt publication
        # fails; pre-existing outputs are never overwritten or deleted.
        if manifest_path.exists() and sha256(manifest_path) == manifest_hash and not receipt_path.exists():
            manifest_path.unlink()
        raise
    if mode(manifest_path) != "600" or mode(receipt_path) != "600" or mode(args.sealed_root) != "700":
        fail("published_permissions_mismatch")
    print(json.dumps({
        "status": "PASS",
        "object_count": 36,
        "split_counts": {"train": 18, "calibration": 9, "confirmatory": 9},
        "sealed_manifest_sha256": manifest_hash,
        "receipt_sha256": sha256(receipt_path),
    }, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except BuildFailure as error:
        print(json.dumps({"status": "FAIL", "error_code": str(error)}, sort_keys=True))
        raise SystemExit(1)
    except Exception:
        print(json.dumps({"status": "FAIL", "error_code": "unexpected_internal_error"}, sort_keys=True))
        raise SystemExit(1)
