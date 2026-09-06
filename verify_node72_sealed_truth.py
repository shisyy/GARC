#!/usr/bin/env python3
"""Independent sealed verifier for node7.2 endpoint truth."""

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
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def key(domain, row):
    return hashlib.sha256(f"{domain}|{row['source']}|{row['object_id']}".encode()).hexdigest()


def eligible(row):
    j = row.get("primary_joint") or {}
    lo, hi = j.get("lower"), j.get("upper")
    return (
        row.get("error") is None and row.get("visual_geometry_count", 0) > 0
        and row.get("primary_candidate_count") == 1
        and j.get("type") in {"revolute", "prismatic"}
        and isinstance(lo, (int, float)) and isinstance(hi, (int, float))
        and math.isfinite(lo) and math.isfinite(hi) and lo < hi
        and j.get("zero_relation") in {"lower", "upper"}
    )


def close(a, b):
    return math.isclose(float(a), float(b), rel_tol=1e-10, abs_tol=1e-10)


def old_fraction(domain, row, bounds):
    raw = bytes.fromhex(key(domain, row))[:8]
    unit = int.from_bytes(raw, "big") / float((1 << 64) - 1)
    return bounds[0] + (bounds[1] - bounds[0]) * unit


def authored_q(joint, fraction):
    lo, hi = joint["lower"], joint["upper"]
    return lo + fraction * (hi - lo) if joint["zero_relation"] == "lower" else hi - fraction * (hi - lo)


def original_urdf_bytes(row, archive_root, njc_root):
    if row["source"] == "articraft":
        path = archive_root / f"{row['object_id']}.tar.gz"
        if digest(path) != row["archive_sha256"]:
            raise RuntimeError("asset_hash")
        with tarfile.open(path, "r:gz") as archive:
            members = [x for x in archive.getmembers() if x.isfile() and x.name.lower().endswith(".urdf")]
            if len(members) != 1:
                raise RuntimeError("urdf_count")
            return archive.extractfile(members[0]).read()
    path = njc_root / row["object_id"]
    urdfs = list(path.rglob("*.urdf"))
    if len(urdfs) != 1 or digest(urdfs[0]) != row["urdf_sha256"]:
        raise RuntimeError("asset_hash")
    return urdfs[0].read_bytes()


def named_joint(payload, name):
    root = ET.fromstring(payload)
    candidates = [x for x in root.findall("joint") if x.get("name") == name]
    if len(candidates) != 1:
        raise RuntimeError("joint_identity")
    node = candidates[0]
    limit = node.find("limit")
    if limit is None:
        raise RuntimeError("joint_limit")
    return node.get("type"), float(limit.get("lower")), float(limit.get("upper"))


def expected_endpoint(joint, q0, q1):
    lo, hi = joint["lower"], joint["upper"]
    span, delta = hi - lo, q1 - q0
    width = abs(delta)
    if delta > 0:
        before, after, before_side, after_side = lo, hi, "lower", "upper"
    else:
        before, after, before_side, after_side = hi, lo, "upper", "lower"
    ext0_units, ext1_units = abs(q0 - before), abs(after - q1)
    ext0, ext1 = ext0_units / width, ext1_units / width
    return {
        "direction": (before_side, after_side),
        "extension_units": (ext0_units, ext1_units),
        "extension_local": (ext0, ext1),
        "scalars": (-ext0, 1.0 + ext1),
        "nmae_scale": width / span,
        "range": span,
    }


def publish(path, payload):
    if path.exists():
        raise FileExistsError(path)
    fd, name = tempfile.mkstemp(prefix=".verify.", suffix=".tmp", dir=path.parent)
    os.fchmod(fd, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    temp = Path(name)
    try:
        os.link(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--inventory", required=True, type=Path)
    p.add_argument("--public-manifest", required=True, type=Path)
    p.add_argument("--node71-preregister", required=True, type=Path)
    p.add_argument("--node71-presentation-salt", required=True, type=Path)
    p.add_argument("--split-salt", required=True, type=Path)
    p.add_argument("--archive-root", required=True, type=Path)
    p.add_argument("--njc-root", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    if stat.S_IMODE(a.output.parent.stat().st_mode) != 0o700 or str(a.output.parent.resolve()).startswith("/data"):
        raise RuntimeError("sealed_root")
    manifest = json.loads(a.manifest.read_text())
    inventory = json.loads(a.inventory.read_text())
    public = json.loads(a.public_manifest.read_text())
    old_prereg = json.loads(a.node71_preregister.read_text())
    presentation_salt, split_salt = a.node71_presentation_salt.read_bytes(), a.split_salt.read_bytes()
    selected = sorted((x for x in inventory["records"] if eligible(x)), key=lambda x: key("node7.1-object-selection-v1", x))[:36]
    by_pair = {(x["source"], x["object_id"]): x for x in selected}
    profiles = {(x["source"], x["object_id"]): x for x in public["profiles"]}
    rows = {(x["source"], x["object_id"]): x for x in manifest["episodes"]}
    if len(by_pair) != 36 or set(by_pair) != set(profiles) or set(by_pair) != set(rows):
        raise RuntimeError("coverage")

    def pkey(row):
        identity = f"|node7.1-presentation-rank-v1|{row['source']}|{row['object_id']}".encode()
        return hashlib.sha256(presentation_salt + identity).hexdigest()

    prefix = selected[:12]
    prank = {(x["source"], x["object_id"]): i for i, x in enumerate(sorted(prefix, key=pkey))}

    def skey(row):
        identity = f"|node7.2-split-rank-v1|{row['source']}|{row['object_id']}".encode()
        return hashlib.sha256(split_salt + identity).hexdigest()

    srank = {(x["source"], x["object_id"]): i for i, x in enumerate(sorted(selected, key=skey))}
    verified_urdfs = verified_directions = verified_targets = 0
    for pair, source in by_pair.items():
        profile, truth = profiles[pair], rows[pair]
        joint = source["primary_joint"]
        joint_type, lo, hi = named_joint(original_urdf_bytes(source, a.archive_root, a.njc_root), joint["name"])
        if joint_type != joint["type"] or not close(lo, joint["lower"]) or not close(hi, joint["upper"]):
            raise RuntimeError("original_joint")
        verified_urdfs += 1
        if profile["existing_node71_profile"]:
            near = old_fraction("node7.1-near-v1", source, old_prereg["physical_states"]["near_band"])
            far = old_fraction("node7.1-far-v1", source, old_prereg["physical_states"]["far_band"])
            nq, fq = authored_q(joint, near), authored_q(joint, far)
            q0, q1 = (nq, fq) if prank[pair] % 2 == 0 else (fq, nq)
        else:
            q0 = lo + (hi - lo) / 3.0
            q1 = lo + 2.0 * (hi - lo) / 3.0
        endpoint = expected_endpoint(joint, q0, q1)
        actual = truth["endpoint_truth"]
        if not close(actual["observed_state_q"]["state0"], q0) or not close(actual["observed_state_q"]["state1"], q1):
            raise RuntimeError("state_mapping")
        direction = actual["local_direction_mapping"]
        if (direction["local_lower_physical_side"], direction["local_upper_physical_side"]) != endpoint["direction"]:
            raise RuntimeError("direction")
        verified_directions += 1
        targets = actual["local_endpoint_targets"]
        observed = (
            targets["extension0_joint_units"], targets["extension1_joint_units"],
            targets["extension0_observation_units"], targets["extension1_observation_units"],
            targets["local_lower_scalar"], targets["local_upper_scalar"],
            actual["target_nmae_normalization"]["local_scalar_error_multiplier"],
            actual["target_nmae_normalization"]["physical_range"],
        )
        expected = (*endpoint["extension_units"], *endpoint["extension_local"], *endpoint["scalars"], endpoint["nmae_scale"], endpoint["range"])
        if not all(close(x, y) for x, y in zip(observed, expected)):
            raise RuntimeError("target_or_normalization")
        verified_targets += 1
        rank = srank[pair]
        split = "train" if rank < 18 else "calibration" if rank < 27 else "confirmatory"
        if truth["split_rank"] != rank or truth["split"] != split or truth["split_order_key"] != skey(source):
            raise RuntimeError("split")

    counts = Counter(x["split"] for x in rows.values())
    if counts != Counter({"train": 18, "calibration": 9, "confirmatory": 9}):
        raise RuntimeError("split_counts")
    receipt = {
        "schema": "splart-node7.2-sealed-endpoint-truth-independent-verification/v1",
        "status": "PASS",
        "object_count": 36,
        "split_counts": {"train": 18, "calibration": 9, "confirmatory": 9},
        "sealed_manifest_sha256": digest(a.manifest),
        "input_hashes": {
            "inventory": digest(a.inventory), "public_manifest": digest(a.public_manifest),
            "node71_preregister": digest(a.node71_preregister),
            "node71_presentation_salt": digest(a.node71_presentation_salt),
            "node72_split_salt": digest(a.split_salt),
        },
        "checks": {
            "object_coverage": len(rows) == 36,
            "original_urdf_joint_limits": verified_urdfs == 36,
            "presentation_direction_mapping": verified_directions == 36,
            "target_and_nmae_normalization": verified_targets == 36,
            "split_hash_and_counts": True,
            "manifest_mode_0600": stat.S_IMODE(a.manifest.stat().st_mode) == 0o600,
            "sealed_directory_mode_0700": stat.S_IMODE(a.output.parent.stat().st_mode) == 0o700,
            "no_public_write": True,
        },
        "score_files_read": [],
        "protected_benchmark_files_read": [],
    }
    publish(a.output, receipt)
    if stat.S_IMODE(a.output.stat().st_mode) != 0o600:
        raise RuntimeError("receipt_mode")
    print(json.dumps({
        "status": "PASS", "object_count": 36,
        "split_counts": {"train": 18, "calibration": 9, "confirmatory": 9},
        "sealed_manifest_sha256": digest(a.manifest), "verification_receipt_sha256": digest(a.output),
    }, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print(json.dumps({"status": "FAIL", "error_code": "independent_verification_failed"}, sort_keys=True))
        raise SystemExit(1)
