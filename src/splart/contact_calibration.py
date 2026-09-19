"""Strict contact-cache loading and endpoint-free interstitial falsification."""
from __future__ import annotations

from dataclasses import asdict
import json
import math
from pathlib import Path
import re

import torch

from splart.relative_search import file_sha256, load_inputs
from splart.surface_contact import CONFIG, TrianglePairQuery, rotation_transform


COUNTS = {"source_train": 13, "source_validation": 6}
CONTACT_HASH_KEYS = {"surface_index_sha256", "base_input_index_sha256", "engine_sha256", "consumer_sha256",
                     "runner_sha256", "backend_receipt_sha256"}
CONTACT_KEYS = {"schema", "object_id", "split", "coordinates", "observed_coordinates", "observed", "sides",
                "observed_seen_pairs", "persistent_all_observed_pairs", "observed_intersection_count",
                "observed_anchor_intersection", "observed_overlapping_or_touching", "physical_validity", "novelty_caution",
                "boundary_confirmed", "endpoint_selection_authorized", "config", "provenance"}
COUNT_KEYS = {"contact_count", "pair_count", "enumeration_capacity", "enumeration_attempts", "observed_seen_pair_count",
              "persistent_observed_pair_count", "novel_pair_count", "observed_region_recombination_count"}
PACK_KEYS = COUNT_KEYS | {"surface_intersection", "surface_clearance", "backend_unsigned_distance", "triangle_pairs",
                          "novel_pairs", "enumeration_complete"}
CALIBRATION_HASH_KEYS = {"contact_index_sha256", "base_input_index_sha256", "surface_index_sha256",
                         "calibrator_sha256", "runner_sha256", "backend_receipt_sha256"}
CALIBRATION_ROW_KEYS = {"object_id", "split", "calibration_passed", "midpoint_new_event_count", "enumeration_complete",
                        "exact_swap_passed", "contact_artifact_sha256", "midpoint_artifact", "midpoint_artifact_sha256"}
MIDPOINTS = tuple((i + 0.5) / 64 for i in range(64))


def _hashes(value, keys):
    if not isinstance(value, dict) or set(value) != keys or any(not isinstance(v, str) or re.fullmatch("[0-9a-f]{64}", v) is None for v in value.values()):
        raise ValueError("unexpected hash-only provenance")


def _safe_file(root: Path, relative: str, digest: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError("artifact must be a safe relative path")
    path = root / path
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("artifact escapes cache")
    if file_sha256(path) != digest:
        raise ValueError("artifact hash mismatch")
    return path


def _pairs(tensor) -> set[tuple[int, int]]:
    if not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.int64 or tensor.device.type != "cpu" or tensor.ndim != 2 or tensor.shape[1] != 2:
        raise ValueError("triangle witnesses must be CPU int64 pairs")
    result = [tuple(pair) for pair in tensor.tolist()]
    if result != sorted(set(result)) or any(min(pair) < 0 for pair in result):
        raise ValueError("triangle witnesses must be sorted unique nonnegative pairs")
    return set(result)


def unseen_face_pairs(pairs: torch.Tensor, observed_seen_pairs: torch.Tensor) -> torch.Tensor:
    seen = _pairs(observed_seen_pairs)
    static, mobile = {p[0] for p in seen}, {p[1] for p in seen}
    retained = sorted((a, b) for a, b in _pairs(pairs) if a not in static and b not in mobile)
    return torch.tensor(retained, dtype=torch.int64).reshape(-1, 2)


def first_event(contact: dict, side: int) -> int | None:
    if side not in (0, 1):
        raise ValueError("side must be 0 or 1")
    for index, pairs in enumerate(contact["sides"][side]["triangle_pairs"]):
        if len(unseen_face_pairs(pairs, contact["observed_seen_pairs"])):
            return index
    return None


def swap_contact(contact: dict) -> dict:
    output = dict(contact, sides=list(reversed(contact["sides"])))
    if "coordinates" in contact:
        output["coordinates"] = contact["coordinates"].flip(0)
    if "observed" in contact:
        output["observed"] = {key: value.flip(0) if isinstance(value, torch.Tensor) else
                              list(reversed(value)) if isinstance(value, list) else value
                              for key, value in contact["observed"].items()}
    if "observed_anchor_intersection" in contact:
        output["observed_anchor_intersection"] = list(reversed(contact["observed_anchor_intersection"]))
    return output


def _validate_pack(pack: dict, length: int, seen: set, persistent: set) -> None:
    if not isinstance(pack, dict) or set(pack) != PACK_KEYS or pack["enumeration_complete"] is not True:
        raise ValueError("contact pack must be exact and complete")
    for key in COUNT_KEYS | {"surface_intersection", "surface_clearance"}:
        value = pack[key]
        dtype = torch.bool if key == "surface_intersection" else torch.float64 if key == "surface_clearance" else torch.int64
        if not isinstance(value, torch.Tensor) or value.dtype != dtype or value.shape != (length,) or value.device.type != "cpu" or not torch.isfinite(value).all() or (value < 0).any():
            raise ValueError("invalid contact scalar array")
    for key in ("triangle_pairs", "novel_pairs", "backend_unsigned_distance"):
        if not isinstance(pack[key], list) or len(pack[key]) != length:
            raise ValueError("invalid contact row count")
    seen_static, seen_mobile = {p[0] for p in seen}, {p[1] for p in seen}
    for i in range(length):
        pairs, novel = _pairs(pack["triangle_pairs"][i]), _pairs(pack["novel_pairs"][i])
        count, capacity = int(pack["contact_count"][i]), int(pack["enumeration_capacity"][i])
        if len(pairs) != int(pack["pair_count"][i]) or count < len(pairs) or not count < capacity <= CONFIG.maximum_contact_capacity or int(pack["enumeration_attempts"][i]) < 1:
            raise ValueError("contact enumeration count/capacity inconsistent")
        if bool(pack["surface_intersection"][i]) != bool(pairs) or novel != pairs - seen:
            raise ValueError("contact witnesses disagree with flags")
        expected = {"observed_seen_pair_count": len(pairs & seen), "persistent_observed_pair_count": len(pairs & persistent),
                    "novel_pair_count": len(novel), "observed_region_recombination_count": sum(a in seen_static and b in seen_mobile for a, b in novel)}
        if any(int(pack[key][i]) != value for key, value in expected.items()):
            raise ValueError("contact derived counts mismatch")
        distance, clearance = pack["backend_unsigned_distance"][i], float(pack["surface_clearance"][i])
        if pairs:
            if distance is not None or clearance != 0:
                raise ValueError("intersecting surfaces must have zero unsigned clearance")
        elif not isinstance(distance, (int, float)) or not math.isfinite(distance) or distance < 0 or clearance != distance:
            raise ValueError("invalid separated surface clearance")


def validate_contact(contact: dict, value: dict, provenance: dict) -> None:
    if not isinstance(contact, dict) or set(contact) != CONTACT_KEYS or contact["schema"] != "surface-contact-measurement/v1":
        raise ValueError("unexpected contact artifact schema")
    if (contact["split"], contact["object_id"]) != (value["split"], value["object_id"]):
        raise ValueError("contact/base identity mismatch")
    _hashes(contact["provenance"], CONTACT_HASH_KEYS | {"surface_artifact_sha256"})
    if any(contact["provenance"][key] != provenance[key] for key in CONTACT_HASH_KEYS):
        raise ValueError("contact/index provenance mismatch")
    if not torch.equal(contact["coordinates"], value["coordinates"]) or not torch.equal(contact["observed_coordinates"], torch.linspace(0, 1, 65, dtype=torch.float64)):
        raise ValueError("contact coordinate grid mismatch")
    if contact["config"] != asdict(CONFIG) or contact["boundary_confirmed"] is not False or contact["endpoint_selection_authorized"] is not False:
        raise ValueError("contact configuration or unauthorized endpoint selection")
    if not isinstance(contact["sides"], list) or len(contact["sides"]) != 2:
        raise ValueError("two contact sides required")
    seen, persistent = _pairs(contact["observed_seen_pairs"]), _pairs(contact["persistent_all_observed_pairs"])
    _validate_pack(contact["observed"], 65, seen, persistent)
    raw = [_pairs(p) for p in contact["observed"]["triangle_pairs"]]
    if seen != set.union(*raw) or persistent != set.intersection(*raw):
        raise ValueError("observed contact atlas was changed")
    for side in contact["sides"]:
        _validate_pack(side, 129, seen, persistent)
    flags = contact["observed"]["surface_intersection"]
    if contact["observed_intersection_count"] != int(flags.sum()) or contact["observed_anchor_intersection"] != [bool(flags[0]), bool(flags[-1])] or contact["observed_overlapping_or_touching"] != bool(flags.any()):
        raise ValueError("observed overlap diagnostics mismatch")


def load_contacts(index_path: Path, base_input_index: Path) -> tuple[dict, dict]:
    index_path, base_input_index = Path(index_path), Path(base_input_index)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if set(index) != {"schema", "rows", "provenance", "split_counts", "audit_sha256"} or index["schema"] != "surface-contact-index/v1" or index["split_counts"] != COUNTS:
        raise ValueError("unexpected complete contact index")
    _hashes(index["provenance"], CONTACT_HASH_KEYS)
    if index["provenance"]["base_input_index_sha256"] != file_sha256(base_input_index):
        raise ValueError("contact/base index hash mismatch")
    _, base = load_inputs(base_input_index)
    values = {(v["split"], v["object_id"]): v for v in base}
    if len(values) != 19 or {s: sum(k[0] == s for k in values) for s in COUNTS} != COUNTS:
        raise ValueError("fixed 13/6 base roster required")
    audit_path = _safe_file(index_path.parent, "audit.json", index["audit_sha256"])
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("provenance") != index["provenance"] or audit.get("config") != asdict(CONFIG) or audit.get("split_counts") != COUNTS or any(audit.get(key) is not True for key in ("exact_swap_passed", "all_contact_enumerations_complete", "complete_fixed_source_roster")) or audit.get("endpoint_labels_read") is not False:
        raise ValueError("contact audit is incomplete or mismatched")
    _safe_file(index_path.parent, "backend_receipt.json", index["provenance"]["backend_receipt_sha256"])
    root = Path(__file__).resolve().parents[2]
    for key, relative in (("engine_sha256", "src/splart/surface_contact.py"), ("consumer_sha256", "src/splart/observed_surfaces.py"), ("runner_sha256", "run_surface_contact.py")):
        if index["provenance"][key] != file_sha256(root / relative):
            raise ValueError("frozen contact producer code mismatch")
    output = {}
    for row in index["rows"]:
        if set(row) != {"object_id", "split", "artifact", "artifact_sha256"}:
            raise ValueError("unexpected contact row keys")
        identity = row["split"], row["object_id"]
        if identity not in values or identity in output:
            raise ValueError("contact roster mismatch")
        artifact_path = _safe_file(index_path.parent, row["artifact"], row["artifact_sha256"])
        contact = torch.load(artifact_path, map_location="cpu", weights_only=True)
        validate_contact(contact, values[identity], index["provenance"])
        output[identity] = contact
    if set(output) != set(values):
        raise ValueError("contact roster incomplete")
    return index, output


def measure_midpoints(surface: dict, contact: dict) -> dict:
    from splart.observed_surfaces import validate_surface
    validate_surface(surface)
    if (surface["split"], surface["object_id"]) != (contact["split"], contact["object_id"]):
        raise ValueError("surface/contact identity mismatch")
    queries = [TrianglePairQuery(surface["static_vertices"].numpy(), surface["static_faces"].numpy(), mesh.numpy(), surface["mobile_faces"].numpy())
               for mesh in surface["mobile_vertices"]]
    axis, pivot = surface["axis"].numpy(), surface["pivot"].numpy()
    rows = []
    for q in MIDPOINTS:
        side = 0 if q < 0.5 else 1
        rotation, translation = rotation_transform(axis, pivot, surface["angular_increment"] * (q - side))
        measured = queries[side].query(rotation, translation)
        pairs = torch.tensor(measured["triangle_pairs"], dtype=torch.int64).reshape(-1, 2)
        unseen = unseen_face_pairs(pairs, contact["observed_seen_pairs"])
        rows.append(dict(measured, both_unseen_pairs=unseen.tolist(), both_unseen_pair_count=len(unseen)))
    failures = sum(row["both_unseen_pair_count"] > 0 for row in rows)
    return {"schema": "contact-midpoint-measurement/v1", "object_id": surface["object_id"], "split": surface["split"],
            "midpoints": list(MIDPOINTS), "queries": rows, "midpoint_new_event_count": failures,
            "calibration_passed": failures == 0, "enumeration_complete": True,
            "atlas_updated": False}


def assert_midpoint_swap(original: dict, swapped: dict) -> None:
    if original["queries"] != list(reversed(swapped["queries"])) or original["calibration_passed"] != swapped["calibration_passed"]:
        raise AssertionError("midpoint calibration exact swap failed")


def validate_midpoints(artifact: dict, row: dict, contact: dict, provenance: dict) -> None:
    keys = {"schema", "object_id", "split", "midpoints", "queries", "midpoint_new_event_count", "calibration_passed",
            "enumeration_complete", "atlas_updated", "exact_swap_passed", "provenance"}
    if set(artifact) != keys or artifact["schema"] != "contact-midpoint-measurement/v1" or artifact["midpoints"] != list(MIDPOINTS) or len(artifact["queries"]) != 64:
        raise ValueError("midpoint artifact schema/grid mismatch")
    if any(artifact[key] != row[key] for key in ("object_id", "split", "midpoint_new_event_count", "calibration_passed", "enumeration_complete", "exact_swap_passed")) or artifact["atlas_updated"] is not False:
        raise ValueError("midpoint artifact/index mismatch or mutated atlas")
    expected_provenance = dict(provenance, contact_artifact_sha256=row["contact_artifact_sha256"])
    if artifact["provenance"] != expected_provenance:
        raise ValueError("midpoint provenance mismatch")
    failures = 0
    for query in artifact["queries"]:
        if set(query) != {"surface_intersection", "surface_clearance", "backend_unsigned_distance", "triangle_pairs", "contact_count", "pair_count",
                          "enumeration_complete", "enumeration_capacity", "enumeration_attempts", "both_unseen_pairs", "both_unseen_pair_count"} or query["enumeration_complete"] is not True:
            raise ValueError("midpoint contact query incomplete")
        for key in ("triangle_pairs", "both_unseen_pairs"):
            if not isinstance(query[key], list) or any(not isinstance(pair, list) or len(pair) != 2 or any(type(index) is not int or index < 0 for index in pair) for pair in query[key]):
                raise ValueError("midpoint witnesses require integer pair lists")
        for key in ("contact_count", "pair_count", "enumeration_capacity", "enumeration_attempts", "both_unseen_pair_count"):
            if type(query[key]) is not int or query[key] < 0:
                raise ValueError("midpoint query counts must be nonnegative integers")
        pairs = torch.tensor(query["triangle_pairs"], dtype=torch.int64).reshape(-1, 2)
        _pairs(pairs)
        expected = unseen_face_pairs(pairs, contact["observed_seen_pairs"]).tolist()
        if query["both_unseen_pairs"] != expected or query["both_unseen_pair_count"] != len(expected):
            raise ValueError("midpoint face-exclusion result mismatch")
        if query["pair_count"] != len(pairs) or query["surface_intersection"] is not bool(len(pairs)) or not len(pairs) <= query["contact_count"] < query["enumeration_capacity"] <= CONFIG.maximum_contact_capacity or query["enumeration_attempts"] < 1:
            raise ValueError("midpoint contact count/cap inconsistent")
        clearance = query["surface_clearance"]
        if not isinstance(clearance, (int, float)) or not math.isfinite(clearance) or clearance < 0:
            raise ValueError("midpoint clearance invalid")
        distance = query["backend_unsigned_distance"]
        if len(pairs):
            if distance is not None or clearance != 0:
                raise ValueError("midpoint intersecting clearance mismatch")
        elif not isinstance(distance, (int, float)) or not math.isfinite(distance) or distance < 0 or distance != clearance:
            raise ValueError("midpoint backend clearance mismatch")
        failures += bool(expected)
    if row["midpoint_new_event_count"] != failures or row["calibration_passed"] != (failures == 0):
        raise ValueError("midpoint calibration decision mismatch")


def load_calibration(index_path: Path, contact_index: Path, base_input_index: Path) -> tuple[dict, dict]:
    index_path, contact_index, base_input_index = map(Path, (index_path, contact_index, base_input_index))
    contacts_index, contacts = load_contacts(contact_index, base_input_index)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if set(index) != {"schema", "rows", "provenance", "split_counts", "audit_sha256"} or index["schema"] != "contact-calibration-index/v1" or index["split_counts"] != COUNTS:
        raise ValueError("unexpected calibration index")
    _hashes(index["provenance"], CALIBRATION_HASH_KEYS)
    expected = {"contact_index_sha256": file_sha256(contact_index), "base_input_index_sha256": file_sha256(base_input_index),
                "surface_index_sha256": contacts_index["provenance"]["surface_index_sha256"],
                "backend_receipt_sha256": contacts_index["provenance"]["backend_receipt_sha256"]}
    if any(index["provenance"][key] != value for key, value in expected.items()):
        raise ValueError("calibration contact/base/backend binding mismatch")
    root = Path(__file__).resolve().parents[2]
    if index["provenance"]["calibrator_sha256"] != file_sha256(Path(__file__)) or index["provenance"]["runner_sha256"] != file_sha256(root / "run_contact_calibration.py"):
        raise ValueError("calibration producer code mismatch")
    audit = json.loads(_safe_file(index_path.parent, "audit.json", index["audit_sha256"]).read_text(encoding="utf-8"))
    if audit.get("provenance") != index["provenance"] or audit.get("rows") != index["rows"] or audit.get("endpoint_labels_read") is not False or audit.get("atlas_updated") is not False:
        raise ValueError("calibration audit mismatch")
    contact_rows = {(row["split"], row["object_id"]): row for row in contacts_index["rows"]}
    output = {}
    for row in index["rows"]:
        if set(row) != CALIBRATION_ROW_KEYS:
            raise ValueError("unexpected calibration row keys")
        identity = row["split"], row["object_id"]
        if identity not in contacts or identity in output or row["contact_artifact_sha256"] != contact_rows[identity]["artifact_sha256"]:
            raise ValueError("calibration roster/contact artifact mismatch")
        if row["enumeration_complete"] is not True or row["exact_swap_passed"] is not True or not isinstance(row["calibration_passed"], bool):
            raise ValueError("calibration completeness/swap unverified")
        artifact = json.loads(_safe_file(index_path.parent, row["midpoint_artifact"], row["midpoint_artifact_sha256"]).read_text(encoding="utf-8"))
        validate_midpoints(artifact, row, contacts[identity], index["provenance"])
        output[identity] = row
    if set(output) != set(contacts):
        raise ValueError("calibration roster incomplete")
    return index, output
