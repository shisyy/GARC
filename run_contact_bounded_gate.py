#!/usr/bin/env python3
"""Seal contact-bounded predictions and routing evidence before independent scoring.

Only strict target-free inputs are accepted. This module neither imports an
endpoint loader nor accepts an endpoint-bearing source index argument.
"""
import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from splart.relative_search import CONFIG, METHODS, file_sha256, load_inputs, predict, swap_input, write_json_new
from splart.contact_bounded_semantics import MODES, bounded_predict, swap_contact
from splart.contact_calibration import load_contacts, load_calibration

COUNTS = {"source_train": 13, "source_validation": 6}
CODE_PATHS = ("src/splart/relative_search.py", "src/splart/geometry_first.py",
              "run_contact_bounded_gate.py", "evaluate_relative_search.py",
              "src/splart/contact_bounded_semantics.py", "src/splart/contact_calibration.py",
              "src/splart/surface_contact.py", "src/splart/observed_surfaces.py")


def donor_map(values):
    """Sorted cyclic opaque IDs within each split, independent of index order."""
    donors = {}
    for split in COUNTS:
        group = sorted((v for v in values if v["split"] == split), key=lambda v: v["object_id"])
        if len(group) < 3:
            raise ValueError("each split needs at least three distinct donor identities")
        for i, value in enumerate(group):
            donors[value["object_id"]] = group[(i + 1) % len(group)]
    return donors


def routing(value, method, mode, donors):
    first = donors[value["object_id"]]
    second = donors[first["object_id"]]
    if mode == "raw":
        return first, None
    semantic = first if method == "object_shuffle" else None
    role = None
    if mode == "donor_semantics" and method in ("joint", "object_shuffle"):
        semantic = second if method == "object_shuffle" else first
    if mode == "shuffled_verification" and method in ("joint", "object_shuffle"):
        role = second if method == "object_shuffle" else first
    return semantic, role


def diagnostic_summary(rows):
    result = {}
    for split in COUNTS:
        result[split] = {}
        for method in METHODS:
            selected = [r for r in rows if r["split"] == split and r["method"] == method]
            sides = [s for r in selected for s in r["sides"]]
            counts = {key: sum(s.get(key) is True for s in sides) for key in (
                "geometric_confirmed", "geometric_available", "calibration_passed", "role_eligible", "semantic_verified", "semantic_activation", "selection_changed")}
            fallback = Counter()
            for side in sides:
                reason = side.get("fallback_reason")
                if reason:
                    fallback[str(reason)] += 1
            result[split][method] = {"objects": len(selected), "sides": len(sides),
                                     **counts, "fallback_reasons": dict(sorted(fallback.items()))}
    return result


def run(inputs: Path, contacts: Path, calibration: Path, output_dir: Path) -> dict:
    if output_dir.exists():
        raise ValueError("output directory must be new")
    torch.set_num_threads(1)
    index, values = load_inputs(inputs)
    _, contact_values = load_contacts(contacts, inputs)
    _, calibration_values = load_calibration(calibration, contacts, inputs)
    counts = {s: sum(v["split"] == s for v in values) for s in COUNTS}
    if counts != COUNTS:
        raise ValueError("full fixed 13/6 whitelist roster required")
    donors = donor_map(values)
    implementation = {p: file_sha256(ROOT / p) for p in CODE_PATHS}
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    bindings = {"input_index_sha256": file_sha256(inputs), "source_index_sha256": index["source_index_sha256"],
                "implementation_sha256": implementation, "contact_index_sha256": file_sha256(contacts),
                "calibration_index_sha256": file_sha256(calibration)}
    manifest = {"schema": "contact-bounded-all-modes-seal/v1", "modes": list(MODES),
                **bindings, "predictions": {}, "diagnostics": {}}
    for mode in MODES:
        rows, diagnostics = [], []
        for value in values:
            first = donors[value["object_id"]]
            for method in METHODS:
                semantic, role = routing(value, method, mode, donors)
                contact = contact_values[(value["split"], value["object_id"])]
                calibrated = calibration_values[(value["split"], value["object_id"])]
                prediction = bounded_predict(value, method, contact, calibrated, mode, semantic, role)
                swapped = bounded_predict(swap_input(value), method, swap_contact(contact), calibrated, mode,
                                                  swap_input(semantic) if semantic is not None else None,
                                                  swap_input(role) if role is not None else None)
                restored = dict(swapped, distance=list(reversed(swapped["distance"])),
                                sides=list(reversed(swapped["sides"])))
                if prediction != restored:
                    raise AssertionError("exact side-swap output/diagnostics invariant failed")
                if mode == "raw" and prediction != predict(value, method, first):
                    raise AssertionError("raw mode must exactly recover original predictor")
                prediction["semantic_donor"] = first["object_id"] if method == "object_shuffle" else None
                rows.append(prediction)
                active_method = method in ("joint", "object_shuffle")
                effective = semantic if active_method and semantic is not None and not (mode == "raw" and method == "joint") else value
                role_owner = role if active_method and role is not None else effective
                diagnostics.append({"object_id": value["object_id"], "split": value["split"], "method": method,
                                    "semantic_owner": effective["object_id"], "routing_owner": role_owner["object_id"],
                                    "first_donor": first["object_id"],
                                    "sides": [s.get("contact_bounded", {}) for s in prediction["sides"]]})
        sidecar = {"schema": "contact-bounded-routing-diagnostics/v1", "mode": mode, **bindings,
                   "endpoint_labels_used": False, "rows": diagnostics, "summary": diagnostic_summary(diagnostics)}
        diagnostic_path = output_dir / mode / "routing_diagnostics.json"
        write_json_new(diagnostic_path, sidecar)
        diagnostic_digest = file_sha256(diagnostic_path)
        report = {"schema": "relative-search-predictions/v1", "config": asdict(CONFIG), "methods": list(METHODS),
                  "contact_bounded_mode": mode, "preregistered_modes": list(MODES), **bindings,
                  "diagnostic": index["diagnostic"], "rows": rows, "exact_swap_error": 0.0,
                  "routing_diagnostics_sha256": diagnostic_digest,
                  "code_commit": commit, "endpoint_labels_used": False, "trained_parameters": 0}
        output = output_dir / mode / "predictions.json"
        write_json_new(output, report)
        digest = file_sha256(output)
        write_json_new(output.with_suffix(".seal.json"), {"schema": "relative-search-seal/v1",
                       "prediction_sha256": digest, "routing_diagnostics_sha256": diagnostic_digest, **bindings})
        manifest["predictions"][mode] = {"path": output.relative_to(output_dir).as_posix(), "sha256": digest}
        manifest["diagnostics"][mode] = {"path": diagnostic_path.relative_to(output_dir).as_posix(), "sha256": diagnostic_digest}
    write_json_new(output_dir / "all_modes.seal.json", manifest)
    verify_all_modes(output_dir)
    return {"output_dir": str(output_dir), "modes": list(MODES), "rows_per_mode": len(rows),
            "all_modes_seal_sha256": file_sha256(output_dir / "all_modes.seal.json")}


def verify_all_modes(output_dir: Path) -> dict:
    """Verify complete predictions, diagnostics and code bindings before labels."""
    manifest = json.loads((output_dir / "all_modes.seal.json").read_text(encoding="utf-8"))
    if (manifest["schema"] != "contact-bounded-all-modes-seal/v1" or manifest["modes"] != list(MODES)
            or set(manifest["predictions"]) != set(MODES) or set(manifest["diagnostics"]) != set(MODES)):
        raise ValueError("all four preregistered modes must be sealed")
    if manifest["implementation_sha256"] != {p: file_sha256(ROOT / p) for p in CODE_PATHS}:
        raise ValueError("current code differs from sealed implementation")
    common_roster = None
    for mode in MODES:
        entry, diagnostic_entry = manifest["predictions"][mode], manifest["diagnostics"][mode]
        if entry["path"] != f"{mode}/predictions.json" or diagnostic_entry["path"] != f"{mode}/routing_diagnostics.json":
            raise ValueError("unexpected sealed artifact path")
        output, diagnostic_path = output_dir / entry["path"], output_dir / diagnostic_entry["path"]
        seal = json.loads(output.with_suffix(".seal.json").read_text(encoding="utf-8"))
        payload = json.loads(output.read_text(encoding="utf-8"))
        diagnostics = json.loads(diagnostic_path.read_text(encoding="utf-8"))
        digest, diagnostic_digest = file_sha256(output), file_sha256(diagnostic_path)
        if (entry["sha256"] != digest or seal["prediction_sha256"] != digest
                or diagnostic_entry["sha256"] != diagnostic_digest
                or seal["routing_diagnostics_sha256"] != diagnostic_digest
                or payload["routing_diagnostics_sha256"] != diagnostic_digest):
            raise ValueError("all-modes prediction/diagnostic seal mismatch")
        if (payload["contact_bounded_mode"] != mode or diagnostics["mode"] != mode
                or payload["preregistered_modes"] != list(MODES)):
            raise ValueError("mode identity mismatch")
        for key in ("source_index_sha256", "implementation_sha256", "input_index_sha256", "contact_index_sha256", "calibration_index_sha256"):
            if any(x[key] != manifest[key] for x in (seal, payload, diagnostics)):
                raise ValueError("shared code/source/input binding mismatch")
        if (payload["schema"] != "relative-search-predictions/v1" or payload["methods"] != list(METHODS)
                or payload["endpoint_labels_used"] or payload["trained_parameters"] != 0
                or payload["exact_swap_error"] != 0 or diagnostics["endpoint_labels_used"]):
            raise ValueError("training-free invariant failed")
        roster = [(r["split"], r["object_id"], r["method"]) for r in payload["rows"]]
        identities = {(s, o) for s, o, _ in roster}
        expected = {(s, o, m) for s, o in identities for m in METHODS}
        if (len(roster) != 95 or set(roster) != expected or len(set(roster)) != 95
                or {s: sum(a == s for a, _ in identities) for s in COUNTS} != COUNTS):
            raise ValueError("full fixed 13/6 all-method roster required")
        if common_roster is not None and roster != common_roster:
            raise ValueError("cross-mode roster mismatch")
        common_roster = roster
        if [(r["split"], r["object_id"], r["method"]) for r in diagnostics["rows"]] != roster:
            raise ValueError("diagnostic roster mismatch")
        if diagnostic_summary(diagnostics["rows"]) != diagnostics["summary"]:
            raise ValueError("diagnostic summary mismatch")
    return {"verified_modes": list(MODES), "all_modes_seal_sha256": file_sha256(output_dir / "all_modes.seal.json")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--contacts", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.inputs, args.contacts, args.calibration, args.output_dir), sort_keys=True))


