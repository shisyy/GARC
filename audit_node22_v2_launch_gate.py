#!/usr/bin/env python3
"""Fail-closed public-artifact audit for the node2.2 v2 evaluator.

This program deliberately opens only explicitly named public source/protocol files.
It never resolves, searches for, or reads the evaluator truth/index inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any


FORBIDDEN = ("b_test", "full22")
REQUIRED_VARIANTS = {
    "shared",
    "coordinate_only",
    "scalar_only_mlp",
    "pooled_summary_mlp",
    "fixed_permutation",
    "unshared_head",
}
REQUIRED_V1_FALSE = (
    "optimizer_initialized",
    "model_created",
    "calibration_started",
    "confirmatory_started",
    "result_emitted",
    "membership_emitted",
    "target_values_emitted",
    "membership_or_target_values_observed_by_human",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_text_sha256(path: Path) -> str:
    """Hash UTF-8 text after CRLF/CR normalization, matching the git blob bytes."""
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path.name}")
    return value


def _hex256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def audit(
    repo: Path,
    v1_receipt_path: Path | None = None,
    independent_p0_path: Path | None = None,
    baseline_export_receipt_path: Path | None = None,
) -> dict[str, Any]:
    """Return a machine-readable decision without opening any private input."""
    evaluator_path = repo / "run_node22_sealed_evaluator.py"
    runtime_path = repo / "src" / "splart" / "node22_head.py"
    protocol_path = repo / "node22_head_protocol_addendum.json"
    supersession_path = repo / "evaluator_v2_supersession.json"
    checklist_path = repo / "node22_v2_launch_gate_checklist.json"
    baseline_schema_path = repo / "node22_baseline_export_schema.json"
    d2_exporter_path = repo / "export_d2_ablation_evidence.py"
    test_path = repo / "tests" / "test_evaluator_v2_schema.py"

    failures: list[str] = []
    evidence: dict[str, Any] = {}

    def require(condition: bool, code: str) -> None:
        if not condition:
            failures.append(code)

    explicit_public_paths = (
        evaluator_path,
        runtime_path,
        protocol_path,
        supersession_path,
        checklist_path,
        baseline_schema_path,
        d2_exporter_path,
        test_path,
    )
    require(all(path.is_file() for path in explicit_public_paths), "PUBLIC-ARTIFACT-MISSING")
    if failures:
        return _result(failures, evidence)

    evaluator = evaluator_path.read_text(encoding="utf-8")
    runtime = runtime_path.read_text(encoding="utf-8")
    protocol = read_json(protocol_path)
    supersession = read_json(supersession_path)
    checklist = read_json(checklist_path)
    baseline_schema = read_json(baseline_schema_path)
    schema_test = test_path.read_text(encoding="utf-8")
    d2_exporter = d2_exporter_path.read_text(encoding="utf-8")
    source_lower = (evaluator + "\n" + runtime).lower()

    evidence.update(
        evaluator_sha256=sha256(evaluator_path),
        evaluator_canonical_text_sha256=canonical_text_sha256(evaluator_path),
        runtime_sha256=sha256(runtime_path),
        protocol_sha256=sha256(protocol_path),
        supersession_sha256=sha256(supersession_path),
        supersession_canonical_text_sha256=canonical_text_sha256(supersession_path),
        checklist_sha256=sha256(checklist_path),
        baseline_schema_canonical_text_sha256=canonical_text_sha256(baseline_schema_path),
    )

    # Protected-input firewall: this audit never receives their paths, and the
    # evaluator must reject their markers before it opens anything.
    require(all(marker in source_lower for marker in FORBIDDEN), "PROTECTED-SPLIT-DENYLIST-MISSING")
    require(set(checklist.get("protected_inputs", ())) >= {"B_test", "Full22"}, "PROTECTED-SPLIT-CHECKLIST-MISSING")
    require(protocol.get("protected_inputs") and {"B_test", "Full22"}.issubset(protocol["protected_inputs"]), "PROTECTED-SPLIT-PROTOCOL-MISSING")

    # Exact public-builder target schema and unit conversion.
    exact_target_keys = (
        "extension0_joint_units",
        "extension0_observation_units",
        "extension1_joint_units",
        "extension1_observation_units",
        "local_lower_scalar",
        "local_upper_scalar",
    )
    require(all(key in evaluator for key in exact_target_keys), "EXACT-TARGET-SCHEMA-NOT-ENFORCED")
    require("extension0_observation_units" in evaluator and "extension1_observation_units" in evaluator, "OBSERVATION-UNIT-TARGETS-MISSING")
    require("physical_endpoint_distances_normalized_by_range" not in evaluator, "WRONG-DISTANCE-DICTIONARY-USED")
    require("target_nmae_normalization" in evaluator and "local_scalar_error_multiplier" in evaluator, "NMAE-MULTIPLIER-MISSING")
    require(
        "local_scalar_error_multiplier" in schema_test
        and "normalized_object_metrics" in schema_test
        and "test_max_side_and_normalized_width" in schema_test,
        "NMAE-SEMANTIC-SYNTHETIC-TEST-MISSING",
    )

    # A max-side point metric must be explicit in the evaluator. A mean across
    # confirmatory objects is allowed only after producing one max-side score/object.
    metric_window = evaluator[evaluator.find("def normalized_object_metrics") : evaluator.find("def main")]
    has_side_max = ".amax(1)" in metric_window or re.search(r"\bmax\s*\(", metric_window) is not None
    require(has_side_max and "multiplier" in metric_window, "PRIMARY-OBJECT-SCORE-NOT-DUAL-SIDE-MAX")
    require("width=(hi-lo)*multiplier[:,None]" in metric_window.replace(" ", ""), "INTERVAL-WIDTH-NOT-NMAE-NORMALIZED")
    require(
        "max(lower_endpoint_nmae,upper_endpoint_nmae)" in evaluator
        and "abs(predicted_local_scalar-target_local_scalar)*local_scalar_error_multiplier" in evaluator,
        "NORMALIZATION-FORMULA-VALUE-NOT-ENFORCED",
    )
    require(
        re.search(r"normalized_object_metrics\s*\(\s*ep\s*,\s*ey\s*,\s*cl\s*,\s*ch\s*,\s*mul\s*\)", evaluator) is not None,
        "CONSTANT-WIDTH-NOT-NMAE-NORMALIZED",
    )

    # Runtime conformal semantics: exactly one max-side score/object and the
    # finite-sample object-level rank. Confirmatory targets never enter fitting.
    require("len(ids)!=9" in runtime and "len(set(ids))!=9" in runtime, "CONFORMAL-OBJECT-UNIQUENESS-MISSING")
    require(".amax(1)" in runtime, "CONFORMAL-SIDE-MAX-MISSING")
    require("ceil((len(ids)+1)*coverage)" in runtime.replace(" ", ""), "CONFORMAL-RANK-FORMULA-MISSING")
    require(math.ceil((9 + 1) * 0.9) == 9, "CONFORMAL-RANK-NOT-NINE")
    require("groups['calibration']" in evaluator and "groups['confirmatory']" in evaluator, "CALIBRATION-CONFIRMATORY-SEPARATION-MISSING")
    require("fit_joint_conformal(ep" not in evaluator, "CONFIRMATORY-LEAKS-INTO-CONFORMAL")
    require(
        re.search(r"len\(episodes\)\s*==\s*len\(\{\s*e\[['\"]object_id['\"]\]\s*for\s*e\s*in\s*episodes\s*\}\)\s*==\s*36", evaluator) is not None
        and "set(payload)==" in evaluator.replace(" ", ""),
        "TRUTH-OBJECT-DISJOINTNESS-NOT-ENFORCED",
    )

    # Head nulls are frozen in source. D2 baselines/ablations and the learned-scale
    # ablation may be bound by a separate frozen public plan, but must exist before
    # broad CVPR contribution claims.
    variant_match = re.search(r"VARIANTS\s*=\s*\((.*?)\)", runtime, re.S)
    variants = set(re.findall(r"['\"]([^'\"]+)['\"]", variant_match.group(1))) if variant_match else set()
    require(variants == REQUIRED_VARIANTS, "HEAD-VARIANT-SET-MISMATCH")
    required_methods = set(baseline_schema.get("required_methods", ()))
    require(required_methods >= {"scratch", "symmetric_linear", "global_prior_train18", "range_prior_train18", "full_d2"}, "D2-BASELINE-PLAN-MISSING")
    require(required_methods >= {"no_contact", "no_penetration", "no_terminal_support", "single_radius"}, "D2-ABLATION-PLAN-MISSING")
    require(bool({"frozen_d2_no_learned_head", "full_d2"} & required_methods), "FROZEN-D2-HEAD-NULL-MISSING")
    require("shared_distance_only" in required_methods and "shared_distance_only" in variants, "LEARNED-SCALE-ABLATION-MISSING")
    require(
        re.search(r"len\(rows\)\s*!=\s*36|len\(rows\)\s*==\s*36", d2_exporter) is not None,
        "D2-EXPORTER-EXACT36-NOT-ENFORCED",
    )
    require("reject_private" in d2_exporter or "PRIVATE_KEYS" in d2_exporter, "D2-EXPORTER-PRIVATE-FIELDS-NOT-REJECTED")
    require("path traversal" in d2_exporter.lower() or "is_relative_to" in d2_exporter or ".name !=" in d2_exporter, "D2-EXPORTER-OBJECT-ID-PATH-UNSAFE")

    if baseline_export_receipt_path is None:
        candidate = repo / "node22_target_free_baseline_export_receipt.json"
        baseline_export_receipt_path = candidate if candidate.is_file() else None
    if baseline_export_receipt_path is None or not baseline_export_receipt_path.is_file():
        failures.append("BASELINE-EXPORTS-NOT-FROZEN-READY")
    else:
        exported = read_json(baseline_export_receipt_path)
        evidence["baseline_export_receipt_sha256"] = sha256(baseline_export_receipt_path)
        require(exported.get("status") == "PASS", "BASELINE-EXPORT-RECEIPT-NOT-PASS")
        require(set(exported.get("methods", ())) >= required_methods, "BASELINE-EXPORT-METHODS-INCOMPLETE")
        require(exported.get("objects_per_method") == 36, "BASELINE-EXPORT-OBJECT-COUNT-MISMATCH")
        require(exported.get("targets_read") == [] and exported.get("split_membership_read") == [], "BASELINE-EXPORT-REPORTS-PRIVATE-READ")
        require(_hex256(exported.get("artifact_index_sha256")), "BASELINE-EXPORT-INDEX-HASH-MISSING")

    # The supersession is a new milestone, never a retry under v1. All corrections
    # must be enumerated because the code reaches both parser and metric semantics.
    require(supersession.get("status") == "FROZEN_BEFORE_V2_TARGET_READ", "V2-NOT-FROZEN-BEFORE-TARGET")
    require(supersession.get("v1_rerun") is False, "V1-RERUN-NOT-FORBIDDEN")
    require(supersession.get("v3_allowed") is False, "V3-NOT-FORBIDDEN")
    require(supersession.get("method_config_seed_split_salt_targets_unchanged") is True, "IMMUTABLE-DESIGN-BINDING-MISSING")
    require(supersession.get("new_output_root_required") is True, "NEW-OUTPUT-ROOT-NOT-REQUIRED")
    require(supersession.get("protected_splits_read") == [] and supersession.get("targets_read") == [], "V2-FREEZE-RECEIPT-REPORTS-PRIVATE-READ")
    repairs = " ".join(supersession.get("implementation_repairs", supersession.get("only_changes", ())))
    require(all(term in repairs for term in ("observation_units", "local_scalar_error_multiplier", "max side", "width")), "V2-REPAIR-ENUMERATION-INCOMPLETE")
    require(_hex256(supersession.get("supersedes_runner_sha256")), "V1-RUNNER-HASH-MISSING")
    require(supersession.get("v2_runner_sha256") == evidence["evaluator_canonical_text_sha256"], "V2-RUNNER-HASH-NOT-FROZEN")
    require("AUTHORIZED_FOR_V2_EVALUATOR" in evaluator, "V2-AUTHORIZATION-STATUS-NOT-ENFORCED")

    # Sanitized v1 evidence is mandatory; absence is a hard failure rather than a
    # reason to inspect private evaluator storage.
    if v1_receipt_path is None or not v1_receipt_path.is_file():
        failures.append("V1-TERMINAL-EVIDENCE-MISSING")
    else:
        v1 = read_json(v1_receipt_path)
        evidence["v1_terminal_receipt_sha256"] = sha256(v1_receipt_path)
        require(v1.get("schema") == "splart-node22-evaluator-v1-terminal-evidence/v1", "V1-EVIDENCE-SCHEMA-MISMATCH")
        require(v1.get("status") == "FAILED_TERMINAL", "V1-NOT-FAILED-TERMINAL")
        require(v1.get("truth_deserialized") is True, "V1-TRUTH-DESERIALIZATION-NOT-RECORDED")
        require(all(v1.get(key) is False for key in REQUIRED_V1_FALSE), "V1-POST-FAILURE-ACTIVITY-NOT-RULED-OUT")
        require(v1.get("runner_sha256") == supersession.get("supersedes_runner_sha256"), "V1-RUNNER-BINDING-MISMATCH")
        require(all(_hex256(v1.get(key)) for key in ("run_guard_sha256", "stdout_sha256", "stderr_sha256", "artifact_inventory_sha256")), "V1-FORENSIC-HASHES-MISSING")
        require(v1.get("error_stage") == "TARGET_SCHEMA_PARSE", "V1-ERROR-STAGE-MISMATCH")
        require(v1.get("forbidden_artifacts_found") == [], "V1-FORBIDDEN-ARTIFACTS-FOUND")

    if independent_p0_path is None:
        candidate = repo / "node22_v2_source_request_p0.json"
        independent_p0_path = candidate if candidate.is_file() else None
    if independent_p0_path is None or not independent_p0_path.is_file():
        failures.append("V2-INDEPENDENT-P0-MISSING")
    else:
        p0 = read_json(independent_p0_path)
        evidence["independent_p0_sha256"] = sha256(independent_p0_path)
        p0_pass = p0.get("status") == "PASS" and p0.get("launch_authorized") is True
        require(p0_pass, "V2-INDEPENDENT-P0-NOT-AUTHORIZED")
        require(p0.get("v2_runner_sha256", p0.get("runner_sha256")) == evidence["evaluator_canonical_text_sha256"], "V2-P0-RUNNER-BINDING-MISMATCH")
        require(p0.get("supersession_sha256") == evidence["supersession_canonical_text_sha256"], "V2-P0-SUPERSESSION-BINDING-MISMATCH")
        require(p0.get("baseline_export_schema_sha256") == evidence["baseline_schema_canonical_text_sha256"], "V2-P0-BASELINE-SCHEMA-BINDING-MISMATCH")
        if p0_pass:
            require(p0.get("one_execution_only") is True, "V2-P0-ONE-SHOT-MISSING")
        no_target_read = p0.get("targets_read", p0.get("target_values_read", [])) == []
        no_membership_read = p0.get("split_membership_read", []) == []
        require(p0.get("protected_splits_read") == [] and no_target_read and no_membership_read, "V2-P0-REPORTS-PRIVATE-READ")

    return _result(failures, evidence)


def _result(failures: list[str], evidence: dict[str, Any]) -> dict[str, Any]:
    unique = sorted(set(failures))
    passed = not unique
    return {
        "schema": "splart-node22-v2-launch-gate/v1",
        "status": "AUTHORIZED_FOR_V2_EVALUATOR" if passed else "FAIL",
        "launch_authorized": passed,
        "one_execution_only": passed,
        "failures": unique,
        "evidence": evidence,
        "private_inputs_opened": False,
        "protected_splits_read": [],
        "targets_read": [],
        "claim_ceiling": (
            "transparent superseding recovery milestone; not a pristine first-attempt one-shot"
            if passed
            else "no v2 evaluation claim"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--v1-terminal-receipt", type=Path)
    parser.add_argument("--independent-p0", type=Path)
    parser.add_argument("--baseline-export-receipt", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.repo.resolve(), args.v1_terminal_receipt, args.independent_p0, args.baseline_export_receipt)
    encoded = json.dumps(result, sort_keys=True, indent=2) + "\n"
    if args.output:
        if args.output.exists():
            raise FileExistsError(args.output)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    if result["status"] != "AUTHORIZED_FOR_V2_EVALUATOR":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
