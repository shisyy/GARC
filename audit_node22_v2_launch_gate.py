#!/usr/bin/env python3
"""Final fail-closed P0 audit for the node2.2 v2 recovery milestone.

Only public source, protocol JSON, and a sanitized remote preflight receipt are
opened. The sealed truth is never parsed by this program.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

RUNNER_SHA = "97c12d076faa3dbfa307b70a1d5feabcf09080600e309babcc1f98612cb0ff5a"
HEAD_SHA = "20cbc6557404506f82e4b1380d98100eac985b852c52dd43bc3a335440de5f76"
SCHEMA_SHA = "b41ebf66c515c0e42dd34501ef4ab3199beeb45823a1078e6ef80ac8391ef034"
SUPERSESSION_SHA = "46f9b99d912fb418d0a7769eb02287a4d4745d70e7b208ded581b994be2553dd"
V1_RUNNER_SHA = "96820781c2cac1692e9e4e02ee3855fed257ee3d9eb13aecd3bd556992d930c5"
TRUTH_SHA = "6d0e4c57871363b69335feaa50d8ed94746d8aa3be4787fafd230f1736c516dc"
INDEX_SHAS = ["328e9dffd2e8d9ad5c7372160bf3ca3ab7f2f076d72520a2fc5cc00d1ced3985", "c539e423db9d957f873dbb6b381728c6eb8840a6d5cbc1479cd388b3b6e61991"]
PROFILE_SET_SHA = "52043406521635d072fce912fb3c7fb0cd2ddf9dc96300270f5097a084491392"
BASELINE_SHA = "efe29fd991f29fa89be762cc5662b962482c06a866fca6045eca917da23cdeb6"
OUTPUT_PATH = "/home/yptang/arbor-runs/splart-node22-sealed-eval-v2/output"
VARIANTS = {"shared", "shared_distance_only", "coordinate_only", "scalar_only_mlp", "pooled_summary_mlp", "fixed_permutation", "unshared_head"}
RAW_METHODS = {"scratch", "symmetric_linear", "full_d2", "single_radius", "no_contact", "no_penetration", "no_terminal_support"}


def canonical_sha(path: Path) -> str:
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(text.encode()).hexdigest()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"not an object: {path}")
    return value


def audit(repo: Path, remote_receipt_path: Path | None = None) -> dict[str, Any]:
    remote_receipt_path = remote_receipt_path or repo / "node22_v2_remote_preflight.json"
    paths = {
        "runner": repo / "run_node22_sealed_evaluator.py",
        "head": repo / "src/splart/node22_head.py",
        "schema": repo / "node22_baseline_export_schema.json",
        "sup": repo / "evaluator_v2_supersession.json",
        "v1": repo / "v1_failure_sanitized_binding.json",
        "protocol": repo / "node22_head_protocol_addendum.json",
        "tests": repo / "tests/test_evaluator_v2_schema.py",
        "remote": remote_receipt_path,
    }
    failures: list[str] = []

    def check(value: bool, code: str) -> None:
        if not value:
            failures.append(code)

    check(all(path.is_file() for path in paths.values()), "ARTIFACT-MISSING")
    if failures:
        return result(failures, {})
    runner = paths["runner"].read_text(encoding="utf-8")
    head = paths["head"].read_text(encoding="utf-8")
    schema, sup, v1, protocol, remote = (load(paths[x]) for x in ("schema", "sup", "v1", "protocol", "remote"))
    tests = paths["tests"].read_text(encoding="utf-8")
    evidence = {
        "runner_sha256": canonical_sha(paths["runner"]),
        "head_sha256": canonical_sha(paths["head"]),
        "baseline_schema_sha256": canonical_sha(paths["schema"]),
        "supersession_sha256": canonical_sha(paths["sup"]),
        "v1_binding_sha256": canonical_sha(paths["v1"]),
        "remote_preflight_sha256": canonical_sha(paths["remote"]),
    }

    check(evidence["runner_sha256"] == RUNNER_SHA == sup.get("v2_runner_sha256"), "RUNNER-HASH")
    check(evidence["head_sha256"] == HEAD_SHA == sup.get("node22_head_sha256"), "HEAD-HASH")
    check(evidence["baseline_schema_sha256"] == SCHEMA_SHA == sup.get("baseline_schema_sha256"), "BASELINE-SCHEMA-HASH")
    check(evidence["supersession_sha256"] == SUPERSESSION_SHA, "SUPERSESSION-HASH")
    repairs = sup.get("implementation_repairs", [])
    check(sup.get("schema") == "splart-node22-evaluator-supersession/v4", "SUPERSESSION-SCHEMA")
    check(sup.get("status") == "FROZEN_BEFORE_V2_TARGET_READ", "SUPERSESSION-STATUS")
    check(len(repairs) == 10 and "neither a V1 retry nor a pristine first-attempt run" in sup.get("reason", ""), "RECOVERY-DISCLOSURE")
    check(sup.get("method_config_seed_split_salt_targets_unchanged") is True, "IMMUTABLE-DESIGN")
    check(sup.get("one_execution_only") is True and sup.get("v1_rerun") is False and sup.get("v3_allowed") is False, "ONE-SHOT-NO-V3")
    check(sup.get("v2_output_absolute_path") == OUTPUT_PATH, "OUTPUT-BINDING")
    check(sup.get("truth_sha256") == TRUTH_SHA and sup.get("profile_index_sha256") == INDEX_SHAS, "PRIVATE-INPUT-HASH-BINDING")
    check(sup.get("profile_set_sha256") == PROFILE_SET_SHA and sup.get("baseline_export_sha256") == BASELINE_SHA, "PUBLIC-INPUT-HASH-BINDING")
    check(sup.get("protected_splits_read") == [] and sup.get("targets_read") == [], "SUPERSESSION-PRIVATE-READ")

    check(v1.get("schema") == "splart-node22-v1-failure-binding/v2" and v1.get("status") == "FAILED_TERMINAL", "V1-TERMINAL")
    check(v1.get("v1_runner_sha256") == V1_RUNNER_SHA == sup.get("supersedes_runner_sha256"), "V1-RUNNER-BINDING")
    check(all(re.fullmatch(r"[0-9a-f]{64}", str(v1.get(k, ""))) for k in ("run_guard_sha256", "stdout_stderr_sha256", "inventory_sha256", "abort_receipt_sha256")), "V1-FORENSIC-HASHES")
    check(v1.get("target_file_deserialized") is True and v1.get("target_values_consumed_for_training") is False, "V1-TARGET-STAGE")
    check(v1.get("optimizer_initialized") is False and v1.get("model_or_score_written") is False, "V1-NO-TRAIN-OR-SCORE")
    check(v1.get("retry_authorized") is False and v1.get("membership_or_target_emitted") is False, "V1-NO-RETRY-OR-LEAK")

    lower = runner.lower()
    check(all(x in lower for x in ("b_test", "full22")), "PROTECTED-PATH-DENYLIST")
    check("a.index+[a.truth,a.authorization,a.baseline_export,a.output]" in runner.replace(" ", ""), "BASELINE-PATH-NOT-SCANNED")
    check("if set(auth)!=set(expected) or auth!=expected" in runner, "AUTH-NOT-EXACT")
    for token in ("runner_sha256", "head_sha256", "baseline_schema_sha256", "supersession_sha256", "profile_indexes", "baseline_export", "output_absolute_path", "one_execution_only", "v3_allowed"):
        check(token in runner, "AUTH-MISSING-" + token.upper())
    check(runner.index("TARGET_VALUES_CONSUMING") < runner.index("y=torch.tensor([target_pair"), "TARGET-STATE-LATE")
    check("failed_from_stage" in runner and "prior=json.loads(state.read_text())" in runner, "FAILURE-NOT-STAGE-AWARE")
    check("if out.exists():raise FileExistsError" in runner and "v3_allowed':False" in runner, "OUTPUT-NOT-WRITE-ONCE")
    check("len(episodes)==len({e['object_id'] for e in episodes})==36" in runner and "==set(payload)" in runner, "OBJECT-SETS-NOT-EXACT")

    for token in ("extension0_observation_units", "extension1_observation_units", "local_scalar_error_multiplier", "max(lower_endpoint_nmae,upper_endpoint_nmae)"):
        check(token in runner, "METRIC-MISSING-" + token.upper())
    check("err.amax(1)" in runner and "scores=((cp-cy).abs()*cm[:,None]).amax(1)" in runner, "MAX-SIDE-MISSING")
    check("r=float(scores.sort().values[8])" in runner and "fit_joint_conformal(cp,cs,cy" in runner, "OBJECT-CONFORMAL-RANK")
    check("normalized_mean_width" in runner and "constant_normalized_mean_width" in runner and "((ch-cl)*em[:,None]).mean()" in runner, "NORMALIZED-WIDTHS")
    for token in ("lower_nmae", "upper_nmae", "endpoint_nmae", "median_endpoint_nmae", "wins_vs_full_d2", "joint_coverage", "swap_error", "wall_time_seconds", "endpoint_nmae_bootstrap95_ci"):
        check(token in runner, "RESULT-MISSING-" + token.upper())
    check("replicates':10000" in runner and "unit':'confirmatory object'" in runner and "selection_use':False" in runner, "BOOTSTRAP-CONTRACT")
    check("per_object_values_emitted':False" in runner and "membership_emitted':False" in runner and "targets_emitted':False" in runner, "AGGREGATE-ONLY-OUTPUT")
    check("tr=groups['train']" in runner and "global_prior_train18" in runner and "range_prior_train18" in runner, "TRAIN18-PRIORS")
    check("sealed train18 prior fit plus confirmatory aggregation" in runner, "PRIOR-RUNTIME")

    variant_match = re.search(r"VARIANTS\s*=\s*\((.*?)\)", head, re.S)
    variants = set(re.findall(r"['\"]([^'\"]+)['\"]", variant_match.group(1))) if variant_match else set()
    check(variants == VARIANTS, "HEAD-VARIANTS")
    check("shared_distance_only" in head and "torch.ones_like" in head, "DISTANCE-ONLY-ABLATION")
    check("on_optimizer_initialized" in head and "opt=torch.optim.AdamW" in head and head.index("opt=torch.optim.AdamW") < head.index("on_optimizer_initialized()"), "OPTIMIZER-STAGE")
    check(set(schema.get("required_methods", [])) == RAW_METHODS, "RAW-BASELINE-METHODS")
    check(set(schema.get("forbidden_fields", [])) >= {"score", "metrics", "aggregate", "target", "split"}, "RAW-BASELINE-FIREWALL")
    check("test_authorization_exact_hash_and_path" in tests and "test_fixed_object_bootstrap_is_reproducible" in tests, "CRITICAL-TESTS-MISSING")
    check({"B_test", "Full22"}.issubset(set(protocol.get("protected_inputs", []))), "PROTOCOL-PROTECTED-SPLITS")

    check(remote.get("schema") == "splart-node22-v2-remote-preflight/v1" and remote.get("status") == "PASS", "REMOTE-PREFLIGHT")
    check(remote.get("source_commit") == "57b931dc53e306221b9bd869f1bb2553b213e3a0", "REMOTE-SOURCE-COMMIT")
    check(remote.get("conda_tests") == {"passed": 18, "failed": 0}, "REMOTE-CONDA-TESTS")
    check(remote.get("runner_sha256") == RUNNER_SHA and remote.get("head_sha256") == HEAD_SHA and remote.get("supersession_sha256") == SUPERSESSION_SHA, "REMOTE-SOURCE-HASHES")
    check(remote.get("truth") == {"path": "/home/yptang/arbor-sealed/splart-node7.2/sealed-truth-v1.json", "sha256": TRUTH_SHA, "parsed": False}, "REMOTE-TRUTH-BINDING")
    check([x.get("sha256") for x in remote.get("profile_indexes", [])] == INDEX_SHAS, "REMOTE-INDEX-HASHES")
    baseline = remote.get("baseline_export", {})
    check(baseline.get("sha256") == BASELINE_SHA and set(baseline.get("methods", [])) == RAW_METHODS, "REMOTE-BASELINE")
    check(baseline.get("rows_per_method") == 36 and baseline.get("id_sets_equal") is True and baseline.get("profile_set_sha256") == PROFILE_SET_SHA, "REMOTE-BASELINE-ROWS")
    check(baseline.get("forbidden_fields_found") == [] and baseline.get("d2_provenance_hashes_valid") is True, "REMOTE-BASELINE-PURITY")
    check(remote.get("output") == {"path": OUTPUT_PATH, "exists": False}, "REMOTE-OUTPUT-NOT-FRESH")
    check(remote.get("sealed_content_parsed") is False and remote.get("evaluator_started") is False, "P0-TOUCHED-EVALUATOR")
    return result(failures, evidence)


def result(failures: list[str], evidence: dict[str, Any]) -> dict[str, Any]:
    failures = sorted(set(failures)); ok = not failures
    return {"schema": "splart-node22-v2-source-request-p0/v2", "status": "PASS" if ok else "FAIL", "launch_authorized": ok, "one_execution_only": ok, "v3_allowed": False, "failures": failures, "evidence": evidence, "evaluator_started": False, "sealed_content_parsed": False, "membership_emitted": False, "targets_emitted": False, "protected_splits_read": [], "claim_ceiling": "transparent v2 recovery; not a pristine first-attempt run" if ok else "no v2 claim"}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parent); parser.add_argument("--remote-preflight", type=Path); parser.add_argument("--output", type=Path); args = parser.parse_args()
    value = audit(args.repo.resolve(), args.remote_preflight); encoded = json.dumps(value, sort_keys=True, indent=2) + "\n"
    if args.output:
        if args.output.exists(): raise FileExistsError(args.output)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    if value["status"] != "PASS": raise SystemExit(1)


if __name__ == "__main__": main()
