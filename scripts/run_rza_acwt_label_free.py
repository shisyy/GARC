#!/usr/bin/env python3
"""Run the unique score-free RZA-ACWT feasibility candidate.

The runner opens feature/provenance payloads only.  It has no label or score
argument and writes an atomic, independently recomputed feasibility receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from scripts.run_smarc_source_gate import (
    FROZEN_CONFIG_SHA256 as SOURCE_CONFIG_SHA256,
    _atomic_json,
    _begin_atomic_directory,
    _finish_atomic_directory,
    conditional_overrides,
    indices_for,
    merge_features,
)
from splart.frozen_visual import sha256_file
from splart.rza_acwt import (
    canonical_object_scalar,
    feature_crossfit_rza_acwt,
    global_row_identity,
    preserve_recipient_displacement,
    receipt_passes,
    rza_acwt_ablation,
)
from splart.smarc_source import fit_feature_preprocessor, hash_ids, transform


FROZEN_CONFIG_SHA256 = "779874c52dc3c64b3e6606ad11c4b0a2e3d07fe5b814054b7d2ce8a0ff852be5"
PREDECESSOR_CONFIG_SHA256 = "e3d6c1b3d59e2fd4c155fcff7e818cce0af95343da81c9d2f01bb2990c0a5acc"
CONTEXTS = {"semantic": "semantic_rza_acwt/final-source-features",
            "mechanical": "mechanical_rza_acwt/final-source-features"}


def text_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def canonical_sha256(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    return hashlib.sha256(payload.encode()).hexdigest()


def verify_provenance_binding(result: dict, receipt: dict) -> bool:
    keys = {"schema", "config_sha256", "feasibility_sha256", "decision", "code_sha256",
            "feature_provenance_sha256", "source_labels_opened",
            "source_labels_hashed", "source_scores_computed", "training_started",
            "remote_execution_started", "execution_environment", "execution_phase",
            "box_labels_read", "protected_splits_read"}
    if set(receipt) != keys or receipt.get("schema") != "splart-rza-acwt-feasibility-receipt/v1":
        return False
    if receipt.get("config_sha256") != FROZEN_CONFIG_SHA256:
        return False
    if result.get("schema") != "splart-rza-acwt-feasibility/v1" or result.get("config_sha256") != FROZEN_CONFIG_SHA256:
        return False
    if receipt.get("decision") not in {"REQUEST_SOURCE_SCORE_AUTHORIZATION", "PRUNE_LABEL_FREE"}:
        return False
    if not isinstance(receipt.get("feasibility_sha256"), str) or len(receipt["feasibility_sha256"]) != 64:
        return False
    if receipt.get("code_sha256") != canonical_sha256(result.get("code_files_sha256")):
        return False
    if receipt.get("feature_provenance_sha256") != canonical_sha256(result.get("feature_provenance")):
        return False
    if result.get("code_sha256") != receipt["code_sha256"] or result.get("feature_provenance_sha256") != receipt["feature_provenance_sha256"]:
        return False
    false_flags = ("source_labels_opened", "source_labels_hashed", "source_scores_computed",
                   "training_started")
    return (all(result.get(key) is False and receipt.get(key) is False for key in false_flags)
            and result.get("remote_execution_started") is True
            and receipt.get("remote_execution_started") is True
            and result.get("execution_environment") == receipt.get("execution_environment") == "CASIA_98"
            and result.get("execution_phase") == receipt.get("execution_phase") == "label_free_feasibility"
            and result.get("box_labels_read") == receipt.get("box_labels_read") == []
            and result.get("protected_splits_read") == receipt.get("protected_splits_read") == [])


def contract(config: dict, source_config: dict, predecessor: dict) -> dict:
    unchanged = source_config["conditional_residual_contract"]["gates"]
    predecessor_gates = predecessor["gates"]
    for key in ("reconstruction_max_error", "orthogonality_max",
                "held_boundary_clipped_fraction_max", "residual_energy_ratio_at_least",
                "shuffle_rms_over_original_sd_at_least", "shuffled_norm_p99_ratio_at_most",
                "train_coverage", "held_coverage", "train_self_rate",
                "held_effective_donors_per_object"):
        if config["gates"][key] != predecessor_gates[key]:
            raise RuntimeError(f"node8.10 gate changed: {key}")
    if (config["crossfit"]["folds"] != predecessor["crossfit"]["folds"]
            or config["crossfit"]["absolute_spearman_at_most"] != predecessor["crossfit"]["absolute_spearman_at_most"]
            or config["crossfit"]["distance_correlation_at_most"] != predecessor["crossfit"]["distance_correlation_at_most"]):
        raise RuntimeError("node8.10 crossfit gate changed")
    if unchanged.get("normalized_residual_mean_max") != 1e-10 or unchanged.get("normalized_residual_z_correlation_max") != 1e-10:
        raise RuntimeError("node8.9 raw gates changed")
    return {
        "rank_relative_tolerance": config["numerics"]["rank_relative_tolerance"],
        "condition_max": config["numerics"]["condition_max"],
        "eigengap_multiplier": config["numerics"]["eigengap_multiplier"],
        "crossfit_folds": config["crossfit"]["folds"],
        "crossfit_absolute_spearman_at_most": config["crossfit"]["absolute_spearman_at_most"],
        "crossfit_distance_correlation_at_most": config["crossfit"]["distance_correlation_at_most"],
        "normalized_residual_mean_max": unchanged["normalized_residual_mean_max"],
        "normalized_residual_raw_z_correlation_max": unchanged["normalized_residual_z_correlation_max"],
        **{key: value for key, value in config["gates"].items()
           if isinstance(value, (int, float)) and not isinstance(value, bool)},
    }


def _canonicalize(rows, indices):
    result = sorted(indices, key=lambda i: global_row_identity(rows[i]))
    identities = [global_row_identity(rows[i]) for i in result]
    if len(identities) != len(set(identities)):
        raise ValueError("global row identity collision")
    return result


def _remap(rows, indices, value):
    return {global_row_identity(rows[i]): value[k] for k, i in enumerate(indices)}


def _bit_equal_mapping(left, right) -> bool:
    return set(left) == set(right) and all(torch.equal(left[key], right[key]) for key in left)


def run_cell(rows, train, held, train_field, held_field, train_z, held_z,
             cfg, context, field_kind):
    first_crossfit = feature_crossfit_rza_acwt(rows, train, field_kind, cfg)
    second_crossfit = feature_crossfit_rza_acwt(rows, train, field_kind, cfg)
    first = rza_acwt_ablation(rows, train, held, train_field, held_field,
                              train_z, held_z, cfg, context, first_crossfit)
    second = rza_acwt_ablation(rows, train, held, train_field, held_field,
                               train_z, held_z, cfg, context, second_crossfit)
    for domain in first[2]["domains"]:
        left = first[2]["domains"][domain]["crossfit"]
        right = second[2]["domains"][domain]["crossfit"]
        if left is right or left["audit_payload"] is right["audit_payload"] or left["folds"] is right["folds"]:
            raise RuntimeError("independent receipts share nested object identity")
    repeat = (torch.equal(first[0], second[0]) and torch.equal(first[1], second[1])
              and first[2] == second[2])
    reverse_train, reverse_held = list(reversed(train)), list(reversed(held))
    reverse_crossfit = feature_crossfit_rza_acwt(rows, reverse_train, field_kind, cfg)
    reverse = rza_acwt_ablation(rows, reverse_train, reverse_held, train_field.flip(0),
                                held_field.flip(0), train_z, held_z, cfg, context, reverse_crossfit)
    invariant = (_bit_equal_mapping(_remap(rows, train, first[0]),
                                    _remap(rows, reverse_train, reverse[0]))
                 and _bit_equal_mapping(_remap(rows, held, first[1]),
                                        _remap(rows, reverse_held, reverse[1]))
                 and first[2] == reverse[2])
    return first[0], first[1], first[2], second[2], {
        "repeat_bit_identical": repeat,
        "row_order_invariant": invariant,
        "candidate_prediction_sha256": canonical_sha256({
            "train": {key: value.tolist() for key, value in _remap(rows, train, first[0]).items()},
            "held": {key: value.tolist() for key, value in _remap(rows, held, first[1]).items()}}),
        "gate_recompute_prediction_sha256": canonical_sha256({
            "train": {key: value.tolist() for key, value in _remap(rows, train, second[0]).items()},
            "held": {key: value.tolist() for key, value in _remap(rows, held, second[1]).items()}}),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--articraft-shards", type=Path, nargs=3, required=True)
    parser.add_argument("--njc-cache", type=Path, required=True)
    parser.add_argument("--articraft-pilc-data", type=Path, required=True)
    parser.add_argument("--njc-pilc-data", type=Path, required=True)
    parser.add_argument("--render-logs", type=Path, nargs=4, required=True)
    parser.add_argument("--dino-checkpoint", type=Path, required=True)
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--predecessor-config", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or (args.output.parent / (args.output.name + ".atomic-tmp")).exists():
        raise FileExistsError(args.output)
    config = json.loads(args.config.read_text())
    source_config = json.loads(args.source_config.read_text())
    predecessor = json.loads(args.predecessor_config.read_text())
    if (text_sha256(args.config) != FROZEN_CONFIG_SHA256
            or text_sha256(args.source_config) != SOURCE_CONFIG_SHA256
            or text_sha256(args.predecessor_config) != PREDECESSOR_CONFIG_SHA256):
        raise RuntimeError("frozen config hash mismatch")
    forbidden_names = {"labels", "scores", "targets"}
    if (forbidden_names & set(config) or config.get("source_labels_opened")
            or config.get("source_labels_hashed") or config.get("source_scores_computed")
            or config.get("training_started") or config.get("remote_execution_started") is not False):
        raise RuntimeError("score-free boundary violated")
    rows, provenance = merge_features(args, source_config)
    if provenance.get("target_payloads_read") != [] or provenance.get("labels_opened") is not False:
        raise RuntimeError("target payload was accessed")
    if [sha256_file(path) for path in args.render_logs] != config["source"]["render_log_sha256"]:
        raise RuntimeError("renderer log hash mismatch")
    if any("overflow" in path.read_text(errors="replace").lower() for path in args.render_logs):
        raise RuntimeError("renderer overflow marker")
    if sha256_file(args.dino_checkpoint) != config["source"]["dino_checkpoint_sha256"]:
        raise RuntimeError("DINO hash mismatch")
    all_identities = [global_row_identity(row) for row in rows]
    if len(all_identities) != len(set(all_identities)):
        raise RuntimeError("global row identity collision")
    train = _canonicalize(rows, indices_for(rows, {"endpoint_pretrain", "njc_train"}))
    held = _canonicalize(rows, indices_for(rows, {"endpoint_validation", "njc_validation"}))
    observed = {
        "articraft_train": hash_ids(sorted({rows[i]["object_group_id"] for i in train if rows[i]["domain"] == "articraft"})),
        "articraft_held": hash_ids(sorted({rows[i]["object_group_id"] for i in held if rows[i]["domain"] == "articraft"})),
        "njc_train": hash_ids(sorted({rows[i]["object_group_id"] for i in train if rows[i]["domain"] == "njc"})),
        "njc_held": hash_ids(sorted({rows[i]["object_group_id"] for i in held if rows[i]["domain"] == "njc"})),
    }
    if observed != config["source"]["object_list_hashes"]:
        raise RuntimeError("frozen object-list hash mismatch")
    preprocessor = fit_feature_preprocessor(rows, train, 16)
    train_mech, train_sem, _ = transform(preprocessor, rows, train)
    held_mech, held_sem, _ = transform(preprocessor, rows, held)
    _, linear_history = conditional_overrides(rows, train, held, preprocessor, source_config)
    cfg = contract(config, source_config, predecessor)
    semantic = run_cell(
        rows, train, held, train_sem, held_sem,
        canonical_object_scalar(rows, train, "semantic"),
        canonical_object_scalar(rows, held, "semantic"), cfg, CONTEXTS["semantic"], "semantic")
    mechanical = run_cell(
        rows, train, held, train_mech[:, :-1], held_mech[:, :-1],
        canonical_object_scalar(rows, train, "mechanical", train),
        canonical_object_scalar(rows, held, "mechanical", train), cfg, CONTEXTS["mechanical"], "mechanical")
    mechanical_train = preserve_recipient_displacement(mechanical[0], train_mech)
    mechanical_held = preserve_recipient_displacement(mechanical[1], held_mech)
    displacement_equal = (torch.equal(mechanical_train[:, -1], train_mech[:, -1])
                          and torch.equal(mechanical_held[:, -1], held_mech[:, -1]))
    domains = {"articraft", "njc"}
    cell_pass, full_receipt_pass, domain_receipt_pass = {}, {}, {}
    for name, cell in (("semantic", semantic), ("mechanical", mechanical)):
        receipt, expected, audit = cell[2], cell[3], cell[4]
        full_receipt_pass[name] = receipt_passes(receipt, expected, cfg, CONTEXTS[name], domains)
        cell_pass[name], domain_receipt_pass[name] = {}, {}
        for domain in sorted(domains):
            single = {"schema": receipt["schema"], "context": receipt["context"],
                      "contract_sha256": receipt["contract_sha256"],
                      "domains": {domain: receipt["domains"][domain]}}
            expected_single = {"schema": expected["schema"], "context": expected["context"],
                               "contract_sha256": expected["contract_sha256"],
                               "domains": {domain: expected["domains"][domain]}}
            domain_gate = receipt_passes(single, expected_single, cfg, CONTEXTS[name], {domain})
            domain_receipt_pass[name][domain] = domain_gate
            cell_pass[name][domain] = (domain_gate and audit["repeat_bit_identical"]
                                       and audit["row_order_invariant"]
                                       and audit["candidate_prediction_sha256"] == audit["gate_recompute_prediction_sha256"]
                                       and (name != "mechanical" or displacement_equal))
    root = Path(__file__).resolve().parents[1]
    code_files = (Path(__file__).resolve(), root / "src" / "splart" / "rza_acwt.py",
                  root / "src" / "splart" / "rqlsot.py", root / "src" / "splart" / "conditional_residual.py",
                  root / "scripts" / "run_rqlsot_feasibility.py", root / "scripts" / "run_smarc_source_gate.py",
                  root / "src" / "splart" / "smarc_source.py", root / "src" / "splart" / "smarc.py",
                  root / "src" / "splart" / "frozen_visual.py", root / "scripts" / "build_smarc_articraft_shard.py",
                  root / "scripts" / "build_smarc_smoke.py")
    code_hashes = {path.relative_to(root).as_posix(): sha256_file(path) for path in code_files}
    feature_sha = canonical_sha256(provenance)
    code_sha = canonical_sha256(code_hashes)
    result = {"schema": "splart-rza-acwt-feasibility/v1", "config_sha256": FROZEN_CONFIG_SHA256,
              "source_config_sha256": SOURCE_CONFIG_SHA256,
              "predecessor_config_sha256": PREDECESSOR_CONFIG_SHA256,
              "feature_provenance": provenance, "feature_provenance_sha256": feature_sha,
              "node89_linear_historical": linear_history,
              "rza_acwt": {"semantic": semantic[2], "mechanical": mechanical[2]},
              "independent_recomputation_sha256": {
                  "semantic": canonical_sha256(semantic[3]), "mechanical": canonical_sha256(mechanical[3])},
              "runtime_audits": {"semantic": semantic[4],
                  "mechanical": {**mechanical[4], "recipient_displacement_bitwise_unchanged": displacement_equal}},
              "full_receipt_pass": full_receipt_pass, "domain_receipt_pass": domain_receipt_pass,
              "cell_pass": cell_pass,
              "all_pass": (all(full_receipt_pass.values())
                           and all(value for part in cell_pass.values() for value in part.values())),
              "object_list_hashes": observed, "global_row_identity_sha256": canonical_sha256(sorted(all_identities)),
              "code_files_sha256": code_hashes, "code_sha256": code_sha,
              "source_labels_opened": False, "source_labels_hashed": False,
              "source_scores_computed": False, "training_started": False,
              "remote_execution_started": True, "execution_environment": "CASIA_98",
              "execution_phase": "label_free_feasibility",
              "box_labels_read": [], "protected_splits_read": []}
    if canonical_sha256(result["feature_provenance"]) != feature_sha or canonical_sha256(result["code_files_sha256"]) != code_sha:
        raise RuntimeError("provenance self-check failed")
    temporary = _begin_atomic_directory(args.output)
    _atomic_json(temporary / "feasibility.json", result)
    output_receipt = {"schema": "splart-rza-acwt-feasibility-receipt/v1",
        "config_sha256": FROZEN_CONFIG_SHA256,
        "feasibility_sha256": sha256_file(temporary / "feasibility.json"),
        "decision": "REQUEST_SOURCE_SCORE_AUTHORIZATION" if result["all_pass"] else "PRUNE_LABEL_FREE",
        "code_sha256": code_sha, "feature_provenance_sha256": feature_sha,
        "source_labels_opened": False, "source_labels_hashed": False,
        "source_scores_computed": False, "training_started": False,
        "remote_execution_started": True, "execution_environment": "CASIA_98",
        "execution_phase": "label_free_feasibility",
        "box_labels_read": [], "protected_splits_read": []}
    if not verify_provenance_binding(result, output_receipt):
        raise RuntimeError("output receipt provenance binding failed")
    _atomic_json(temporary / "receipt.json", output_receipt)
    _finish_atomic_directory(temporary, args.output)
    print(json.dumps({"output": str(args.output), "all_pass": result["all_pass"],
                      "cell_pass": cell_pass}))


if __name__ == "__main__":
    main()
