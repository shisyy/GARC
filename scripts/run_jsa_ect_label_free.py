#!/usr/bin/env python3
"""Score-free JSA-ECT feasibility runner; source truth is intentionally absent."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from scripts.run_smarc_source_gate import (_atomic_json, _begin_atomic_directory,
                                            _finish_atomic_directory, indices_for,
                                            merge_features)
from splart.frozen_visual import sha256_file
from splart.jsa_ect import (canonical_object_scalar, feature_crossfit_jsa_ect,
                            global_row_identity, jsa_ect_ablation,
                            preserve_recipient_displacement, receipt_passes)
from splart.smarc_source import fit_feature_preprocessor, hash_ids, transform


FROZEN_CONFIG_SHA256 = "7d820961ae786a08991d03c69348cfe9cd5e50d4ad6d7737eb557411efe4ba43"
CONTEXTS = {"semantic": "semantic_jsa_ect/final-source-features", "mechanical": "mechanical_jsa_ect/final-source-features"}


def canonical_sha256(value) -> str:
    return hashlib.sha256((json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()).hexdigest()


def text_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def contract(config: dict) -> dict:
    return {"rank_relative_tolerance": config["numerics"]["rank_relative_tolerance"],
        "condition_max": config["numerics"]["condition_max"], "eigengap_multiplier": config["numerics"]["eigengap_multiplier"],
        "numeric_tolerance": config["numerics"]["numeric_tolerance"], "crossfit_folds": config["crossfit"]["folds"],
        "crossfit_absolute_spearman_at_most": config["crossfit"]["absolute_spearman_at_most"],
        "crossfit_distance_correlation_at_most": config["crossfit"]["distance_correlation_at_most"],
        **{k: v for k, v in config["gates"].items() if isinstance(v, (int, float)) and not isinstance(v, bool)}}


def _remap(rows, indices, values):
    return {global_row_identity(rows[i]): values[k] for k, i in enumerate(indices)}


def run_cell(rows, train, held, train_field, held_field, train_z, held_z, cfg, context, kind):
    first_cf = feature_crossfit_jsa_ect(rows, train, kind, cfg); second_cf = feature_crossfit_jsa_ect(rows, train, kind, cfg)
    first = jsa_ect_ablation(rows, train, held, train_field, held_field, train_z, held_z, cfg, context, first_cf)
    second = jsa_ect_ablation(rows, train, held, train_field, held_field, train_z, held_z, cfg, context, second_cf)
    if any(first[2]["domains"][d]["crossfit"] is second[2]["domains"][d]["crossfit"] for d in first[2]["domains"]): raise RuntimeError("receipt alias")
    reverse = jsa_ect_ablation(rows, list(reversed(train)), list(reversed(held)), train_field.flip(0), held_field.flip(0), train_z, held_z, cfg, context,
                               feature_crossfit_jsa_ect(rows, list(reversed(train)), kind, cfg))
    same = lambda a, b: set(a) == set(b) and all(torch.equal(a[k], b[k]) for k in a)
    audit = {"repeat_bit_identical": torch.equal(first[0], second[0]) and torch.equal(first[1], second[1]) and first[2] == second[2],
             "row_order_invariant": same(_remap(rows, train, first[0]), _remap(rows, list(reversed(train)), reverse[0])) and same(_remap(rows, held, first[1]), _remap(rows, list(reversed(held)), reverse[1])) and first[2] == reverse[2]}
    return first[0], first[1], first[2], second[2], audit


def verify_provenance_binding(result: dict, receipt: dict) -> bool:
    return (receipt.get("schema") == "splart-jsa-ect-feasibility-receipt/v1" and receipt.get("config_sha256") == FROZEN_CONFIG_SHA256
            and receipt.get("code_sha256") == result.get("code_sha256") == canonical_sha256(result.get("code_files_sha256"))
            and receipt.get("feature_provenance_sha256") == result.get("feature_provenance_sha256") == canonical_sha256(result.get("feature_provenance"))
            and all(result.get(k) is False and receipt.get(k) is False for k in ("source_labels_opened", "source_labels_hashed", "source_scores_computed", "training_started"))
            and result.get("remote_execution_started") is True and receipt.get("remote_execution_started") is True
            and result.get("execution_environment") == receipt.get("execution_environment") == "CASIA_98"
            and result.get("execution_phase") == receipt.get("execution_phase") == "label_free_feasibility"
            and result.get("box_labels_read") == receipt.get("box_labels_read") == [] and result.get("protected_splits_read") == receipt.get("protected_splits_read") == [])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--articraft-shards", type=Path, nargs=3, required=True); parser.add_argument("--njc-cache", type=Path, required=True)
    parser.add_argument("--articraft-pilc-data", type=Path, required=True); parser.add_argument("--njc-pilc-data", type=Path, required=True)
    parser.add_argument("--render-logs", type=Path, nargs=4, required=True); parser.add_argument("--dino-checkpoint", type=Path, required=True)
    parser.add_argument("--source-config", type=Path, required=True); parser.add_argument("--config", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); config = json.loads(args.config.read_text()); source_config = json.loads(args.source_config.read_text())
    if text_sha256(args.config) != FROZEN_CONFIG_SHA256: raise RuntimeError("frozen config hash mismatch")
    if config.get("remote_execution_started") is not False or config.get("source_labels_opened") or config.get("source_scores_computed") or config.get("training_started"): raise RuntimeError("score-free snapshot violated")
    rows, provenance = merge_features(args, source_config)
    if provenance.get("target_payloads_read") != [] or provenance.get("labels_opened") is not False: raise RuntimeError("target payload accessed")
    if [sha256_file(p) for p in args.render_logs] != config["source"]["render_log_sha256"] or sha256_file(args.dino_checkpoint) != config["source"]["dino_checkpoint_sha256"]: raise RuntimeError("frozen source hash mismatch")
    train = indices_for(rows, {"endpoint_pretrain", "njc_train"}); held = indices_for(rows, {"endpoint_validation", "njc_validation"})
    prep = fit_feature_preprocessor(rows, train, 16); tm, ts, _ = transform(prep, rows, train); hm, hs, _ = transform(prep, rows, held); cfg = contract(config)
    semantic = run_cell(rows, train, held, ts, hs, canonical_object_scalar(rows, train, "semantic"), canonical_object_scalar(rows, held, "semantic"), cfg, CONTEXTS["semantic"], "semantic")
    mechanical = run_cell(rows, train, held, tm[:, :-1], hm[:, :-1], canonical_object_scalar(rows, train, "mechanical", train), canonical_object_scalar(rows, held, "mechanical", train), cfg, CONTEXTS["mechanical"], "mechanical")
    d_exact = torch.equal(preserve_recipient_displacement(mechanical[0], tm)[:, -1], tm[:, -1]) and torch.equal(preserve_recipient_displacement(mechanical[1], hm)[:, -1], hm[:, -1])
    domains = {"articraft", "njc"}; cells = {}; full = {}
    for name, cell in (("semantic", semantic), ("mechanical", mechanical)):
        full[name] = receipt_passes(cell[2], cell[3], cfg, CONTEXTS[name], domains)
        cells[name] = {d: full[name] and cell[4]["repeat_bit_identical"] and cell[4]["row_order_invariant"] and (name != "mechanical" or d_exact) for d in domains}
    root = Path(__file__).resolve().parents[1]; code_files = [Path(__file__).resolve(), root/"src"/"splart"/"jsa_ect.py", root/"src"/"splart"/"rza_acwt.py", root/"src"/"splart"/"smarc_source.py", root/"scripts"/"run_smarc_source_gate.py"]
    code = {p.relative_to(root).as_posix(): sha256_file(p) for p in code_files}; feature_sha = canonical_sha256(provenance); code_sha = canonical_sha256(code)
    result = {"schema": "splart-jsa-ect-feasibility/v1", "config_sha256": FROZEN_CONFIG_SHA256, "feature_provenance": provenance, "feature_provenance_sha256": feature_sha,
        "jsa_ect": {"semantic": semantic[2], "mechanical": mechanical[2]}, "runtime_audits": {"semantic": semantic[4], "mechanical": {**mechanical[4], "recipient_displacement_bitwise_unchanged": d_exact}},
        "full_receipt_pass": full, "cell_pass": cells, "all_pass": all(full.values()) and all(v for part in cells.values() for v in part.values()), "code_files_sha256": code, "code_sha256": code_sha,
        "source_labels_opened": False, "source_labels_hashed": False, "source_scores_computed": False, "training_started": False, "remote_execution_started": True,
        "execution_environment": "CASIA_98", "execution_phase": "label_free_feasibility", "box_labels_read": [], "protected_splits_read": []}
    temporary = _begin_atomic_directory(args.output); _atomic_json(temporary/"feasibility.json", result)
    receipt = {"schema": "splart-jsa-ect-feasibility-receipt/v1", "config_sha256": FROZEN_CONFIG_SHA256, "feasibility_sha256": sha256_file(temporary/"feasibility.json"),
        "decision": "REQUEST_SOURCE_SCORE_AUTHORIZATION" if result["all_pass"] else "PRUNE_LABEL_FREE", "code_sha256": code_sha, "feature_provenance_sha256": feature_sha,
        "source_labels_opened": False, "source_labels_hashed": False, "source_scores_computed": False, "training_started": False, "remote_execution_started": True,
        "execution_environment": "CASIA_98", "execution_phase": "label_free_feasibility", "box_labels_read": [], "protected_splits_read": []}
    if not verify_provenance_binding(result, receipt): raise RuntimeError("provenance binding failed")
    _atomic_json(temporary/"receipt.json", receipt); _finish_atomic_directory(temporary, args.output)


if __name__ == "__main__": main()
