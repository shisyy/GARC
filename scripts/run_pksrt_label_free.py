#!/usr/bin/env python3
"""Score-free PKSRT feasibility runner; source truth is intentionally absent."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from scripts.run_smarc_source_gate import (FROZEN_CONFIG_SHA256 as SOURCE_CONFIG_SHA256,
                                            _atomic_json, _begin_atomic_directory,
                                            _finish_atomic_directory, indices_for,
                                            merge_features)
from splart.frozen_visual import sha256_file
from splart.pksrt import (canonical_object_scalar, domain_receipt_passes,
                          feature_crossfit_pksrt, global_receipt_passes,
                          global_row_identity, pksrt_ablation,
                          preserve_recipient_displacement)
from splart.smarc_source import fit_feature_preprocessor, hash_ids, transform


FROZEN_CONFIG_SHA256 = "62c69b18a3c2171bf57fa71acdcb4fa43b186377212a719ff118b09f913f8570"
PREDECESSOR_CONFIG_SHA256 = "a89543b1cf954cfc2de191fb50dda265ca8182dcac9e1a7d14a8faed90f853b2"
CONTEXTS = {"semantic": "semantic_pksrt/final-source-features", "mechanical": "mechanical_pksrt/final-source-features"}


def canonical_sha256(value) -> str:
    return hashlib.sha256((json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()).hexdigest()


def text_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def contract(config: dict) -> dict:
    return {"rank_relative_tolerance": config["numerics"]["rank_relative_tolerance"],
        "condition_max": config["numerics"]["condition_max"], "numeric_tolerance": config["numerics"]["numeric_tolerance"],
        "minimum_effective_sample_size": config["numerics"]["minimum_effective_sample_size"],
        "crossfit_folds": config["crossfit"]["folds"],
        "crossfit_absolute_spearman_at_most": config["crossfit"]["absolute_spearman_at_most"],
        "crossfit_distance_correlation_at_most": config["crossfit"]["distance_correlation_at_most"],
        **{k: v for k, v in config["gates"].items() if isinstance(v, (int, float)) and not isinstance(v, bool)}}


def verify_frozen_source_payloads(args, config: dict, predecessor: dict) -> None:
    source = config["source"]
    keys = ("dino_checkpoint_sha256", "render_log_sha256", "object_list_hashes", "articraft_inputs_sha256",
            "articraft_manifest_sha256", "njc_inputs_sha256", "njc_manifest_sha256", "render_feature_payload_sha256")
    for key in keys:
        if source[key] != predecessor["source"][key]: raise RuntimeError(f"source contract changed: {key}")
    caches = list(args.articraft_shards) + [args.njc_cache]
    for path, expected in zip(caches, source["render_feature_payload_sha256"]):
        actual = {"inputs": sha256_file(path / "inputs.pt"), "manifest": sha256_file(path / "manifest.json"),
                  "render_bank": sha256_file(path / "neutral_render_bank.pt")}
        if actual != expected: raise RuntimeError("render feature payload hash mismatch")
    for path, prefix in ((args.articraft_pilc_data, "articraft"), (args.njc_pilc_data, "njc")):
        if sha256_file(path / "inputs.pt") != source[f"{prefix}_inputs_sha256"] or sha256_file(path / "manifest.json") != source[f"{prefix}_manifest_sha256"]:
            raise RuntimeError("PILC feature payload hash mismatch")


def _remap(rows, indices, values):
    order = sorted(indices, key=lambda i: global_row_identity(rows[i]))
    return {global_row_identity(rows[i]): values[k] for k, i in enumerate(order)}


def run_cell(rows, train, held, train_field, held_field, train_z, held_z, cfg, context, kind):
    first_cf = feature_crossfit_pksrt(rows, train, kind, cfg)
    second_cf = feature_crossfit_pksrt(rows, train, kind, cfg)
    first = pksrt_ablation(rows, train, held, train_field, held_field, train_z, held_z, cfg, context, first_cf)
    second = pksrt_ablation(rows, train, held, train_field, held_field, train_z, held_z, cfg, context, second_cf)
    expected_sha = canonical_sha256(second[2])
    if first[2] is second[2] or first_cf is second_cf: raise RuntimeError("receipt alias")
    from splart.pksrt import _shares_container_identity
    if _shares_container_identity(first[2], second[2]): raise RuntimeError("nested receipt alias")
    reverse_train, reverse_held = list(reversed(train)), list(reversed(held))
    reverse = pksrt_ablation(rows, reverse_train, reverse_held, train_field.flip(0), held_field.flip(0), train_z, held_z,
                             cfg, context, feature_crossfit_pksrt(rows, reverse_train, kind, cfg))
    canonical_train = sorted(train, key=lambda i: global_row_identity(rows[i])); canonical_held = sorted(held, key=lambda i: global_row_identity(rows[i]))
    same = lambda a, b: set(a) == set(b) and all(torch.equal(a[k], b[k]) for k in a)
    audit = {"repeat_bit_identical": torch.equal(first[0], second[0]) and torch.equal(first[1], second[1]) and first[2] == second[2],
             "row_order_invariant": same(_remap(rows, canonical_train, first[0]), _remap(rows, canonical_train, reverse[0])) and
                                    same(_remap(rows, canonical_held, first[1]), _remap(rows, canonical_held, reverse[1])) and first[2] == reverse[2]}
    audit["candidate_prediction_sha256"] = canonical_sha256({"train": first[0].tolist(), "held": first[1].tolist()})
    audit["gate_recompute_prediction_sha256"] = canonical_sha256({"train": second[0].tolist(), "held": second[1].tolist()})
    return first[0], first[1], first[2], second[2], expected_sha, audit


def verify_provenance_binding(result: dict, receipt: dict) -> bool:
    receipt_keys = {"schema", "config_sha256", "source_config_sha256", "predecessor_config_sha256", "feasibility_sha256", "decision",
                    "code_sha256", "feature_provenance_sha256", "source_labels_opened", "source_labels_hashed", "source_scores_computed",
                    "training_started", "remote_execution_started", "execution_environment", "execution_phase", "box_labels_read", "protected_splits_read"}
    result_keys = {"schema", "config_sha256", "source_config_sha256", "predecessor_config_sha256", "feature_provenance",
                   "feature_provenance_sha256", "pksrt", "independent_recomputation_sha256", "runtime_audits", "global_receipt_pass",
                   "domain_receipt_pass", "cell_pass", "all_pass", "object_list_hashes", "global_row_identity_sha256",
                   "code_files_sha256", "code_sha256", "source_labels_opened", "source_labels_hashed", "source_scores_computed",
                   "training_started", "remote_execution_started", "execution_environment", "execution_phase", "box_labels_read", "protected_splits_read"}
    try:
        if set(receipt) != receipt_keys or set(result) != result_keys: return False
        if receipt["schema"] != "splart-pksrt-feasibility-receipt/v1" or result["schema"] != "splart-pksrt-feasibility/v1": return False
        if receipt["config_sha256"] != result["config_sha256"] or result["config_sha256"] != FROZEN_CONFIG_SHA256: return False
        if receipt["source_config_sha256"] != result["source_config_sha256"] or result["source_config_sha256"] != SOURCE_CONFIG_SHA256: return False
        if receipt["predecessor_config_sha256"] != result["predecessor_config_sha256"] or result["predecessor_config_sha256"] != PREDECESSOR_CONFIG_SHA256: return False
        if receipt["decision"] != ("REQUEST_SOURCE_SCORE_AUTHORIZATION" if result["all_pass"] else "PRUNE_LABEL_FREE"): return False
        if receipt["feasibility_sha256"] != canonical_sha256(result): return False
        if receipt["code_sha256"] != result["code_sha256"] or result["code_sha256"] != canonical_sha256(result["code_files_sha256"]): return False
        if receipt["feature_provenance_sha256"] != result["feature_provenance_sha256"] or result["feature_provenance_sha256"] != canonical_sha256(result["feature_provenance"]): return False
        return (all(result[k] is False and receipt[k] is False for k in ("source_labels_opened", "source_labels_hashed", "source_scores_computed", "training_started"))
                and result["remote_execution_started"] is receipt["remote_execution_started"] is True
                and result["execution_environment"] == receipt["execution_environment"] == "CASIA_98"
                and result["execution_phase"] == receipt["execution_phase"] == "label_free_feasibility"
                and result["box_labels_read"] == receipt["box_labels_read"] == []
                and result["protected_splits_read"] == receipt["protected_splits_read"] == [])
    except (KeyError, TypeError): return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--articraft-shards", type=Path, nargs=3, required=True); parser.add_argument("--njc-cache", type=Path, required=True)
    parser.add_argument("--articraft-pilc-data", type=Path, required=True); parser.add_argument("--njc-pilc-data", type=Path, required=True)
    parser.add_argument("--render-logs", type=Path, nargs=4, required=True); parser.add_argument("--dino-checkpoint", type=Path, required=True)
    parser.add_argument("--source-config", type=Path, required=True); parser.add_argument("--predecessor-config", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); config = json.loads(args.config.read_text()); source_config = json.loads(args.source_config.read_text()); predecessor = json.loads(args.predecessor_config.read_text())
    if text_sha256(args.config) != FROZEN_CONFIG_SHA256 or text_sha256(args.source_config) != SOURCE_CONFIG_SHA256 or text_sha256(args.predecessor_config) != PREDECESSOR_CONFIG_SHA256: raise RuntimeError("frozen config hash mismatch")
    if config["source_config_sha256"] != SOURCE_CONFIG_SHA256 or config["predecessor_jsa_config_sha256"] != PREDECESSOR_CONFIG_SHA256 or predecessor.get("schema") != "splart-jsa-ect-label-free-preregister/v1": raise RuntimeError("predecessor binding mismatch")
    if (config.get("remote_execution_started") is not False or config.get("source_labels_opened") or config.get("source_labels_hashed")
            or config.get("source_scores_computed") or config.get("training_started") or config.get("box_labels_read") != []
            or config.get("protected_splits_read") != []): raise RuntimeError("score-free snapshot violated")
    verify_frozen_source_payloads(args, config, predecessor); rows, provenance = merge_features(args, source_config)
    if provenance.get("target_payloads_read") != [] or provenance.get("labels_opened") is not False: raise RuntimeError("target payload accessed")
    if [sha256_file(p) for p in args.render_logs] != config["source"]["render_log_sha256"] or sha256_file(args.dino_checkpoint) != config["source"]["dino_checkpoint_sha256"]: raise RuntimeError("frozen source hash mismatch")
    if any("overflow" in p.read_text(errors="replace").lower() for p in args.render_logs): raise RuntimeError("renderer overflow marker")
    train = indices_for(rows, {"endpoint_pretrain", "njc_train"}); held = indices_for(rows, {"endpoint_validation", "njc_validation"})
    observed = {"articraft_train": hash_ids(sorted({rows[i]["object_group_id"] for i in train if rows[i]["domain"] == "articraft"})),
        "articraft_held": hash_ids(sorted({rows[i]["object_group_id"] for i in held if rows[i]["domain"] == "articraft"})),
        "njc_train": hash_ids(sorted({rows[i]["object_group_id"] for i in train if rows[i]["domain"] == "njc"})),
        "njc_held": hash_ids(sorted({rows[i]["object_group_id"] for i in held if rows[i]["domain"] == "njc"}))}
    if observed != config["source"]["object_list_hashes"]: raise RuntimeError("object-list hash mismatch")
    prep = fit_feature_preprocessor(rows, train, 16); tm, ts, _ = transform(prep, rows, train); hm, hs, _ = transform(prep, rows, held); cfg = contract(config)
    kept = torch.nonzero(prep.mechanical_keep).flatten().tolist()
    if not bool(prep.mechanical_keep[-1]) or kept[-1] != prep.mechanical_keep.numel() - 1: raise RuntimeError("mechanical displacement contract failed")
    semantic = run_cell(rows, train, held, ts, hs, canonical_object_scalar(rows, train, "semantic"), canonical_object_scalar(rows, held, "semantic"), cfg, CONTEXTS["semantic"], "semantic")
    mechanical = run_cell(rows, train, held, tm[:, :-1], hm[:, :-1], canonical_object_scalar(rows, train, "mechanical", train), canonical_object_scalar(rows, held, "mechanical", train), cfg, CONTEXTS["mechanical"], "mechanical")
    d_exact = torch.equal(preserve_recipient_displacement(mechanical[0], tm)[:, -1], tm[:, -1]) and torch.equal(preserve_recipient_displacement(mechanical[1], hm)[:, -1], hm[:, -1])
    domains = {"articraft", "njc"}; cells = {}; full = {}
    for name, cell in (("semantic", semantic), ("mechanical", mechanical)):
        full[name] = global_receipt_passes(cell[2], cell[3], cell[4], cfg, CONTEXTS[name], domains)
        cells[name] = {d: full[name] and domain_receipt_passes(cell[2], cell[3], cell[4], cfg, CONTEXTS[name], domains, d)
                       and cell[5]["repeat_bit_identical"] and cell[5]["row_order_invariant"]
                       and cell[5]["candidate_prediction_sha256"] == cell[5]["gate_recompute_prediction_sha256"]
                       and (name != "mechanical" or d_exact) for d in domains}
    root = Path(__file__).resolve().parents[1]
    code_files = [Path(__file__).resolve(), root/"src"/"splart"/"pksrt.py", root/"scripts"/"run_jsa_ect_label_free.py", root/"src"/"splart"/"jsa_ect.py",
                  root/"scripts"/"run_rza_acwt_label_free.py", root/"src"/"splart"/"rza_acwt.py", root/"scripts"/"run_rqlsot_feasibility.py",
                  root/"src"/"splart"/"rqlsot.py", root/"src"/"splart"/"conditional_residual.py", root/"src"/"splart"/"smarc_source.py",
                  root/"src"/"splart"/"smarc.py", root/"src"/"splart"/"frozen_visual.py", root/"scripts"/"run_smarc_source_gate.py",
                  root/"scripts"/"build_smarc_articraft_shard.py", root/"scripts"/"build_smarc_njc_cache.py", root/"scripts"/"build_smarc_smoke.py"]
    code = {p.relative_to(root).as_posix(): sha256_file(p) for p in code_files}; feature_sha = canonical_sha256(provenance); code_sha = canonical_sha256(code)
    result = {"schema": "splart-pksrt-feasibility/v1", "config_sha256": FROZEN_CONFIG_SHA256, "source_config_sha256": SOURCE_CONFIG_SHA256,
        "predecessor_config_sha256": PREDECESSOR_CONFIG_SHA256, "feature_provenance": provenance, "feature_provenance_sha256": feature_sha,
        "pksrt": {"semantic": semantic[2], "mechanical": mechanical[2]}, "runtime_audits": {"semantic": semantic[5], "mechanical": {**mechanical[5], "recipient_displacement_bitwise_unchanged": d_exact}},
        "independent_recomputation_sha256": {"semantic": semantic[4], "mechanical": mechanical[4]}, "global_receipt_pass": full,
        "domain_receipt_pass": {name: {d: domain_receipt_passes(cell[2], cell[3], cell[4], cfg, CONTEXTS[name], domains, d) for d in domains} for name, cell in (("semantic", semantic), ("mechanical", mechanical))},
        "cell_pass": cells, "all_pass": all(full.values()) and all(v for part in cells.values() for v in part.values()), "object_list_hashes": observed,
        "global_row_identity_sha256": canonical_sha256(sorted(global_row_identity(r) for r in rows)), "code_files_sha256": code, "code_sha256": code_sha,
        "source_labels_opened": False, "source_labels_hashed": False, "source_scores_computed": False, "training_started": False,
        "remote_execution_started": True, "execution_environment": "CASIA_98", "execution_phase": "label_free_feasibility", "box_labels_read": [], "protected_splits_read": []}
    temporary = _begin_atomic_directory(args.output); _atomic_json(temporary / "feasibility.json", result)
    receipt = {"schema": "splart-pksrt-feasibility-receipt/v1", "config_sha256": FROZEN_CONFIG_SHA256, "source_config_sha256": SOURCE_CONFIG_SHA256,
        "predecessor_config_sha256": PREDECESSOR_CONFIG_SHA256, "feasibility_sha256": canonical_sha256(result),
        "decision": "REQUEST_SOURCE_SCORE_AUTHORIZATION" if result["all_pass"] else "PRUNE_LABEL_FREE", "code_sha256": code_sha,
        "feature_provenance_sha256": feature_sha, "source_labels_opened": False, "source_labels_hashed": False, "source_scores_computed": False,
        "training_started": False, "remote_execution_started": True, "execution_environment": "CASIA_98", "execution_phase": "label_free_feasibility",
        "box_labels_read": [], "protected_splits_read": []}
    if not verify_provenance_binding(result, receipt): raise RuntimeError("provenance binding failed")
    _atomic_json(temporary / "receipt.json", receipt); _finish_atomic_directory(temporary, args.output)


if __name__ == "__main__": main()
