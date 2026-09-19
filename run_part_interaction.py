#!/usr/bin/env python3
"""Seal all four interaction modes; no endpoint/source-label arguments."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from splart.relative_search import CONFIG, METHODS, file_sha256, load_inputs, swap_input, write_json_new
from splart.part_interaction import MODES, interaction_predict, load_interactions, residual, swap_artifact


def run(inputs: Path, cache_index: Path, base_semantic_index: Path, output_dir: Path) -> dict:
    if output_dir.exists():
        raise ValueError("output directory must be new")
    torch.set_num_threads(1)
    index, values = load_inputs(inputs)
    counts = {s: sum(v["split"] == s for v in values) for s in ("source_train", "source_validation")}
    if counts != {"source_train": 13, "source_validation": 6}:
        raise ValueError("full fixed 13/6 whitelist roster required")
    cache = load_interactions(cache_index, base_semantic_index, values)
    donors = {}
    for split in counts:
        group = [v for v in values if v["split"] == split]
        for i, value in enumerate(group):
            donors[value["object_id"]] = group[(i + 1) % len(group)]
    implementation = {str(p.relative_to(ROOT)): file_sha256(p) for p in (
        ROOT / "src/splart/relative_search.py", ROOT / "src/splart/part_interaction.py",
        ROOT / "run_part_interaction.py", ROOT / "evaluate_relative_search.py")}
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    manifest = {"schema": "part-interaction-all-modes-seal/v1", "modes": list(MODES),
                "input_index_sha256": file_sha256(inputs), "interaction_index_sha256": file_sha256(cache_index),
                "base_semantic_index_sha256": file_sha256(base_semantic_index), "implementation_sha256": implementation,
                "source_index_sha256": index["source_index_sha256"], "predictions": {}}
    for mode in MODES:
        rows, diagnostics = [], []
        for value in values:
            original_donor = donors[value["object_id"]]
            for method in METHODS:
                owner = original_donor if method == "object_shuffle" else value
                if mode == "donor_interaction":
                    owner = donors[owner["object_id"]]
                artifact = cache[(owner["split"], owner["object_id"])]
                prediction = interaction_predict(value, method, mode, artifact, owner)
                swapped = interaction_predict(swap_input(value), method, mode, swap_artifact(artifact), swap_input(owner))
                if prediction["sides"] != list(reversed(swapped["sides"])):
                    raise AssertionError("exact side-swap evidence/output invariant failed")
                prediction["semantic_donor"] = original_donor["object_id"] if method == "object_shuffle" else None
                rows.append(prediction)
                if mode != "raw" and method not in ("coordinate_only", "geometry_only"):
                    r = residual(owner, artifact, mode)
                    diagnostics.append({"object_id": value["object_id"], "split": value["split"], "method": method,
                                        "effective_semantic_owner": owner["object_id"],
                                        "residual_l2_mean": float(r.norm(dim=-1).mean()),
                                        "residual_anchor_change_l2_mean": float((r[0, 0] - r[1, 0]).norm(dim=-1).mean())})
        report = {"schema": "relative-search-predictions/v1", "config": asdict(CONFIG), "methods": list(METHODS),
                  "interaction_mode": mode, "preregistered_modes": list(MODES), "source_index_sha256": index["source_index_sha256"],
                  "input_index_sha256": file_sha256(inputs), "interaction_index_sha256": file_sha256(cache_index),
                  "base_semantic_index_sha256": file_sha256(base_semantic_index), "diagnostic": index["diagnostic"],
                  "rows": rows, "interaction_diagnostics": diagnostics, "exact_swap_error": 0.0,
                  "code_commit": commit, "implementation_sha256": implementation, "endpoint_labels_used": False, "trained_parameters": 0}
        output = output_dir / mode / "predictions.json"
        write_json_new(output, report)
        digest = file_sha256(output)
        write_json_new(output.with_suffix(".seal.json"), {"schema": "relative-search-seal/v1", "prediction_sha256": digest,
                       "source_index_sha256": index["source_index_sha256"], "implementation_sha256": implementation})
        manifest["predictions"][mode] = {"path": output.relative_to(output_dir).as_posix(), "sha256": digest}
    write_json_new(output_dir / "all_modes.seal.json", manifest)
    verify_all_modes(output_dir)
    return {"output_dir": str(output_dir), "modes": list(MODES), "rows_per_mode": len(rows),
            "all_modes_seal_sha256": file_sha256(output_dir / "all_modes.seal.json")}


def verify_all_modes(output_dir: Path) -> dict:
    """Verify every mode seal before any independently launched label evaluator."""
    manifest = json.loads((output_dir / "all_modes.seal.json").read_text(encoding="utf-8"))
    if manifest["schema"] != "part-interaction-all-modes-seal/v1" or manifest["modes"] != list(MODES) or set(manifest["predictions"]) != set(MODES):
        raise ValueError("all four preregistered modes must be sealed")
    for mode in MODES:
        row = manifest["predictions"][mode]
        if row["path"] != f"{mode}/predictions.json":
            raise ValueError("unexpected sealed prediction path")
        output = output_dir / row["path"]
        seal = json.loads(output.with_suffix(".seal.json").read_text(encoding="utf-8"))
        payload = json.loads(output.read_text(encoding="utf-8"))
        digest = file_sha256(output)
        if row["sha256"] != digest or seal["prediction_sha256"] != digest:
            raise ValueError("all-modes prediction seal mismatch")
        if payload["interaction_mode"] != mode or payload["preregistered_modes"] != list(MODES):
            raise ValueError("mode identity mismatch")
        for key in ("source_index_sha256", "implementation_sha256"):
            if seal[key] != manifest[key] or payload[key] != manifest[key]:
                raise ValueError("shared code/source binding mismatch")
        for key in ("input_index_sha256", "interaction_index_sha256", "base_semantic_index_sha256"):
            if payload[key] != manifest[key]:
                raise ValueError("shared input binding mismatch")
    return {"verified_modes": list(MODES), "all_modes_seal_sha256": file_sha256(output_dir / "all_modes.seal.json")}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--interaction-cache", type=Path, required=True)
    p.add_argument("--base-semantic-index", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(run(a.inputs, a.interaction_cache, a.base_semantic_index, a.output_dir), sort_keys=True))
