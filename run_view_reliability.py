#!/usr/bin/env python3
"""Seal all four weighting modes before any independent endpoint evaluation."""
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
from splart.view_reliability import MODES, view_weights, weighted_predict


def run(inputs: Path, output_dir: Path) -> dict:
    if output_dir.exists():
        raise ValueError("output directory must be new")
    torch.set_num_threads(1)
    index, values = load_inputs(inputs)
    counts = {s: sum(v["split"] == s for v in values) for s in ("source_train", "source_validation")}
    if counts != {"source_train": 13, "source_validation": 6}:
        raise ValueError("node 9.2 requires immutable full 13/6 input roster")
    peers = {s: [v for v in values if v["split"] == s] for s in counts}
    donors = {}
    for group in peers.values():
        for i, v in enumerate(group):
            donors[v["object_id"]] = group[(i + 1) % len(group)]
    implementation = {str(p.relative_to(ROOT)): file_sha256(p) for p in (
        ROOT / "src/splart/relative_search.py", ROOT / "src/splart/view_reliability.py",
        ROOT / "run_view_reliability.py", ROOT / "evaluate_relative_search.py")}
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    manifest = {"schema": "view-reliability-all-modes-seal/v1", "modes": list(MODES),
                "input_index_sha256": file_sha256(inputs), "implementation_sha256": implementation,
                "source_index_sha256": index["source_index_sha256"], "predictions": {}}
    for mode in MODES:
        rows, diagnostics = [], []
        for value in values:
            semantic_donor = donors[value["object_id"]]
            weight_owner = semantic_donor if mode == "shuffled_weights" else value
            _, diag = view_weights(weight_owner, mode)
            diagnostics.append({"object_id": value["object_id"], "split": value["split"],
                                "weight_object_id": weight_owner["object_id"], **diag})
            for method in METHODS:
                semantic_owner = semantic_donor if method == "object_shuffle" else value
                weight_donor = donors[semantic_owner["object_id"]]
                prediction = weighted_predict(value, method, mode, semantic_donor, weight_donor)
                swapped = weighted_predict(swap_input(value), method, mode,
                                           swap_input(semantic_donor), swap_input(weight_donor))
                if prediction["sides"] != list(reversed(swapped["sides"])):
                    raise AssertionError("exact swap evidence/output invariant failed")
                prediction["semantic_donor"] = semantic_donor["object_id"] if method == "object_shuffle" else None
                rows.append(prediction)
        report = {"schema": "relative-search-predictions/v1", "config": asdict(CONFIG),
                  "methods": list(METHODS), "weighting_mode": mode, "preregistered_modes": list(MODES),
                  "source_index_sha256": index["source_index_sha256"], "input_index_sha256": file_sha256(inputs),
                  "diagnostic": index["diagnostic"], "rows": rows, "weight_diagnostics": diagnostics,
                  "exact_swap_error": 0.0, "code_commit": commit, "implementation_sha256": implementation,
                  "endpoint_labels_used": False, "trained_parameters": 0}
        output = output_dir / mode / "predictions.json"
        write_json_new(output, report)
        digest = file_sha256(output)
        seal = {"schema": "relative-search-seal/v1", "prediction_sha256": digest,
                "source_index_sha256": index["source_index_sha256"], "implementation_sha256": implementation}
        write_json_new(output.with_suffix(".seal.json"), seal)
        manifest["predictions"][mode] = {"path": str(output.relative_to(output_dir)), "sha256": digest}
    # This exists only after all four complete prediction files are sealed.
    write_json_new(output_dir / "all_modes.seal.json", manifest)
    return {"output_dir": str(output_dir), "modes": list(MODES), "rows_per_mode": len(rows),
            "all_modes_seal_sha256": file_sha256(output_dir / "all_modes.seal.json")}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(run(a.inputs, a.output_dir), sort_keys=True))
