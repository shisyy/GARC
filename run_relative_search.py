#!/usr/bin/env python3
"""Predict from target-free exports; this executable cannot load source labels."""
from dataclasses import asdict
from pathlib import Path
import argparse
import json
import subprocess
import sys

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from splart.relative_search import CONFIG, METHODS, file_sha256, load_inputs, predict, swap_input, write_json_new


def run(inputs: Path, output: Path) -> dict:
    torch.set_num_threads(1)
    index, values = load_inputs(inputs)
    rows, swap_error = [], 0.0
    for value in values:
        peers = [x for x in values if x["split"] == value["split"]]
        if len(peers) < 2:
            raise ValueError("all controls require at least two objects in each evaluated split")
        position = next(i for i, x in enumerate(peers) if x["object_id"] == value["object_id"])
        donor = peers[(position + 1) % len(peers)]
        for method in METHODS:
            prediction = predict(value, method, donor)
            swapped = predict(swap_input(value), method, swap_input(donor))
            error = max(abs(a - b) for a, b in zip(prediction["distance"], reversed(swapped["distance"])))
            swap_error = max(swap_error, error)
            if prediction["sides"] != list(reversed(swapped["sides"])):
                raise AssertionError("exact side-swap evidence/output invariant failed")
            prediction["semantic_donor"] = donor["object_id"] if method == "object_shuffle" else None
            rows.append(prediction)
    report = {"schema": "relative-search-predictions/v1", "config": asdict(CONFIG), "methods": list(METHODS),
              "source_index_sha256": index["source_index_sha256"], "input_index_sha256": file_sha256(inputs),
              "diagnostic": index["diagnostic"], "rows": rows, "exact_swap_error": swap_error,
              "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "implementation_sha256": {p.name: file_sha256(p) for p in [ROOT / "src/splart/relative_search.py", ROOT / "run_relative_search.py"]},
              "endpoint_labels_used": False, "trained_parameters": 0}
    write_json_new(output, report)
    seal = {"schema": "relative-search-seal/v1", "prediction_sha256": file_sha256(output),
            "source_index_sha256": index["source_index_sha256"], "implementation_sha256": report["implementation_sha256"]}
    write_json_new(output.with_suffix(".seal.json"), seal)
    return {"predictions": str(output), "sha256": seal["prediction_sha256"], "rows": len(rows), "exact_swap_error": swap_error}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(run(a.inputs, a.output), sort_keys=True))
