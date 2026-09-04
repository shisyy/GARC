#!/usr/bin/env python3
"""Re-certify the frozen D2-CEA v2 endpoint scalars with PCFG evidence."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from posthoc_endpoint_adapter import (
    ADAPTER_COMMIT,
    BASE_COMMIT,
    FINAL_STEP,
    git_provenance,
    load_public_validator,
    model_state_sha256,
    sha256_file,
    write_exclusive,
)
from splart.pcfg_certificate import pcfg_certificate
from splart_renderer import SplartRenderer


PRIOR_PREDICTION_SHA256 = "700ce30b7244cafd25ee00b5d2337d4768893c84825670554428982112c2db1d"
FROZEN_LOWER = -0.8012999296188354
FROZEN_UPPER = 1.3892436027526855
QUERY_IDS = ("outside_state_0", "outside_state_1")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior-prediction", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--public-scene", type=Path, required=True)
    parser.add_argument("--base-source", type=Path, required=True)
    parser.add_argument("--certificate-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    launch_text = "\n".join(str(value) for value in vars(args).values()).lower()
    if any(marker in launch_text for marker in ("sealed", "evaluator_truth", "full22", "b_test")):
        raise ValueError("model-facing PCFG launch contains evaluator-only path marker")

    prior_path = args.prior_prediction.resolve()
    if sha256_file(prior_path) != PRIOR_PREDICTION_SHA256:
        raise ValueError("PCFG must consume the single frozen D2-CEA v2 prediction")
    prior = json.loads(prior_path.read_text())
    prior_rows = {row["query_id"]: row for row in prior["endpoint_predictions"]}
    if (
        float(prior_rows[QUERY_IDS[0]]["predicted_local_scalar"]) != FROZEN_LOWER
        or float(prior_rows[QUERY_IDS[1]]["predicted_local_scalar"]) != FROZEN_UPPER
    ):
        raise ValueError("frozen D2-CEA scalar binding mismatch")

    base_source = args.base_source.resolve()
    certificate_source = args.certificate_source.resolve()
    base_provenance = git_provenance(base_source, BASE_COMMIT)
    certificate_provenance = git_provenance(certificate_source, ADAPTER_COMMIT)
    public = load_public_validator(base_source).validate_public_scene(args.public_scene)
    checkpoint_dir = args.checkpoint_dir.resolve()
    checkpoint = checkpoint_dir / f"step-{FINAL_STEP:09d}.ckpt"
    if sha256_file(checkpoint) != prior["candidate"]["checkpoint"]["sha256"]:
        raise ValueError("base checkpoint changed since the frozen D2-CEA prediction")

    renderer = SplartRenderer(checkpoint_dir, load_step=FINAL_STEP, data_dir=None, device="cuda")
    for parameter in renderer.model.parameters():
        parameter.requires_grad_(False)
    before_hash = model_state_sha256(renderer.model)
    certificate = pcfg_certificate(renderer.model, FROZEN_LOWER, FROZEN_UPPER, renderer.ns_s_raw)
    after_hash = model_state_sha256(renderer.model)
    if before_hash != after_hash or any(parameter.grad is not None for parameter in renderer.model.parameters()):
        raise RuntimeError("PCFG changed or differentiated through the frozen SplArt model")

    closed_query = {
        "lower": QUERY_IDS[0],
        "upper": QUERY_IDS[1],
        "unknown": None,
    }[certificate.closed_end]
    prediction = dict(prior)
    prediction["candidate"] = dict(prior["candidate"])
    prediction["candidate"]["source"] = certificate_provenance
    prediction["closed_prediction"] = {
        "status": "predicted" if closed_query is not None else "unknown",
        "query_id": closed_query,
    }
    prediction["endpoint_predictions"] = [
        {
            "query_id": QUERY_IDS[0],
            "local_direction": -1,
            "predicted_local_scalar": FROZEN_LOWER,
            "predicted_closed": closed_query == QUERY_IDS[0],
        },
        {
            "query_id": QUERY_IDS[1],
            "local_direction": 1,
            "predicted_local_scalar": FROZEN_UPPER,
            "predicted_closed": closed_query == QUERY_IDS[1],
        },
    ]
    prediction["renderer"] = "original-splart-plus-frozen-d2-cea-v2-scalars-plus-pcfg"
    prediction["training"] = dict(prior["training"])
    prediction["training"]["pcfg_posthoc_only"] = True
    prediction["pcfg"] = {
        "schema": "splart-pcfg-certificate/v1",
        "prior_prediction": {"path": str(prior_path), "sha256": PRIOR_PREDICTION_SHA256},
        "base_checkpoint_sha256": sha256_file(checkpoint),
        "base_source": base_provenance,
        "certificate_source": certificate_provenance,
        "module_sha256": sha256_file(certificate_source / "src" / "splart" / "pcfg_certificate.py"),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "model_state_sha256_before": before_hash,
        "model_state_sha256_after": after_hash,
        "evidence": asdict(certificate),
    }
    # Reassert that the only scene input remains the audited public tree.
    prediction["public_input"] = {
        "root": public["scene_dir"],
        "tree_sha256": public["public_tree_sha256"],
        "file_count": len(public["public_tree"]),
    }
    write_exclusive(args.output.resolve(), prediction)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "lower": FROZEN_LOWER,
                "upper": FROZEN_UPPER,
                "closed": closed_query,
                "lower_terminal": certificate.lower.terminal_valid,
                "upper_terminal": certificate.upper.terminal_valid,
            }
        )
    )


if __name__ == "__main__":
    main()
