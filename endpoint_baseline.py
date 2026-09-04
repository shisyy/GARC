"""CLI for leakage-safe endpoint baseline predictions and launch manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from splart.endpoint_baselines import (
    BASELINES,
    SOURCE_COMMIT,
    build_splart_training_argv,
    make_prediction,
    validate_public_scene,
    write_json_atomic,
)


def _validate(args: argparse.Namespace) -> None:
    print(json.dumps(validate_public_scene(args.scene_dir), indent=2, sort_keys=True))


def _predict(args: argparse.Namespace) -> None:
    prediction = make_prediction(args.scene_dir, args.baseline, args.checkpoint_dir, args.source_dir)
    write_json_atomic(args.output, prediction)
    print(args.output)


def _training_request(args: argparse.Namespace) -> None:
    public = validate_public_scene(args.scene_dir)
    argv = build_splart_training_argv(args.scene_dir, args.output_dir, args.experiment_name)
    request = {
        "schema_version": "splart-middle-training-request/v1",
        "source_commit": SOURCE_COMMIT,
        "scene_id": public["scene_id"],
        "public_scene_dir": public["scene_dir"],
        "public_tree_sha256": public["public_tree_sha256"],
        "public_file_count": len(public["public_tree"]),
        "argv": argv,
        "invariants": {
            "fresh_scratch": True,
            "author_checkpoint": False,
            "endpoint_physics_enabled": False,
            "model_reads_evaluator_manifest": False,
        },
    }
    write_json_atomic(args.output, request)
    print(args.output)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-public")
    validate.add_argument("--scene-dir", type=Path, required=True)
    validate.set_defaults(func=_validate)

    predict = subparsers.add_parser("predict")
    predict.add_argument("--scene-dir", type=Path, required=True)
    predict.add_argument("--baseline", choices=BASELINES, required=True)
    predict.add_argument("--checkpoint-dir", type=Path)
    predict.add_argument("--source-dir", type=Path)
    predict.add_argument("--output", type=Path, required=True)
    predict.set_defaults(func=_predict)

    request = subparsers.add_parser("training-request")
    request.add_argument("--scene-dir", type=Path, required=True)
    request.add_argument("--output-dir", type=Path, required=True)
    request.add_argument("--experiment-name", required=True)
    request.add_argument("--output", type=Path, required=True)
    request.set_defaults(func=_training_request)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
