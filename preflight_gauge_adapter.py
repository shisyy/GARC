#!/usr/bin/env python3
"""Hash and validate the public node7.1 payload before any GPU export."""

import argparse
import json
from pathlib import Path

from export_gauge_profiles import preflight_public_materialization, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public-manifest", type=Path, required=True)
    parser.add_argument("--materialization-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = preflight_public_materialization(args.public_manifest, args.materialization_receipt)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "sha256": sha256_file(args.output), "objects": len(result["objects"])}))


if __name__ == "__main__":
    main()
