#!/usr/bin/env python3
"""Two-iteration GPU3 runtime smoke; never an endpoint baseline score."""

from pathlib import PurePosixPath

from control import endpoint_middle_cap as cap

cap.PHYSICAL_GPU = 3
cap.RUN_ROOT = PurePosixPath("/home/yptang/arbor-runs/splart-endpoint-middle-diagnostic-box-runtime-smoke")
cap.EXPERIMENT_NAME = "100247-Box/runtime-smoke"
cap.MAX_ITERATIONS = 2
cap.RECEIPT_SCHEMA = "splart-endpoint-middle-runtime-smoke-cap/v1"


if __name__ == "__main__":
    raise SystemExit(cap.main())
