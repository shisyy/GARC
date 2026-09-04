#!/usr/bin/env python3
"""Launch the fixed Box baseline under an 8-GiB PyTorch allocator cap."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import Mapping, Sequence

from splart.endpoint_baselines import assert_model_process_boundary

CAP_BYTES = 8 * 1024**3
CAP_NUMERATOR_MIB = 8192
CAP_DENOMINATOR_MIB = 49140
ALLOC_CONF = "expandable_segments:True"
PHYSICAL_GPU = 2
RUN_ROOT = PurePosixPath("/home/yptang/arbor-runs/splart-endpoint-middle-baseline-box-scratch25000")
SOURCE_PARENT = PurePosixPath("/home/yptang/.arbor-worktrees")
SOURCE_PREFIX = "splart_endpoint_middle_baseline_"
SCENE = "100247-Box"
EXPERIMENT_NAME = "100247-Box/baseline"


def canonical_bytes(payload: Mapping[str, object]) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode()


def exclusive_write(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)


def single_flag_value(argv: Sequence[str], flag: str) -> str:
    assignment_prefix = flag + "="
    if any(token.startswith(assignment_prefix) for token in argv):
        raise RuntimeError(f"assignment form is forbidden for critical flag {flag}")
    positions = [index for index, token in enumerate(argv) if token == flag]
    if len(positions) != 1 or positions[0] + 1 >= len(argv):
        raise RuntimeError(f"critical flag must occur exactly once: {flag}")
    return argv[positions[0] + 1]


def validate_launch(argv: Sequence[str], env: Mapping[str, str]) -> dict[str, str]:
    if not argv or argv[0] != "splart":
        raise RuntimeError("baseline cap accepts only the SplArt training method")
    if env.get("CUDA_VISIBLE_DEVICES") != str(PHYSICAL_GPU):
        raise RuntimeError("physical GPU binding changed")
    if env.get("PYTORCH_CUDA_ALLOC_CONF") != ALLOC_CONF:
        raise RuntimeError("allocator configuration changed")
    run_root = PurePosixPath(env.get("SPLART_RUN_ROOT", ""))
    if run_root != RUN_ROOT:
        raise RuntimeError("fresh Box run root changed")
    source_dir = PurePosixPath(env.get("SPLART_SOURCE_DIR", ""))
    if source_dir.parent != SOURCE_PARENT or not source_dir.name.startswith(SOURCE_PREFIX):
        raise RuntimeError("source directory escaped the reviewed worktree family")
    public_scene = PurePosixPath(env.get("SPLART_PUBLIC_SCENE_DIR", ""))
    if public_scene.name != SCENE or not public_scene.is_absolute():
        raise RuntimeError("public Box scene binding changed")
    data_value = single_flag_value(argv, "--data")
    output_value = single_flag_value(argv, "--output-dir")
    experiment_value = single_flag_value(argv, "--experiment-name")
    if data_value != str(public_scene):
        raise RuntimeError("training argv must use only the reviewed public scene")
    if PurePosixPath(output_value) != RUN_ROOT / "model_ckpts":
        raise RuntimeError("training output escaped the fresh run root")
    if experiment_value != EXPERIMENT_NAME or ".." in PurePosixPath(experiment_value).parts:
        raise RuntimeError("experiment name changed or contains traversal")
    forbidden_flags = {
        "--load-dir",
        "--load-config",
        "--load-checkpoint",
        "--checkpoint",
        "--resume",
        "--seed",
        "--machine.seed",
        "--view",
    }
    if forbidden_flags.intersection(argv) or any(
        any(token.startswith(flag + "=") for flag in forbidden_flags) for token in argv
    ):
        raise RuntimeError("checkpoint/seed/view selection flag is forbidden")
    if any("endpoint-physics" in token.lower() or "contact-endpoint" in token.lower() for token in argv):
        raise RuntimeError("physics/endpoint module must be disabled in the baseline")
    assert_model_process_boundary(argv, env)

    source_commit = env.get("SPLART_SOURCE_COMMIT", "")
    if len(source_commit) != 40 or any(ch not in "0123456789abcdef" for ch in source_commit):
        raise RuntimeError("source commit is malformed")
    return {
        "run_root": str(run_root),
        "source_dir": str(source_dir),
        "source_commit": source_commit,
        "public_scene_dir": str(public_scene),
    }


def main() -> int:
    forwarded = sys.argv[1:]
    bindings = validate_launch(forwarded, os.environ)
    source_dir = Path(bindings["source_dir"])
    actual_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=source_dir, check=True, capture_output=True, text=True, timeout=30
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=source_dir, check=True, capture_output=True, text=True, timeout=30
    ).stdout.strip()
    if actual_commit != bindings["source_commit"] or dirty:
        raise RuntimeError("reviewed source commit/cleanliness check failed")

    import torch

    if torch.cuda.device_count() != 1:
        raise RuntimeError("worker must expose exactly one CUDA device")
    torch.cuda.set_device(0)
    total_bytes = int(torch.cuda.get_device_properties(0).total_memory)
    fraction = CAP_NUMERATOR_MIB / CAP_DENOMINATOR_MIB
    effective_ceiling = math.floor(total_bytes * fraction)
    if effective_ceiling > CAP_BYTES:
        raise RuntimeError("effective allocator ceiling exceeds 8 GiB")
    torch.cuda.set_per_process_memory_fraction(fraction, 0)

    receipt_path = Path(str(RUN_ROOT / "evidence" / "gpu-safety" / "cap-receipt.json"))
    receipt_path.parent.mkdir(parents=True, exist_ok=False)
    receipt = {
        "schema": "splart-endpoint-middle-cap/v1",
        "physical_gpu": PHYSICAL_GPU,
        "visible_gpu": 0,
        "requested_max_bytes": CAP_BYTES,
        "effective_ceiling_bytes": effective_ceiling,
        "fraction_numerator_mib": CAP_NUMERATOR_MIB,
        "fraction_denominator_mib": CAP_DENOMINATOR_MIB,
        "pytorch_cuda_alloc_conf": ALLOC_CONF,
        "source_commit": actual_commit,
        "public_scene_dir": bindings["public_scene_dir"],
        "argv_sha256": hashlib.sha256(canonical_bytes({"argv": forwarded})).hexdigest(),
        "pid": os.getpid(),
    }
    receipt["content_sha256"] = hashlib.sha256(canonical_bytes(receipt)).hexdigest()
    exclusive_write(receipt_path, canonical_bytes(receipt))

    sys.argv = [sys.argv[0], *forwarded]
    from nerfstudio.scripts.train import entrypoint

    entrypoint()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
