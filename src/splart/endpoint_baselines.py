"""Leakage-safe baselines for the intermediate-state endpoint task.

This module deliberately has no torch/nerfstudio dependency.  The predictor is
part of the model-facing process and therefore only consumes the public scene
directory.  Ground-truth endpoint scalars are consumed separately by
``endpoint_eval.py``.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

SCHEMA_VERSION = "splart-endpoint-prediction/v1"
QUERY_IDS = ("outside_state_0", "outside_state_1")
BASELINES = ("observed-span", "symmetric-linear", "splart-middle")
SOURCE_COMMIT = "2e5e286b3ffc027d37c9c9da1a5dc87d18301849"
FINAL_TRAIN_STEP = 24_999

# These fields are evaluator-only.  Match normalized spellings so that minor
# punctuation changes cannot accidentally punch a hole through the boundary.
_FORBIDDEN_PUBLIC_KEYS = {
    "physicalfraction",
    "physicalfractions",
    "physicalendpoint",
    "physicalendpoints",
    "fullrange",
    "fulljointrange",
    "truejointlimit",
    "truejointlimits",
    "urdfjointlimit",
    "urdfjointlimits",
    "gtscalar",
    "gtscalars",
    "localgtscalar",
    "localgtscalars",
    "closedquery",
    "closedside",
    "closedendpoint",
    "endpointgroundtruth",
    "endpointassets",
    "heldoutendpoint",
    "heldoutendpoints",
    "testfilenames",
    "axis",
    "pivot",
    "angle",
    "dist",
}

_PUBLIC_ROOT_FILES = {"transforms.json", "endpoint_queries.json", "PROXY_RECEIPT.json", "COMPLETE.json"}
_PUBLIC_MODALITIES = {"color", "depth", "part-seg"}
_PUBLIC_SPLITS = {"train", "val"}
_PUBLIC_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def _normalise_key(key: object) -> str:
    return "".join(ch for ch in str(key).lower() if ch.isalnum())


def _walk_keys(value: Any, prefix: str = "$") -> Iterable[tuple[str, str]]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}"
            yield child_prefix, _normalise_key(key)
            yield from _walk_keys(child, child_prefix)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            yield from _walk_keys(child, f"{prefix}[{index}]")


def assert_no_evaluator_fields(value: Any) -> None:
    """Reject evaluator-only keys from a model-facing JSON object."""

    violations = [path for path, key in _walk_keys(value) if key in _FORBIDDEN_PUBLIC_KEYS]
    if violations:
        raise ValueError("evaluator-only field(s) in public input: " + ", ".join(violations))


def assert_model_process_boundary(
    argv: Sequence[str], env: Mapping[str, str] | None = None, sealed_paths: Iterable[Path] = ()
) -> None:
    """Fail closed if evaluator-only material reaches a model process launch."""

    env = {} if env is None else env
    assert_no_evaluator_fields({"argv": list(argv), "env": dict(env)})
    launch_text = "\n".join([*(str(item) for item in argv), *(f"{key}={value}" for key, value in env.items())]).lower()
    denied_markers = (
        "sealed-evaluator",
        "sealed_evaluator",
        "/arbor-reviews/",
        "\\arbor-reviews\\",
        "b_test",
        "full22",
    )
    present = [marker for marker in denied_markers if marker in launch_text]
    resolved_sealed = [str(Path(path).resolve()).lower() for path in sealed_paths]
    present.extend(path for path in resolved_sealed if path and path in launch_text)
    if present:
        raise ValueError("evaluator-only path/marker in model process launch: " + ", ".join(present))


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def content_sha256(payload: Any) -> str:
    # This is byte-for-byte the benchmark builder's canonical_sha256.  In
    # particular, canonical JSON is *not* terminated by a newline.
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode()).hexdigest()


def audit_public_tree(scene_dir: Path) -> dict[str, str]:
    """Hash a strict allowlist; an unexpected sidecar is a launch blocker."""

    scene_dir = Path(scene_dir).resolve()
    files: dict[str, str] = {}
    for path in sorted(scene_dir.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"public scene contains symlink: {path}")
        relative = path.relative_to(scene_dir)
        parts = relative.parts
        if path.is_dir():
            if len(parts) == 1 and parts[0] in _PUBLIC_MODALITIES:
                continue
            if len(parts) == 2 and parts[0] in _PUBLIC_MODALITIES and parts[1] in _PUBLIC_SPLITS:
                continue
            raise ValueError(f"unexpected directory in public scene: {relative.as_posix()}")
        if len(parts) == 1 and parts[0] in _PUBLIC_ROOT_FILES:
            assert_no_evaluator_fields(_load_json(path))
        elif (
            len(parts) == 3
            and parts[0] in _PUBLIC_MODALITIES
            and parts[1] in _PUBLIC_SPLITS
            and path.suffix.lower() in _PUBLIC_IMAGE_SUFFIXES
        ):
            pass
        else:
            raise ValueError(f"unexpected file in public scene: {relative.as_posix()}")
        files[relative.as_posix()] = sha256_file(path)
    missing = _PUBLIC_ROOT_FILES.difference(files)
    if missing:
        raise ValueError(f"public scene is missing required files: {sorted(missing)}")
    image_files = [name for name in files if Path(name).suffix.lower() in _PUBLIC_IMAGE_SUFFIXES]
    if len(image_files) != 660:
        raise ValueError(f"public scene must contain exactly 660 image payloads; found {len(image_files)}")
    for modality in _PUBLIC_MODALITIES:
        count = sum(name.startswith(modality + "/") for name in image_files)
        if count != 220:
            raise ValueError(f"public {modality} payload count must be 220; found {count}")
    return files


def _is_binary_state(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and float(value) in (0.0, 1.0)


def validate_public_scene(scene_dir: Path) -> dict[str, Any]:
    """Validate the only directory that the baseline/model process may read.

    The public episode exposes two observed states, relabelled exactly 0/1.
    Kinematic parameters other than the articulation type are rejected.
    """

    scene_dir = Path(scene_dir).resolve()
    transforms_path = scene_dir / "transforms.json"
    queries_path = scene_dir / "endpoint_queries.json"
    if not transforms_path.is_file():
        raise FileNotFoundError(f"missing public transforms: {transforms_path}")
    if not queries_path.is_file():
        raise FileNotFoundError(f"missing public query directions: {queries_path}")

    transforms = _load_json(transforms_path)
    queries = _load_json(queries_path)
    public_files = audit_public_tree(scene_dir)
    assert_no_evaluator_fields(transforms)
    assert_no_evaluator_fields(queries)

    articulation = transforms.get("articulation")
    if articulation is not None:
        if not isinstance(articulation, Mapping) or set(articulation) != {"type"}:
            raise ValueError("public articulation metadata may contain only 'type'")

    frames = transforms.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("public transforms must contain non-empty frames")
    seen_states: set[int] = set()
    for index, frame in enumerate(frames):
        if not isinstance(frame, Mapping):
            raise ValueError(f"frame {index} is not an object")
        if "state" not in frame or not _is_binary_state(frame["state"]):
            raise ValueError(f"frame {index} must expose state 0 or 1")
        state = int(frame["state"])
        if "local_state" in frame and (
            not _is_binary_state(frame["local_state"]) or int(frame["local_state"]) != state
        ):
            raise ValueError(f"frame {index} local_state must equal state")
        seen_states.add(state)
    if seen_states != {0, 1}:
        raise ValueError(f"both relabelled states are required; found {sorted(seen_states)}")

    raw_queries = queries.get("queries") if isinstance(queries, Mapping) else None
    if not isinstance(raw_queries, list) or len(raw_queries) != 2:
        raise ValueError("endpoint_queries.json must contain exactly two queries")
    query_by_id: dict[str, Mapping[str, Any]] = {}
    for query in raw_queries:
        if not isinstance(query, Mapping):
            raise ValueError("each public endpoint query must be an object")
        extra = set(query).difference({"query_id", "local_direction"})
        if extra:
            raise ValueError(f"public endpoint query has unsupported fields: {sorted(extra)}")
        query_id = query.get("query_id")
        if query_id in query_by_id:
            raise ValueError(f"duplicate endpoint query: {query_id}")
        query_by_id[query_id] = query
    if set(query_by_id) != set(QUERY_IDS):
        raise ValueError(f"query ids must be {QUERY_IDS}")
    expected_directions = {QUERY_IDS[0]: -1, QUERY_IDS[1]: 1}
    for query_id, direction in expected_directions.items():
        if query_by_id[query_id].get("local_direction") != direction:
            raise ValueError(f"{query_id} must have local_direction {direction}")

    return {
        "scene_id": scene_dir.name,
        "scene_dir": str(scene_dir),
        "frame_count": len(frames),
        "state_counts": {str(state): sum(int(frame["state"]) == state for frame in frames) for state in (0, 1)},
        "queries": [{"query_id": query_id, "local_direction": expected_directions[query_id]} for query_id in QUERY_IDS],
        "public_tree": public_files,
        "public_tree_sha256": content_sha256(public_files),
    }


def validate_query_scalar(value: float) -> float:
    """Validate but never clamp a render query (extrapolation is intentional)."""

    value = float(value)
    if not math.isfinite(value):
        raise ValueError("endpoint query scalar must be finite")
    return value


@dataclass(frozen=True)
class BaselinePolicy:
    name: str
    lower_scalar: float
    upper_scalar: float
    renderer: str
    requires_trained_model: bool

    @classmethod
    def from_name(cls, name: str) -> "BaselinePolicy":
        if name == "observed-span":
            return cls(name, 0.0, 1.0, "none", False)
        if name == "symmetric-linear":
            return cls(name, -0.5, 1.5, "none", False)
        if name == "splart-middle":
            return cls(name, 0.0, 1.0, "original-splart", True)
        raise ValueError(f"unknown baseline {name!r}; choose one of {BASELINES}")

    def predictions(self) -> tuple[dict[str, Any], dict[str, Any]]:
        scalars = (self.lower_scalar, self.upper_scalar)
        directions = (-1, 1)
        return tuple(
            {
                "query_id": query_id,
                "local_direction": direction,
                "predicted_local_scalar": validate_query_scalar(scalar),
                # Original SplArt has no closed-side prediction head and must
                # abstain. Scalar-only diagnostics retain a separately labelled
                # blind fixed prior, never promoted as the learned baseline.
                "predicted_closed": (None if self.name == "splart-middle" else query_id == "outside_state_0"),
            }
            for query_id, direction, scalar in zip(QUERY_IDS, directions, scalars)
        )  # type: ignore[return-value]


def _candidate_provenance(checkpoint_dir: Path, source_dir: Path) -> dict[str, Any]:
    checkpoint_dir = Path(checkpoint_dir).resolve()
    source_dir = Path(source_dir).resolve()
    checkpoint = checkpoint_dir / f"step-{FINAL_TRAIN_STEP:09d}.ckpt"
    config = checkpoint_dir.parent / "config.yml"
    dataparser = checkpoint_dir.parent / "dataparser_transforms.json"
    for label, path in (("checkpoint", checkpoint), ("config", config), ("dataparser transforms", dataparser)):
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(f"missing ordinary final candidate {label}: {path}")
    all_steps = sorted(checkpoint_dir.glob("step-*.ckpt"))
    if checkpoint not in all_steps:
        raise ValueError("fixed final checkpoint is absent")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=source_dir, check=True, capture_output=True, text=True, timeout=30
    ).stdout.strip()
    tree = subprocess.run(
        ["git", "rev-parse", "HEAD^{tree}"], cwd=source_dir, check=True, capture_output=True, text=True, timeout=30
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=source_dir, check=True, capture_output=True, text=True, timeout=30
    ).stdout.strip()
    if dirty:
        raise ValueError("candidate source worktree is not clean")
    return {
        "checkpoint": {"path": str(checkpoint), "sha256": sha256_file(checkpoint), "step": FINAL_TRAIN_STEP},
        "config": {"path": str(config), "sha256": sha256_file(config)},
        "dataparser_transforms": {"path": str(dataparser), "sha256": sha256_file(dataparser)},
        "source": {"path": str(source_dir), "commit": head, "tree": tree},
    }


def make_prediction(
    scene_dir: Path, baseline: str, checkpoint_dir: Path | None = None, source_dir: Path | None = None
) -> dict[str, Any]:
    public = validate_public_scene(scene_dir)
    policy = BaselinePolicy.from_name(baseline)
    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir).resolve()
    if policy.requires_trained_model and checkpoint_dir is None:
        raise ValueError("splart-middle requires a freshly trained checkpoint directory")
    if not policy.requires_trained_model and checkpoint_dir is not None:
        raise ValueError(f"{baseline} is scalar-only and must not receive a checkpoint")
    if policy.requires_trained_model and source_dir is None:
        raise ValueError("splart-middle requires the clean candidate source directory")
    candidate = (
        _candidate_provenance(checkpoint_dir, source_dir)  # type: ignore[arg-type]
        if policy.requires_trained_model
        else None
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "scene_id": public["scene_id"],
        "baseline": policy.name,
        "source_commit": SOURCE_COMMIT,
        "training": {
            "from_scratch": policy.requires_trained_model,
            "author_checkpoint": False,
            "endpoint_physics_enabled": False,
            "seed_or_checkpoint_selection": False,
        },
        "renderer": policy.renderer,
        "closed_prediction": {
            "status": "unknown" if policy.name == "splart-middle" else "blind-fixed-prior-diagnostic",
            "query_id": None if policy.name == "splart-middle" else QUERY_IDS[0],
        },
        "candidate": candidate,
        "public_input": {
            "root": public["scene_dir"],
            "tree_sha256": public["public_tree_sha256"],
            "file_count": len(public["public_tree"]),
        },
        "observed_states": [0, 1],
        "endpoint_predictions": list(policy.predictions()),
    }


def build_splart_training_argv(scene_dir: Path, output_dir: Path, experiment_name: str) -> list[str]:
    """Construct the canonical original-SplArt, fresh-scratch training argv."""

    public = validate_public_scene(scene_dir)
    if not experiment_name or experiment_name.startswith(("/", "\\")):
        raise ValueError("experiment_name must be a non-empty relative name")
    argv = [
        "ns-train",
        "splart",
        "--output-dir",
        str(Path(output_dir).resolve()),
        "--experiment-name",
        experiment_name,
        "--vis",
        "tensorboard",
        "--data",
        public["scene_dir"],
        "--max-num-iterations",
        "25000",
        "--pipeline.model.num-random",
        "999999",
        "--pipeline.model.random-scale",
        "1.3",
    ]
    forbidden_flags = {"--load-dir", "--load-config", "--load-checkpoint", "--resume"}
    if forbidden_flags.intersection(argv):
        raise AssertionError("fresh-scratch argv unexpectedly contains resume/checkpoint input")
    # This also rejects accidentally templating evaluator secrets into argv.
    assert_model_process_boundary(argv)
    return argv


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)
