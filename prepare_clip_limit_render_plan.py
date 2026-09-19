#!/usr/bin/env python3
"""Create a hash-bound, label-independent counterfactual render plan."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from run_glpdt_source_gate import FORBIDDEN_MARKERS, sha256_file
from splart.clip_limit_cache import fixed_candidate_grid


PLAN_SCHEMA = "splart-c-clip-ld-render-plan/v2"
AZIMUTH_ELEVATION_DEGREES = ((0, 0), (90, 0), (180, 0), (270, 0), (45, 35), (225, -35))
ROW_KEYS = {
    "asset",
    "axis_orientation",
    "episode_id",
    "joint_selector",
    "object_id",
    "split",
    "state0_fraction",
    "state1_fraction",
    "state_order",
}
ART_SELECTION_SHA256 = "b39aba89d3670b2930df7e77fdc8fac0d493e43aea3ca0e2acf340a46b2b5089"
NJC_PREREGISTER_SHA256 = "1928eed04e666e3162d8306ace62a6b195c7202a0ddf84d939dc98d3b1db8f25"


def _regular_json(path: Path, description: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{description} must be a regular non-symlink file")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{description} must contain a JSON object")
    return value


def articraft_fractions(object_id: str) -> tuple[float, float]:
    """Reproduce node8.4's public object-id-only episode selection exactly."""

    digest = int(hashlib.sha256(object_id.encode()).hexdigest()[:8], 16)
    lower, upper = (0.15, 0.25, 0.35), (0.65, 0.75, 0.85)
    f0, f1 = lower[digest % 3], upper[(digest // 3) % 3]
    if f1 - f0 < 0.35:
        f1 = 0.85
    return f0, f1


def reconstruct_njc_rows(preregister: dict[str, Any]) -> list[dict[str, Any]]:
    """Reconstruct the public PILC episode IDs without labels or target endpoints."""

    if preregister.get("schema") != "splart-pilc-preregister-v1":
        raise ValueError("unexpected NJC preregistration schema")
    split = preregister.get("object_split")
    grid = preregister.get("episode_grid")
    if not isinstance(split, dict) or not isinstance(grid, dict):
        raise ValueError("NJC preregistration is missing split/grid metadata")
    if len(split.get("train", [])) != 11 or len(split.get("calibration", [])) != 4:
        raise ValueError("NJC preregistration is not the frozen 11/4 split")
    required_grid = {
        "state0_fraction",
        "state1_fraction",
        "minimum_span",
        "axis_orientation_augmentation",
        "state_order_augmentation",
    }
    if not required_grid.issubset(grid):
        raise ValueError("NJC preregistration is missing episode-grid fields")
    rows = []
    for prereg_split, output_split in (("train", "source_train"), ("calibration", "source_validation")):
        for object_id in split[prereg_split]:
            object_count = 0
            for f0 in grid["state0_fraction"]:
                for f1 in grid["state1_fraction"]:
                    if f1 - f0 < grid["minimum_span"]:
                        continue
                    for orientation in grid["axis_orientation_augmentation"]:
                        base = f"{prereg_split}:{object_id}:{f0:.2f}:{f1:.2f}:axis{orientation:+d}"
                        for order in grid["state_order_augmentation"]:
                            episode_id = hashlib.sha256(f"splart-pilc-v1:{base}:{order}".encode()).hexdigest()
                            if order == "forward":
                                start, end = f0, f1
                            elif order == "reverse":
                                start, end = f1, f0
                            else:
                                raise ValueError(f"unsupported NJC state order: {order}")
                            rows.append(
                                {
                                    "split": output_split,
                                    "object_id": object_id,
                                    "episode_id": episode_id,
                                    "state0_fraction": float(start),
                                    "state1_fraction": float(end),
                                    "axis_orientation": int(orientation),
                                    "state_order": order,
                                    "joint_selector": {"kind": "exact_name", "name": "lid_hinge"},
                                    "asset": {
                                        "kind": "njc_directory",
                                        "relative_path": object_id,
                                        "required_files": ["object.urdf", "base_final.obj", "lid_final.obj", "qc.json"],
                                    },
                                }
                            )
                            object_count += 1
            if object_count != 32:
                raise ValueError("each NJC object must reconstruct exactly 32 episodes")
    if len(rows) != 480 or len({row["episode_id"] for row in rows}) != 480:
        raise ValueError("NJC preregistration did not reconstruct 480 unique episodes")
    return rows


def reconstruct_articraft_rows(selection: dict[str, Any]) -> list[dict[str, Any]]:
    selected = selection.get("selection")
    if not isinstance(selected, dict):
        raise ValueError("unexpected Articraft selection schema")
    rows = []
    for key, split in (("endpoint_pretrain", "source_train"), ("endpoint_validation", "source_validation")):
        names = selected.get(key)
        if not isinstance(names, list) or not all(isinstance(name, str) and name.endswith(".gz") for name in names):
            raise ValueError("Articraft selection contains invalid archive names")
        for archive in names:
            object_id = Path(archive).stem
            f0, f1 = articraft_fractions(object_id)
            rows.append(
                {
                    "split": split,
                    "object_id": object_id,
                    "episode_id": object_id,
                    "state0_fraction": f0,
                    "state1_fraction": f1,
                    "axis_orientation": 1,
                    "state_order": "forward",
                    "joint_selector": {"kind": "first_bounded_revolute_document_order"},
                    "asset": {"kind": "articraft_archive", "relative_path": archive},
                }
            )
    identities = [(row["split"], row["object_id"], row["episode_id"]) for row in rows]
    if len(set(identities)) != len(rows):
        raise ValueError("Articraft selection has duplicate object identities")
    return rows


def _geometry_identities(source_index: Path, domain: str) -> set[tuple[str, str, str]]:
    if domain == "articraft":
        from run_glpdt_source_gate import validate_index

        rows = validate_index(source_index.resolve())
        return {(row["split"], row["object_id"], row["object_id"]) for row, _ in rows}
    if domain == "njc":
        from run_cr_fpl_njc_gate import validate_njc_index

        rows = validate_njc_index(source_index.resolve())
        return {(row["split"], row["object_id"], row["episode_id"]) for row, _ in rows}
    raise ValueError("domain must be articraft or njc")


def expected_rows(
    source_index: Path, domain: str, *, selection_path: Path | None = None, preregister_path: Path | None = None
) -> tuple[list[dict[str, Any]], dict[str, dict[str, str]]]:
    if domain == "articraft":
        if selection_path is None or preregister_path is not None:
            raise ValueError("Articraft requires only --selection")
        selection = _regular_json(selection_path, "Articraft selection")
        selection_sha256 = sha256_file(selection_path)
        if selection_sha256 != ART_SELECTION_SHA256:
            raise ValueError("Articraft selection digest mismatch")
        rows = reconstruct_articraft_rows(selection)
        bindings = {"selection": {"path": str(selection_path.resolve()), "sha256": selection_sha256}}
    elif domain == "njc":
        if preregister_path is None or selection_path is not None:
            raise ValueError("NJC requires only --preregister")
        preregister = _regular_json(preregister_path, "NJC preregistration")
        preregister_sha256 = sha256_file(preregister_path)
        if preregister_sha256 != NJC_PREREGISTER_SHA256:
            raise ValueError("NJC preregistration digest mismatch")
        rows = reconstruct_njc_rows(preregister)
        bindings = {"preregister": {"path": str(preregister_path.resolve()), "sha256": preregister_sha256}}
    else:
        raise ValueError("domain must be articraft or njc")
    geometry = _geometry_identities(source_index, domain)
    by_identity = {(row["split"], row["object_id"], row["episode_id"]): row for row in rows}
    if not geometry.issubset(by_identity):
        raise ValueError("public pose metadata does not cover the frozen geometry index")
    selected_rows = [by_identity[identity] for identity in sorted(geometry)]
    return selected_rows, bindings


def _validate_row(row: dict[str, Any]) -> None:
    if not isinstance(row, dict) or set(row) != ROW_KEYS:
        raise ValueError("render plan row has missing or extra pose metadata")
    if row["split"] not in ("source_train", "source_validation"):
        raise ValueError("render plan row has an invalid split")
    if not all(isinstance(row[key], str) and row[key] for key in ("object_id", "episode_id", "state_order")):
        raise ValueError("render plan row has invalid string metadata")
    if row["state_order"] not in ("forward", "reverse") or row["axis_orientation"] not in (-1, 1):
        raise ValueError("render plan row has invalid orientation/order")
    if not all(isinstance(row[key], (int, float)) for key in ("state0_fraction", "state1_fraction")):
        raise ValueError("render plan row has invalid state fractions")
    if row["state0_fraction"] == row["state1_fraction"]:
        raise ValueError("render plan row states must differ")
    if not isinstance(row["joint_selector"], dict) or not isinstance(row["asset"], dict):
        raise ValueError("render plan row has invalid joint/asset metadata")


def validate_plan(
    plan: dict[str, Any],
    source_index: Path,
    domain: str,
    *,
    selection_path: Path | None = None,
    preregister_path: Path | None = None,
) -> None:
    if plan.get("schema") != PLAN_SCHEMA or plan.get("domain") != domain:
        raise ValueError("render plan schema/domain mismatch")
    if plan.get("source_index_sha256") != sha256_file(source_index.resolve()):
        raise ValueError("render plan is bound to a different source index")
    if plan.get("target_labels_used") is not False:
        raise ValueError("render plan must be label independent")
    if plan.get("views") != [list(value) for value in AZIMUTH_ELEVATION_DEGREES]:
        raise ValueError("render plan must use the exact fixed six-view schedule")
    expected_shape = [2, len(fixed_candidate_grid()), len(AZIMUTH_ELEVATION_DEGREES), 3, 224, 224]
    if plan.get("required_render_shape") != expected_shape:
        raise ValueError("render plan must require uint8 candidate views with shape [3,224,224]")
    expected, bindings = expected_rows(
        source_index, domain, selection_path=selection_path, preregister_path=preregister_path
    )
    if plan.get("input_bindings") != bindings:
        raise ValueError("render plan public metadata binding mismatch")
    rows = plan.get("rows")
    if not isinstance(rows, list):
        raise ValueError("render plan has no rows")
    for row in rows:
        _validate_row(row)
    if rows != expected:
        raise ValueError("render plan pose metadata differs from public reconstruction")
    grid = fixed_candidate_grid().tolist()
    if plan.get("candidate_coordinates") != grid:
        raise ValueError("render plan does not use the fixed candidate grid")
    if plan.get("canonical_q") != {"side0": [-value for value in grid], "side1": [1.0 + value for value in grid]}:
        raise ValueError("render plan canonical q contract mismatch")


def prepare(
    source_index: Path,
    domain: str,
    output: Path,
    *,
    selection_path: Path | None = None,
    preregister_path: Path | None = None,
) -> dict[str, Any]:
    if output.exists():
        raise ValueError("render plan output already exists")
    rows, bindings = expected_rows(
        source_index, domain, selection_path=selection_path, preregister_path=preregister_path
    )
    grid = fixed_candidate_grid().tolist()
    plan = {
        "schema": PLAN_SCHEMA,
        "domain": domain,
        "source_index": str(source_index.resolve()),
        "source_index_sha256": sha256_file(source_index.resolve()),
        "input_bindings": bindings,
        "coordinate_semantics": "normalized outward distance from each observed boundary",
        "candidate_coordinates": grid,
        "canonical_q": {"side0": [-value for value in grid], "side1": [1.0 + value for value in grid]},
        "pose_formula": "lower+(state0_fraction+q*(state1_fraction-state0_fraction))*(upper-lower)",
        "target_labels_used": False,
        "views": [list(value) for value in AZIMUTH_ELEVATION_DEGREES],
        "required_render_shape": [2, len(grid), len(AZIMUTH_ELEVATION_DEGREES), 3, 224, 224],
        "rows": rows,
        "next_stage": "stream real renderer mini-batches directly into build_clip_limit_cache.py",
        "streaming_contract": "iter_candidate_batches receives the complete audited row and emits contiguous uint8 [N,6,3,224,224] batches",
        "protected_splits_read": [],
        "box_labels_read": False,
    }
    validate_plan(plan, source_index, domain, selection_path=selection_path, preregister_path=preregister_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return plan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-index", type=Path, required=True)
    parser.add_argument("--domain", choices=("articraft", "njc"), required=True)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--preregister", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    launch = "\n".join(str(value).lower() for value in vars(args).values() if value is not None)
    if any(marker in launch for marker in FORBIDDEN_MARKERS):
        raise ValueError("render-plan launch contains a protected path marker")
    print(
        json.dumps(
            prepare(
                args.source_index,
                args.domain,
                args.output,
                selection_path=args.selection,
                preregister_path=args.preregister,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
