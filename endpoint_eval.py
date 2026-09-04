"""Evaluator-only metric aggregation for endpoint extrapolation.

Unlike :mod:`endpoint_baseline`, this program is allowed to consume a sealed
target record.  Keep it in a separate process and never place its argv or
environment into a model/training launch.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable, Mapping

from splart.endpoint_baselines import QUERY_IDS, write_json_atomic

RENDER_METRICS = ("psnr", "ssim", "lpips", "depth_mae", "static_iou", "mobile_iou", "background_iou", "miou")


def validate_complete_measurement(measurement: Mapping[str, Any], articulation_type: int | str) -> None:
    """Reject partial candidate evidence before any authoritative score exists."""

    endpoints = measurement.get("endpoint_metrics")
    if not isinstance(endpoints, Mapping) or set(endpoints) != set(QUERY_IDS):
        raise ValueError("complete measurement requires exactly both endpoint metric records")
    for query_id in QUERY_IDS:
        record = endpoints[query_id]
        views = record.get("views") if isinstance(record, Mapping) else None
        if not isinstance(views, list) or len(views) != 10:
            raise ValueError(f"complete measurement requires exactly 10 views for {query_id}")
        for view in views:
            if not isinstance(view, Mapping):
                raise ValueError("endpoint view metric must be an object")
            canonical = _canonicalise_view(view)
            missing = set(RENDER_METRICS).difference(canonical)
            if missing:
                raise ValueError(f"endpoint view is missing render metrics: {sorted(missing)}")
            artifacts = view.get("candidate_artifacts")
            if not isinstance(artifacts, Mapping) or len(artifacts) != 3:
                raise ValueError("endpoint view lacks three hashed candidate render artifacts")
            if any(
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
                for value in artifacts.values()
            ):
                raise ValueError("candidate render artifact hash is malformed")

    physics = measurement.get("physics")
    queries = physics.get("queries") if isinstance(physics, Mapping) else None
    if not isinstance(queries, Mapping) or set(queries) != set(QUERY_IDS):
        raise ValueError("physical certificate must cover exactly both endpoint queries")
    required_physics = {
        "valid",
        "terminal_contact_valid",
        "contact_fraction",
        "anchor_contact_fraction",
        "contact_fraction_gain",
        "penetration_fraction",
        "penetration_q99_m",
        "penetration_depth",
        "static_count",
        "mobile_count",
        "static_sample_count",
        "mobile_sample_count",
        "sample_cap_per_part",
        "sampling_rule",
    }
    for query_id, certificate in queries.items():
        if not isinstance(certificate, Mapping) or not required_physics.issubset(certificate):
            raise ValueError(f"physical certificate is incomplete for {query_id}")
        if certificate.get("valid") is not True or not isinstance(certificate.get("terminal_contact_valid"), bool):
            raise ValueError(f"physical certificate is invalid for {query_id}")
        for key in required_physics.difference({"valid", "terminal_contact_valid"}):
            if key != "sampling_rule":
                _finite_float(certificate[key], f"physics.{query_id}.{key}")
        if certificate["sampling_rule"] != "original-index-even-stride-v1":
            raise ValueError(f"physical certificate sampling rule is invalid for {query_id}")
        if certificate["sample_cap_per_part"] != 4096:
            raise ValueError(f"physical certificate sample cap is invalid for {query_id}")
        for part in ("static", "mobile"):
            total = certificate[f"{part}_count"]
            sampled = certificate[f"{part}_sample_count"]
            if (
                not isinstance(total, int)
                or isinstance(total, bool)
                or not isinstance(sampled, int)
                or isinstance(sampled, bool)
            ):
                raise ValueError(f"physical certificate {part} counts must be integers for {query_id}")
            if total <= 0 or sampled != min(total, 4096):
                raise ValueError(f"physical certificate {part} sample coverage is invalid for {query_id}")

    articulation = measurement.get("articulation")
    if not isinstance(articulation, Mapping) or not {"valid", "type_correct", "axis"}.issubset(articulation):
        raise ValueError("candidate articulation measurement is incomplete")
    if not isinstance(articulation["valid"], bool) or not isinstance(articulation["type_correct"], bool):
        raise ValueError("candidate articulation validity fields must be boolean")
    type_name = str(articulation_type).lower()
    if type_name in {"1", "revolute"}:
        required_articulation = {"pivot", "r"}
    elif type_name in {"2", "prismatic"}:
        required_articulation = {"t"}
    else:
        raise ValueError(f"unsupported benchmark articulation type: {articulation_type}")
    if not required_articulation.issubset(articulation):
        raise ValueError("candidate articulation measurement lacks scene-specific errors")
    for key in {"axis", *required_articulation}:
        _finite_float(articulation[key], f"articulation.{key}")


def _finite_float(value: Any, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return fmean(values) if values else None


def _load(path: Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _scene_record(payload: Mapping[str, Any], scene_id: str) -> Mapping[str, Any]:
    """Select one scene from either a per-scene or multi-scene sealed record."""

    if payload.get("scene_id") == scene_id:
        return payload
    scenes = payload.get("scenes")
    if isinstance(scenes, Mapping) and scene_id in scenes:
        record = scenes[scene_id]
        if isinstance(record, Mapping):
            return {"scene_id": scene_id, **record}
    if isinstance(scenes, list):
        matches = [item for item in scenes if isinstance(item, Mapping) and item.get("scene_id") == scene_id]
        if len(matches) == 1:
            return matches[0]
    episodes = payload.get("episodes")
    if isinstance(episodes, Mapping) and scene_id in episodes:
        record = episodes[scene_id]
        if isinstance(record, Mapping):
            return {"scene_id": scene_id, **record}
    raise ValueError(f"sealed evaluator record does not contain scene {scene_id!r}")


def _target(record: Mapping[str, Any]) -> tuple[dict[str, float], str]:
    target = record.get("target", record)
    if not isinstance(target, Mapping):
        raise ValueError("target must be an object")
    raw_scalars = target.get("local_scalars", target.get("local_gt_scalars"))
    if raw_scalars is None:
        raw_scalars = record.get("local_gt_scalars")
    if isinstance(raw_scalars, (list, tuple)) and len(raw_scalars) == 2:
        raw_scalars = dict(zip(QUERY_IDS, raw_scalars))
    if not isinstance(raw_scalars, Mapping) or set(raw_scalars) != set(QUERY_IDS):
        raise ValueError(f"sealed target must provide local scalars for {QUERY_IDS}")
    scalars = {query_id: _finite_float(raw_scalars[query_id], f"target scalar {query_id}") for query_id in QUERY_IDS}
    if not scalars[QUERY_IDS[0]] < scalars[QUERY_IDS[1]]:
        raise ValueError("target local endpoint scalars must be strictly ordered")
    closed_query = target.get("closed_query", record.get("closed_query"))
    if closed_query not in QUERY_IDS:
        raise ValueError("sealed target closed_query must identify one public query")
    return scalars, str(closed_query)


def _prediction_map(prediction: Mapping[str, Any]) -> tuple[dict[str, float], str | None]:
    rows = prediction.get("endpoint_predictions")
    if not isinstance(rows, list) or len(rows) != 2:
        raise ValueError("prediction must contain exactly two endpoint predictions")
    by_id = {row.get("query_id"): row for row in rows if isinstance(row, Mapping)}
    if set(by_id) != set(QUERY_IDS):
        raise ValueError(f"prediction query ids must be {QUERY_IDS}")
    scalars = {
        query_id: _finite_float(by_id[query_id].get("predicted_local_scalar"), f"prediction {query_id}")
        for query_id in QUERY_IDS
    }
    closed = [query_id for query_id in QUERY_IDS if by_id[query_id].get("predicted_closed") is True]
    status = prediction.get("closed_prediction", {})
    if len(closed) == 0 and isinstance(status, Mapping) and status.get("status") == "unknown":
        return scalars, None
    if len(closed) != 1:
        raise ValueError("prediction must declare exactly one closed query or explicit unknown abstention")
    return scalars, closed[0]


def _canonicalise_view(view: Mapping[str, Any]) -> dict[str, float]:
    aliases = {"bg_iou": "background_iou", "part_seg_miou": "miou", "m_iou": "miou"}
    result: dict[str, float] = {}
    for raw_key, raw_value in view.items():
        key = aliases.get(raw_key, raw_key)
        if key in RENDER_METRICS:
            result[key] = _finite_float(raw_value, key)
    part_ious = view.get("part_seg_ious")
    if part_ious is not None:
        if not isinstance(part_ious, (list, tuple)) or len(part_ious) != 3:
            raise ValueError("part_seg_ious must be [static, mobile, background]")
        result.update(
            {
                "static_iou": _finite_float(part_ious[0], "static_iou"),
                "mobile_iou": _finite_float(part_ious[1], "mobile_iou"),
                "background_iou": _finite_float(part_ious[2], "background_iou"),
            }
        )
    if "miou" not in result and all(key in result for key in ("static_iou", "mobile_iou", "background_iou")):
        result["miou"] = fmean(result[key] for key in ("static_iou", "mobile_iou", "background_iou"))
    return result


def _endpoint_render_metrics(measurement: Mapping[str, Any]) -> dict[str, float | None]:
    endpoint_metrics = measurement.get("endpoint_metrics", measurement.get("render_metrics"))
    if endpoint_metrics is None:
        return {key: None for key in RENDER_METRICS}
    if not isinstance(endpoint_metrics, Mapping):
        raise ValueError("endpoint_metrics must be keyed by public query id")
    canonical_views: list[dict[str, float]] = []
    for query_id in QUERY_IDS:
        query_record = endpoint_metrics.get(query_id)
        if query_record is None:
            raise ValueError(f"missing endpoint metrics for {query_id}")
        if isinstance(query_record, Mapping) and "views" in query_record:
            query_record = query_record["views"]
        if isinstance(query_record, Mapping):
            query_record = [query_record]
        if not isinstance(query_record, list) or not query_record:
            raise ValueError(f"endpoint metrics for {query_id} must contain at least one view")
        for view in query_record:
            if not isinstance(view, Mapping):
                raise ValueError(f"endpoint metric view for {query_id} must be an object")
            canonical_views.append(_canonicalise_view(view))
    return {key: _mean(view[key] for view in canonical_views if key in view) for key in RENDER_METRICS}


def evaluate_scene(
    prediction: Mapping[str, Any],
    sealed_payload: Mapping[str, Any],
    measurement_payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    scene_id = prediction.get("scene_id")
    if not isinstance(scene_id, str) or not scene_id:
        raise ValueError("prediction scene_id is required")
    record = _scene_record(sealed_payload, scene_id)
    target_scalars, closed_query = _target(record)
    predicted_scalars, predicted_closed = _prediction_map(prediction)
    measurement = _scene_record(measurement_payload, scene_id) if measurement_payload is not None else {}
    target_range = target_scalars[QUERY_IDS[1]] - target_scalars[QUERY_IDS[0]]
    limit_errors = {
        query_id: abs(predicted_scalars[query_id] - target_scalars[query_id]) / target_range for query_id in QUERY_IDS
    }
    metrics: dict[str, Any] = {
        "normalized_limit_error_lower": limit_errors[QUERY_IDS[0]],
        "normalized_limit_error_upper": limit_errors[QUERY_IDS[1]],
        "normalized_limit_error": fmean(limit_errors.values()),
        "closed_endpoint_accuracy": None if predicted_closed is None else float(predicted_closed == closed_query),
        "closed_endpoint_coverage": float(predicted_closed is not None),
        **_endpoint_render_metrics(measurement),
    }

    physics = measurement.get("physics", measurement.get("physical_certificate", {}))
    if physics is None:
        physics = {}
    if not isinstance(physics, Mapping):
        raise ValueError("physics must be an object")
    terminal = physics.get("terminal_contact_valid")
    metrics["terminal_contact_valid"] = None if terminal is None else float(bool(terminal))
    penetration = physics.get("penetration_depth", physics.get("max_penetration"))
    metrics["penetration_depth"] = None if penetration is None else _finite_float(penetration, "penetration_depth")

    articulation = measurement.get("articulation", {})
    if articulation is None:
        articulation = {}
    if not isinstance(articulation, Mapping):
        raise ValueError("articulation must be an object")
    valid = articulation.get("valid", articulation.get("is_valid"))
    metrics["articulation_valid"] = None if valid is None else float(bool(valid))
    for key, value in articulation.items():
        if key in {"valid", "is_valid"}:
            continue
        if isinstance(value, bool):
            metrics[f"articulation_{key}"] = float(value)
        elif isinstance(value, (int, float)):
            metrics[f"articulation_{key}"] = _finite_float(value, f"articulation.{key}")

    return {"scene_id": scene_id, "baseline": prediction.get("baseline"), "metrics": metrics}


def aggregate_scenes(scene_results: list[Mapping[str, Any]]) -> dict[str, Any]:
    if not scene_results:
        raise ValueError("at least one evaluated scene is required")
    seen: set[str] = set()
    for result in scene_results:
        scene_id = result.get("scene_id")
        if scene_id in seen:
            raise ValueError(f"duplicate scene result: {scene_id}")
        seen.add(scene_id)
    metric_names = sorted({key for result in scene_results for key in result["metrics"]})
    macro = {
        key: _mean(float(result["metrics"][key]) for result in scene_results if result["metrics"].get(key) is not None)
        for key in metric_names
    }
    coverage = {key: sum(result["metrics"].get(key) is not None for result in scene_results) for key in metric_names}
    return {
        "schema_version": "splart-endpoint-evaluation/v1",
        "scene_order": [result["scene_id"] for result in scene_results],
        "scenes": scene_results,
        "macro": macro,
        "coverage": coverage,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", type=Path, action="append", required=True)
    parser.add_argument("--sealed-evaluator-record", type=Path, required=True)
    parser.add_argument(
        "--measurement-record",
        type=Path,
        help="post-render numeric metrics; never the GT asset descriptors in the sealed manifest",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sealed = _load(args.sealed_evaluator_record)
    measurements = _load(args.measurement_record) if args.measurement_record else None
    scene_results = [evaluate_scene(_load(path), sealed, measurements) for path in args.prediction]
    write_json_atomic(args.output, aggregate_scenes(scene_results))
    print(args.output)


if __name__ == "__main__":
    main()
