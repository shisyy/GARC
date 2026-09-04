import json

import numpy as np

from benchmark.audit_public_proxy import walk_keys
from benchmark.endpoint_proxy import (
    camera_points,
    classify_static_mobile,
    Cloud,
    geometry_certificate,
    PROXY_QUALITY_THRESHOLDS,
    public_queries,
    to_world,
    transform_mobile,
)


def test_public_queries_expose_directions_not_true_scalars():
    payload = public_queries()
    assert payload["queries"] == [
        {"query_id": "outside_state_0", "local_direction": -1},
        {"query_id": "outside_state_1", "local_direction": 1},
    ]
    assert not walk_keys(payload)
    raw = json.dumps(payload).lower()
    assert "closed_query" not in raw and "local_scalars" not in raw


def test_opengl_backprojection_roundtrip_identity():
    depth = np.zeros((5, 7), dtype=np.uint16)
    depth[1, 5] = 2000
    points, valid = camera_points(depth, (4.0, 4.0, 3.0, 2.0), 1)
    assert valid.sum() == 1
    point = points[0]
    assert np.allclose(point, [1.0, 0.5, -2.0])
    assert np.allclose(to_world(points, np.eye(4)), points)
    assert np.allclose([4 * point[0] / -point[2] + 3, 2 - 4 * point[1] / -point[2]], [5, 1])


def test_prismatic_endpoint_clouds_meet_at_same_interior_state():
    articulation = {"type": 2, "axis": [1.0, 0.0, 0.0], "dist": 2.0}
    lower = transform_mobile(np.array([[0, 0, 0]], np.float32), articulation, 0.3)
    upper = transform_mobile(np.array([[2, 0, 0]], np.float32), articulation, -0.7)
    assert np.allclose(lower, upper, atol=1e-6)


def test_revolute_endpoint_clouds_meet_at_same_interior_state():
    articulation = {"type": 1, "axis": [0, 0, 1], "pivot": [0, 0, 0], "angle": np.pi / 2}
    lower = transform_mobile(np.array([[1, 0, 0]], np.float32), articulation, 0.25)
    upper = transform_mobile(np.array([[0, 1, 0]], np.float32), articulation, -0.75)
    assert np.allclose(lower, upper, atol=1e-6)


def test_geometry_hypotheses_separate_static_and_prismatic_mobile():
    static0 = np.array([[0, y, 0] for y in np.linspace(-1, 1, 9)], np.float32)
    static1 = static0.copy()
    mobile0 = np.array([[2, y, 0] for y in np.linspace(-1, 1, 9)], np.float32)
    mobile1 = mobile0 + np.array([0.5, 0, 0], np.float32)
    points = np.concatenate((static0, mobile0, static1, mobile1))
    states = np.array([0] * 18 + [1] * 18, np.uint8)
    labels = classify_static_mobile(
        points, states, {"type": 2, "axis": [1, 0, 0], "dist": 0.5}, voxel=0.01
    )
    assert np.array_equal(labels[:9], np.zeros(9))
    assert np.array_equal(labels[9:18], np.ones(9))


def test_proxy_quality_thresholds_are_preregistered():
    assert PROXY_QUALITY_THRESHOLDS == {
        "foreground_iou_min": 0.70,
        "static_iou_min": 0.70,
        "mobile_iou_min": 0.50,
        "psnr_min": 16.0,
        "depth_mae_m_max": 0.025,
    }


def test_geometry_certificate_reports_symmetric_endpoint_alignment():
    static = np.array([[0, y, 0] for y in np.linspace(-0.04, 0.04, 9)], np.float32)
    mobile0 = np.array([[0.2, y, 0] for y in np.linspace(-0.04, 0.04, 9)], np.float32)
    mobile1 = mobile0 + np.array([0.05, 0, 0], np.float32)
    cloud = Cloud(
        points=np.concatenate((static, mobile0, static, mobile1)),
        colors=np.zeros((36, 3), np.uint8),
        labels=np.array([0] * 9 + [1] * 9 + [0] * 9 + [1] * 9, np.uint8),
        source_states=np.array([0] * 18 + [1] * 18, np.uint8),
    )
    result = geometry_certificate(
        cloud, {"type": 2, "axis": [1, 0, 0], "dist": 0.05}, voxel=0.004
    )
    assert set(result["alignment"]["directional"]) == {
        "static_0_to_1", "static_1_to_0", "mobile_0_to_1", "mobile_1_to_0"
    }
    assert result["alignment_pass"]
    assert not result["closure"]["identifiable"]


def test_counterfactual_contact_is_required_to_certify_closed_endpoint():
    yz = np.array([[y, z] for y in np.linspace(-0.04, 0.04, 9) for z in np.linspace(-0.04, 0.04, 9)])
    static = np.c_[np.full(len(yz), -0.02), yz].astype(np.float32)
    mobile0 = np.c_[np.zeros(len(yz)), yz].astype(np.float32)
    mobile1 = mobile0 + np.array([0.4, 0, 0], np.float32)
    cloud = Cloud(
        points=np.concatenate((static, mobile0, static, mobile1)),
        colors=np.zeros((4 * len(yz), 3), np.uint8),
        labels=np.array([0] * len(yz) + [1] * len(yz) + [0] * len(yz) + [1] * len(yz), np.uint8),
        source_states=np.array([0] * (2 * len(yz)) + [1] * (2 * len(yz)), np.uint8),
    )
    result = geometry_certificate(
        cloud, {"type": 2, "axis": [1, 0, 0], "dist": 0.4}, voxel=0.004
    )
    assert result["closure"]["contact_vote"] == 0
    assert result["closure"]["closed_query"] == "outside_state_0"
    assert result["closure"]["identifiable"]
