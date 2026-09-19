import pytest
import torch

from splart.clip_limit_gate import (
    add_cr_fpl_object_macro_metrics,
    add_object_macro_metrics,
    apply_semantic_control,
    binary_auroc,
    normalize_semantic_by_train_rms,
    semantic_endpoint_auroc,
)


def test_controls_are_executable_zero_sign_flip_coordinate_and_object_derangement():
    coordinates = torch.linspace(0, 2, 129).view(1, 1, 129).expand(4, 2, -1).clone()
    coefficient = torch.arange(1, 5, dtype=torch.float32).view(4, 1, 1)
    evidence = (coefficient * coordinates.square() + torch.arange(4).view(4, 1, 1)).unsqueeze(-1)
    objects = ("a", "a", "b", "b")
    episodes = ("a0", "a1", "b0", "b1")
    true, _ = apply_semantic_control(evidence, coordinates, objects, episodes, "cstr")
    zero, zero_coordinates = apply_semantic_control(evidence, coordinates, objects, episodes, "zero")
    swapped, _ = apply_semantic_control(evidence, coordinates, objects, episodes, "sign_flip")
    coordinate_only, _ = apply_semantic_control(evidence, coordinates, objects, episodes, "coordinate_only")
    shuffled, _ = apply_semantic_control(evidence, coordinates, objects, episodes, "object_trajectory_shuffle")
    assert torch.equal(zero, torch.zeros_like(true))
    assert torch.equal(zero_coordinates, coordinates)
    assert torch.allclose(swapped, true)
    assert coordinate_only.shape == true.shape and not torch.equal(coordinate_only, true)
    assert torch.allclose(shuffled[:2], true[2:]) and torch.allclose(shuffled[2:], true[:2])


def test_semantic_scale_preserves_zero_and_sign_and_auroc_is_rank_based():
    evidence = torch.tensor([[[[-2.0], [0.0], [2.0]], [[-1.0], [0.0], [1.0]]]])
    coordinates = torch.tensor([[[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]]])
    data = {
        "source_train": (evidence, coordinates, ("a",), ("a",)),
        "source_validation": (evidence, coordinates, ("b",), ("b",)),
    }
    normalized = normalize_semantic_by_train_rms(data)["source_validation"][0]
    assert torch.equal(normalized == 0, evidence == 0)
    assert torch.equal(torch.sign(normalized), torch.sign(evidence))
    target = torch.tensor([[1.5, 1.5]])
    assert semantic_endpoint_auroc(evidence, coordinates, target) == 1.0
    assert binary_auroc(evidence[..., 0], torch.zeros_like(evidence[..., 0], dtype=torch.bool)) != 0.0


def test_object_shuffle_fails_when_episode_counts_do_not_match():
    evidence = torch.zeros(3, 2, 3, 1)
    coordinates = torch.zeros(3, 2, 3)
    with pytest.raises(ValueError, match="equal trajectories"):
        apply_semantic_control(evidence, coordinates, ("a", "a", "b"), ("a0", "a1", "b0"), "object_trajectory_shuffle")


def test_njc_object_macro_metrics_weight_objects_and_report_one_loop():
    class Output:
        distance = torch.tensor([[1.0, 1.0], [3.0, 3.0], [9.0, 9.0]])
        loop_distances = torch.stack((distance + 1.0, distance), dim=-1)

    metrics = {}
    target = torch.zeros(3, 2)
    objects = ("a", "a", "b")
    add_object_macro_metrics(metrics, Output(), target, objects)
    add_cr_fpl_object_macro_metrics(metrics, Output(), target, objects)
    assert metrics["candidate_object_macro_mean_side_nmae"] == pytest.approx(5.5)
    assert metrics["candidate_object_macro_one_loop_mean_side_nmae"] == pytest.approx(6.5)
    assert metrics["cr_fpl_control_object_macro_one_loop_mean_side_nmae"] == pytest.approx(6.5)
