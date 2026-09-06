from types import SimpleNamespace

import torch

from splart.gauge_energy_profile import (
    GaugeEquivariantProfileHead, fields_to_profile, fit_object_split_conformal,
    make_profile_null, object_macro_nmae, swap_profiles,
)


def _inputs(batch=3, radii=3, samples=17, channels=9):
    torch.manual_seed(4)
    features = torch.randn(batch, 2, radii, samples, channels)
    lower = -torch.linspace(0.05, 2.0, samples).repeat(batch, radii, 1)
    upper = 1.0 + torch.linspace(0.05, 2.0, samples).repeat(batch, radii, 1)
    return features, torch.stack((lower, upper), 1)


def test_shared_head_has_numerically_exact_swap_equivariance() -> None:
    features, scalars = _inputs()
    head = GaugeEquivariantProfileHead().double()
    direct = head(features.double(), scalars.double())
    sf, ss = swap_profiles(features.double(), scalars.double())
    swapped = head(sf, ss)
    assert torch.equal(swapped, direct.flip(1))
    endpoints = head.endpoints(features.double(), scalars.double())
    swapped_endpoints = head.endpoints(sf, ss)
    assert torch.allclose(swapped_endpoints[:, 0], 1.0 - endpoints[:, 1], rtol=0, atol=2e-16)
    assert torch.allclose(swapped_endpoints[:, 1], 1.0 - endpoints[:, 0], rtol=0, atol=2e-16)


def test_distances_are_nonnegative_and_profile_permutation_is_not_invariant() -> None:
    features, scalars = _inputs()
    head = GaugeEquivariantProfileHead()
    prediction = head(features, scalars)
    permutation = torch.randperm(features.shape[-2])
    permuted = head(features[..., permutation, :], scalars)
    assert (prediction >= 0).all()
    assert not torch.allclose(prediction, permuted)


def test_preregistered_nulls_preserve_shape_and_coordinate_channel() -> None:
    features, _ = _inputs()
    assert torch.count_nonzero(make_profile_null(features, "zero_geometry")) == 0
    assert torch.count_nonzero(make_profile_null(features, "order_only")) == 0
    permutation = torch.arange(features.shape[-2] - 1, -1, -1)
    assert torch.equal(make_profile_null(features, "profile_permutation", permutation), features[..., permutation, :])


def test_object_macro_weights_objects_not_rows() -> None:
    predicted = torch.tensor([[0.0, 0.0], [2.0, 2.0], [0.0, 0.0]])
    target = torch.zeros_like(predicted)
    assert torch.allclose(object_macro_nmae(predicted, target, ["a", "a", "b"]), torch.tensor(0.5))


def test_object_split_conformal_has_simultaneous_object_coverage() -> None:
    target = torch.ones(10, 2)
    prediction = target.clone()
    prediction[:, 0] += torch.arange(10) / 100
    prediction[:, 1] -= torch.arange(10) / 200
    conformal = fit_object_split_conformal(prediction, target, [f"o{i}" for i in range(10)], 0.9)
    lo, hi = conformal.distance_interval(prediction)
    covered = ((target >= lo) & (target <= hi)).all(1).float().mean()
    assert covered >= 0.9
    assert conformal.radius.ndim == 0


def test_complete_multiradius_profile_export() -> None:
    def field(offset):
        values = {name: torch.arange(5, dtype=torch.float32) + offset for name in (
            "signed_gap", "contact_energy", "penetration_energy", "support_energy",
            "inside_penetration_energy", "contact_mass", "support_rise", "total_energy", "posterior")}
        return SimpleNamespace(scalars=torch.linspace(-2, -0.1, 5) + offset, **values)
    features, scalars = fields_to_profile((field(0), field(1)), (field(2), field(3)))
    assert features.shape == (2, 2, 5, 9)
    assert scalars.shape == (2, 2, 5)
