import json
from pathlib import Path

import numpy as np
import pytest
import torch

from oscillator.kinematics import feet_world
from oscillator.model import HarmonicDecoder, RFFDecoder
from oscillator.runtime import OscillatorRuntime


@pytest.mark.parametrize("model", [HarmonicDecoder(6), RFFDecoder(seed=17)])
def test_periodic_pose_and_total_derivative(model):
    model = model.double()
    model.target_scale.copy_(torch.linspace(0.1, 2.0, 28))
    phi = torch.tensor([0.0, 0.99, 0.3], dtype=torch.float64)
    z = torch.randn(3, 4, dtype=torch.float64)
    rate = torch.randn_like(z)
    hz = torch.tensor([1.1, 0.9, 1.3], dtype=torch.float64)
    velocity = model.reference_derivative(phi, z, hz, rate)
    torch.testing.assert_close(
        model.decode(phi, z), model.decode(phi + 1, z), atol=1e-12, rtol=1e-12
    )
    torch.testing.assert_close(velocity, model.reference_derivative(phi + 1, z, hz, rate))
    eps = 1e-6
    difference = (
        model.decode(phi + hz * eps, z + rate * eps) - model.decode(phi - hz * eps, z - rate * eps)
    ) / (2 * eps)
    torch.testing.assert_close(velocity, difference, atol=1e-8, rtol=1e-7)


def test_ridge_fit_and_fixed_projection_reproducibility():
    torch.manual_seed(9)
    model = RFFDecoder(seed=17).double()
    assert model.parameter_count == 2268
    assert HarmonicDecoder(6).parameter_count == 3168
    other = RFFDecoder(seed=17).double()
    torch.testing.assert_close(model.omega, other.omega, atol=0, rtol=0)
    phi = torch.rand(100, dtype=torch.float64)
    context = torch.randn(100, 4, dtype=torch.float64)
    truth = torch.randn(80, 28, dtype=torch.float64)
    target = model.features(phi, context) @ truth + torch.linspace(-1, 1, 28)
    omega = model.omega.clone()
    model.fit_ridge(phi, context, target, 1e-7)
    torch.testing.assert_close(model.omega, omega, atol=0, rtol=0)
    assert float((model(phi, context) - target).square().mean().detach()) < 1e-4
    # Check the ridge optimum, including an unpenalised intercept.
    loss = (model(phi, context, normalised=True) - target).square().mean()
    loss += 1e-7 * model.readout.weight.square().sum()
    gradients = torch.autograd.grad(loss, tuple(model.readout.parameters()))
    assert max(float(g.abs().max()) for g in gradients) < 1e-12
    with pytest.raises(ValueError, match="Ridge"):
        model.fit_ridge(phi, context, target, 0)


def test_rff_runtime_smooths_inputs_and_clones_preview():
    model = RFFDecoder(seed=3)
    runtime = OscillatorRuntime(model, 0.2)
    state = runtime.initial_state([0.8, 0, 0, 0.9])
    before = state.clone()
    ref = runtime.advance(state, [1.2, 0.2, 0.3, 0.9], 0.1)
    expected = np.array([1.2, 0.2, 0.3, 0.9]) + (
        np.array([0.8, 0, 0, 0.9]) - np.array([1.2, 0.2, 0.3, 0.9])
    ) * np.exp(-0.5)
    np.testing.assert_allclose(state.z.numpy(), expected, atol=1e-7)
    assert np.isfinite(ref.coordinate_velocity).all()
    torch.testing.assert_close(before.z, torch.tensor([0.8, 0, 0, 0.9]))
    clone = state.clone()
    runtime.advance(clone, [0.1, -0.2, -1.0, 0.7], 0.3)
    assert state.time == pytest.approx(0.1)
    assert not torch.equal(clone.z, state.z)


def test_foot_diagnostics_use_source_kinematics_without_alignment():
    directory = Path(__file__).resolve().parents[1] / "web/assets"
    skeleton = json.loads((directory / "skeleton.json").read_text())
    cases = json.loads((directory / "validation.json").read_text())["kinematics"]
    actual = feet_world(
        skeleton,
        np.array([c["joints"] for c in cases]),
        np.array([c["position"] for c in cases]),
        np.array([c["quaternion"] for c in cases]),
    )
    np.testing.assert_allclose(actual, [c["feet"] for c in cases], atol=2e-6)
    offset = np.array([1.0, -2.0, 0.0])
    shifted = feet_world(
        skeleton,
        np.array([c["joints"] for c in cases]),
        np.array([c["position"] for c in cases]) + offset,
        np.array([c["quaternion"] for c in cases]),
    )
    np.testing.assert_allclose(shifted, actual + offset, atol=1e-12)
