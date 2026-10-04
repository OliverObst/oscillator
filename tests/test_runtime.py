import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from oscillator.model import CadenceNetwork, HarmonicDecoder
from oscillator.runtime import OscillatorRuntime, StepTimings, integrate_planar_path


def runtime():
    torch.manual_seed(2)
    return OscillatorRuntime(HarmonicDecoder().double(), tau_z=0.2)


def test_preview_clone_does_not_mutate_shared_mixture_or_root_state():
    r = runtime()
    state = r.initial_state([1, 0, 0, 1])
    preview = state.clone()
    previous_z = state.z.clone()
    r.advance(preview, [2, 0.2, 0.3, 0.5], 0.1)
    torch.testing.assert_close(state.z, previous_z)
    np.testing.assert_array_equal(state.path_pos, [0, 0, 0])
    assert state.phase == state.time == 0
    assert not torch.equal(state.z, preview.z)
    assert preview.z.data_ptr() != state.z.data_ptr()


def test_smoothing_and_reference_derivative_follow_exact_exponential():
    r = runtime()
    r.cadence_rate_limit = 100
    state = r.initial_state([1, 0, 0, 1])
    old_z = state.z.clone()
    new_context = torch.tensor([2, 0, 0, 1], dtype=torch.float64)
    target = r.decoder.mixture(new_context).detach()
    result = r.advance(state, new_context, 0.1)
    torch.testing.assert_close(state.z, target + (old_z - target) * np.exp(-0.5))
    eps = 1e-6
    before, after = state.clone(), state.clone()
    # The returned instantaneous derivative should match the next very small runtime step.
    y_before = r.decoder.decode(torch.tensor(state.phase), state.z).detach().numpy()
    y_after = r.advance(after, new_context, eps).coordinates
    np.testing.assert_allclose(
        (y_after - y_before) / eps, result.coordinate_velocity, atol=1e-4, rtol=1e-4
    )
    assert before.time == state.time


def test_rate_limit_integrates_ramp_and_does_not_compete_with_step_timings():
    r = runtime()
    state = r.initial_state([1, 0, 0, 1])
    r.advance(state, [1, 0, 0, 0.5], 0.2)
    assert state.nominal_hz == pytest.approx(1.2)
    assert state.phase == pytest.approx(0.22)
    # Ramp reaches its target mid-step: integrate ramp and remaining constant frequency.
    r.advance(state, [1, 0, 0, 1 / 1.3], 0.2)
    assert state.nominal_hz == pytest.approx(1.3)
    assert state.phase == pytest.approx(0.475)

    class ForbiddenCadence(CadenceNetwork):
        def forward(self, command):
            raise AssertionError("Cadence must never run under supplied timings")

    schedule = StepTimings((0.0, 0.3, 1.0, 1.3, 2.0))
    r.cadence = ForbiddenCadence().double()
    state = r.initial_state([1, 0, 0, 0.2], timings=schedule)
    reference = r.advance(state, [1, 0, 0, 0.2], 0.3, timings=schedule)
    assert state.phase == pytest.approx(0.5)
    assert reference.phase_rate == pytest.approx(0.5 / 0.7)
    r.advance(state, [1, 0, 0, 0.2], 0.7, timings=schedule)
    assert state.phase == pytest.approx(0.0)


def test_step_schedule_exhaustion_is_atomic_and_right_anchor_supported():
    r = runtime()
    schedule = StepTimings((0.0, 0.4, 1.0), first_foot="right")
    state = r.initial_state([0, 0, 0, 1], timings=schedule)
    assert state.phase == 0.5
    before = state.clone()
    with pytest.raises(ValueError, match="cover"):
        r.advance(state, [1, 0, 0, 1], 1.1, timings=schedule)
    assert state.time == before.time
    torch.testing.assert_close(state.z, before.z)
    with pytest.raises(ValueError, match="increasing"):
        StepTimings((0, 0.4, 0.4))


def test_planar_root_path_exact_turn_integration():
    position, yaw = integrate_planar_path(np.zeros(3), 0, [1, 0, 1], np.pi / 2)
    np.testing.assert_allclose(position, [1, 1, 0], atol=1e-12)
    assert yaw == np.pi / 2
    position, yaw = integrate_planar_path(position, yaw, [1, 0, 0], 2)
    np.testing.assert_allclose(position, [1, 3, 0], atol=1e-12)


def test_adjustment_receives_smoothed_reference():
    r = runtime()
    state = r.initial_state([1, 0, 0, 1])
    previous = state.z.clone()
    seen = []

    def adjust(reference, current):
        seen.append(current.z.clone())
        reference.coordinates[0] = 42
        return reference

    result = r.advance(state, [3, 0, 0, 1], 0.01, adjustment=adjust)
    assert result.coordinates[0] == 42  # No subsequent per-joint filtering.
    assert not torch.equal(seen[0], previous)
    target = r.decoder.mixture(torch.tensor([3, 0, 0, 1], dtype=torch.float64))
    assert not torch.equal(seen[0], target)


def test_world_root_derivatives_include_path_motion_and_rotation_jacobian():
    r = runtime()
    context = [1.2, 0.2, 0.5, 0.8]
    state = r.initial_state(context)
    reference = r.advance(state, context, 0.3)
    eps = 1e-6
    later = r.advance(state.clone(), context, eps)
    np.testing.assert_allclose(
        (later.root_pos - reference.root_pos) / eps,
        reference.root_linear_velocity,
        atol=1e-5,
        rtol=1e-5,
    )
    rotation_delta = (
        Rotation.from_quat(later.root_rot_xyzw) * Rotation.from_quat(reference.root_rot_xyzw).inv()
    ).as_rotvec() / eps
    np.testing.assert_allclose(
        rotation_delta, reference.root_angular_velocity, atol=1e-5, rtol=1e-5
    )
