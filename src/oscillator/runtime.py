"""Cloneable shared runtime state; smooth mixtures before reference adjustment."""

from dataclasses import dataclass

import numpy as np
import torch

from .model import CadenceNetwork, HarmonicDecoder
from .preprocess import reconstruct_root, yaw_rotation


@dataclass(frozen=True)
class StepTimings:
    """Alternating supplied strikes. First strike anchors phase 0 (left) or .5 (right).

    Unequal half-cycle interpolation is explicitly enabled by supplying this schedule.
    The caller/contact supervisor owns the timings; no inferred cadence is blended in.
    """

    times: tuple[float, ...]
    first_foot: str = "left"

    def __post_init__(self):
        if (
            len(self.times) < 3
            or not np.isfinite(self.times).all()
            or np.any(np.diff(self.times) <= 0)
        ):
            raise ValueError("Supply at least three strictly increasing finite strike times")
        if self.first_foot not in {"left", "right"}:
            raise ValueError("first_foot must be left or right")

    def sample(self, time: float) -> tuple[float, float, float]:
        if not self.times[0] <= time <= self.times[-1]:
            raise ValueError("Step schedule does not cover the requested time")
        i = min(int(np.searchsorted(self.times, time, side="right")) - 1, len(self.times) - 2)
        duration = self.times[i + 1] - self.times[i]
        phase = (0.0 if self.first_foot == "left" else 0.5) + 0.5 * i
        phase += 0.5 * (time - self.times[i]) / duration
        other = i + 1 if i + 2 < len(self.times) else i - 1
        period = duration + self.times[other + 1] - self.times[other]
        return phase % 1, 0.5 / duration, period


@dataclass
class RuntimeState:
    time: float
    phase: float
    nominal_hz: float
    z: torch.Tensor
    path_pos: np.ndarray
    path_yaw: float

    def clone(self):
        return RuntimeState(
            self.time,
            self.phase,
            self.nominal_hz,
            self.z.clone(),
            self.path_pos.copy(),
            self.path_yaw,
        )


@dataclass
class Reference:
    coordinates: np.ndarray  # 22 joints + 6 root residuals
    coordinate_velocity: np.ndarray
    root_pos: np.ndarray
    root_rot_xyzw: np.ndarray
    root_linear_velocity: np.ndarray  # world frame; includes progression of the path
    root_angular_velocity: np.ndarray  # world frame; rotvec rates use the SO(3) Jacobian
    phase_rate: float


def root_world_velocities(residual, residual_rate, path_yaw, command):
    rotation = yaw_rotation(np.array([path_yaw]))
    yaw_velocity = np.array([0.0, 0.0, command[2]])
    linear_local = np.array([command[0], command[1], 0.0]) + residual_rate[:3]
    linear_local += np.cross(yaw_velocity, residual[:3])
    vector, vector_rate = residual[3:], residual_rate[3:]
    theta = np.linalg.norm(vector)
    if theta < 1e-4:
        a, b = 0.5 - theta**2 / 24, 1 / 6 - theta**2 / 120
    else:
        a, b = (1 - np.cos(theta)) / theta**2, (theta - np.sin(theta)) / theta**3
    local_angular = vector_rate + a * np.cross(vector, vector_rate)
    local_angular += b * np.cross(vector, np.cross(vector, vector_rate))
    return rotation.apply(linear_local)[0], yaw_velocity + rotation.apply(local_angular)[0]


def integrate_planar_path(position, yaw: float, command, dt: float):
    """Exact SE(2) integration of constant body-frame vx, vy, yaw rate."""
    turn = float(command[2]) * dt
    midpoint = yaw + turn / 2
    distance_scale = dt * np.sinc(turn / (2 * np.pi))
    c, s = np.cos(midpoint), np.sin(midpoint)
    vx, vy = command[:2]
    result = position.copy()
    result[:2] += distance_scale * np.array([c * vx - s * vy, s * vx + c * vy])
    return result, yaw + turn


class OscillatorRuntime:
    def __init__(
        self,
        decoder: HarmonicDecoder,
        tau_z: float,
        cadence: CadenceNetwork | None = None,
        cadence_rate_limit: float = 1.0,
    ):
        if not np.isfinite(tau_z) or tau_z <= 0:
            raise ValueError("Mixture smoothing time constant must be positive")
        if not np.isfinite(cadence_rate_limit) or cadence_rate_limit <= 0:
            raise ValueError("Cadence rate limit must be positive")
        self.decoder = decoder.eval()
        self.cadence = cadence.eval() if cadence is not None else None
        self.tau_z = tau_z
        self.cadence_rate_limit = cadence_rate_limit

    def _context(self, context):
        x = torch.as_tensor(
            context, dtype=self.decoder.waveforms.dtype, device=self.decoder.waveforms.device
        )
        if x.shape != (4,) or not torch.isfinite(x).all() or x[3] <= 0:
            raise ValueError("Context must be finite [vx, vy, yaw_rate, positive_cycle_period]")
        return x

    def _frequency(self, context):
        return (
            float(self.cadence(context[:3])) if self.cadence is not None else 1 / float(context[3])
        )

    @torch.no_grad()
    def initial_state(
        self,
        context,
        phase=0.0,
        time=0.0,
        path_pos=None,
        path_yaw=0.0,
        timings: StepTimings | None = None,
    ):
        context = self._context(context).clone()
        if timings is not None:
            phase, _, period = timings.sample(time)
            context[3] = period
            hz = 1 / period
        else:
            hz = self._frequency(context)
            context[3] = 1 / hz
        position = np.zeros(3) if path_pos is None else np.array(path_pos, dtype=float, copy=True)
        if (
            position.shape != (3,)
            or not np.isfinite(position).all()
            or not np.isfinite([phase, time, path_yaw]).all()
        ):
            raise ValueError("Invalid initial runtime state")
        return RuntimeState(
            float(time),
            float(phase) % 1,
            hz,
            self.decoder.mixture(context),
            position,
            float(path_yaw),
        )

    @torch.no_grad()
    def advance(
        self,
        state: RuntimeState,
        context,
        dt: float,
        timings: StepTimings | None = None,
        adjustment=None,
    ) -> Reference:
        """Advance in place. adjustment(reference, state) may apply downstream kinematics.

        No per-joint target filtering follows the adjustment hook. An adjustment must also
        provide consistent velocities; foot planting/IK is outside this baseline.
        """
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be positive and finite")
        context = self._context(context).clone()
        new_time = state.time + dt
        if timings is not None:
            timings.sample(state.time)  # Fail before mutating state on schedule exhaustion.
            phase, phase_rate, period = timings.sample(new_time)
            context[3] = period
            # External timings are the sole phase control. Do not call the cadence network.
            new_hz = state.nominal_hz
        else:
            target_hz = self._frequency(context)
            delta = target_hz - state.nominal_hz
            ramp_time = min(dt, abs(delta) / self.cadence_rate_limit)
            new_hz = state.nominal_hz + np.sign(delta) * self.cadence_rate_limit * ramp_time
            phase_advance = (state.nominal_hz + new_hz) * ramp_time / 2
            phase_advance += new_hz * (dt - ramp_time)
            phase = (state.phase + phase_advance) % 1
            phase_rate = new_hz
            # Keep the decoder's cycle period consistent with the actual nominal clock.
            context[3] = 1 / new_hz
        target_z = self.decoder.mixture(context)
        z = target_z + (state.z - target_z) * np.exp(-dt / self.tau_z)
        z_rate = (target_z - z) / self.tau_z
        path_pos, path_yaw = integrate_planar_path(
            state.path_pos, state.path_yaw, context.cpu().numpy(), dt
        )
        phi = self.decoder.waveforms.new_tensor(phase)
        y = self.decoder.decode(phi, z).cpu().numpy()
        dy = (
            self.decoder.reference_derivative(
                phi,
                z,
                phi.new_tensor(phase_rate),
                z_rate,
            )
            .cpu()
            .numpy()
        )
        root_pos, root_quat = reconstruct_root(y[None, 22:], path_pos[None], np.array([path_yaw]))
        root_linear, root_angular = root_world_velocities(
            y[22:], dy[22:], path_yaw, context.cpu().numpy()
        )
        state.time, state.phase, state.nominal_hz, state.z = new_time, phase, float(new_hz), z
        state.path_pos, state.path_yaw = path_pos, path_yaw
        reference = Reference(
            y, dy, root_pos[0], root_quat[0], root_linear, root_angular, float(phase_rate)
        )
        return reference if adjustment is None else adjustment(reference, state)
