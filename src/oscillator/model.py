"""Whole-body harmonic and periodic RFF decoders with analytic reference derivatives."""

import math

import torch
from torch import nn


def harmonic_features(phase: torch.Tensor, derivative=False, harmonics=3) -> torch.Tensor:
    k = torch.arange(1, harmonics + 1, dtype=phase.dtype, device=phase.device)
    angle = 2 * math.pi * phase[..., None] * k
    if derivative:
        paired = torch.stack(
            [-angle.sin() * 2 * math.pi * k, angle.cos() * 2 * math.pi * k], dim=-1
        )
        dc = torch.zeros_like(phase[..., None])
    else:
        paired = torch.stack([angle.cos(), angle.sin()], dim=-1)
        dc = torch.ones_like(phase[..., None])
    return torch.cat([dc, paired.flatten(-2)], dim=-1)


class HarmonicDecoder(nn.Module):
    def __init__(self, harmonics=3):
        super().__init__()
        if not isinstance(harmonics, int) or harmonics < 1:
            raise ValueError("Harmonic count must be a positive integer")
        self.harmonics = harmonics
        self.context_net = nn.Sequential(
            nn.Linear(4, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 4),
        )
        self.waveforms = nn.Parameter(torch.randn(5, 28, 1 + 2 * harmonics) * 0.02)
        self.register_buffer("context_mean", torch.zeros(4))
        self.register_buffer("context_scale", torch.ones(4))
        self.register_buffer("target_mean", torch.zeros(28))
        self.register_buffer("target_scale", torch.ones(28))

    @property
    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def set_normalisation(self, context, target):
        for prefix, data in (("context", context), ("target", target)):
            getattr(self, f"{prefix}_mean").copy_(data.mean(0))
            getattr(self, f"{prefix}_scale").copy_(data.std(0, correction=0).clamp_min(1e-3))

    def mixture(self, context):
        return self.context_net((context - self.context_mean) / self.context_scale)

    def basis(self, phase, derivative=False):
        return torch.einsum(
            "...h,rdh->...rd", harmonic_features(phase, derivative, self.harmonics), self.waveforms
        )

    def decode(self, phase, z, normalised=False):
        basis = self.basis(phase)
        result = basis[..., 0, :] + torch.einsum("...r,...rd->...d", z, basis[..., 1:, :])
        return result if normalised else result * self.target_scale + self.target_mean

    def forward(self, phase, context, normalised=False):
        return self.decode(phase, self.mixture(context), normalised=normalised)

    def reference_derivative(self, phase, z, phase_rate, z_rate):
        """dy/dt = partial_phi(y) phi_dot + sum_r B_r z_dot_r, in physical units."""
        basis = self.basis(phase)
        derivative = self.basis(phase, derivative=True)
        phase_term = derivative[..., 0, :] + torch.einsum(
            "...r,...rd->...d", z, derivative[..., 1:, :]
        )
        mixture_term = torch.einsum("...r,...rd->...d", z_rate, basis[..., 1:, :])
        return (phase_term * phase_rate[..., None] + mixture_term) * self.target_scale

    def regularisation(self, z):
        # Suppress large mixtures and high-frequency energy/acceleration.
        k = torch.arange(1, self.harmonics + 1, device=self.waveforms.device)
        weights = torch.cat([k.new_zeros(1), k.pow(4).repeat_interleave(2)])
        return z.square().mean(), (self.waveforms.square() * weights).mean()


class RFFDecoder(nn.Module):
    """Periodic phase/context random features with a regularised linear readout.

    The four runtime state values are normalised context, rather than waveform mixtures.
    Smooth them before decoding; derivatives include their evolution. Projection matrices
    are fixed buffers. Phase enters on the unit circle, so all derivatives are periodic.
    Bandwidths are in radians per unit-circle / normalised-context coordinate.
    """

    def __init__(self, projections=40, phase_bandwidth=1.0, context_bandwidth=0.5, seed=7):
        super().__init__()
        if projections < 1 or not isinstance(projections, int):
            raise ValueError("Projection count must be a positive integer")
        if not all(math.isfinite(b) and b > 0 for b in (phase_bandwidth, context_bandwidth)):
            raise ValueError("RFF bandwidths must be positive and finite")
        self.projections = projections
        self.phase_bandwidth, self.context_bandwidth = phase_bandwidth, context_bandwidth
        self.seed = seed
        generator = torch.Generator().manual_seed(seed)
        omega = torch.randn(projections, 6, generator=generator)
        omega[:, :2] *= phase_bandwidth
        omega[:, 2:] *= context_bandwidth
        self.register_buffer("omega", omega)
        self.readout = nn.Linear(2 * projections, 28)
        for prefix, width in (("context", 4), ("target", 28)):
            self.register_buffer(f"{prefix}_mean", torch.zeros(width))
            self.register_buffer(f"{prefix}_scale", torch.ones(width))

    @property
    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())

    set_normalisation = HarmonicDecoder.set_normalisation

    def mixture(self, context):
        return (context - self.context_mean) / self.context_scale

    def features(self, phase, z):
        angle = 2 * math.pi * phase
        circle = torch.stack([angle.cos(), angle.sin()], dim=-1)
        shape = torch.broadcast_shapes(circle.shape[:-1], z.shape[:-1])
        x = torch.cat([circle.expand(*shape, 2), z.expand(*shape, 4)], dim=-1)
        projection = x @ self.omega.T
        return torch.cat([projection.cos(), projection.sin()], dim=-1) / math.sqrt(self.projections)

    def decode(self, phase, z, normalised=False):
        result = self.readout(self.features(phase, z))
        return result if normalised else result * self.target_scale + self.target_mean

    def forward(self, phase, context, normalised=False):
        return self.decode(phase, self.mixture(context), normalised)

    def reference_derivative(self, phase, z, phase_rate, z_rate):
        angle = 2 * math.pi * phase
        circle = torch.stack([angle.cos(), angle.sin()], dim=-1)
        circle_rate = torch.stack([-angle.sin(), angle.cos()], dim=-1)
        circle_rate = circle_rate * (2 * math.pi * phase_rate[..., None])
        shape = torch.broadcast_shapes(circle.shape[:-1], z.shape[:-1], z_rate.shape[:-1])
        projection = torch.cat([circle.expand(*shape, 2), z.expand(*shape, 4)], -1) @ self.omega.T
        rate = (
            torch.cat([circle_rate.expand(*shape, 2), z_rate.expand(*shape, 4)], -1) @ self.omega.T
        )
        derivative = torch.cat([-projection.sin() * rate, projection.cos() * rate], -1)
        return (
            (derivative @ self.readout.weight.T) / math.sqrt(self.projections) * self.target_scale
        )

    @torch.no_grad()
    def fit_ridge(self, phase, context, target, ridge):
        """Minimise mean squared error + ridge * sum(weight²); intercept is unpenalised."""
        if not math.isfinite(ridge) or ridge <= 0:
            raise ValueError("Ridge penalty must be positive and finite")
        features = self.features(phase, self.mixture(context)).double()
        target = ((target - self.target_mean) / self.target_scale).double()
        mean_x, mean_y = features.mean(0), target.mean(0)
        x, y = features - mean_x, target - mean_y
        identity = torch.eye(x.shape[1], dtype=x.dtype, device=x.device)
        # Loss is averaged over frames and all 28 coordinates, like harmonic training.
        weights = torch.linalg.solve(x.T @ x + len(x) * 28 * ridge * identity, x.T @ y)
        self.readout.weight.copy_(weights.T)
        self.readout.bias.copy_(mean_y - mean_x @ weights)


def decoder_spec(model):
    if isinstance(model, RFFDecoder):
        return {
            "kind": "rff",
            "projections": model.projections,
            "phase_bandwidth": model.phase_bandwidth,
            "context_bandwidth": model.context_bandwidth,
            "seed": model.seed,
        }
    return {"kind": "harmonic", "harmonics": model.harmonics}


def make_decoder(spec):
    options = {k: v for k, v in spec.items() if k != "kind"}
    if spec["kind"] == "harmonic":
        return HarmonicDecoder(**options)
    if spec["kind"] == "rff":
        return RFFDecoder(**options)
    raise ValueError(f"Unknown decoder kind: {spec['kind']}")


class CadenceNetwork(nn.Module):
    """Positive bounded cycle frequency, never a step frequency."""

    def __init__(self, min_hz=0.5, max_hz=3.0):
        super().__init__()
        if not 0 < min_hz < max_hz:
            raise ValueError("Invalid cycle frequency bounds")
        self.min_hz, self.max_hz = float(min_hz), float(max_hz)
        self.net = nn.Sequential(nn.Linear(3, 16), nn.Tanh(), nn.Linear(16, 1))
        self.register_buffer("command_mean", torch.zeros(3))
        self.register_buffer("command_scale", torch.ones(3))

    def forward(self, command):
        x = (command - self.command_mean) / self.command_scale
        return self.min_hz + (self.max_hz - self.min_hz) * self.net(x).squeeze(-1).sigmoid()
