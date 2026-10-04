"""The specified 2,328-parameter conditional harmonic decoder."""

import math

import torch
from torch import nn


def harmonic_features(phase: torch.Tensor, derivative=False) -> torch.Tensor:
    k = torch.arange(1, 4, dtype=phase.dtype, device=phase.device)
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
    def __init__(self):
        super().__init__()
        self.context_net = nn.Sequential(
            nn.Linear(4, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 4),
        )
        self.waveforms = nn.Parameter(torch.randn(5, 28, 7) * 0.02)
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
        return torch.einsum("...h,rdh->...rd", harmonic_features(phase, derivative), self.waveforms)

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
        weights = self.waveforms.new_tensor([0, 1, 1, 16, 16, 81, 81])
        return z.square().mean(), (self.waveforms.square() * weights).mean()


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
