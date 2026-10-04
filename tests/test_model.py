import numpy as np
import torch

from oscillator.model import CadenceNetwork, HarmonicDecoder


def test_architecture_counts_and_periodicity():
    torch.manual_seed(1)
    model = HarmonicDecoder().double()
    assert model.waveforms.numel() == 980
    assert sum(p.numel() for p in model.context_net.parameters()) == 1348
    assert model.parameter_count == 2328
    cadence = CadenceNetwork().double()
    assert sum(p.numel() for p in cadence.parameters()) == 81
    phi = torch.linspace(-2, 2, 20, dtype=torch.float64)
    x = torch.randn(20, 4, dtype=torch.float64)
    torch.testing.assert_close(model(phi, x), model(phi + 1, x))
    frequency = cadence(torch.tensor([[-1e5, 0.0, 0.0], [1e5, 0.0, 0.0]]).double())
    assert torch.all((frequency >= cadence.min_hz) & (frequency <= cadence.max_hz))


def test_reference_derivative_includes_mixture_change():
    torch.manual_seed(2)
    model = HarmonicDecoder().double()
    model.target_scale.copy_(torch.linspace(0.1, 3, 28))
    phi = torch.tensor([0.23, 0.89], dtype=torch.float64)
    z = torch.randn(2, 4, dtype=torch.float64)
    z_rate = torch.randn(2, 4, dtype=torch.float64)
    phase_rate = torch.tensor([1.3, 0.9], dtype=torch.float64)
    eps = 1e-6
    difference = (
        model.decode(phi + eps * phase_rate, z + eps * z_rate)
        - model.decode(phi - eps * phase_rate, z - eps * z_rate)
    ) / (2 * eps)
    expected = model.reference_derivative(phi, z, phase_rate, z_rate)
    torch.testing.assert_close(difference, expected, atol=1e-8, rtol=1e-7)
    phase_only = model.reference_derivative(phi, z, phase_rate, torch.zeros_like(z_rate))
    assert np.max(np.abs((expected - phase_only).detach().numpy())) > 1e-3


def test_normalisation_roundtrip_and_constant_channels():
    model = HarmonicDecoder()
    x = torch.tensor([[1, 2, 3, 4], [1, 3, 5, 7]], dtype=torch.float32)
    y = torch.arange(56, dtype=torch.float32).reshape(2, 28)
    model.set_normalisation(x, y)
    assert model.context_scale[0] > 0
    phi = torch.tensor([0.2, 0.7])
    torch.testing.assert_close(
        model(phi, x), model(phi, x, normalised=True) * model.target_scale + model.target_mean
    )
