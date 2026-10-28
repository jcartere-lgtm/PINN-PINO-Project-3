"""Independent analytic checks for the complex operator and GPE residual."""

import unittest

import numpy as np
import torch
from torch import nn

from project3_operator import ComplexDeepONet, fit_operator, physics_residual, predict_operator


class _NonlinearConstantWave(nn.Module):
    """Exact zero-potential periodic solution, including its nonlinear phase."""

    length = 16.0
    T = 2.5
    g = 1.0

    def forward(self, knots, coordinates):
        t = coordinates[:, 1]
        amplitude = self.length ** -0.5
        phase = t / self.length
        return amplitude * torch.stack((torch.cos(phase), -torch.sin(phase)), dim=1)

    def control_at(self, knots, t):
        return torch.zeros_like(t)


class _LinearHarmonicWave(nn.Module):
    """Exact u=1, g=0 harmonic ground state with chemical potential 1/2."""

    length = 16.0
    T = 2.5
    g = 0.0

    def forward(self, knots, coordinates):
        x, t = coordinates[:, 0], coordinates[:, 1]
        amplitude = torch.exp(-x.square() / 2) / torch.pi ** 0.25
        return amplitude[:, None] * torch.stack((torch.cos(t / 2), -torch.sin(t / 2)), dim=1)

    def control_at(self, knots, t):
        return torch.ones_like(t)


class OperatorTests(unittest.TestCase):
    def test_nonlinear_constant_wave_has_zero_complex_residual(self):
        # Checks the cubic term and negative phase evolution independently of
        # numerical training or spectral propagation. Samples include nonzero x.
        coordinates = torch.tensor(
            [[-7.1, 0.0], [-1.0, 0.4], [0.0, 1.0], [3.0, 2.5]],
            dtype=torch.float64,
        )
        residual = physics_residual(
            _NonlinearConstantWave(), torch.zeros((4, 3), dtype=torch.float64), coordinates
        )
        self.assertLess(float(residual.detach().abs().max()), 1e-12)

    def test_linear_harmonic_wave_has_zero_complex_residual(self):
        # Both kinetic/potential cancellation and i*psi_t signs must be right.
        coordinates = torch.tensor(
            [[-3.0, 0.2], [-0.7, 1.0], [0.0, 2.0], [1.7, 2.4]],
            dtype=torch.float64,
        )
        residual = physics_residual(
            _LinearHarmonicWave(), torch.ones((4, 3), dtype=torch.float64), coordinates
        )
        self.assertLess(float(residual.detach().abs().max()), 1e-12)

    def test_whole_control_train_validation_overlap_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "disjoint"):
            fit_operator({}, train_indices=[0, 1], val_indices=[1, 2], steps=1)

    def test_empty_training_split_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "nonempty"):
            fit_operator({}, train_indices=[], val_indices=[0], steps=1)

    def test_single_time_prediction_uses_physical_time(self):
        torch.manual_seed(42)
        model = ComplexDeepONet(
            [0.0, 1.25, 2.5],
            {"length": 16.0, "T": 2.5, "width": 8, "depth": 1,
             "latent": 4, "frequencies": [1]},
        )
        x = np.linspace(-8.0, 8.0, 32, endpoint=False)
        knots = np.array([1.0, 7.0, 25.0])
        requested_time = 0.37
        single = predict_operator(model, knots, x, [requested_time])
        multiple = predict_operator(model, knots, x, [requested_time, 1.0, 2.5])
        self.assertEqual(single.shape, (1, x.size))
        self.assertTrue(np.iscomplexobj(single))
        np.testing.assert_allclose(single[0], multiple[0], rtol=2e-6, atol=2e-7)
        self.assertTrue(model.training, "Inference should restore the caller's training/eval state")


if __name__ == "__main__":
    unittest.main()
