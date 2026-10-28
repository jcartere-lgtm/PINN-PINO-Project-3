"""Physical checks for the conventional GPE reference, using unittest.

Run from the project root with ``python -B -m unittest discover -s tests -v``.
These tests check conservation, exact solutions, stationary PDEs, and the
meaning of the control objective; they do not establish production-grid
accuracy for an arbitrary optimized trajectory.
"""

import unittest

import numpy as np

from project3_gpe import (
    SimulationConfig,
    control_derivative_integral,
    control_knots,
    normalize,
    objective_components,
    propagate,
    solve_ground_state,
)


class ConventionalGPEPhysicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = SimulationConfig(n=256, length=16.0, T=0.5, dt=0.005, g=1.0)
        cls.x, cls.k, cls.dx = cls.config.grid()
        cls.ground_states = {
            u: solve_ground_state(cls.config, u, tol=1e-7, max_iterations=60000)
            for u in (1.0, 100.0)
        }

    def test_normalized_complex_state_has_unit_mass(self):
        initial = np.exp(-self.x**2) * np.exp(0.7j * self.x)
        psi = normalize(initial, self.dx)
        # Integrate independently of the module's mass helper.
        self.assertAlmostEqual(float(np.sum(np.abs(psi)**2) * self.dx), 1.0, places=13)

    def test_free_fourier_mode_has_exact_complex_phase(self):
        config = SimulationConfig(n=256, length=16.0, T=0.73, dt=0.0073, g=0.0)
        x, _, dx = config.grid()
        wave_number = 2 * np.pi * 3 / config.length
        initial = np.exp(1j * wave_number * x) / np.sqrt(config.length)
        solution = propagate(config, initial, [0.0, 0.0], save_times=[0.0, 0.17, config.T])
        for t, numerical in zip(solution["t"], solution["psi"]):
            with self.subTest(time=float(t)):
                exact = initial * np.exp(-0.5j * wave_number**2 * t)
                error = np.sqrt(dx * np.sum(np.abs(numerical - exact)**2))
                self.assertLess(error, 2e-12)

    def test_nonlinear_ground_states_satisfy_stationary_pde(self):
        for u, result in self.ground_states.items():
            with self.subTest(trap_strength=u):
                self.assertTrue(result["converged"], msg=str(result))
                psi = result["psi"]
                mass = self.dx * np.sum(np.abs(psi)**2)
                self.assertAlmostEqual(float(mass), 1.0, places=12)
                # Evaluate the stationary GPE rather than trusting its returned
                # convergence flag or a small imaginary-time update alone.
                hpsi = np.fft.ifft(0.5 * self.k**2 * np.fft.fft(psi))
                hpsi += (0.5 * u * self.x**2 + np.abs(psi)**2) * psi
                chemical_potential = float(np.real(self.dx * np.vdot(psi, hpsi)) / mass)
                residual = np.sqrt(self.dx * np.sum(np.abs(hpsi - chemical_potential * psi)**2))
                self.assertLess(residual, 1.2e-7)

    def test_stationary_density_is_preserved_and_time_refinement_helps(self):
        initial = self.ground_states[1.0]["psi"]
        initial_density = np.abs(initial)**2
        errors = []
        for dt in (0.02, 0.005):
            config = SimulationConfig(n=256, length=16.0, T=0.5, dt=dt, g=1.0)
            final = propagate(config, initial, [1.0, 1.0], save_times=[0.0, config.T])["psi"][-1]
            errors.append(np.linalg.norm(np.abs(final)**2 - initial_density)
                          / np.linalg.norm(initial_density))
        self.assertLess(errors[1], 2e-6)
        # A fourfold step reduction should noticeably improve a second-order
        # propagator; the margin allows the independently computed state's error.
        self.assertLess(errors[1], errors[0] / 8)

    def test_mass_is_conserved_under_changing_nonlinear_trap(self):
        solution = propagate(self.config, self.ground_states[1.0]["psi"],
                             [1.0, 4.0, 2.0, 5.0], save_every=5)
        independently_integrated_mass = self.dx * np.sum(np.abs(solution["psi"])**2, axis=1)
        self.assertLess(np.max(np.abs(independently_integrated_mass - 1.0)), 2e-12)

    def test_linear_ramp_has_analytic_derivative_cost(self):
        expected_integral = 99.0**2 / 2.5
        for number_of_knots in (2, 3, 7, 19):
            with self.subTest(number_of_knots=number_of_knots):
                knots = np.linspace(1.0, 100.0, number_of_knots)
                self.assertAlmostEqual(control_derivative_integral(knots, 2.5),
                                       expected_integral, places=8)
        config = SimulationConfig(n=256, length=16.0, T=2.5, dt=0.01)
        state = self.ground_states[1.0]["psi"]
        components = objective_components(config, state, state, [1.0, 100.0], gamma=1e-4)
        self.assertAlmostEqual(components["total"], 0.5e-4 * expected_integral, places=12)

    def test_fidelity_ignores_global_phase_but_detects_spatial_phase(self):
        uniform = np.ones(self.config.n, dtype=complex) / np.sqrt(self.config.length)
        global_phase = uniform * np.exp(1.234j)
        spatial_phase = uniform * np.exp(2j * np.pi * self.x / self.config.length)
        global_components = objective_components(self.config, global_phase, uniform,
                                                 [1.0, 1.0], gamma=0.0)
        spatial_components = objective_components(self.config, spatial_phase, uniform,
                                                  [1.0, 1.0], gamma=0.0)
        self.assertAlmostEqual(global_components["fidelity"], 1.0, places=13)
        np.testing.assert_allclose(np.abs(spatial_phase)**2, np.abs(uniform)**2,
                                   rtol=0.0, atol=1e-16)
        self.assertLess(spatial_components["fidelity"], 1e-25)
        self.assertAlmostEqual(spatial_components["infidelity"], 1.0, places=13)

    def test_control_construction_preserves_endpoints_and_allows_nonmonotone_paths(self):
        knots = control_knots([1.0, 0.0, 0.9], u0=1.0, uT=100.0)
        self.assertEqual(knots[0], 1.0)
        self.assertEqual(knots[-1], 100.0)
        self.assertTrue(np.all((knots >= 1.0) & (knots <= 100.0)))
        self.assertGreater(knots[1], knots[2])
        self.assertEqual(control_derivative_integral([5.0, 5.0], 2.5), 0.0)


if __name__ == "__main__":
    unittest.main()
