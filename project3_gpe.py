"""Conventional, dimensionless 1-D Gross--Pitaevskii control benchmark.

The equation is i psi_t = -psi_xx/2 + u(t) x**2 psi/2 + g |psi|**2 psi.
FFT differentiation uses a periodic computational box.  Decay at its edges and
grid/time refinement must therefore be checked before accepting results.

Controls are continuous, piecewise linear on equally spaced time knots.  Their
endpoints are fixed; interior values are restricted to the interval between
the endpoint values.  This box restriction is an explicit additional modeling
choice beyond the admissible class in Adriazola and Goodman, Eqs. (2)--(4).
The interior knots need not be monotone.

This benchmark uses full nonlinear ground states for its initial and target
states.  The paper's two-mode initialization is a different approximation.
No computation or file writing happens when this module is imported.
"""

from dataclasses import dataclass
import math
from time import perf_counter

import numpy as np
from scipy.optimize import minimize


@dataclass(frozen=True)
class SimulationConfig:
    n: int = 256
    length: float = 16.0
    T: float = 2.5
    dt: float = 0.002
    g: float = 1.0

    def __post_init__(self):
        if not isinstance(self.n, (int, np.integer)) or self.n < 16:
            raise ValueError("n must be an integer of at least 16")
        if not all(np.isfinite(v) and v > 0 for v in (self.length, self.T, self.dt)):
            raise ValueError("length, T and dt must be finite and positive")
        if not np.isfinite(self.g) or self.g < 0:
            raise ValueError("This repulsive-interaction benchmark requires g >= 0")

    def grid(self):
        """Return x, angular Fourier wavenumbers k, and dx; omit the duplicate endpoint."""
        dx = self.length / self.n
        x = np.linspace(-self.length / 2, self.length / 2, self.n, endpoint=False)
        k = 2 * np.pi * np.fft.fftfreq(self.n, d=dx)
        return x, k, dx


def mass(psi, dx):
    """Discrete integral of |psi|**2; history arrays return one mass per row."""
    return np.sum(np.abs(np.asarray(psi)) ** 2, axis=-1) * dx


def normalize(psi, dx):
    """Return a normalized copy of one wavefunction."""
    psi = np.asarray(psi, dtype=np.complex128)
    if psi.ndim != 1 or not np.all(np.isfinite(psi)):
        raise ValueError("psi must be a finite one-dimensional wavefunction")
    m = float(mass(psi, dx))
    if not np.isfinite(m) or m <= 0:
        raise ValueError("Cannot normalize a zero or nonfinite wavefunction")
    return psi / np.sqrt(m)


def _check_wavefunction(config, psi):
    psi = np.asarray(psi, dtype=np.complex128)
    if psi.shape != (config.n,) or not np.all(np.isfinite(psi)):
        raise ValueError(f"Expected a finite wavefunction of shape ({config.n},)")
    return psi


def _hamiltonian_action(config, psi, u):
    x, k, _ = config.grid()
    kinetic = np.fft.ifft(0.5 * k**2 * np.fft.fft(psi))
    return kinetic + (0.5 * u * x**2 + config.g * np.abs(psi) ** 2) * psi


def energy(config, psi, u):
    """Full GPE energy: integral [|psi_x|**2/2 + V|psi|**2 + g|psi|**4/2]."""
    psi = _check_wavefunction(config, psi)
    x, k, dx = config.grid()
    psi_x = np.fft.ifft(1j * k * np.fft.fft(psi))
    rho = np.abs(psi) ** 2
    return float(dx * np.sum(0.5 * np.abs(psi_x) ** 2 + 0.5 * u * x**2 * rho
                             + 0.5 * config.g * rho**2))


def stationary_residual(config, psi, u):
    """Return ||H(psi)psi - mu psi||_2, with mu determined from the state."""
    psi = _check_wavefunction(config, psi)
    _, _, dx = config.grid()
    hpsi = _hamiltonian_action(config, psi, u)
    m = float(mass(psi, dx))
    if m <= 0:
        raise ValueError("Stationary residual is undefined for the zero state")
    mu = float(np.real(dx * np.vdot(psi, hpsi)) / m)
    return float(np.sqrt(dx * np.sum(np.abs(hpsi - mu * psi) ** 2)))


def solve_ground_state(config, u, tol=1e-6, max_iterations=60000,
                       imaginary_dt=0.01, initial=None):
    """Find a unit-mass nonlinear ground state by normalized imaginary time.

    A Strang split alternates normalized potential/nonlinear flow and spectral
    kinetic flow.  The nonlinear subflow uses an exponential midpoint step.
    The imaginary step is halved when a discrete fixed point is reached but
    its *stationary PDE residual* is still too large; a small update alone is
    never accepted as proof of convergence.  An optional initial guess is
    replaced by its absolute value to seek the nodeless ground state rather
    than preserve an excited state's phase/sign symmetry. Inspect
    ``converged`` before use.
    """
    if not np.isfinite(u) or u <= 0:
        raise ValueError("A positive trapping coefficient u is required")
    if not np.isfinite(tol) or tol <= 0 or imaginary_dt <= 0 or max_iterations < 1:
        raise ValueError("tol, imaginary_dt, and max_iterations must be positive")
    x, k, dx = config.grid()
    if initial is None:
        # Linear Gaussian is only the starting guess, not the claimed g>0 ground state.
        psi = normalize(np.exp(-0.5 * np.sqrt(u) * x**2), dx)
    else:
        psi = normalize(np.abs(_check_wavefunction(config, initial)), dx)
    potential = 0.5 * u * x**2
    tau = min(float(imaginary_dt), 0.1 / max(1.0, np.sqrt(u)))
    start = perf_counter()
    residual = stationary_residual(config, psi, u)
    iterations = 0
    refinements = 0
    delta_rate = 0.0

    def normalized_potential_step(state, step):
        # Predict density at the half time, then integrate the log-amplitude
        # using that midpoint density. Normalization removes the scalar mu term.
        midpoint = normalize(
            state * np.exp(-0.5 * step * (potential + config.g * np.abs(state)**2)), dx)
        return normalize(
            state * np.exp(-step * (potential + config.g * np.abs(midpoint)**2)), dx)

    while residual > tol and iterations < max_iterations:
        old = psi
        psi = normalized_potential_step(psi, tau / 2)
        psi = normalize(np.fft.ifft(np.exp(-0.5 * tau * k**2) * np.fft.fft(psi)), dx)
        psi = normalized_potential_step(psi, tau / 2)
        iterations += 1
        delta_rate = float(np.sqrt(dx * np.sum(np.abs(psi - old)**2)) / tau)
        if iterations % 25 == 0:
            residual = stationary_residual(config, psi, u)
            if residual > tol and delta_rate < tol / 10:
                tau /= 2
                refinements += 1
    residual = stationary_residual(config, psi, u)
    # Choose a reproducible global phase without changing the physical state.
    psi *= np.exp(-1j * np.angle(psi[np.argmax(np.abs(psi))]))
    return {
        "psi": psi,
        "energy": energy(config, psi, u),
        "residual": residual,
        "iterations": iterations,
        "converged": bool(residual <= tol),
        "imaginary_dt": tau,
        "step_refinements": refinements,
        "update_rate": delta_rate,
        "elapsed_seconds": perf_counter() - start,
        "u": float(u),
    }


def control_knots(interior, u0=1.0, uT=100.0):
    """Map fractional interior knots in [0,1] to physical trap coefficients.

    Endpoints are exactly u0 and uT. Fractions need not be ordered, so this is
    a bounded control family, not a monotonicity constraint.
    """
    interior = np.asarray(interior, dtype=float)
    if interior.ndim != 1 or not np.all(np.isfinite(interior)):
        raise ValueError("interior must be a finite one-dimensional array")
    if np.any(interior < 0) or np.any(interior > 1):
        raise ValueError("Fractional interior knot values must lie in [0,1]")
    if not all(np.isfinite(v) and v >= 0 for v in (u0, uT)):
        raise ValueError("Nonnegative, finite endpoint trap coefficients are required")
    return np.concatenate(([u0], u0 + (uT - u0) * interior, [uT]))


def _checked_knots(knot_values):
    values = np.asarray(knot_values, dtype=float)
    if values.ndim != 1 or values.size < 2 or not np.all(np.isfinite(values)):
        raise ValueError("At least two finite physical knot values are required")
    if np.any(values < 0):
        raise ValueError("Trap coefficients must remain nonnegative")
    return values


def control_derivative_integral(knot_values, T):
    """Exact integral of |du/dt|**2 for equally spaced, piecewise-linear knots."""
    values = _checked_knots(knot_values)
    if not np.isfinite(T) or T <= 0:
        raise ValueError("T must be positive and finite")
    knot_dt = T / (values.size - 1)
    return float(np.sum(np.diff(values)**2) / knot_dt)


def propagate(config, psi0, knot_values, save_every=25, save_times=None):
    """Propagate with second-order Strang splitting and midpoint trap values.

    ``psi`` in the returned dictionary has shape (saved_times, spatial_points)
    and retains the full complex field. Requested ``save_times`` must increase
    strictly from 0 to T. Each saved interval is subdivided so no step exceeds
    config.dt, and control-knot times are always respected internally. Norms
    are measured without renormalizing during real-time propagation.

    ``edge_mass`` is mass in the outer 10 percent of the box (5 percent at
    each edge); it diagnoses possible contamination by periodic boundaries.
    """
    start = perf_counter()
    values = _checked_knots(knot_values)
    x, k, dx = config.grid()
    psi = _check_wavefunction(config, psi0).copy()
    if save_times is None:
        if not isinstance(save_every, (int, np.integer)) or save_every < 1:
            raise ValueError("save_every must be a positive integer")
        nsteps = int(math.ceil(config.T / config.dt))
        indices = np.unique(np.r_[np.arange(0, nsteps, save_every), nsteps])
        times = indices * (config.T / nsteps)
    else:
        times = np.asarray(save_times, dtype=float).copy()
        if (times.ndim != 1 or times.size < 2 or not np.all(np.isfinite(times))
                or np.any(np.diff(times) <= 0)
                or not np.isclose(times[0], 0, rtol=0, atol=1e-12)
                or not np.isclose(times[-1], config.T, rtol=0, atol=1e-12)):
            raise ValueError("save_times must increase strictly from 0 to T")
        times[0], times[-1] = 0.0, config.T
    knot_times = np.linspace(0, config.T, values.size)
    history = np.empty((times.size, config.n), dtype=np.complex128)
    history[0] = psi
    actual_steps = 0
    for output_index, (left, right) in enumerate(zip(times[:-1], times[1:]), start=1):
        # Splitting a step at a control corner avoids assuming smooth u there.
        internal = knot_times[(knot_times > left + 1e-13) & (knot_times < right - 1e-13)]
        boundaries = np.r_[left, internal, right]
        for a, b in zip(boundaries[:-1], boundaries[1:]):
            steps = max(1, int(math.ceil((b - a) / config.dt - 1e-12)))
            h = (b - a) / steps
            kinetic_factor = np.exp(-0.5j * h * k**2)
            for j in range(steps):
                midpoint = a + (j + 0.5) * h
                u_mid = np.interp(midpoint, knot_times, values)
                V = 0.5 * u_mid * x**2
                psi *= np.exp(-0.5j * h * (V + config.g * np.abs(psi)**2))
                psi = np.fft.ifft(kinetic_factor * np.fft.fft(psi))
                psi *= np.exp(-0.5j * h * (V + config.g * np.abs(psi)**2))
            actual_steps += steps
        history[output_index] = psi
    edge_mask = np.abs(x) >= 0.45 * config.length
    return {
        "psi": history,
        "t": times,
        "x": x,
        "mass": mass(history, dx),
        "edge_mass": mass(history[:, edge_mask], dx),
        "elapsed_seconds": perf_counter() - start,
        "steps": actual_steps,
        "knot_times": knot_times,
        "knot_values": values.copy(),
    }


def objective_components(config, final_state, target, knot_values, gamma=1e-4):
    """Eq. (2): 0.5 * (complex-state infidelity + gamma * integral(du/dt)**2)."""
    final_state = _check_wavefunction(config, final_state)
    target = _check_wavefunction(config, target)
    _, _, dx = config.grid()
    if not np.isfinite(gamma) or gamma < 0:
        raise ValueError("gamma must be finite and nonnegative")
    if any(abs(float(mass(s, dx)) - 1) > 1e-7 for s in (final_state, target)):
        raise ValueError("Fidelity requires unit-normalized final and target states")
    fidelity = float(np.abs(dx * np.vdot(target, final_state))**2)
    if fidelity > 1 + 1e-8:
        raise ValueError("Fidelity exceeds one; check state normalization")
    fidelity = min(1.0, max(0.0, fidelity))  # Remove only floating-point overshoot.
    infidelity = 1.0 - fidelity
    derivative_integral = control_derivative_integral(knot_values, config.T)
    regularization = float(gamma * derivative_integral)
    return {
        "fidelity": fidelity,
        "infidelity": infidelity,
        "regularization": regularization,
        "total": 0.5 * (infidelity + regularization),
        "derivative_integral": derivative_integral,
    }


def optimize_control(config, psi0, target, u0=1.0, uT=100.0, gamma=1e-4,
                     n_interior=5, initial_interior=None, maxiter=50,
                     maxfun=1000, max_seconds=None, ftol=1e-10):
    """Locally optimize fractional knots with L-BFGS-B and finite differences.

    Return the best *evaluated* candidate, its conventional-solver final field,
    scalar objective history, and honest optimizer termination metadata. This
    does not certify a global optimum or guarantee lower infidelity separately
    from the total objective. ``max_seconds`` is checked between PDE solves.
    """
    psi0 = _check_wavefunction(config, psi0)
    target = _check_wavefunction(config, target)
    if not isinstance(n_interior, (int, np.integer)) or n_interior < 1:
        raise ValueError("n_interior must be a positive integer")
    if max_seconds is not None and max_seconds <= 0:
        raise ValueError("max_seconds must be positive when supplied")
    initial = (np.linspace(0, 1, n_interior + 2)[1:-1] if initial_interior is None
               else np.asarray(initial_interior, dtype=float))
    if initial.shape != (n_interior,):
        raise ValueError("initial_interior has the wrong number of knots")
    control_knots(initial, u0, uT)  # Validate before spending time on propagation.
    start = perf_counter()
    cache = {}
    evaluations = []
    history = []
    best = None

    class _TimeLimit(Exception):
        pass

    def evaluate(fractions):
        nonlocal best
        fractions = np.asarray(fractions, dtype=float)
        key = fractions.tobytes()
        if key in cache:
            return cache[key]
        if max_seconds is not None and evaluations and perf_counter() - start >= max_seconds:
            raise _TimeLimit
        values = control_knots(fractions, u0, uT)
        solution = propagate(config, psi0, values, save_times=[0.0, config.T])
        final = solution["psi"][-1].copy()
        components = objective_components(config, final, target, values, gamma)
        record = {"evaluation": len(evaluations), "elapsed_seconds": perf_counter() - start,
                  "interior": fractions.tolist(), **components}
        evaluations.append(record)
        candidate = {"interior": fractions.copy(), "knot_values": values,
                     "components": components, "final_state": final}
        cache[key] = candidate
        if best is None or components["total"] < best["components"]["total"]:
            best = candidate
        return candidate

    initial_candidate = evaluate(initial)
    history.append({"iteration": 0, "elapsed_seconds": perf_counter() - start,
                    "interior": initial.tolist(), **initial_candidate["components"]})

    def scalar_objective(fractions):
        return evaluate(fractions)["components"]["total"]

    def callback(fractions):
        candidate = evaluate(fractions)
        history.append({"iteration": len(history), "elapsed_seconds": perf_counter() - start,
                        "interior": np.asarray(fractions).tolist(), **candidate["components"]})

    result = None
    timed_out = False
    try:
        result = minimize(scalar_objective, initial, method="L-BFGS-B",
                          bounds=[(0.0, 1.0)] * n_interior, callback=callback,
                          options={"maxiter": int(maxiter), "maxfun": int(maxfun),
                                   "ftol": ftol, "gtol": 1e-6, "maxls": 20})
    except _TimeLimit:
        timed_out = True
    metadata = {
        "method": "L-BFGS-B",
        "success": bool(result.success) if result is not None else False,
        "message": str(result.message) if result is not None else "Wall-time budget reached between PDE solves",
        "nit": int(result.nit) if result is not None else len(history) - 1,
        "nfev": int(result.nfev) if result is not None else len(evaluations),
        "timed_out": timed_out,
        "evaluated_candidates": len(evaluations),
    }
    return {
        "interior": best["interior"].copy(),
        "knot_values": best["knot_values"].copy(),
        "final_state": best["final_state"].copy(),
        "initial_components": initial_candidate["components"].copy(),
        "final_components": best["components"].copy(),
        "history": history,
        "evaluations": evaluations,
        "elapsed_seconds": perf_counter() - start,
        "optimizer_result": metadata,
        "success": metadata["success"],
        "message": metadata["message"],
        "selected_candidate": "lowest total objective among evaluated candidates",
    }
