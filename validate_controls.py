"""Independently refine saved control candidates and measure post-control breathing.

This script reads a completed experiment's results.json and writes only a new
control_validation.json and post_control_breathing.png beside it.  It never
updates the existing results or report, and never retrains the neural model.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np

from project3_gpe import (
    SimulationConfig, control_knots, mass, objective_components, propagate,
    solve_ground_state,
)


def _ground_summary(state):
    return {name: value for name, value in state.items() if name != "psi"}


def _relative_l2(prediction, reference):
    return float(np.linalg.norm(prediction - reference) / np.linalg.norm(reference))


def _widths(trajectory, dx):
    """Centered standard deviation, with expectation values divided by measured mass."""
    rho = np.abs(trajectory["psi"])**2
    measured_mass = np.sum(rho, axis=1) * dx
    center = (rho @ trajectory["x"]) * dx / measured_mass
    second = (rho @ (trajectory["x"]**2)) * dx / measured_mass
    variance = np.maximum(0.0, second - center**2)
    return np.sqrt(variance), center, measured_mass


def validate_controls(results_path="output/project3/results.json"):
    """Validate controls in one completed experiment and return scalar diagnostics.

    ``results_path`` may name results.json or its containing directory. A
    doubled spatial grid and halved maximum time step estimate discretization
    sensitivity for each individual saved control. No global-phase alignment
    is applied to the complex-state errors.

    All terminal states are then held in the final trap for one additional
    time unit on the refined grid. The finite-window width variation is a
    diagnostic of residual breathing, not a proof that all excitations vanish.
    """
    started = perf_counter()
    path = Path(results_path).expanduser().resolve()
    if path.is_dir():
        path = path / "results.json"
    results = json.loads(path.read_text())
    output = path.parent
    settings = results["settings"]
    # Current runner fixes length=16, T=2.5, g=1. Future saved configs override
    # these explicit defaults rather than silently inferring physical units.
    saved_config = results.get("simulation_config", results.get("config", {}))
    config = SimulationConfig(
        n=int(settings["n"]), dt=float(settings["dt"]),
        length=float(saved_config.get("length", 16.0)),
        T=float(saved_config.get("T", 2.5)),
        g=float(saved_config.get("g", 1.0)),
    )
    fine_config = replace(config, n=2 * config.n, dt=config.dt / 2)
    times = np.linspace(0, config.T, int(settings.get("nt", 65)))
    conventional_knots = np.asarray(results["traditional"]["knot_values"], dtype=float)
    surrogate_knots = np.asarray(results["surrogate_control"]["knot_values"], dtype=float)
    u0, uT = float(conventional_knots[0]), float(conventional_knots[-1])
    if not np.allclose(surrogate_knots[[0, -1]], [u0, uT], atol=1e-12, rtol=0):
        raise ValueError("Conventional and learned controls must have the same endpoint traps")
    # Recover gamma from the actually saved conventional objective, if available.
    old_components = results["traditional"].get("final_components", {})
    old_integral = float(old_components.get("derivative_integral", 0.0))
    gamma = (float(old_components["regularization"]) / old_integral
             if old_integral > 0 else float(results.get("gamma", 1e-4)))
    controls = {
        "ramp": control_knots(np.array([0.25, 0.5, 0.75]), u0=u0, uT=uT),
        "traditional": conventional_knots,
        "surrogate": surrogate_knots,
    }
    ground_states = {}
    for name, cfg in (("base", config), ("refined", fine_config)):
        ground_states[name] = {
            "initial": solve_ground_state(cfg, u0),
            "target": solve_ground_state(cfg, uT),
        }
        if not all(state["converged"] for state in ground_states[name].values()):
            raise RuntimeError(f"The {name} ground states failed their stationary residual tolerance")

    control_results = {}
    refined_terminal = {}
    for name, knots in controls.items():
        base = propagate(config, ground_states["base"]["initial"]["psi"], knots,
                         save_times=times)
        fine = propagate(fine_config, ground_states["refined"]["initial"]["psi"], knots,
                         save_times=times)
        base_components = objective_components(
            config, base["psi"][-1], ground_states["base"]["target"]["psi"], knots, gamma)
        fine_components = objective_components(
            fine_config, fine["psi"][-1], ground_states["refined"]["target"]["psi"], knots, gamma)
        field_error = _relative_l2(base["psi"], fine["psi"][:, ::2])
        terminal_error = _relative_l2(base["psi"][-1], fine["psi"][-1, ::2])
        fidelity_difference = abs(base_components["fidelity"] - fine_components["fidelity"])
        total_difference = abs(base_components["total"] - fine_components["total"])
        control_results[name] = {
            "knot_values": knots.tolist(),
            "base_components": base_components,
            "refined_components": fine_components,
            "space_time_complex_relative_l2": field_error,
            "terminal_complex_relative_l2": terminal_error,
            "absolute_fidelity_difference": fidelity_difference,
            "absolute_total_objective_difference": total_difference,
            "base_max_mass_drift": float(np.max(np.abs(base["mass"] - 1))),
            "refined_max_mass_drift": float(np.max(np.abs(fine["mass"] - 1))),
            "base_max_edge_mass": float(np.max(base["edge_mass"])),
            "refined_max_edge_mass": float(np.max(fine["edge_mass"])),
            "base_propagation_seconds": float(base["elapsed_seconds"]),
            "refined_propagation_seconds": float(fine["elapsed_seconds"]),
            "passes_refinement_check": bool(
                field_error < 1e-3 and terminal_error < 1e-3
                and fidelity_difference < 1e-3 and total_difference < 5e-4
                and np.max(fine["edge_mass"]) < 1e-8
            ),
        }
        refined_terminal[name] = fine["psi"][-1].copy()

    post_config = replace(fine_config, T=1.0)
    post_times = np.linspace(0.0, 1.0, 201)
    _, _, fine_dx = post_config.grid()
    target = ground_states["refined"]["target"]["psi"]
    target_width = float(_widths({"psi": target[None, :], "x": post_config.grid()[0]}, fine_dx)[0][0])
    post_results = {}
    for name, state in {**refined_terminal, "stationary_target": target}.items():
        trajectory = propagate(post_config, state, [uT, uT], save_times=post_times)
        width, center, measured_mass = _widths(trajectory, fine_dx)
        fidelities = np.abs(fine_dx * (trajectory["psi"] @ np.conjugate(target)))**2
        peak_to_peak = float(np.ptp(width))
        post_results[name] = {
            "width_initial": float(width[0]),
            "width_mean": float(np.mean(width)),
            "width_min": float(np.min(width)),
            "width_max": float(np.max(width)),
            "width_peak_to_peak": peak_to_peak,
            "width_peak_to_peak_relative_to_target_width": peak_to_peak / target_width,
            "max_center_displacement": float(np.max(np.abs(center - center[0]))),
            "max_mass_drift_from_one": float(np.max(np.abs(measured_mass - 1))),
            "max_mass_drift_from_initial": float(np.max(np.abs(measured_mass - measured_mass[0]))),
            "max_edge_mass": float(np.max(trajectory["edge_mass"])),
            "target_fidelity_initial": float(fidelities[0]),
            "target_fidelity_min": float(np.min(fidelities)),
            "target_fidelity_max": float(np.max(fidelities)),
            "propagation_seconds": float(trajectory["elapsed_seconds"]),
            "width_trace": width.tolist(),
        }

    diagnostic = {
        "source_results": str(path),
        "base_config": asdict(config),
        "refined_config": asdict(fine_config),
        "gamma": gamma,
        "ground_states": {name: {kind: _ground_summary(state) for kind, state in group.items()}
                          for name, group in ground_states.items()},
        "controls": control_results,
        "all_controls_pass_refinement_check": all(
            item["passes_refinement_check"] for item in control_results.values()),
        "thresholds": {"complex_relative_l2": 1e-3, "absolute_fidelity_difference": 1e-3,
                       "absolute_total_objective_difference": 5e-4, "edge_mass": 1e-8},
        "post_control": {
            "hold_duration": 1.0,
            "held_trap_coefficient": uT,
            "stationary_target_width": target_width,
            "time_since_control_end": post_times.tolist(),
            "states": post_results,
        },
        "notes": [
            "Refinement differences estimate discretization sensitivity, not a rigorous error bound.",
            "Complex errors use matching grids with no global-phase alignment or field normalization.",
            "The one-unit post-control hold uses the refined grid and time step for every state.",
            "Centered widths divide their expectation values by measured mass; fields are never renormalized during propagation.",
            "Finite-window width oscillation diagnoses breathing; it does not characterize all excitations or prove their elimination.",
            "The stationary target continuation measures numerical breathing introduced by the split-step propagation itself.",
        ],
        "elapsed_seconds": perf_counter() - started,
    }

    os.environ.setdefault("MPLCONFIGDIR", str(output / ".mplconfig"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True, constrained_layout=True)
    labels = {"ramp": "Linear ramp", "traditional": "Conventional optimized",
              "surrogate": "Learned candidate (numerically propagated)",
              "stationary_target": "Stationary target (numerical sanity check)"}
    for name, item in post_results.items():
        width = np.asarray(item["width_trace"])
        style = "--" if name == "stationary_target" else "-"
        line = axes[0].plot(post_times, width, style, label=labels[name])[0]
        axes[1].plot(post_times, 100 * (width - width[0]) / target_width,
                     style, color=line.get_color(), label=labels[name])
    axes[0].set(ylabel="Centered standard deviation of x",
                title=f"Post-control breathing: trap held at u={uT:g} for one time unit")
    axes[1].set(xlabel="Time since control ended", ylabel="Width change / target width (%)")
    axes[0].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.savefig(output / "post_control_breathing.png", dpi=150)
    plt.close(fig)
    diagnostic["elapsed_seconds"] = perf_counter() - started
    (output / "control_validation.json").write_text(
        json.dumps(diagnostic, indent=2, allow_nan=False) + "\n")
    return diagnostic


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_path", nargs="?", default="output/project3/results.json",
                        help="Path to results.json or its containing output directory")
    arguments = parser.parse_args()
    diagnostics = validate_controls(arguments.results_path)
    print("All saved controls pass refinement:", diagnostics["all_controls_pass_refinement_check"])
    for name, result in diagnostics["post_control"]["states"].items():
        print(f"{name}: width peak-to-peak={result['width_peak_to_peak']:.6g}; "
              f"mass drift={result['max_mass_drift_from_one']:.3g}")
