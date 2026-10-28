"""Reproducible, deliberately small GPE control / neural-operator experiment.

This is a minimal direct-PDE study of Eqs. (2)-(4), not the paper's reduced
Hamiltonian algorithm. All report numbers are produced by actual execution.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
from pathlib import Path
import statistics
import time

import numpy as np
import scipy
from scipy.optimize import minimize
import torch

from project3_gpe import (
    SimulationConfig, control_knots, objective_components, optimize_control,
    propagate, solve_ground_state,
)
from project3_operator import fit_operator, predict_operator


PROFILES = {
    "quick": dict(n=256, dt=0.002, nt=65, per_knob=12, train=8, val=2,
                  steps=1000, opt_iterations=35, width=64, latent=48),
    "full": dict(n=256, dt=0.001, nt=129, per_knob=40, train=28, val=6,
                 steps=5000, opt_iterations=80, width=96, latent=64),
}


def relative_error(prediction, reference):
    """Full complex field L2 error on the same uniform space-time grid."""
    return float(np.linalg.norm(prediction - reference) / np.linalg.norm(reference))


def serializable(value):
    if isinstance(value, dict):
        return {str(k): serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, data):
    path.write_text(json.dumps(serializable(data), indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def ground_summary(state):
    return {k: serializable(v) for k, v in state.items()
            if k != "psi" and np.isscalar(v)}


def researcher_context_markdown():
    """Team-supplied interview facts, kept separate from measured run data."""
    return Path(__file__).with_name("RESEARCHER_CONTEXT.md").read_text()


def query_accounting(results):
    """Describe saved calls without treating them as the researcher's workload.

    The original learned counter includes one post-search detailed prediction.
    Older saved results therefore need no alteration or rerun to expose that
    distinction. Conventional evaluations are the actual cached PDE solves.
    """
    traditional = results["traditional"]
    metadata = traditional.get("optimizer_result", {})
    surrogate = results["surrogate_control"]
    learned_predictions = int(surrogate["objective_evaluations"])
    return {
        "conventional_unique_pde_evaluations": len(traditional["evaluations"]),
        "conventional_optimizer_iterations": int(metadata["nit"]),
        "learned_search_objective_predictions": learned_predictions - 1,
        "learned_post_search_assessment_predictions": 1,
        "learned_total_recorded_predictions": learned_predictions,
        "learned_numerical_verification_solves": 1,
        "family_reference_pde_solves": sum(len(values) for values in results["splits"].values()),
    }


def validate_reference(config, psi0, times, output):
    """Separate time, spatial, and finite-box errors at the hardest ramp."""
    knots = control_knots(np.array([0.25, 0.5, 0.75]), uT=100.0)
    base = propagate(config, psi0, knots, save_times=times)
    time_cfg = SimulationConfig(n=config.n, length=config.length,
                                T=config.T, dt=config.dt / 2, g=config.g)
    refined_time = propagate(time_cfg, psi0, knots, save_times=times)
    fine_cfg = SimulationConfig(n=2 * config.n, length=config.length,
                                T=config.T, dt=config.dt / 2, g=config.g)
    fine_ground = solve_ground_state(fine_cfg, 1.0)
    fine = propagate(fine_cfg, fine_ground["psi"], knots, save_times=times)
    # Same dx as the spatially refined box: the central slice coincides exactly.
    large_cfg = SimulationConfig(n=3 * config.n, length=1.5 * config.length,
                                 T=config.T, dt=config.dt / 2, g=config.g)
    large_ground = solve_ground_state(large_cfg, 1.0)
    large = propagate(large_cfg, large_ground["psi"], knots, save_times=times)
    start = config.n // 2
    large_central = large["psi"][:, start:start + 2 * config.n]
    measurements = {
        "time_refinement_relative_l2": relative_error(base["psi"], refined_time["psi"]),
        "space_refinement_relative_l2": relative_error(refined_time["psi"], fine["psi"][:, ::2]),
        "combined_refinement_relative_l2": relative_error(base["psi"], fine["psi"][:, ::2]),
        "larger_domain_relative_l2": relative_error(fine["psi"], large_central),
        "max_mass_drift": float(np.max(np.abs(base["mass"] - 1))),
        "max_edge_mass": float(np.max(base["edge_mass"])),
        "base_solve_seconds": float(base["elapsed_seconds"]),
        "reference_error_tolerance": 1e-3,
        "caveat": "Refinement differences estimate accuracy; they are not a rigorous error bound.",
    }
    measurements["passes_reference_check"] = bool(
        measurements["combined_refinement_relative_l2"] < 1e-3
        and measurements["larger_domain_relative_l2"] < 1e-4
        and measurements["max_edge_mass"] < 1e-8
        and fine_ground["converged"] and large_ground["converged"]
    )
    write_json(output / "p1_reference_validation.json", measurements)
    if not measurements["passes_reference_check"]:
        raise RuntimeError("Reference failed accuracy checks; inspect p1_reference_validation.json before training.")
    return measurements


def make_dataset(config, settings, psi0, seed):
    rng = np.random.default_rng(seed)
    times = np.linspace(0, config.T, settings["nt"])
    x, _, dx = config.grid()
    trajectories, controls, records = [], [], []
    train, val, test = [], [], []
    started = time.perf_counter()
    for uT in [5.0, 25.0, 100.0]:
        for local_id in range(settings["per_knob"]):
            fractions = (np.array([0.25, 0.5, 0.75]) if local_id == 0
                         else rng.uniform(0.0, 1.0, 3))
            knots = control_knots(fractions, uT=uT)
            trajectory = propagate(config, psi0, knots, save_times=times)
            index = len(controls)
            split = ("train" if local_id < settings["train"] else
                     "validation" if local_id < settings["train"] + settings["val"] else "test")
            {"train": train, "validation": val, "test": test}[split].append(index)
            controls.append(knots)
            trajectories.append(trajectory["psi"])
            records.append(dict(control_id=index, uT=uT, split=split,
                                solve_seconds=trajectory["elapsed_seconds"],
                                max_mass_drift=float(np.max(np.abs(trajectory["mass"] - 1))),
                                max_edge_mass=float(np.max(trajectory["edge_mass"]))))
    return {
        "x": x, "t": times, "psi": np.array(trajectories),
        "knot_values": np.array(controls), "psi0": psi0,
        "train_indices": np.array(train), "val_indices": np.array(val),
        "test_indices": np.array(test), "dx": dx,
    }, records, time.perf_counter() - started


def evaluate_model(model, dataset, test_indices, mode):
    rows, predictions = [], {}
    x, t, dx = dataset["x"], dataset["t"], float(dataset["dx"])
    # Warmup is excluded from steady-state latency; training cost is separate.
    predict_operator(model, dataset["knot_values"][test_indices[0]], x, t)
    for index in test_indices:
        repeats = []
        for _ in range(3):
            began = time.perf_counter()
            prediction = predict_operator(model, dataset["knot_values"][index], x, t)
            repeats.append(time.perf_counter() - began)
        reference = dataset["psi"][index]
        masses = np.sum(np.abs(prediction)**2, axis=1) * dx
        norm_last = float(masses[-1])
        overlap = abs(dx * np.vdot(reference[-1], prediction[-1]))**2
        normalized_fidelity = float(overlap / max(norm_last, 1e-15))
        rows.append(dict(model=mode, control_id=int(index),
                         uT=float(dataset["knot_values"][index, -1]),
                         relative_l2=relative_error(prediction, reference),
                         density_relative_l2=relative_error(abs(prediction)**2, abs(reference)**2),
                         terminal_normalized_fidelity=min(1.0, normalized_fidelity),
                         max_mass_error=float(np.max(np.abs(masses - 1))),
                         inference_seconds=statistics.median(repeats)))
        predictions[int(index)] = prediction
    return rows, predictions


def surrogate_control(config, model, psi0, target, gamma, maxiter):
    """Optimize a learned prediction, then verify it with the numerical solver.

    Learned fields need not have unit norm. Selection therefore uses explicitly
    normalized overlap plus a mass penalty. Evaluation always reports raw mass
    and the true Eqs. (2)-(4) objective after numerical propagation.
    """
    x, _, dx = config.grid()
    calls = 0
    def predicted_cost(interior, detailed=False):
        nonlocal calls
        calls += 1
        knots = control_knots(interior, uT=100.0)
        state = predict_operator(model, knots, x, np.array([config.T]))[0]
        mass = float(np.sum(abs(state)**2) * dx)
        fidelity = float(abs(dx * np.vdot(target, state))**2 / max(mass, 1e-15))
        h = config.T / (len(knots) - 1)
        regularization = float(gamma * np.sum(np.diff(knots)**2) / h)
        penalty = 0.1 * (mass - 1)**2
        total = 0.5 * (1 - min(fidelity, 1.0) + regularization) + penalty
        details = dict(normalized_fidelity=min(fidelity, 1.0), raw_mass=mass,
                       regularization=regularization, mass_penalty=penalty,
                       selection_objective=total)
        return details if detailed else total
    began = time.perf_counter()
    result = minimize(predicted_cost, np.array([0.25, 0.5, 0.75]), method="L-BFGS-B",
                      bounds=[(0.0, 1.0)] * 3,
                      options=dict(maxiter=maxiter, eps=1e-3, ftol=1e-9))
    selection_seconds = time.perf_counter() - began
    knots = control_knots(result.x, uT=100.0)
    predicted = predicted_cost(result.x, detailed=True)
    verified = propagate(config, psi0, knots, save_times=np.linspace(0, config.T, 65))
    return dict(knot_values=knots, predicted=predicted,
                verified_components=objective_components(config, verified["psi"][-1], target, knots, gamma),
                scipy_success=bool(result.success), scipy_message=str(result.message),
                selection_seconds=selection_seconds, verification_seconds=verified["elapsed_seconds"],
                objective_evaluations=calls,
                caution="Learned optimization is exploratory. Only numerically verified components are physical control results."), verified


def make_plots(output, dataset, loss_logs, errors, predictions, controls, trajectories):
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".mplconfig"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 120, "font.size": 10})
    figures = []
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for ax, (name, logs) in zip(axes, loss_logs.items()):
        for key in ["data", "pde", "ic", "bc"]:
            # The operator logs raw losses under these explicit names.
            candidates = [key, "loss_" + key, "L_" + key]
            match = next((candidate for candidate in candidates if candidate in logs[0]), None)
            if match and any(r[match] is not None for r in logs):
                kept = [(i, r) for i, r in enumerate(logs) if r[match] is not None]
                ax.semilogy([r.get("step", r.get("epoch", i)) for i, r in kept],
                            [max(float(r[match]), 1e-16) for i, r in kept], label=key)
        ax.set(title=name + " (raw loss diagnostics)", xlabel="Training step", ylabel="Loss")
        ax.legend(); ax.grid(alpha=.25)
    path = output / "p2_loss_components.png"; fig.savefig(path); plt.close(fig); figures.append(path.name)

    fig, ax = plt.subplots(figsize=(6.5, 4), constrained_layout=True)
    for model_index, name in enumerate(loss_logs):
        color = f"C{model_index}"
        for uT in [5, 25, 100]:
            values = [r["relative_l2"] for r in errors if r["model"] == name and r["uT"] == uT]
            ax.scatter([uT]*len(values), values, alpha=.45, s=25, color=color)
        averages = [np.mean([r["relative_l2"] for r in errors if r["model"] == name and r["uT"] == uT])
                    for uT in [5, 25, 100]]
        ax.plot([5, 25, 100], averages, "o-", label=name + " mean", color=color)
    ax.set(xlabel="Final trap strength uT (difficulty knob)", ylabel="Relative complex-field L2 error",
           title="P3: Entire held-out control functions")
    ax.legend(); ax.grid(alpha=.25)
    path = output / "p3_error_vs_knob.png"; fig.savefig(path); plt.close(fig); figures.append(path.name)

    hardest = [r for r in errors if r["model"] == "physics" and r["uT"] == 100]
    worst = max(hardest, key=lambda row: row["relative_l2"])
    index = worst["control_id"]
    error = predictions["physics"][index] - dataset["psi"][index]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    # Plot only the occupied center for readability; spectrum uses the full box.
    center = np.abs(dataset["x"]) <= 3.0
    field = axes[0].pcolormesh(dataset["x"][center], dataset["t"], abs(error[:, center]), shading="auto")
    axes[0].set(xlabel="x", ylabel="t", title=f"P4: |prediction-reference|, control {index}")
    fig.colorbar(field, ax=axes[0], label="Complex error magnitude")
    frequencies = np.fft.fftshift(np.fft.fftfreq(len(dataset["x"]), float(dataset["dx"])))
    power = np.mean(abs(np.fft.fftshift(np.fft.fft(error, axis=1), axes=1) / len(dataset["x"]))**2, axis=0)
    axes[1].semilogy(frequencies, np.maximum(power, 1e-20))
    axes[1].set(xlabel="Spatial frequency (cycles / unit length)", ylabel="Mean error power over time",
                title="P4: Fourier spectrum of the full-box error")
    axes[1].grid(alpha=.25)
    path = output / "p4_error_field_and_spectrum.png"; fig.savefig(path); plt.close(fig); figures.append(path.name)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    t = np.linspace(0, 2.5, 300)
    for name, knots in controls.items():
        axes[0].plot(t, np.interp(t, np.linspace(0, 2.5, len(knots)), knots), label=name)
        trajectory = trajectories[name]
        axes[1].plot(trajectory["x"], abs(trajectory["psi"][-1])**2, label=name)
    axes[0].set(xlabel="t", ylabel="u(t)", title="Continuous controls with fixed endpoints")
    axes[1].plot(dataset["x"], abs(trajectories["target"])**2, "k--", label="Nonlinear target")
    axes[1].set(xlim=(-2, 2), xlabel="x", ylabel="Terminal density", title="Numerically verified terminal states")
    for ax in axes: ax.legend(fontsize=8); ax.grid(alpha=.25)
    path = output / "control_comparison.png"; fig.savefig(path); plt.close(fig); figures.append(path.name)
    return figures


def write_report(output, results):
    p1 = results["p1"]
    traditional = results["traditional"]
    surrogate = results["surrogate_control"]
    rows = results["test_errors"]
    queries = query_accounting(results)
    table = []
    for name in ["physics", "supervised"]:
        for uT in [5, 25, 100]:
            group = [r for r in rows if r["model"] == name and r["uT"] == uT]
            table.append(f"| {name} | {uT} | {np.mean([r['relative_l2'] for r in group]):.4f} | "
                         f"{np.mean([r['max_mass_error'] for r in group]):.4f} |")
    cost_rows = []
    for name, cost in results["cost"].items():
        breakeven = f"{cost['break_even_queries']:.0f}" if cost["break_even_queries"] is not None else "No positive break-even"
        cost_rows.append(f"| {name} | {cost['training_seconds']:.2f} | {cost['inference_seconds']:.6f} | {breakeven} |")
    report = f"""# MAE 598 Project 3: measured pilot and report draft

Run profile: **{results['profile']}**. Seed: {results['seed']}. This file is generated from
an actual run; it does not certify completion of the outreach or path requirements.

{researcher_context_markdown()}

## Minimal-case formulation and problem brief

Dr. Jimmie Adriazola suggested the optimal-control formulation in Eqs. (2)-(4) of
Adriazola and Goodman, *A Reduction-Based Strategy for Optimal Control of Bose-Einstein
Condensates* (arXiv:2111.14266). Repeated control queries motivate testing a learned
forward operator. The October 16 interview supports that research question; it
does not provide a runtime benchmark, typical query count, or quantitative
accuracy threshold. This written problem brief has **not yet been reviewed or
approved by Dr. Adriazola**. The measurements below belong to our local minimal
case, rather than to his current research solver.

We solve i psi_t = -0.5 psi_xx + (0.5 u(t) x^2 + |psi|^2) psi, with g=1,
T=2.5, on a periodic box [-8,8). Ground states decay near the box edges. The initial
state is the full nonlinear ground state at u=1; the target is the full nonlinear
ground state at u=uT. This deliberately differs from the paper's two-mode initial
state and does not reproduce its Galerkin/CRAB/DE/GRAPE algorithm. The dimensional
reduction from the paper's general 3D formulation to 1D keeps nonlinear controlled
wave dynamics and the sharpening features caused by stronger confinement.

Five equally spaced control knots are connected continuously by straight lines.
Endpoint values are fixed; the three interior fractions are optimized. The extra
constraint 1<=u(t)<=uT is a study restriction, not a consequence of endpoint data.
The objective is 0.5*(1-fidelity + gamma*integral(udot^2)), with gamma=1e-4 a study
choice. The control derivative penalty is integrated exactly for this parameterization.
The difficulty knob is uT in {{5,25,100}}. Larger uT makes narrower spatial features;
whether it is harder for this trained model is established by measured error.

## P1: conventional baseline

The preserved Strang split-step Fourier engine now uses midpoint controls, full complex
states, and an independently computed target. Nonlinear ground-state residuals and
convergence status are stored in results.json.

- Time-refinement relative L2 difference: {p1['time_refinement_relative_l2']:.3e}.
- Spatial-refinement difference: {p1['space_refinement_relative_l2']:.3e}.
- Combined refinement difference: {p1['combined_refinement_relative_l2']:.3e}.
- Larger-domain difference: {p1['larger_domain_relative_l2']:.3e}.
- Maximum reference norm drift: {p1['max_mass_drift']:.3e}.
- Maximum edge mass: {p1['max_edge_mass']:.3e}.
- Hardest-ramp solve time: {p1['base_solve_seconds']:.4f} s.

Refinement differences estimate error; no exact nonlinear solution or rigorous error
bound is claimed. Validation covers the hardest ramp, not every possible control.

For uT=100, conventional SciPy optimization changed fidelity from
{traditional['initial_components']['fidelity']:.6f} to {traditional['final_components']['fidelity']:.6f},
and total objective from {traditional['initial_components']['total']:.6f} to
{traditional['final_components']['total']:.6f}. The solver's success flag is
{traditional['scipy_success']}; message: {traditional['scipy_message']}.
This is a finite-dimensional local search, not a global-optimality certificate.

## P2-P4: operator, loss diagnostics, and held-out errors

The model is a **physics-informed DeepONet**, not the published FNO-based PINO
architecture. Its branch receives the whole control's knots; its trunk receives x,t.
It predicts real and imaginary wave components. All controls use the same initial
state. Entire controls, rather than points from known trajectories, are held out.
Counts: {len(results['splits']['train'])} training, {len(results['splits']['validation'])}
validation, {len(results['splits']['test'])} test controls.

PDE, initial, boundary, and data losses are logged separately. The supervised comparison
uses the same architecture, seed, data split, initial-state labels, and optimizer-step
budget, but removes PDE/BC terms from its optimized loss. The physics training objective
is data + IC + PDE/100^2 + 0.1*BC; the optional norm weight is zero. The raw PDE loss
remains visible. Half the spatial
training samples emphasize |x|<=3, while evaluation uses the full box. These are
explicit numerical training choices. This is a controlled loss-composition
comparison; it does not on its own establish the unique cause of a failure.

Reported errors use the **raw complex fields**, with no normalization or per-time phase
alignment to conceal errors. Final normalized fidelity is a supplementary metric;
raw norm errors are also reported. Values below are means across held-out controls.

| Model | uT | Relative complex L2 error | Maximum norm error |
|---|---:|---:|---:|
{os.linesep.join(table)}

![P2 loss terms](p2_loss_components.png)
![P3 error versus knob](p3_error_vs_knob.png)
![P4 pointwise error and error spectrum](p4_error_field_and_spectrum.png)

The P4 spectrum is the spatial FFT power of the **complex prediction error**, averaged
over time. The pictured error field is the worst held-out uT=100 physics-model case;
only its occupied central region is shown, while its spectrum uses the whole box.

## Learned control checked against the numerical solver

The learned model proposes a control using normalized predicted overlap plus an explicit
raw-norm penalty; its selection cost is therefore a surrogate diagnostic, not an exact
evaluation of the paper's cost. Its raw predicted terminal norm was
{surrogate['predicted']['raw_mass']:.6f}, and normalized predicted fidelity was
{surrogate['predicted']['normalized_fidelity']:.6f}. Rerunning that control through the
numerical GPE solver produced fidelity {surrogate['verified_components']['fidelity']:.6f}
and true total objective {surrogate['verified_components']['total']:.6f}.
Only the rerun values support a physical control claim.

![Control comparison](control_comparison.png)

## P5: cost

Hardware: {results['hardware']}. Versions: {results['versions']}.
Steady-state inference timings use three repeats after warmup, on the same full
space-time output grid as each conventional query. Median conventional query cost:
{results['baseline_median_seconds']:.6f} s. Upfront reference-data generation cost:
{results['training_data_seconds']:.3f} s, including the shared initial ground state
and, conservatively, all training, validation, and test control trajectories.
Break-even includes that cost plus the corresponding model's training time.

| Model | Training seconds | Inference seconds/query | Break-even queries |
|---|---:|---:|---:|
{os.linesep.join(cost_rows)}

These are hardware-specific forward-query comparisons. They do not establish an
end-to-end control-optimization speedup or an accuracy-matched speedup.

### Recorded workload in our optimization experiment

| Work | Recorded count | Interpretation |
|---|---:|---|
| Conventional optimization | {queries['conventional_unique_pde_evaluations']} distinct PDE evaluations | {queries['conventional_optimizer_iterations']} optimizer iterations; includes objective/finite-difference evaluations recorded by the conventional optimizer. |
| Learned search | {queries['learned_search_objective_predictions']} neural objective predictions | Excludes the separately recorded final assessment below. The learned search did not produce an accuracy-qualified control advantage. |
| Learned final assessment | {queries['learned_post_search_assessment_predictions']} neural prediction | The saved total counter is {queries['learned_total_recorded_predictions']}, including this post-search diagnostic call. |
| Learned candidate verification | {queries['learned_numerical_verification_solves']} PDE solve | Checks the proposed control with the conventional GPE solver. |
| Reference-family generation | {queries['family_reference_pde_solves']} PDE solves | Training, validation, and test references; an upfront cost rather than optimization-loop queries. |

These counts exclude additional ground-state, refinement, plotting, and post-control
validation work. The recorded conventional search took {traditional['elapsed_seconds']:.3f} s;
learned selection took {surrogate['selection_seconds']:.3f} s and its subsequent PDE
verification took {surrogate['verification_seconds']:.3f} s. Those different, unequally
accurate searches do not demonstrate an end-to-end speedup. In particular, the learned
search termination was `{surrogate['scipy_message']}`. Dr. Adriazola's typical query
count remains **unknown**; these are observations from this project only.

## Path and limitations

This is a **preliminary diagnostic experiment**. Do not label it successful Path A
without a researcher-useful accuracy threshold and an honest value argument. Path B
requires a failure trend and evidence distinguishing candidate causes; a short run or
inconclusive supervised comparison alone is insufficient. No Path C result is claimed.

Limitations include small data, fixed initial state/time/domain, three trap strengths,
one seed, one architecture/budget, extra control bounds, and limited reference checks.
The few held-out controls per knob do not establish a general operator guarantee.
Terminal squeezing alone does not establish suppressed post-control oscillations.

## Remaining team information and review

- The October 16 interview is recorded above; contact method, reservation history,
  any other outreach, and permission to publish interview material remain unavailable.
- Exact research-solver runtime/hardware, typical query count, and numerical accuracy
  threshold were not provided in that interview and remain explicitly unknown.
- This written brief has not yet been reviewed or approved by Dr. Adriazola.
- A defensible path conclusion supported by the actual plots and controlled comparison.
- Final one-page memo, whether it was sent, and any actual response.
- Public repository link and Canvas submission, performed by the team.

## Draft recommendation to Dr. Adriazola (not sent)

We retained and validated the spectral GPE solver, corrected the fidelity objective and
admissible controls, and tested a small physics-informed control-to-wave operator.
The tables above quantify its actual predictive accuracy and cost. We do not yet have
evidence sufficient to recommend replacing the numerical solver in an optimization
loop. Every learned candidate should be rerun numerically. The next experiment should
be chosen after comparing the physics and supervised curves: extend a fixed-budget
learning curve and repeat across seeds before attributing error to a specific mechanism.
Please correct our simplified benchmark and advise what accuracy/query count would
make an operator useful to your work. You requested results during the October 16
interview; this draft has not been sent and does not imply written-brief approval.

## Reproduce

Run `python -m pip install -r requirements.txt`, then
`python run_project3.py --profile {results['profile']} --seed {results['seed']}`.
Run `python -m unittest discover -s tests -v` for independent physical checks.
The original notebook is preserved separately; all figures/numbers here are generated
from the corrected pipeline. No interview, correspondence, or unmeasured performance
result has been invented.
"""
    (output / "EXPERIMENT_REPORT.md").write_text(report)


def run_experiment(profile="quick", seed=598, steps=None, output_dir="output/project3", device="cpu"):
    if profile not in PROFILES:
        raise ValueError("profile must be quick or full")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    settings = dict(PROFILES[profile])
    if steps is not None:
        settings["steps"] = int(steps)
    np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(min(2, os.cpu_count() or 1))
    config = SimulationConfig(n=settings["n"], dt=settings["dt"])
    began = time.perf_counter()
    print("Computing nonlinear stationary states...", flush=True)
    initial_started = time.perf_counter()
    initial = solve_ground_state(config, 1.0)
    initial_seconds = time.perf_counter() - initial_started
    targets = {uT: solve_ground_state(config, uT) for uT in [5.0, 25.0, 100.0]}
    if not initial["converged"] or not all(state["converged"] for state in targets.values()):
        raise RuntimeError("A stationary state failed residual convergence; do not train on it.")
    times = np.linspace(0, config.T, settings["nt"])
    print("Checking time/grid/domain refinement...", flush=True)
    p1 = validate_reference(config, initial["psi"], times, output)
    print("Optimizing the corrected conventional objective...", flush=True)
    traditional = optimize_control(config, initial["psi"], targets[100.0]["psi"],
                                   gamma=1e-4, n_interior=3, maxiter=settings["opt_iterations"])
    traditional["scipy_success"] = traditional["optimizer_result"]["success"]
    traditional["scipy_message"] = traditional["optimizer_result"]["message"]
    print("Generating separate training, validation, and test controls...", flush=True)
    dataset, records, dataset_seconds = make_dataset(config, settings, initial["psi"], seed)
    np.savez_compressed(output / "reference_dataset.npz", **dataset)
    write_csv(output / "reference_controls.csv", [dict(r, knots=dataset["knot_values"][r["control_id"]].tolist()) for r in records])
    all_errors, predictions, models, histories, training = [], {}, {}, {}, {}
    for mode in ["physics", "supervised"]:
        print(f"Training {mode} operator ({settings['steps']} steps)...", flush=True)
        model, logs, summary = fit_operator(
            dataset, dataset["train_indices"], dataset["val_indices"],
            config=dict(length=config.length, T=config.T, seed=seed, width=settings["width"],
                        latent=settings["latent"], log_every=max(1, settings["steps"] // 40)),
            mode=mode, steps=settings["steps"], device=device)
        models[mode], histories[mode], training[mode] = model, logs, summary
        write_csv(output / f"{mode}_loss_history.csv", logs)
        torch.save(dict(state_dict=model.state_dict(), summary=summary,
                        config=getattr(model, "config", {})), output / f"{mode}_operator.pt")
        errors, model_predictions = evaluate_model(model, dataset, dataset["test_indices"], mode)
        predictions[mode] = model_predictions
        all_errors.extend(errors)
    write_csv(output / "p3_p5_heldout_errors.csv", all_errors)
    print("Checking a learned control with the conventional solver...", flush=True)
    surrogate, surrogate_trajectory = surrogate_control(config, models["physics"], initial["psi"],
                                                       targets[100.0]["psi"], 1e-4, 25)
    ramp_knots = control_knots(np.array([.25, .5, .75]), uT=100.0)
    traditional_knots = np.asarray(traditional["knot_values"])
    controls = {"Linear ramp": ramp_knots, "Conventional optimized": traditional_knots,
                "Learned candidate (verified)": np.asarray(surrogate["knot_values"])}
    trajectories = {name: propagate(config, initial["psi"], knots, save_times=times)
                    for name, knots in controls.items()}
    trajectories["target"] = targets[100.0]["psi"]
    figures = make_plots(output, dataset, histories, all_errors, predictions, controls, trajectories)
    baseline_median = statistics.median(r["solve_seconds"] for r in records)
    # Conservative upfront cost includes *all* generated family references,
    # not just the subset used as training labels.
    training_data_seconds = initial_seconds + dataset_seconds
    costs = {}
    for mode in models:
        summary = training[mode]
        elapsed = float(summary.get("training_seconds", summary.get("elapsed_seconds", 0)))
        if elapsed <= 0:
            raise RuntimeError("Operator must report actual training_seconds")
        inference = statistics.median(r["inference_seconds"] for r in all_errors if r["model"] == mode)
        costs[mode] = dict(training_seconds=elapsed, inference_seconds=inference,
                           training_data_seconds=training_data_seconds,
                           break_even_queries=((elapsed + training_data_seconds) / (baseline_median - inference)
                                               if baseline_median > inference else None))
    # Keep compact scalar optimization metadata; trajectories are saved separately.
    traditional_clean = {key: value for key, value in traditional.items()
                         if key not in {"psi", "final_state", "final_wave", "trajectory", "scipy_result"}}
    results = dict(profile=profile, settings=settings, seed=seed,
                   hardware=f"{platform.system()} {platform.machine()}, {os.cpu_count()} logical CPUs; torch device={device}, threads={torch.get_num_threads()}",
                   versions=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__, torch=torch.__version__),
                   ground_states={"initial_u1": ground_summary(initial),
                                  **{f"target_u{uT:g}": ground_summary(state) for uT, state in targets.items()}},
                   p1=p1, traditional=traditional_clean, surrogate_control=surrogate,
                   splits=dict(train=dataset["train_indices"], validation=dataset["val_indices"], test=dataset["test_indices"]),
                   dataset_generation_seconds=dataset_seconds,
                   training_data_seconds=training_data_seconds,
                   baseline_median_seconds=baseline_median,
                   training=training, cost=costs, test_errors=all_errors, figures=figures,
                   total_seconds=time.perf_counter() - began)
    write_json(output / "results.json", results)
    write_report(output, serializable(results))
    print(f"Completed in {results['total_seconds']:.1f}s. Report: {output / 'EXPERIMENT_REPORT.md'}", flush=True)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=list(PROFILES), default="quick")
    parser.add_argument("--seed", type=int, default=598)
    parser.add_argument("--steps", type=int, default=None, help="Override training budget for both matched models")
    parser.add_argument("--output-dir", default="output/project3")
    parser.add_argument("--device", default="cpu", help="CPU is the reproducible default; cuda requires available CUDA")
    args = parser.parse_args()
    run_experiment(args.profile, args.seed, args.steps, args.output_dir, args.device)
