"""Long-budget fit of one known training trajectory, without test controls.

The predeclared experiment uses all labels of the quick dataset's uT=100
training ramp, the unchanged 64/48 DeepONet, and exactly 50,000 supervised Adam
steps. Dense error is measured every 5,000 steps without checkpoint selection
or early stopping. This is a fitting/representability diagnostic; it is not a
comparison at the original operator-training budget or a generalization test.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from project3_operator import ComplexDeepONet, predict_operator


def run_capacity_diagnostic():
    source = Path("output/quick_pilot/reference_dataset.npz")
    output = Path("output/diagnostic")
    output.mkdir(parents=True, exist_ok=True)
    with np.load(source) as stored:
        data = {key: stored[key] for key in stored.files}
    control_id = 24
    if control_id not in data["train_indices"]:
        raise ValueError("Predeclared control 24 must belong to the training split")
    x, t = data["x"], data["t"]
    knots = data["knot_values"][control_id]
    truth = data["psi"][control_id]
    if truth.shape != (65, 256) or not np.allclose(knots, np.linspace(1, 100, 5)):
        raise ValueError("Use the preserved quick dataset's 65x256 uT=100 training ramp")
    seed, steps, measure_every = 598, 50000, 5000
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    config = dict(length=16.0, T=2.5, seed=seed, width=64, latent=48, depth=2,
                  frequencies=[1, 2, 4, 8], u_min=1.0, u_max=100.0, g=1.0)
    model = ComplexDeepONet(np.linspace(0, 2.5, len(knots)), config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    knot_tensor = torch.as_tensor(knots, dtype=torch.float32)
    central_indices = np.flatnonzero(np.abs(x) <= 3.0)

    def sample_grid(n):
        indices = np.concatenate((rng.choice(central_indices, n // 2),
                                  rng.integers(0, len(x), n - n // 2)))
        return rng.permutation(indices)

    def tensor(array):
        return torch.as_tensor(array, dtype=torch.float32)

    history = []
    began = time.perf_counter()

    def measure(step, data_loss=None, ic_loss=None):
        prediction = predict_operator(model, knots, x, t)
        relative = float(np.linalg.norm(prediction - truth) / np.linalg.norm(truth))
        density_relative = float(np.linalg.norm(abs(prediction)**2 - abs(truth)**2)
                                 / np.linalg.norm(abs(truth)**2))
        mass = np.sum(abs(prediction)**2, axis=1) * float(x[1] - x[0])
        row = dict(step=int(step), relative_complex_l2=relative,
                   relative_density_l2=density_relative,
                   max_mass_error=float(np.max(np.abs(mass - 1.0))),
                   last_batch_data_loss=data_loss, last_batch_ic_loss=ic_loss,
                   elapsed_seconds=time.perf_counter() - began)
        history.append(row)
        print(f"Capacity diagnostic step {step}: raw dense complex L2={relative:.6f}, "
              f"max mass error={row['max_mass_error']:.6f}", flush=True)

    measure(0)
    for step in range(1, steps + 1):
        # Match the pilot's random-family draw ordering. The unused draws below
        # keep sampling comparable; they neither enter the loss nor use labels.
        rng.integers(0, 1, 256)
        time_indices = rng.integers(0, len(t), 256)
        space_indices = sample_grid(256)
        rng.integers(0, 1, 256)
        px = np.concatenate((rng.uniform(-3, 3, 128), rng.uniform(-8, 8, 128)))
        rng.permutation(px)
        rng.uniform(0, 2.5, 256)
        rng.integers(0, 1, 128)
        ic_indices = sample_grid(128)
        rng.integers(0, 1, 32)
        rng.uniform(0, 2.5, 32)
        rng.integers(0, 1, 2)
        rng.uniform(0, 2.5, 2)
        optimizer.zero_grad(set_to_none=True)
        coordinates = tensor(np.column_stack((x[space_indices], t[time_indices])))
        prediction = model(knot_tensor, coordinates)
        observed = truth[time_indices, space_indices]
        labels = tensor(np.column_stack((observed.real, observed.imag)))
        data_loss = (prediction - labels).square().sum(dim=1).mean()
        initial_coordinates = tensor(np.column_stack((x[ic_indices], np.zeros(128))))
        initial_prediction = model(knot_tensor, initial_coordinates)
        initial_truth = truth[0, ic_indices]
        initial_labels = tensor(np.column_stack((initial_truth.real, initial_truth.imag)))
        ic_loss = (initial_prediction - initial_labels).square().sum(dim=1).mean()
        objective = data_loss + ic_loss
        if not bool(torch.isfinite(objective)):
            raise FloatingPointError(f"Nonfinite objective at step {step}")
        objective.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        if step % measure_every == 0:
            measure(step, float(data_loss.detach()), float(ic_loss.detach()))

    training_seconds = time.perf_counter() - began
    summary = dict(
        source=str(source), source_control_id=control_id, source_split="training",
        knot_values=knots.tolist(), configuration=config, seed=seed, steps=steps,
        optimizer="Adam", learning_rate=1e-3, gradient_clip=10.0, num_threads=1,
        architecture="unchanged complex DeepONet, width64/depth2/latent48",
        parameter_count=sum(parameter.numel() for parameter in model.parameters()),
        all_label_points=int(truth.size), initial_state_labels=int(x.size),
        loss="raw complex data MSE + equally weighted initial-state MSE",
        spatial_sampling="50% central grid |x|<=3 and50% full-grid; initial-state batches use same mixture",
        physics_penalties=False, prediction_renormalized=False, phase_aligned=False,
        measurement_interval=measure_every, early_stopping=False, checkpoint_selection=False,
        training_seconds=training_seconds, history=history,
        final_relative_complex_l2=history[-1]["relative_complex_l2"],
        final_below_predeclared_0_05=bool(history[-1]["relative_complex_l2"] < 0.05),
        interpretation_limits=[
            "Fitting one known training trajectory does not measure generalization to unseen controls.",
            "This longer budget is separate from the matched 5,000-step supervised/physics ablation.",
            "A good fit demonstrates that this architecture can approximate this trajectory under this training configuration.",
            "Failure to fit cannot distinguish limited representation from optimizer or loss-design limitations.",
            "No test control labels or errors entered fitting, model selection, or stopping.",
        ],
    )
    (output / "capacity_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".mplconfig"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].semilogy([row["step"] for row in history],
                     [row["relative_complex_l2"] for row in history], "o-")
    axes[0].axhline(.05, color="gray", linestyle="--", label="Predeclared 0.05 reference")
    axes[0].set(xlabel="Supervised Adam steps", ylabel="Dense raw complex L2 error",
                title="One known training trajectory")
    axes[0].legend(fontsize=8)
    axes[1].semilogy([row["step"] for row in history],
                     [row["max_mass_error"] for row in history], "o-")
    axes[1].set(xlabel="Supervised Adam steps", ylabel="Maximum raw norm error",
                title="No normalization of predictions")
    for axis in axes:
        axis.grid(alpha=.25)
    fig.savefig(output / "capacity_learning_curve.png", dpi=140)
    plt.close(fig)
    return summary


if __name__ == "__main__":
    run_capacity_diagnostic()
