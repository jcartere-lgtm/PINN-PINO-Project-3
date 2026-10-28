"""One predeclared, controlled single-training-trajectory fit diagnostic.

This does not measure neural-operator generalization: the uT=100 linear ramp is
already a training control. Both modes see identical alternating
space/time labels from that trajectory. The remaining grid points are held out
only to check interpolation of the same known trajectory. No original test
control is used for fitting, model selection, or this comparison.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import time

import numpy as np

from project3_operator import fit_operator, predict_operator


def _relative_l2(prediction, reference):
    return float(np.linalg.norm(prediction - reference) / np.linalg.norm(reference))


def run_diagnostic(source="output/quick_pilot/reference_dataset.npz",
                   output="output/diagnostic", steps=5000):
    began = time.perf_counter()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with np.load(source) as stored:
        data = {key: stored[key] for key in stored.files}
    ramp = np.linspace(1, 100, data["knot_values"].shape[1])
    candidates = [int(i) for i in data["train_indices"]
                  if np.allclose(data["knot_values"][i], ramp)]
    if len(candidates) != 1:
        raise ValueError("The predeclared uT=100 linear ramp must be one training control")
    control_id = candidates[0]
    knots = data["knot_values"][control_id]
    x, t = data["x"], data["t"]
    truth = data["psi"][control_id]
    x_indices = np.arange(0, len(x), 2)
    t_indices = np.arange(0, len(t), 2)
    if t_indices[-1] != len(t) - 1:
        raise ValueError("Use an odd number of time samples so the subset contains T")
    single = {
        "x": x[x_indices], "t": t[t_indices],
        "psi": truth[np.ix_(t_indices, x_indices)][None, :, :],
        "psi0": truth[0, x_indices], "knot_values": knots[None, :],
    }
    config = dict(length=float((x[1] - x[0]) * len(x)), T=float(t[-1]),
                  seed=598, width=64, latent=48, depth=2, num_threads=1,
                  log_every=250)
    heldout = np.ones(truth.shape, dtype=bool)
    heldout[np.ix_(t_indices, x_indices)] = False
    result = {
        "source": str(source), "source_control_id": control_id,
        "source_split": "training", "knot_values": knots.tolist(),
        "question": "Can the fixed architecture and budget fit one known hardest training trajectory?",
        "config": config, "steps_per_mode": int(steps),
        "label_points": int(len(x_indices) * len(t_indices)),
        "heldout_points_same_control": int(heldout.sum()),
        "comparison": "Identical architecture, initialization, label batches, IC samples, Adam, and step budget; physics adds PDE and periodic BC penalties.",
        "limitations": [
            "One known training control is not an operator-generalization test.",
            "Remaining points assess interpolation of the same trajectory, not unseen control functions.",
            "One seed and one budget cannot separate representation limits from optimization limits.",
            "The comparison changes PDE and BC penalties together; it does not isolate either term individually.",
            "No test controls were used to choose this diagnostic or fit either model.",
        ],
        "models": {},
    }
    predictions, histories = {}, {}
    for mode in ("physics", "supervised"):
        print(f"Single known training-control diagnostic: {mode}, {steps} steps", flush=True)
        model, logs, summary = fit_operator(single, [0], [], config,
                                            mode=mode, steps=steps, device="cpu")
        prediction = predict_operator(model, knots, x, t)
        predictions[mode], histories[mode] = prediction, logs
        metrics = {
            "training_grid_relative_l2": _relative_l2(
                prediction[np.ix_(t_indices, x_indices)], truth[np.ix_(t_indices, x_indices)]),
            "heldout_points_relative_l2": _relative_l2(prediction[heldout], truth[heldout]),
            "full_trajectory_relative_l2": _relative_l2(prediction, truth),
            "max_mass_error": float(np.max(np.abs(
                np.sum(np.abs(prediction) ** 2, axis=1) * (x[1] - x[0]) - 1))),
            "training_summary": summary,
        }
        result["models"][mode] = metrics
        with (output / f"{mode}_loss_history.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=logs[0].keys())
            writer.writeheader()
            writer.writerows(logs)
        print(mode, {key: value for key, value in metrics.items() if key != "training_summary"}, flush=True)

    os.environ.setdefault("MPLCONFIGDIR", str(output / ".mplconfig"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    axes[0].plot(x, abs(truth[-1]) ** 2, "k--", label="Reference")
    for mode in ("physics", "supervised"):
        axes[0].plot(x, abs(predictions[mode][-1]) ** 2, label=mode)
        axes[1].semilogy([row["step"] for row in histories[mode]],
                         [row["loss_data"] for row in histories[mode]], label=mode)
    axes[0].set(xlim=(-2, 2), xlabel="x", ylabel="Terminal density", title="Same known control")
    axes[1].set(xlabel="Training step", ylabel="Raw complex data MSE", title="Identical label samples")
    labels = ["physics", "supervised"]
    axes[2].bar(labels, [result["models"][mode]["heldout_points_relative_l2"] for mode in labels])
    axes[2].set(ylabel="Relative complex L2 error", title="Remaining points of this control")
    for axis in axes[:2]:
        axis.legend()
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.savefig(output / "single_training_control_diagnostic.png", dpi=140)
    plt.close(fig)
    result["elapsed_seconds"] = time.perf_counter() - began
    (output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="output/quick_pilot/reference_dataset.npz")
    parser.add_argument("--output", default="output/diagnostic")
    parser.add_argument("--steps", type=int, default=5000)
    args = parser.parse_args()
    run_diagnostic(args.source, args.output, args.steps)
