"""Four-way, matched-budget loss ablation on the frozen full GPE dataset.

Uses the verified training implementation unchanged. All four conditions run
mode='physics', so raw PDE and boundary diagnostics are computed even when a
term has zero weight. Only the optimized loss weights vary. No old artifact
is updated and no test control is used for training or checkpoint selection.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import time
from unittest.mock import patch

import numpy as np
import torch

import project3_operator as operator


CONDITIONS = {
    "A": dict(label="Data + IC", pde=0.0, bc=0.0),
    "B": dict(label="Data + IC + BC", pde=0.0, bc=0.1),
    "C": dict(label="Data + IC + PDE", pde=1.0, bc=0.0),
    "D": dict(label="Data + IC + BC + PDE", pde=1.0, bc=0.1),
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def state_hash(model):
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        array = value.detach().cpu().numpy()
        h.update(name.encode()); h.update(str(array.dtype).encode())
        h.update(str(array.shape).encode()); h.update(array.tobytes())
    return h.hexdigest()


class AuditedGenerator:
    """Record the exact arrays returned by the unchanged sampler's RNG."""
    def __init__(self, generator, seed, deadline):
        self.generator, self.seed, self.deadline = generator, seed, deadline
        self.digest = hashlib.sha256()
        self.draw_calls = 0

    def __getattr__(self, name):
        value = getattr(self.generator, name)
        if not callable(value):
            return value
        def call(*args, **kwargs):
            if time.perf_counter() > self.deadline:
                raise TimeoutError("Predeclared ablation wall-clock budget exceeded; incomplete run is not a matched result")
            result = value(*args, **kwargs)
            array = np.asarray(result)
            self.digest.update(name.encode())
            self.digest.update(str(array.dtype).encode())
            self.digest.update(str(array.shape).encode())
            self.digest.update(array.tobytes())
            self.draw_calls += 1
            return result
        return call


@contextmanager
def audit_sampling_and_initialization(deadline):
    original_rng = np.random.default_rng
    original_model = operator.ComplexDeepONet
    generators, initial_hashes = [], []
    def audited_rng(seed=None):
        rng = AuditedGenerator(original_rng(seed), seed, deadline)
        generators.append(rng)
        return rng
    class AuditedDeepONet(original_model):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            initial_hashes.append(state_hash(self))
    with patch.object(np.random, "default_rng", audited_rng), patch.object(operator, "ComplexDeepONet", AuditedDeepONet):
        yield generators, initial_hashes


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def relative_l2(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def evaluate(model, dataset, indices, split):
    rows = []
    dx = float(dataset["dx"])
    for index in indices:
        prediction = operator.predict_operator(model, dataset["knot_values"][index], dataset["x"], dataset["t"])
        truth = dataset["psi"][index]
        masses = np.sum(abs(prediction)**2, axis=1) * dx
        rows.append(dict(split=split, control_id=int(index),
                         uT=float(dataset["knot_values"][index, -1]),
                         relative_l2=relative_l2(prediction, truth),
                         density_relative_l2=relative_l2(abs(prediction)**2, abs(truth)**2),
                         max_mass_error=float(np.max(abs(masses - 1))),
                         terminal_mass=float(masses[-1])))
    return rows


def make_plots(output, results, histories):
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".mplconfig"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for index, (name, condition) in enumerate(results["conditions"].items()):
        groups = condition["by_knob"]
        axes[0].plot([5, 25, 100], [groups[str(u)]["mean_relative_l2"] for u in [5, 25, 100]],
                     "o-", color=f"C{index}", label=name + ": " + CONDITIONS[name]["label"])
        axes[1].plot([5, 25, 100], [groups[str(u)]["mean_max_mass_error"] for u in [5, 25, 100]],
                     "o-", color=f"C{index}", label=name)
    axes[0].set(xlabel="Final trap strength uT", ylabel="Held-out relative complex L2 error",
                title="Ablation: same controls, points, initialization, budget")
    axes[1].set(xlabel="Final trap strength uT", ylabel="Mean maximum |mass - 1|",
                title="Raw predictions; no renormalization")
    for ax in axes: ax.legend(fontsize=8); ax.grid(alpha=.25)
    fig.savefig(output / "ablation_errors.png", dpi=140); plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    for ax, name in zip(axes.flat, CONDITIONS):
        logs = histories[name]
        for term in ["data", "ic", "pde", "bc"]:
            ax.semilogy([r["step"] for r in logs],
                        [max(r["loss_" + term], 1e-16) for r in logs], label=term)
        ax.set(title=name + ": " + CONDITIONS[name]["label"], xlabel="Optimizer step", ylabel="Raw diagnostic MSE")
        ax.legend(fontsize=8); ax.grid(alpha=.25)
    fig.savefig(output / "ablation_loss_components.png", dpi=140); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for ax, pairs, title in zip(axes, [[("B", "A"), ("D", "C")], [("C", "A"), ("D", "B")]],
                                ["Adding boundary penalty", "Adding PDE penalty"]):
        for added, base in pairs:
            differences = [results["conditions"][added]["by_knob"][str(u)]["mean_relative_l2"]
                           - results["conditions"][base]["by_knob"][str(u)]["mean_relative_l2"] for u in [5, 25, 100]]
            ax.plot([5, 25, 100], differences, "o-", label=added + " - " + base)
        ax.axhline(0, color="k", lw=.8)
        ax.set(title=title, xlabel="Final trap strength uT", ylabel="Change in relative L2 (positive = worse)")
        ax.legend(); ax.grid(alpha=.25)
    fig.savefig(output / "ablation_effects.png", dpi=140); plt.close(fig)


def run(source="output/project3/reference_dataset.npz", output_dir="output/loss_ablation_4way",
        steps=5000, seed=598, budget_seconds=300):
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Use a new output directory; existing experiment will not be overwritten: {output}")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    deadline = started + budget_seconds
    with np.load(source) as stored:
        dataset = {key: stored[key] for key in stored.files}
    train, validation, test = (dataset[key] for key in ["train_indices", "val_indices", "test_indices"])
    if any(np.intersect1d(a, b).size for a, b in [(train, validation), (train, test), (validation, test)]):
        raise ValueError("Whole-control splits overlap")
    cfg = dict(length=16.0, T=2.5, seed=seed, width=96, depth=2, latent=64,
               num_threads=2, log_every=125, residual_scale=100.0,
               loss_weights=dict(data=1.0, ic=1.0, norm=0.0),
               lr=1e-3, gradient_clip=10.0)
    results = dict(source=str(source), source_sha256=sha256(source),
                   operator_source_sha256=sha256(Path(operator.__file__)),
                   steps=steps, seed=seed, base_config=cfg, conditions={},
                   train_indices=train.tolist(), validation_indices=validation.tolist(), test_indices=test.tolist(),
                   dataset_shape=list(dataset["psi"].shape),
                   hardware=f"{platform.system()} {platform.machine()}, {os.cpu_count()} logical CPUs; CPU with2torchthreads",
                   versions=dict(python=platform.python_version(), numpy=np.__version__, torch=torch.__version__),
                   budget_seconds=budget_seconds,
                   design_note="Same full family, architecture, initialization, sampled arrays, seed, Adam, clipping and steps; only PDE/BC loss weights change. All raw diagnostics computed in every condition.",
                   test_usage="Test labels used only for final fixed-budget evaluation. No tuning, early stopping or checkpoint selection.")
    histories = {}
    for name, condition in CONDITIONS.items():
        print(f"Training {name}: {condition['label']}; {steps} steps", flush=True)
        config = {**cfg, "loss_weights": {**cfg["loss_weights"], "pde": condition["pde"], "bc": condition["bc"]}}
        with audit_sampling_and_initialization(deadline) as (generators, initial_hashes):
            model, logs, summary = operator.fit_operator(dataset, train, validation, config,
                                                         mode="physics", steps=steps, device="cpu")
        audit = dict(initial_model_sha256=initial_hashes[0],
                     generators=[dict(seed=int(g.seed), calls=g.draw_calls, sha256=g.digest.hexdigest()) for g in generators])
        # Check the optimized scalar loss, not just the intended configuration.
        maximum_identity_error = max(abs(row["loss_total"] - (
            row["loss_data"] + row["loss_ic"] + condition["pde"] * row["loss_pde"] / 10000
            + condition["bc"] * row["loss_bc"])) for row in logs)
        if maximum_identity_error > 1e-4:
            raise AssertionError("Observed loss does not match the declared ablation")
        rows = evaluate(model, dataset, validation, "validation") + evaluate(model, dataset, test, "test")
        groups = {}
        for u in [5, 25, 100]:
            group = [row for row in rows if row["split"] == "test" and row["uT"] == u]
            groups[str(u)] = dict(n=len(group), mean_relative_l2=float(np.mean([r["relative_l2"] for r in group])),
                                  std_relative_l2=float(np.std([r["relative_l2"] for r in group], ddof=1)),
                                  mean_max_mass_error=float(np.mean([r["max_mass_error"] for r in group])))
        write_csv(output / f"{name}_loss_history.csv", logs)
        write_csv(output / f"{name}_metrics.csv", rows)
        torch.save(dict(state_dict=model.state_dict(), config=config,
                        knot_times=np.linspace(0, 2.5, dataset["knot_values"].shape[1]).tolist()), output / f"{name}_model.pt")
        results["conditions"][name] = dict(label=condition["label"], weights=config["loss_weights"],
                                          audit=audit, by_knob=groups, metrics=rows,
                                          training_seconds=summary["training_seconds"],
                                          parameter_count=summary["parameter_count"],
                                          loss_identity_max_abs_error=maximum_identity_error,
                                          final_raw_losses={key: logs[-1][key] for key in ["loss_data", "loss_ic", "loss_pde", "loss_bc", "loss_bc_value", "loss_bc_derivative"]})
        histories[name] = logs
        print(name, {u: round(v["mean_relative_l2"], 6) for u, v in groups.items()}, flush=True)
    audits = [c["audit"] for c in results["conditions"].values()]
    if any(audit != audits[0] for audit in audits[1:]):
        raise AssertionError("Sampling streams or initial weights differ; comparison is invalid")
    results["identical_initialization_and_sampling_verified"] = True
    results["elapsed_seconds"] = time.perf_counter() - started
    results["limitations"] = [
        "One seed and one optimizer-step budget; no population-level causal or unique-mechanism claim.",
        "Same number of steps, not equal wall-clock budgets; physical losses require additional derivatives.",
        "Conditional loss effects can identify a harmful term under these settings, not NTK, spectral bias or a unique lower-level cause.",
        "This follow-up was motivated by previously inspected test errors; it is exploratory, not a new confirmatory holdout.",
        "No inference speed or useful speedup is inferred from this loss comparison."]
    make_plots(output, results, histories)
    (output / "results.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    print(f"Completed all four matched conditions in {results['elapsed_seconds']:.2f}s", flush=True)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="output/project3/reference_dataset.npz")
    parser.add_argument("--output-dir", default="output/loss_ablation_4way")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=598)
    parser.add_argument("--budget-seconds", type=float, default=300)
    args = parser.parse_args()
    run(args.source, args.output_dir, args.steps, args.seed, args.budget_seconds)
