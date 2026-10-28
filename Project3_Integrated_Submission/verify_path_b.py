"""Independently check frozen metrics, checkpoints and figures without replacing them."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np
from PIL import Image
import torch

from project3_operator import ComplexDeepONet, predict_operator
from project3_gpe import SimulationConfig, propagate, solve_ground_state
from run_project3 import make_plots
from validate_controls import validate_controls


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with Path(path).open() as stream:
        return [{k: (None if v == "" else float(v)) for k, v in row.items()} for row in csv.DictReader(stream)]


def check_preservation(require_all=False):
    frozen = json.loads(Path("output/submission_checks/preserved_before_path_b.json").read_text())
    # README is a report entry point, not a verified experiment artifact.
    checked = {name: item for name, item in frozen.items()
               if name != "README.md" and not name.endswith(".DS_Store")}
    missing = [name for name in checked if not Path(name).is_file()]
    changed = [name for name, item in checked.items()
               if Path(name).is_file() and sha(name) != item["sha256"]]
    if changed or (require_all and missing):
        raise AssertionError("Pre-existing files changed or missing: " + repr(changed + missing))
    return dict(files_checked=len(checked) - len(missing), changed_files=changed,
                missing_preserved_files=missing, strict=require_all)


def verify(output_dir="output/submission_checks/verification",
           dataset="output/project3/reference_dataset.npz", strict_preservation=False):
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a fresh verification output directory")
    output.mkdir(parents=True, exist_ok=True)
    frozen_dir = Path("output/project3")
    full = json.loads((frozen_dir / "results.json").read_text())
    with np.load(dataset) as stored:
        data = {k: stored[k] for k in stored.files}
    assert full["profile"] == "full" and full["settings"]["steps"] == 5000
    assert data["psi"].shape == (120, 129, 256)
    splits = [data[key] for key in ["train_indices", "val_indices", "test_indices"]]
    assert list(map(len, splits)) == [84, 18, 18]
    assert not any(np.intersect1d(a, b).size for a, b in [(splits[0], splits[1]), (splits[0], splits[2]), (splits[1], splits[2])])
    assert np.all(data["psi"][:, 0] == data["psi0"])
    # The CSV model field is textual, so parse it separately.
    with (frozen_dir / "p3_p5_heldout_errors.csv").open() as stream:
        csv_rows = list(csv.DictReader(stream))
    differences, predictions, histories = [], {}, {}
    torch.set_num_threads(2)
    for mode in ["physics", "supervised"]:
        checkpoint = torch.load(frozen_dir / f"{mode}_operator.pt", map_location="cpu", weights_only=True)
        model = ComplexDeepONet(checkpoint["state_dict"]["knot_times"], checkpoint["config"])
        model.load_state_dict(checkpoint["state_dict"])
        assert sum(p.numel() for p in model.parameters()) == 39650
        predictions[mode] = {}
        for index in data["test_indices"]:
            pred = predict_operator(model, data["knot_values"][index], data["x"], data["t"])
            predictions[mode][int(index)] = pred
            truth = data["psi"][index]
            error = float(np.linalg.norm(pred - truth) / np.linalg.norm(truth))
            norm_error = float(np.max(abs(np.sum(abs(pred)**2, axis=1) * float(data["dx"]) - 1)))
            row = next(r for r in full["test_errors"] if r["model"] == mode and r["control_id"] == index)
            stored_row = next(r for r in csv_rows if r["model"] == mode and int(r["control_id"]) == index)
            differences += [abs(error - row["relative_l2"]), abs(norm_error - row["max_mass_error"]),
                            abs(float(stored_row["relative_l2"]) - row["relative_l2"])]
        histories[mode] = read_csv(frozen_dir / f"{mode}_loss_history.csv")
        assert len(histories[mode]) == 5000
    # On different NumPy/PyTorch platforms, float32 mass statistics can differ
    # by a few ulps; preserve strict validation while allowing 2e-6 absolute drift.
    assert max(differences) < 2e-6, max(differences)
    assert json.loads((frozen_dir / "p1_reference_validation.json").read_text()) == full["p1"]
    for mode, cost in full["cost"].items():
        median = statistics.median(r["inference_seconds"] for r in full["test_errors"] if r["model"] == mode)
        assert median == cost["inference_seconds"]
        denominator = full["baseline_median_seconds"] - median
        expected = ((cost["training_seconds"] + full["training_data_seconds"]) / denominator
                    if denominator > 0 else None)
        assert expected == cost["break_even_queries"]

    cfg = SimulationConfig(n=256, dt=.001)
    initial = solve_ground_state(cfg, 1.0)
    target = solve_ground_state(cfg, 100.0)
    controls = {"Linear ramp": np.linspace(1, 100, 5),
                "Conventional optimized": np.array(full["traditional"]["knot_values"]),
                "Learned candidate (verified)": np.array(full["surrogate_control"]["knot_values"])}
    trajectories = {name: propagate(cfg, initial["psi"], knots, save_times=data["t"])
                    for name, knots in controls.items()}
    trajectories["target"] = target["psi"]
    figure_dir = output / "recreated_figures"
    figure_dir.mkdir()
    make_plots(figure_dir, data, histories, full["test_errors"], predictions, controls, trajectories)
    (figure_dir / "results.json").write_text(json.dumps(full))
    validate_controls(figure_dir / "results.json")
    figure_checks = {}
    for name in [*full["figures"], "post_control_breathing.png"]:
        original = np.asarray(Image.open(frozen_dir / name).convert("RGBA"))
        recreated = np.asarray(Image.open(figure_dir / name).convert("RGBA"))
        identical = original.shape == recreated.shape and np.array_equal(original, recreated)
        figure_checks[name] = dict(pixel_identical=bool(identical),
                                  original_sha256=sha(frozen_dir / name),
                                  recreated_sha256=sha(figure_dir / name))
    result = dict(full_run_profile=full["profile"], dataset_shape=list(data["psi"].shape),
                  split_sizes=list(map(len, splits)), raw_checkpoint_metric_max_abs_difference=max(differences),
                  full_run_csv_metrics_match=True, p1_json_matches=True, break_even_arithmetic_matches=True,
                  figures=figure_checks,
                  all_figures_pixel_identical=all(r["pixel_identical"] for r in figure_checks.values()),
                  preservation=check_preservation(strict_preservation))
    (output / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="output/submission_checks/verification")
    parser.add_argument("--dataset", default="output/project3/reference_dataset.npz")
    parser.add_argument("--strict-preservation", action="store_true")
    args = parser.parse_args()
    verify(args.output_dir, args.dataset, args.strict_preservation)
