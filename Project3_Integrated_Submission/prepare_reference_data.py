"""Regenerate only the synthetic frozen full-run references; do not retrain."""
import argparse
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from project3_gpe import SimulationConfig, solve_ground_state
from run_project3 import PROFILES, make_dataset


def prepare(output_dir="output/reproduced_references"):
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new reference output directory")
    output.mkdir(parents=True, exist_ok=True)
    source = json.loads(Path("output/project3/results.json").read_text())
    if source["settings"] != PROFILES["full"]:
        raise ValueError("Runner defaults differ from frozen full-run settings")
    started = perf_counter()
    config = SimulationConfig(n=256, dt=.001)
    initial = solve_ground_state(config, 1.0)
    if not initial["converged"]:
        raise RuntimeError("Initial stationary state did not converge")
    data, records, elapsed = make_dataset(config, source["settings"], initial["psi"], source["seed"])
    np.savez_compressed(output / "reference_dataset.npz", **data)
    (output / "generation.json").write_text(json.dumps(dict(seed=source["seed"],
        reference_controls=len(records), dataset_shape=list(data["psi"].shape),
        elapsed_seconds=perf_counter() - started, propagation_generation_seconds=elapsed), indent=2) + "\n")
    print("Generated", output / "reference_dataset.npz")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="output/reproduced_references")
    prepare(parser.parse_args().output_dir)
