"""Reproduce the preserved known-trajectory diagnostics in a fresh directory.

The legacy capacity script has fixed relative paths. Running it inside a new
experiment directory preserves both its implementation and every old result.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from diagnostic_operator import run_diagnostic
from capacity_diagnostic import run_capacity_diagnostic
from project3_gpe import SimulationConfig, solve_ground_state
from run_project3 import PROFILES, make_dataset


def reproduce(output_dir="output/reproduced_legacy", references_only=False):
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new output directory; legacy evidence is never replaced")
    references = output / "output/quick_pilot"
    references.mkdir(parents=True)
    config = SimulationConfig(n=256, dt=.002)
    initial = solve_ground_state(config, 1.0)
    if not initial["converged"]:
        raise RuntimeError("Initial ground state did not converge")
    data, records, elapsed = make_dataset(config, PROFILES["quick"], initial["psi"], 598)
    dataset = references / "reference_dataset.npz"
    np.savez_compressed(dataset, **data)
    (output / "generation.json").write_text(json.dumps(dict(
        seed=598, reference_controls=len(records), propagation_seconds=elapsed,
        shape=list(data["psi"].shape)), indent=2) + "\n")
    if references_only:
        return dataset
    previous = Path.cwd()
    try:
        os.chdir(output)
        run_diagnostic()  # Reads only the new quick references.
        run_capacity_diagnostic()  # Uses fixed paths inside this new directory.
    finally:
        os.chdir(previous)
    return output / "output/diagnostic"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="output/reproduced_legacy")
    parser.add_argument("--references-only", action="store_true")
    args = parser.parse_args()
    print(reproduce(args.output_dir, args.references_only))
