"""Lightweight, independent integration check. Never overwrites frozen experiments.

Recomputes a held-out PINO error using a fresh conventional PDE trajectory,
recomputes optimized conventional control fidelity, and creates a supplementary
wave-packet width plot inspired by the independently supplied Colab notebook.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from project3_gpe import SimulationConfig, objective_components, propagate, solve_ground_state
from project3_operator import ComplexDeepONet, predict_operator


def width(psi, x, dx):
    """RMS width about the density mean, not raw variance around x=0."""
    rho = np.abs(psi) ** 2
    total = np.sum(rho) * dx
    mean = np.sum(x * rho) * dx / total
    return float(np.sqrt(np.sum((x - mean) ** 2 * rho) * dx / total))


def run(output_dir):
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Use a fresh, empty output directory to protect results.')
    output.mkdir(parents=True, exist_ok=True)
    saved = json.loads(Path('output/project3/results.json').read_text())
    config = SimulationConfig(n=256, dt=0.001)
    x, _, dx = config.grid()
    times = np.linspace(0, config.T, 129)
    initial = solve_ground_state(config, 1.0)
    target = solve_ground_state(config, 100.0)
    assert initial['converged'] and target['converged']
    comparison = {}
    flows = {}
    for name, knots in {
        'Linear ramp': np.linspace(1, 100, 5),
        'Traditional optimized': saved['traditional']['knot_values'],
        'Learned candidate, PDE verified': saved['surrogate_control']['knot_values'],
    }.items():
        result = propagate(config, initial['psi'], knots, save_times=times)
        obj = objective_components(config, result['psi'][-1], target['psi'], knots)
        comparison[name] = {
            'fidelity': obj['fidelity'],
            'objective': obj['total'],
            'starting_width': width(result['psi'][0], x, dx),
            'ending_width': width(result['psi'][-1], x, dx),
            'max_mass_drift': float(np.max(np.abs(result['mass'] - 1))),
        }
        flows[name] = result
    # Reconstruct one held-out control directly from submitted controls, not cached reference data.
    with Path('output/project3/reference_controls.csv').open(newline='') as stream:
        row = next(r for r in csv.DictReader(stream) if int(r['control_id']) == 34)
    assert row['split'] == 'test'
    knots = np.asarray(json.loads(row['knots']), dtype=float)
    ground_truth = propagate(config, initial['psi'], knots, save_times=times)['psi']
    checkpoint = torch.load('output/project3/physics_operator.pt', map_location='cpu', weights_only=True)
    model = ComplexDeepONet(checkpoint['state_dict']['knot_times'], checkpoint['config'])
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    torch.set_num_threads(2)
    pred = predict_operator(model, knots, x, times)
    relative_l2 = float(np.linalg.norm((pred - ground_truth).ravel()) / np.linalg.norm(ground_truth.ravel()))
    stored_row = next(r for r in saved['test_errors'] if r['model'] == 'physics' and r['control_id'] == 34)
    discrepancy = abs(relative_l2 - stored_row['relative_l2'])
    fidelity_delta = abs(comparison['Traditional optimized']['fidelity'] - saved['traditional']['final_components']['fidelity'])
    assert discrepancy < 3e-5, f'PINO error differs from frozen result: {discrepancy}'
    assert fidelity_delta < 3e-5, f'Conventional fidelity differs from frozen result: {fidelity_delta}'
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, flow in flows.items():
        ax.plot(flow['t'], [width(p, x, dx) for p in flow['psi']], label=name)
    ax.axhline(width(target['psi'], x, dx), color='black', ls='--', label='Target stationary width')
    ax.set(xlabel='Time (dimensionless)', ylabel='Density RMS spatial width',
           title='Supplementary P1: same-model wave-packet squeezing')
    ax.legend(fontsize=8); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(output / 'squeezing_width_comparison.png', dpi=160)
    plt.close(fig)
    data = {
        'status': 'recomputed in this run, using source files and frozen checkpoints',
        'reference': 'fresh conventional nonlinear GPE trajectories, 1D periodic box [-8,8), N=256, dt=.001',
        'models': comparison,
        'target_width': width(target['psi'], x, dx),
        'heldout_id': 34,
        'heldout_relative_complex_l2': relative_l2,
        'frozen_heldout_relative_complex_l2': stored_row['relative_l2'],
        'heldout_error_discrepancy': discrepancy,
        'frozen_optimized_fidelity': saved['traditional']['final_components']['fidelity'],
        'optimized_fidelity_discrepancy': fidelity_delta,
        'note': 'Standalone Colab notebook is excluded from quantitative comparison: different initial state, control parameterization, training data and objective.'
    }
    (output / 'integration_results.json').write_text(json.dumps(data, indent=2) + '\n')
    print(json.dumps({k: data[k] for k in ['heldout_relative_complex_l2', 'heldout_error_discrepancy', 'optimized_fidelity_discrepancy']}, indent=2))
    return data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', default='output/integration_check')
    args = parser.parse_args()
    run(args.output_dir)
