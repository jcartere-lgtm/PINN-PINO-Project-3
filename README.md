# MAE 598 Project 3 — integrated Path B submission

**Purpose.** Diagnose when a physics-informed neural operator helps or fails to approximate the controlled one-dimensional Gross–Pitaevskii equation. The team compared a conventional Fourier split-step solver and PDE-constrained optimization against a complex space–time DeepONet, including matched data / IC / PDE / boundary-loss ablations. The current evidence supports **Path B**, not a successful acceleration or a demonstrated fix.

**Start here:** [Single final report](REPORT.md) · [Source comparison and integration decisions](INTEGRATION_AUDIT.md) · [Contribution map](CONTRIBUTIONS.md) · [Assignment checklist](ASSIGNMENT_CHECKLIST.md)

## Install and run checks (Python 3.10+ recommended)

From this folder in VS Code, Terminal, or a Colab runtime with the repository downloaded:

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\\Scripts\\activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python integration_validation.py --output-dir output/integration_check
```

These commands do **not** retrain the PINO or overwrite the frozen results. The last command independently reruns the conventional optimized control, verifies one saved held-out operator prediction, and creates `output/integration_check/squeezing_width_comparison.png`. Its verified reference results and a labeled figure are already saved at `integration_evidence/`.

If you're running in **Colab**, clone/download the repository and run the last three commands from the project directory in notebook cells (prepend `!` to shell commands). The submission's primary entry point is Python/Markdown rather than a stateful notebook. Neither an end-to-end Colab run nor GPU training was verified during the integration.

## Layout

- `REPORT.md`: one Markdown report with P1–P5, Path B ablation, memo draft and integration appendix.
- `project3_gpe.py`: dimensionless 1-D nonlinear GPE, stationary ground states, continuous bounded control and traditional optimization.
- `project3_operator.py`, `run_project3.py`: complex space–time DeepONet and original full experiment pipeline.
- `run_loss_ablation.py`: A/B/C/D matched 5,000-step loss experiments.
- `tests/`: solver and operator tests, including analytical residual and convergence checks.
- `output/project3/`, `output/loss_ablation_4way/`, `output/diagnostic/`: supplied frozen plots, numerical results, trained checkpoints, loss histories.
- `integration_validation.py`, `integration_evidence/`: newly rerun independent fidelity/error checks and same-model RMS wave-packet width diagnostic.
- `INTEGRATION_AUDIT.md`, `CONTRIBUTIONS.md`, `ASSIGNMENT_CHECKLIST.md`: reconciliation and provenance.

The *independently supplied Colab notebook* is preserved as its original user upload and is **not run or incorporated into this repository's primary experiment**: its Gaussian initial condition, sine-basis controls, supervised FNO-style architecture, training set and loss are not equivalent. See `INTEGRATION_AUDIT.md` before using its plots or timing numbers.

## Deeper reproducibility

Use fresh output directories so frozen results stay intact:

```bash
# Regenerate all 120 full-run reference trajectories (larger intermediate NPZ).
python prepare_reference_data.py --output-dir output/reproduced_references

# Check saved operator checkpoint metrics and regenerate the paper's P1-P5 figures.
python verify_path_b.py \
  --dataset output/reproduced_references/reference_dataset.npz \
  --output-dir output/reproduced_verification

# Re-run the matched four-way ablation (CPU intensive; around 154 s on the original hardware).
python run_loss_ablation.py \
  --source output/reproduced_references/reference_dataset.npz \
  --output-dir output/reproduced_ablation --steps 5000 --seed 598 --budget-seconds 300

# Optional: re-train both main operators and regenerate the full benchmark from scratch.
python run_project3.py --profile full --seed 598 --output-dir output/reproduced_full
```

The saved full run used **seed 598**, NumPy/SciPy/PyTorch CPU execution, and recorded data generation time; `requirements-tested.txt` retains the **original author's** exact library versions and Apple M2 Pro hardware. Results/timing on other machines are not necessarily identical. The integrated verifier uses a strict but portability-appropriate absolute tolerance of `2e-6` for float32 metric checks; pixel-perfect rendering is *not* guaranteed across environments. If ablation exceeds the CPU budget, rerun all four conditions with an increased identical budget and a new directory.

## Publishing and submission

The files in this integrated folder are a proposed **reviewed public subset**, not evidence that anybody has published it or that the researcher gave publication consent. Ask the team to review names/interview attribution (or anonymize), send and date the memo if actually sent, and submit the repository URL via the course. In particular, **do not claim** that Dr. Adriazola reviewed the written brief or gave a numerical usefulness threshold. The original source uploads and old notebook screenshots have not been modified.
