# MAE 598 Project 3: BEC squeezing and neural-model diagnosis

A one-dimensional Gross–Pitaevskii case study for **Path B: Diagnosis**. A conventional Fourier solver is compared with a control-conditioned PINN and a matching data-only model. A supervised final-state FNO is supplementary.

## Start here

- **[Read the complete report](REPORT.md)**: outreach, researcher brief, equations, P1–P5, controlled experiments, recommendation, and limitations.
- **[Open the runnable notebook](Project3CodeSolution.ipynb)**: complete code and saved outputs, copied from the team's final `ccnew.ipynb`.
- **[Figures](figures/)**: saved notebook images embedded by the report.

## Run in Google Colab

1. Open [Google Colab](https://colab.research.google.com).
2. Select **File → Open notebook → GitHub** and paste this repository's URL, or upload `Project3CodeSolution.ipynb` directly.
3. Use a fresh CPU session and run cells from top to bottom. Install NumPy, SciPy, pandas, Matplotlib, and PyTorch if the runtime lacks them.
4. Allow several minutes; the traditional optimization alone took about two minutes in one recorded run. Numerical outputs and timings may vary.

The notebook generates synthetic data itself. No external experimental dataset is needed. Preserve cell order: model-definition cells reset weights, so predictions must follow the relevant training cells. Detailed settings and evidence limits are in [Report Section 7](REPORT.md#7-reproduction-and-submission-status).

## Main recorded findings

- Maximum solver time-refinement wave difference: **0.007970%** in the checked cases.
- Main PINN space–time validation error: **99.54%**, with near-zero predicted probability.
- PINN error increases from **10.81%** at horizon 0.1 to **99.52%** at 2.5 under fixed-budget training.
- Supplementary FNO exploratory final-state error: **65.75%**. Its fast inference is not an accuracy-qualified replacement for the solver.

These are saved notebook measurements, not independently retrained results. The report separates earlier optimization outputs from final-model findings and does not mix the separate DeepONet implementation into this study.
