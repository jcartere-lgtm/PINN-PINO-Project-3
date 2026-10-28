# MAE 598 — Project 3: PINN/PINO in the Field

**Arizona State University | Design Optimization**

## Start Here — Final Project Submission

Our team's completed Project 3, including the final report, Python implementations, experimental results, figures, and validation tests, is located in the folder below.

### [OPEN FINAL PROJECT FOLDER](Project3_Integrated_Submission/)

**To view our final submission:**

1. Click the **Project3_Integrated_Submission** folder above.
2. Open **REPORT.md** to read our complete final report.
3. Refer to the other files and folders for supporting code, figures, numerical results, and experiments.

### [READ THE FINAL REPORT DIRECTLY](Project3_Integrated_Submission/REPORT.md)

## Project Overview

For Project 3, our team collaborated with Dr. Jimmie Adriazola of Arizona State University to investigate whether physics-informed neural operators could improve the efficiency of PDE-constrained optimal control.

Our case study focuses on the one-dimensional Gross–Pitaevskii equation and the problem of controlling Bose–Einstein condensate wave packets.

We compared a conventional split-step Fourier numerical solver and optimization approach with a physics-informed DeepONet neural operator.

**Project Path: B — Diagnosis**

Our experiments investigate the limitations of the neural operator as the problem becomes more difficult, including how different physics-informed loss components affect prediction accuracy.

## Project Files

| File or Folder | Contents |
|---|---|
| [REPORT.md](Project3_Integrated_Submission/REPORT.md) | Complete final report, mathematical formulation, experiments, P1–P5 diagnostics, conclusions, and recommendation memo |
| [README.md](Project3_Integrated_Submission/README.md) | Setup instructions and steps to reproduce the experiments |
| [output/](Project3_Integrated_Submission/output/) | Experimental figures, loss histories, model results, and numerical data |
| [integration_evidence/](Project3_Integrated_Submission/integration_evidence/) | Independently verified calculations and supporting figures |
| [tests/](Project3_Integrated_Submission/tests/) | Numerical solver and neural-operator validation tests |
| [CONTRIBUTIONS.md](Project3_Integrated_Submission/CONTRIBUTIONS.md) | Team contribution information |
| [ASSIGNMENT_CHECKLIST.md](Project3_Integrated_Submission/ASSIGNMENT_CHECKLIST.md) | Project 3 requirements and completion status |

## Reproducing Our Results

For complete installation instructions, required Python packages, and commands to reproduce our numerical experiments, see:

**[Project Setup and Reproducibility Instructions](Project3_Integrated_Submission/README.md)**

The project uses Python, NumPy, SciPy, PyTorch, and Matplotlib. Saved experimental results are included so readers can examine the findings without retraining the neural networks.

## Main Conclusion

Our conventional numerical optimization successfully improves target-state fidelity. However, the physics-informed neural operator did not demonstrate a reliable replacement or practical speedup under the tested conditions.

Our controlled experiments indicate that the PDE residual penalty contributed to increased prediction error for the more difficult cases at the tested training budget.

These findings support a Path B diagnosis and identify directions for future improvements.

---

**For grading and project review, please begin with [REPORT.md](Project3_Integrated_Submission/REPORT.md).**
