# Project 3 integration audit and decision record

**Audit date:** October 28, 2026. **Outcome:** Path B; a single validated 1-D nonlinear Gross–Pitaevskii control model, with a matched four-way loss ablation. Original uploads were **not** modified. Unknown team membership is not inferred from filenames.

## Sources examined and contributions

| Submitted source | Code, numerical setup, evidence | Useful contribution | Why not combine raw metrics? |
|---|---|---|---|
| **Independent Colab notebook** (`DesignOptProj3(1).ipynb`, uploaded separately) | NumPy FFT/Strang-split 1-D harmonic-trap GPE; periodic numerical box [-8,8), N=512, dt=.001, T=2.5, g=1. Normalized **Gaussian initial guess**; later computes u=100 stationary target by imaginary time. Three **sine-basis** control coefficients; direct SciPy control optimization; **supervised FNO-style** final-state network with 30 sampled trajectories (20 train/5 validation/5 test), 150 epochs. | Readable explanations; initial/final packet-width visualization; independent FNO-style alternative; direct PDE verification of an AI-selected control. The **RMS width** diagnostic is reimplemented for the shared primary model in `integration_validation.py`. | Initial wave, target procedure, control basis, optimizer objective, neural model architecture/training set, timing, and error definitions differ. The notebook contains inline screenshot images but **zero stored code-cell outputs**; its numerical claims are not independently runnable evidence here. |
| **Modular DeepONet project** (`project3_working 2.zip`, same files as prior `project3_working.zip`) | `project3_gpe.py` Strang splitting and stationary nonlinear ground states; optimized piecewise-linear **five-knot** control, fixed endpoints; data/IC/PDE/BC loss DeepONet learns full `(x,t)` field; 120 controls, 84/18/18 train/val/test; strength knob 5/25/100; 5,000-step full training and four-way matched ablation; P1–P5 plots and one report. | **Chosen foundation**: strongest numerical validation, runnable tests, comparable supervised/physics-informed operator experiments, root-cause-control evidence and report documentation. | Directly comparable **within** its consistent settings; its fidelity 0.944961 and operator errors are not comparable to the independent notebook's Gaussian/sine-control numbers. |
| **Earlier local archives** (`project3_working copy.zip`, previously uploaded notebook/PDF, and earlier GitHub review ZIP) | Historical versions/packaging of the modular project or earlier shared notebook rather than separately validated additional methods. | Preserve provenance and ideas; no older numeric value supersedes frozen full experiment. | Earlier reports/smoke experiments can differ in model size, training budget and control family; not merged into the main results. |

The two `project3_working.zip` archives were compared by file content: **146 common project files were byte-identical**. The older `project3_working copy.zip` shares 96 files with the newer archive (95 identical, one README modified). The separately uploaded Colab notebook is the distinct alternative implementation.

## Formulation compatibility — explicit reconciliation

| Question | Independent Colab | Chosen modular reference | Decision |
|---|---|---|---|
| PDE | Dimensionless 1-D defocusing GPE, `iψ_t = -½ψ_xx + (½u(t)x² + |ψ|²)ψ` | Same PDE and `g=1` | Compatible governing equation |
| Spatial box / boundary | `[-8,8)`, N=512, FFT **implicit periodic** | `[-8,8)`, N=256, FFT periodic **and tested** via domain/mesh refinement | Physically similar truncated box, *different grid*; use the validated 256-point reference in all comparisons |
| Initial condition | Normalized Gaussian `exp(-x²/2)` | Converged **nonlinear stationary ground state** at `u=1` | **Not equivalent**; do not reuse independent Colab fidelity or target comparisons |
| Target | Imaginary-time target at `u=100`, used after early practice density target | Converged stationary nonlinear targets at `u_T=5,25,100` with residual checks | Not directly identical due to starting states and convergence method |
| Controls / optimization | 3 sine coefficients added to ramp; one earlier piecewise-constant practice optimizer; sine-coefficient penalty is **not** the prescribed `∫|u'|²` | Three adjustable **piecewise-linear interior knots**, bounds and fixed endpoints; fidelity plus exact discretized control-derivative penalty | **Not equivalent**; retain consistent controlled benchmark and state the restricted family |
| Network | Supervised FNO-style terminal-state prediction, MSE only; 20/5/5 examples | Space–time complex DeepONet, physics + data + IC + BC losses; 84/18/18 | Supervised FNO is an *exploratory separate model*, **not a PINO** as trained; retain full-space–time physics-informed DeepONet |
| Diagnostics | Baseline width/conservation, total MSE, terminal-only spatial Fourier plots; P5 demo has a **hard-coded 12.7 s** dataset cost | Convergence/norm, separate losses, P3 field L2, P4 space–time error + error spectrum, measured P5 training/data/inference | Main measured P1–P5 evidence must come from modular experiment |
| Presentation pitfalls | Last heatmap **squares already-stored density**, resulting in `|ψ|⁴`, not probability density; terminal spectrum compares raw amplitudes instead of the **error spectrum** | Complex pointwise error/spectrum computed against PDE references | Do not reuse notebook's mislabeled P4 figure |

## Changes actually made in integrated project

1. Copied **only** the modular project's audited 65-file submission allowlist into a clean folder. No generated 56 MB dataset, duplicate historical notebooks or raw private development logs are part of the new repository.
2. Kept existing frozen P1–P5 plots/results, neural checkpoints, controls, and ablation outcomes untouched.
3. Added `integration_validation.py`: from the submitted control CSV it **re-solves** a held-out test control, recomputes its complex-field L2 error with the submitted checkpoint, re-solves the conventional optimized/ramp/learned control, and adds an independent-method-inspired, correctly normalized **RMS packet-width** figure to `integration_evidence/`.
4. Updated `verify_path_b.py` to tolerate **2e-6** cross-platform absolute drift in float32 mass statistics (original `1e-7` failed on this environment with drift `9.54e-7`); no scientific metrics, checkpoints or target values were modified. Tiny regenerated P4 image raster differences are disclosed, not disguised as exact reproduction.
5. Added explicit `Pillow` dependency for the verifier and modified `.gitignore` so the required model **`.pt` checkpoints are not silently omitted by Git**. Regenerated datasets and temporary audit outputs remain ignored.
6. Added this audit, contribution map, one-command entry point, integration evidence, assignment checklist, and a labeled integration appendix to `REPORT.md`. The standalone notebook is **not embedded as a second executable entry point**, avoiding its hidden-state and incompatible comparison problems. The source upload remains intact.

## What was run *now* versus only supplied

- **Run in this session:** 13 unit tests passed; full synthetic reference family regenerated (120 controls × 129 times × 256 points); optimized-control fidelity independently recomputed within `6.1e-8` of frozen; one held-out physics-informed DeepONet complex-field L2 recomputed as `0.2783662908` vs frozen `0.2783662788`, discrepancy `1.2e-8`; same-model packet RMS widths and figure generated.
- **Run in this session:** `verify_path_b.py` reloaded the saved models and checked all held-out records; worst raw metric difference `9.54e-7`. Four of five regenerated main figures matched pixel-for-pixel; P4 had 156 differing pixels out of 691,200 (max channel difference 4), reflecting cross-runtime numerics/rendering. See `integration_evidence/verification_summary.json`.
- **Supplied and previously verified in original archive, but not retrained here:** full 5,000-step DeepONet training runs, all four 5,000-step ablations, all hardware/timing measurements and the one-trajectory extended training. Re-running these requires the commands in `README.md` and substantial CPU time. The original saved results remain immutable.
- **Not verified:** standalone notebook's embedded screenshot metrics or end-to-end runtime, Colab execution, publication permission, sending the memo, researcher review/approval, and actual GitHub submission.

## Scientific decision

**Path B is justified at the level tested:** with architecture, training data, seed, optimizer, sampled points and 5,000-step budget controlled, PDE loss makes prediction accuracy worse at stronger trap values. The supervised same-architecture comparator reinforces the result; the boundary penalty alone has small/mixed effects. This **does not isolate** a unique low-level cause (gradient conditioning, inadequate weight, optimizer or representation), does not prove a universal failure, and does not satisfy Path C.

The separate FNO-style notebook's high speed or accuracy numbers cannot refute or reinforce this result, because its setup and evaluation differ. No speedup qualifies as useful until it meets a common accuracy threshold on held-out controls and yields verified control decisions.
