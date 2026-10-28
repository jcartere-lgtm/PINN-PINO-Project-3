"""Build the single Path B report from preserved measurements and new ablation.

Does not run an experiment or change any notebook or saved measurement.
"""
import argparse
import json
from pathlib import Path


def load(path):
    return json.loads(Path(path).read_text())


def build(output="REPORT.md", replace=False):
    destination = Path(output)
    if destination.exists() and not replace:
        raise FileExistsError("Use --replace explicitly to rebuild the report")
    full = load("output/project3/results.json")
    ablation = load("output/loss_ablation_4way/results.json")
    single = load("output/diagnostic/summary.json")
    capacity = load("output/diagnostic/capacity_summary.json")
    validation = load("submission_evidence/control_validation.json")
    verification = load("output/submission_checks/verification/verification.json")
    tests = load("output/submission_checks/unit_tests.json")
    assert full["profile"] == "full"
    assert verification["raw_checkpoint_metric_max_abs_difference"] == 0
    assert verification["all_figures_pixel_identical"]
    assert ablation["identical_initialization_and_sampling_verified"]
    assert tests["passed"] and tests["tests_run"] == 13
    for condition, model in [("A", "supervised"), ("D", "physics")]:
        for row in ablation["conditions"][condition]["metrics"]:
            if row["split"] == "test":
                frozen = next(r for r in full["test_errors"]
                              if r["model"] == model and r["control_id"] == row["control_id"])
                assert row["relative_l2"] == frozen["relative_l2"]

    sweep = []
    for u in [5, 25, 100]:
        cells = []
        for model in ["physics", "supervised"]:
            group = [r["relative_l2"] for r in full["test_errors"] if r["model"] == model and r["uT"] == u]
            cells.append(f"{100 * sum(group)/len(group):.2f}%")
        sweep.append(f"| {u} | {' | '.join(cells)} | 6 |")
    ablation_rows, loss_rows, effects = [], [], []
    for name, condition in ablation["conditions"].items():
        means = [f"{100*condition['by_knob'][str(u)]['mean_relative_l2']:.2f}%" for u in [5, 25, 100]]
        ablation_rows.append(f"| {name}: {condition['label']} | {' | '.join(means)} | {condition['training_seconds']:.2f} |")
        losses = condition["final_raw_losses"]
        loss_rows.append(f"| {name} | " + " | ".join(f"{losses['loss_'+k]:.6g}" for k in ['data','ic','pde','bc']) + " |")
    for added, base, label in [("B", "A", "BC, no PDE"), ("D", "C", "BC, with PDE"),
                               ("C", "A", "PDE, no BC"), ("D", "B", "PDE, with BC")]:
        values, counts = [], []
        for u in [5, 25, 100]:
            x = {r['control_id']:r for r in ablation['conditions'][added]['metrics'] if r['split']=='test' and r['uT']==u}
            y = {r['control_id']:r for r in ablation['conditions'][base]['metrics'] if r['split']=='test' and r['uT']==u}
            differences = [x[k]['relative_l2']-y[k]['relative_l2'] for k in x]
            values.append(f"{100*sum(differences)/len(differences):+.2f}")
            counts.append(str(sum(d > 0 for d in differences)))
        effects.append(f"| {added} − {base}: add {label} | {' | '.join(values)} | {' / '.join(counts)} |")
    cost_rows = []
    for model, cost in full['cost'].items():
        cost_rows.append(f"| {model} | {cost['training_seconds']:.3f} | {cost['inference_seconds']:.6f} | {cost['break_even_queries']:.1f} |")
    traditional = full['traditional']
    ramp, optimized = traditional['initial_components'], traditional['final_components']
    learned = full['surrogate_control']['verified_components']
    hold = validation['post_control']['states']
    replacements = {
        'SWEEP': '\n'.join(sweep), 'ABLATION': '\n'.join(ablation_rows),
        'LOSSES': '\n'.join(loss_rows), 'EFFECTS': '\n'.join(effects), 'COST': '\n'.join(cost_rows),
        'ABLATION_SECONDS': f"{ablation['elapsed_seconds']:.2f}",
        'RAMP_F': f"{ramp['fidelity']:.6f}", 'OPT_F': f"{optimized['fidelity']:.6f}",
        'RAMP_J': f"{ramp['total']:.6f}", 'OPT_J': f"{optimized['total']:.6f}",
        'LEARNED_F': f"{learned['fidelity']:.6f}", 'LEARNED_J': f"{learned['total']:.6f}",
        'CLASSICAL_SECONDS': f"{traditional['elapsed_seconds']:.3f}",
        'BREATHING_REDUCTION': f"{100*(1-hold['traditional']['width_peak_to_peak']/hold['ramp']['width_peak_to_peak']):.2f}",
        'SINGLE_PHYSICS': f"{100*single['models']['physics']['full_trajectory_relative_l2']:.2f}",
        'SINGLE_SUPERVISED': f"{100*single['models']['supervised']['full_trajectory_relative_l2']:.2f}",
        'LONG_5K': f"{100*capacity['history'][1]['relative_complex_l2']:.2f}",
        'LONG_50K': f"{100*capacity['final_relative_complex_l2']:.2f}",
        'TEST_SECONDS': f"{tests['elapsed_seconds']:.2f}",
    }
    report = r'''# MAE 598 Project 3 — Path B: diagnosis of a physics-informed GPE operator

**Final report for the October 21 submission.** Prepared October 20, 2026. This is the single submission report; the Python files, JSON/CSV records, and linked figures are supporting evidence. Earlier notebooks and generated report drafts remain preserved locally and are outside the proposed public subset.

The conventional solver and control optimization work: terminal fidelity improves from **@@RAMP_F@@ to @@OPT_F@@**. The physics-informed DeepONet does not provide a useful replacement: mean error increases with trap strength, and its selected control is worse than the linear ramp after independent PDE verification. Our Path B diagnosis is **underfitting at the chosen training budget, with a harmful contribution from the PDE penalty at its current weight for the harder cases**. A controlled four-way experiment separates that effect from the boundary penalty. It does not identify a unique mathematical or optimizer-level root cause. We claim neither a successful Path A nor a demonstrated remedy.

## 1. Researcher contact and outreach log

Source: [the team's interview account](RESEARCHER_CONTEXT.md), supplied by the team. Statements below are paraphrases, not a transcript. Numerical experiments elsewhere in this report are our own measurements.

| Date/activity | Evidence and status |
|---|---|
| Initial contact and researcher reservation | Contact method, date, reservation record, and earlier outreach activity are unavailable. No emails or reservations are reconstructed. |
| October 16, 2026 interview | The team reports meeting Dr. Jimmie Adriazola. He proposed GPE optimal control, directed us to Equations (2)–(4), and recommended a conventional-versus-neural-operator comparison. |
| Solver/workload discussion | He discussed pseudo-spectral Fourier Strang splitting, mentioned exponential time differencing, and described 1D solves as relatively inexpensive. Repeated PDE evaluations during optimization motivated the operator question. |
| Physical objectives | He recommended squeezing and splitting, with approximately stationary final packets. This submission implements squeezing only. |
| Written problem brief | Discussed problem: yes. Review or approval of a written brief: **not yet obtained**. |
| Follow-up | He offered help and requested results. The memo in Section 8 is an **unsent draft**. Delivery, a response, and permission to publish interview material are unavailable. |

**Unknowns remain unknown:** exact research solver runtime and hardware, typical control-query or iteration count, and a quantitative useful-accuracy threshold. Our CPU timings and query counts are not estimates of his research workflow. The October 16 date is recorded as supplied; we do not claim earlier assignment milestones were completed on schedule.

## 2. Problem brief and minimal case

The research question is whether learning the map from an entire trap-control function to a complex GPE solution can reduce the cost of repeated control queries. For a condensate, tightening a trap can narrow the density packet but also excite oscillations. A useful controller should reach the desired state, avoid unnecessarily rapid control changes, and leave little motion after the control ends. Because 1D numerical solves are already cheap, predictive accuracy and the number of repeated queries matter as much as a fast network evaluation.

We follow the direct PDE-constrained formulation in [Adriazola and Goodman, *A Reduction-Based Strategy for Optimal Control of Bose–Einstein Condensates*](https://arxiv.org/abs/2111.14266), Equations (2)–(4), with these explicit project simplifications:

$$
i\partial_t\psi=-\tfrac12\partial_{xx}\psi+
\left[\tfrac12u(t)x^2+g|\psi|^2\right]\psi,\qquad g=1.
$$

All quantities are dimensionless. The real-line problem is approximated by a periodic Fourier box $x\in[-8,8)$ and $t\in[0,2.5]$. We impose both $\psi(-8,t)=\psi(8,t)$ and matching spatial derivatives in the Fourier representation. Negligible edge density and a larger-box check support this approximation for the tested trajectories; it is not a hard-wall boundary model.

The initial state is the normalized nonlinear ground state $\phi_1$ at $u=1$. Each target is independently computed as the normalized nonlinear ground state $\phi_{u_T}$:

$$
\left[-\tfrac12\partial_{xx}+\tfrac12u x^2+|\phi_u|^2\right]\phi_u
=\mu_u\phi_u,\qquad \int|\phi_u|^2\,dx=1,\qquad \psi(x,0)=\phi_1(x).
$$

Imaginary-time iterations adapt their step until the stationary residual is below $10^{-6}$. A stationary density can accumulate a global complex phase. A Gaussian is a starting guess for the ground-state calculation, not an assumed exact nonlinear stationary state.

Our full nonlinear initial ground state differs from the paper's two-mode initial-state approximation. We solve the full 1D PDE; we do not reproduce its reduced Hamiltonian, CRAB/differential-evolution/GRAPE workflow, or all published examples. There is no splitting experiment, 3D geometry, finite-temperature model, or additional interaction parameter sweep.

The traditional objective is

$$
F=\left|\int\phi_{u_T}^*(x)\psi(x,T)\,dx\right|^2,\qquad
J=\tfrac12(1-F)+\tfrac{\gamma}{2}\int_0^T|\dot u(t)|^2\,dt,
\qquad\gamma=10^{-4}.
$$

The discrete overlap includes $\Delta x$. It compares complex states and ignores an overall phase, while still detecting spatial phase differences. The reported `regularization` field is $\gamma\int|\dot u|^2$; the total objective halves **both** infidelity and that field. It is not a density-only or pure-fidelity optimization. The value of $\gamma$ is our study choice, not a researcher-approved setting.

Controls are continuous, piecewise linear through five equally spaced times, with $u(0)=1$, $u(T)=u_T$, and three adjustable interior values. We add the bounds $1\le u(t)\le u_T$; those bounds are a project restriction, not a required implication of the paper. Nonmonotone paths are allowed. The derivative penalty is evaluated exactly as $\sum_j(u_{j+1}-u_j)^2/\Delta t_j$. Fixed endpoints are built into this parameterization.

**Difficulty knob:** $u_T\in\{5,25,100\}$. Increasing confinement produces a narrower target and more demanding spatial and temporal evolution. Each knob value has its own target. The error trend is an observed difficulty trend for this family, not proof that every individual control becomes harder monotonically.

## 3. Conventional reference and P1

The preserved simulator uses a midpoint, second-order Strang split-step Fourier scheme, with $N=256$, $\Delta x=0.0625$, and nominal $\Delta t=0.001$. It uses complex128 arithmetic and does not renormalize states during real-time propagation. Existing FFT propagation, nonlinear phase evolution, and norm checks were retained. Norm conservation alone is insufficient; we also check phase-sensitive solution error.

P1 compares the hardest linear ramp against half the time step, twice the grid resolution, and a larger domain. These are discretization-sensitivity estimates, not exact error bounds. [Saved P1 measurements](output/project3/p1_reference_validation.json):

| Diagnostic | Measured value |
|---|---:|
| Time-refinement relative complex L2 difference | $1.8794\times10^{-5}$ |
| Space-refinement difference after time refinement | $2.1653\times10^{-11}$ |
| Combined refinement difference | $1.8794\times10^{-5}$ |
| Larger-domain difference | $6.7263\times10^{-13}$ |
| Maximum mass drift | $2.2515\times10^{-13}$ |
| Maximum edge mass | $7.2342\times10^{-24}$ |
| Base ramp propagation time | 0.063407 s |

The team-selected reference tolerance is $10^{-3}$, not Dr. Adriazola's usefulness threshold. Refinement is strongest for the hardest ramp and the three selected controls; it is not an exhaustive convergence proof over all possible controls.

For $u_T=100$, bounded L-BFGS-B searches the three interior controls, using finite-difference objective evaluations. It reports convergence after **9 iterations and 60 actual PDE evaluations**, taking **@@CLASSICAL_SECONDS@@ s**. This is a local result within the specified control family, not a global optimum.

| Control | Terminal fidelity $F$ | Total objective $J$ | Interpretation |
|---|---:|---:|---|
| Linear ramp | @@RAMP_F@@ | @@RAMP_J@@ | Starting control |
| Conventional optimized | @@OPT_F@@ | @@OPT_J@@ | Improves the intended objective |
| Learned candidate, checked with PDE | @@LEARNED_F@@ | @@LEARNED_J@@ | Worse than the ramp |

![Continuous controls and numerically verified terminal densities](output/project3/control_comparison.png)

An [independent refinement record](submission_evidence/control_validation.json) checks all three controls at $N=512$, $\Delta t=0.0005$: optimized fidelity is **0.944959964**, and every base-to-refined fidelity difference is at most $1.26\times10^{-6}$. Tiny differences from the main table also reflect separately converged stationary-state calculations. The public record changes only a local absolute source path; all numerical measurements are preserved.

Holding $u=100$ for one additional time unit gives width peak-to-peak variations of 0.293453 (ramp), 0.156011 (conventional), and 0.297819 (learned candidate). The optimized control reduces this breathing by **@@BREATHING_REDUCTION@@%**, but substantial oscillations remain. The stationary target varies by only $6.15\times10^{-7}$ in the same test. A finite hold does not establish long-time stability or researcher-approved stationarity.

![Post-control width and density stability](output/project3/post_control_breathing.png)

## 4. Operator experiment and P2–P5

The model is a physics-informed **DeepONet**, with a branch encoding all five control knots and a trunk encoding $(x,t)$ using raw coordinates and Fourier features at frequencies 1, 2, 4, and 8. Its two-layer Tanh networks have width 96 and latent dimension 64, totaling **39,650 parameters**. Outputs are real and imaginary wavefunction parts. This is an operator model with physics penalties; it is not the published FNO-based PINO implementation. Branch/trunk structure follows the [DeepONet primary paper](https://www.nature.com/articles/s42256-021-00302-5).

The frozen full run uses 120 reference controls, 40 per knob: **84 training, 18 validation, and 18 test**. Whole control functions are separated across splits. Within each knob, the first training example is a ramp; the others draw three interior fractions uniformly. Seed is 598. Every trajectory has 129 time samples and 256 spatial points. The initial state, domain, nonlinearity, and duration are fixed. This is interpolation in the sampled control family; no extrapolation to different physics is established.

Both models use the same initialization, architecture, label sampling, Adam learning rate $10^{-3}$, clipping threshold 10, and 5,000 optimizer steps. Batches contain 256 data labels, 128 initial-state labels, 256 PDE points, and 32 boundary points per step. Data, initial-state, and PDE spatial sampling mix 50% central $|x|\le3$ points with 50% full-box points. The physics loss is

$$
L=L_{\rm data}+L_{\rm IC}+\frac{L_{\rm PDE}}{100^2}+0.1L_{\rm BC}.
$$

The PDE residual is computed by differentiating the raw real and imaginary network outputs in physical coordinates. The boundary term penalizes periodic value **and derivative** mismatch. No explicit norm penalty is enabled, and IC/BC are soft penalties rather than exact architectural conditions. The supervised comparison optimizes data + IC only; its saved configuration retains nominal physics weights, but `mode="supervised"` disables those terms in the implementation. No validation-based checkpoint selection or early stopping is used.

**P2 — loss components.** The preserved figure displays raw data, IC, PDE, and boundary losses on logarithmic axes; raw PDE values must be divided by 10,000 before interpreting their contribution to the total. Disabled supervised terms have no optimized contribution. A smaller residual by itself does not imply a more accurate solution.

![P2 raw training loss components](output/project3/p2_loss_components.png)

**P3 — accuracy versus difficulty.** For each held-out control we compute $\|\hat\psi-\psi\|_2/\|\psi\|_2$ over the complete uniform space–time grid, including complex phase and the full box. No prediction renormalization or phase alignment is applied. Table entries are arithmetic means of six individual control errors, not an error of the mean field.

| Final trap $u_T$ | Physics-informed mean error | Supervised mean error | Test controls |
|---|---:|---:|---:|
@@SWEEP@@

![P3 difficulty sweep with individual test controls](output/project3/p3_error_vs_knob.png)

The dot spread is variation among controls, not uncertainty over training seeds. Neither approach achieves a demonstrated useful-accuracy threshold. Physics-informed predictions also have large mass errors; at $u_T=100$, mean maximum absolute mass error is 0.9725. We retain this failure rather than hide it by normalizing predictions.

**P4 — error location and spatial frequencies.** The field plot selects the worst physics-informed test control at $u_T=100$ and displays the occupied center for readability. The right panel is the time-averaged power spectrum of the **complex prediction error over the full box**, not the spectrum of density or the reference solution. These plots locate the failure but do not establish spectral bias as its cause.

![P4 space–time error and Fourier spectrum of the error](output/project3/p4_error_field_and_spectrum.png)

**P5 — measured cost, separated from predictive quality.** The frozen hardware record is Darwin arm64, 10 logical CPUs, CPU execution with two PyTorch threads; the local machine is an Apple M2 Pro. Versions: Python 3.9.6, NumPy 2.0.2, SciPy 1.13.1, PyTorch 2.8.0. Timings are single-machine observations, not researcher benchmarks or guaranteed Colab timings.

| Model | Training (s) | Median full-field inference (s) | Arithmetic break-even queries |
|---|---:|---:|---:|
@@COST@@

The conventional median solve takes **0.060973 s**. The saved upfront reference-data charge is **7.576307 s** (family propagation alone: 7.484056 s). Break-even uses

$$
Q_* = \frac{t_{\rm training}+t_{\rm reference\ data}}
{t_{\rm PDE}-t_{\rm inference}}.
$$

This compares time for complete 129×256 field queries. It excludes hyperparameter search, failed runs, the extra diagnostic experiments, reporting, and repeated control verification. Values above are continuous arithmetic estimates (round up for an integer query count). Because accuracy is poor, these are **not accuracy-qualified speedups or a recommendation to replace the solver**. The researcher's actual query demand is unknown.

The learned search used 232 optimization predictions plus one final assessment, followed by one numerical verification. Its SciPy status is `ABNORMAL_TERMINATION_IN_LNSRCH`. The selected network prediction had raw mass **0.0206494**; the exploratory selection criterion used normalized fidelity and an extra mass penalty, so it is not an identical-objective optimization comparison. Only its PDE-verified $F$ and $J$ in Section 3 are physical results. Comparing its short selection time against the successful conventional optimization would be misleading.

## 5. Controlled Path B diagnosis

### Existing evidence before the new ablation

The [known-trajectory diagnostic](output/diagnostic/summary.json) uses the quick dataset's hardest **training** ramp, a smaller width-64/latent-48 model (19,282 parameters), and 5,000 steps. Its 4,224 fitting points are an alternating space–time grid; the remaining 12,416 points belong to the same trajectory. The physics-informed model's fitting-grid error is 43.03%, with full-trajectory error **@@SINGLE_PHYSICS@@%**. The supervised values are 30.85% and **@@SINGLE_SUPERVISED@@%**. Thus the failure is not solely an inability to generalize to unseen controls: the setup struggles even on labels for a known trajectory. This experiment changed PDE and BC penalties together and could not separate their effects.

![Known-training-trajectory diagnostic](output/diagnostic/single_training_control_diagnostic.png)

The [longer supervised experiment](output/diagnostic/capacity_summary.json) uses that same smaller architecture and known ramp, now with all 16,640 grid labels. Within its fixed 50,000-step run, error drops from **@@LONG_5K@@% at 5,000 steps to @@LONG_50K@@% at 50,000 steps**, costing 77.639 s. It still misses its predeclared 5% reference. This shows training-budget sensitivity for this known function. It does not prove adequate capacity for the full control family. Compared with the earlier sparse-label diagnostic, both label density and budget differ; only points on the longer run's own learning curve isolate additional training steps.

![Longer supervised learning curve for one known training trajectory](output/diagnostic/capacity_learning_curve.png)

### New four-way loss ablation: what changes and what is held fixed

We ran all four conditions on the **frozen full dataset** with the full 39,650-parameter architecture. The only treatment is the PDE/BC loss weight. Data and IC weights remain one; norm weight remains zero.

| Condition | $L_{data}$ | $L_{IC}$ | Raw $L_{PDE}$ multiplier | $L_{BC}$ multiplier |
|---|---:|---:|---:|---:|
| A: Data + IC | 1 | 1 | 0 | 0 |
| B: Data + IC + BC | 1 | 1 | 0 | 0.1 |
| C: Data + IC + PDE | 1 | 1 | 0.0001 | 0 |
| D: Data + IC + BC + PDE | 1 | 1 | 0.0001 | 0.1 |

All conditions share seed 598, complete control splits, parameter initialization, every sampled array, Adam, clipping, and 5,000 steps. Initialization and random-stream hashes match exactly across the four conditions; the training sampler records 90,000 calls per condition. Raw PDE/BC diagnostics are computed even when disabled, allowing all loss components to be recorded. This intentionally adds unused diagnostic work to A/B, so the ablation runtimes are **not** efficiency comparisons with the original supervised run. The full ablation finished in **@@ABLATION_SECONDS@@ s**, below the predeclared 300 s CPU budget.

| Condition | Mean test error, $u_T=5$ | $u_T=25$ | $u_T=100$ | Training with diagnostics (s) |
|---|---:|---:|---:|---:|
@@ABLATION@@

A reproduces the frozen supervised test errors exactly; D reproduces the frozen physics-informed errors exactly. These are deterministic reproductions, not independent replicates. Validation and test per-control complex L2, density L2, mass error, and terminal mass are in the [new experiment records](output/loss_ablation_4way/results.json), alongside CSV logs and checkpoints.

![Four-way ablation: predictive error and raw mass error](output/loss_ablation_4way/ablation_errors.png)

The paired differences below subtract each control's base-condition error from its added-penalty error, then average. Units are **percentage points**; positive means worse. The final column counts worsened controls out of six at each knob. These are descriptive comparisons for one fixed seed, not significance tests.

| Added penalty and comparison | $u_T=5$ | $u_T=25$ | $u_T=100$ | Worse count at 5 / 25 / 100 |
|---|---:|---:|---:|---|
@@EFFECTS@@

![Separated effects of adding BC and PDE penalties](output/loss_ablation_4way/ablation_effects.png)

Adding the PDE penalty worsens all six harder controls in both comparisons at $u_T=25$ and $100$. At $u_T=100$ it increases mean error by roughly **18 percentage points**, whether or not BC is present. Boundary effects are small and mixed: they do not support BC as the principal cause of this observed accuracy gap. The easy-knob PDE effects are also mixed, so we do not assert that physics penalties always hurt.

The last logged minibatch losses are below. They are raw MSE diagnostics, not full-grid averages; compare total contributions using the multipliers above. At the same step budget, adding PDE lowers the measured residual while increasing data error. This supports a tradeoff in the selected objective, not a proof of conflicting gradients.

| Condition | Data | IC | PDE | BC |
|---|---:|---:|---:|---:|
@@LOSSES@@

![Raw loss histories in every ablation condition](output/loss_ablation_4way/ablation_loss_components.png)

### Supported mechanisms and unresolved alternatives

| Candidate explanation | Evidence and conclusion |
|---|---|
| More severe confinement exposes failure | Supported by the preserved held-out difficulty sweep for this family. |
| Generalization alone explains failure | Insufficient: the known-trajectory fitting labels also have large errors at 5,000 steps. |
| PDE loss contributes to harder-case underfitting | Supported at this residual scaling, architecture, optimizer, seed, and step budget by C−A and D−B. |
| Boundary penalty is the main harmful term | Not supported: B−A and D−C are much smaller and change sign. Boundary accuracy itself can improve without improving field accuracy. |
| More training can improve fitting | Supported within the 50,000-step supervised learning curve for one known trajectory. It is not a demonstrated operator remedy. |
| Architecture has an absolute capacity ceiling | Unproven. Training and representation limitations are not separated. |
| Spectral bias, NTK imbalance, bad conditioning, gradient conflict, or lack of temporal causality is the unique root cause | Unproven: we did not measure frequency-learning rates, kernel/gradient spectra, gradient alignment, or run controlled causal/conditioning interventions. |
| Reference discretization or residual-sign error explains the entire gap | Existing refinement and analytic residual tests argue against these as the sole explanation for the tested cases, but they are not an exhaustive proof of implementation correctness. |

The operational diagnosis is therefore **a loss-dependent fitting failure under the available training budget**. Prior PINN work discusses conditioning-related failures, but that literature is context, not evidence that our experiment has the same lower-level cause. See the [primary failure-modes study](https://arxiv.org/abs/2109.01050).

Limits: one training seed, one budget, one weighting, three difficulty levels, and six test controls per level. This follow-up was motivated by test errors already inspected in the original project; it is exploratory rather than a fresh confirmatory holdout. Test labels do not enter training, early stopping, or checkpoint choice. No weight sweep, new seed study, gradient diagnosis, trained capacity sweep, or successful mitigation has been completed. We do not claim a universal or unique root cause, and the instructor must judge whether this bounded operational diagnosis meets Path B's mechanism requirement.

## 6. Evidence integrity and P1–P5 coverage

The authoritative original run is [output/project3/results.json](output/project3/results.json), profile `full`, seed 598, 5,000 steps. [Verification](output/submission_checks/verification/verification.json) reloads both saved checkpoints, recomputes every held-out error and mass metric, and checks the CSVs and cost arithmetic. The largest metric discrepancy is **0.0**. All five main figures, including the post-control figure, regenerate **pixel-for-pixel identically** in a separate directory. No original figure or measurement was replaced.

The [final integrity checks](output/submission_checks/final_checks.json) also reload all four new ablation checkpoints and reproduce all 36 validation/test records per condition exactly, including density and mass metrics; they check every logged loss against its declared weights. A local rehearsal containing **only the 65 allowlisted files** passes the same 13 tests and reproduces the frozen main metrics and figures using regenerated references. The [reference-regeneration comparison](output/submission_checks/dataset_regeneration.json) confirms every full and quick dataset array is exactly equal to its preserved original on this machine. This rehearsal is not a GitHub upload.

| Requirement | Evidence | Status and limit |
|---|---|---|
| P1: numerical baseline accuracy and runtime | Section 3; refinement JSON, optimized-versus-ramp figure, post-hold check | Measured, with estimated rather than rigorous error bounds |
| P2: separate losses on log axes | Section 4 loss figure and original CSVs; ablation raw components | Measured; raw versus weighted residual is explained |
| P3: error versus difficulty | Three-knob table and preserved scatter/mean figure | Measured on entire held-out controls, one seed |
| P4: space–time error and error spectrum | Preserved worst-case complex-error figure | Measured; spectrum is not a cause diagnosis |
| P5: training/inference cost, hardware, break-even | Section 4, frozen JSON and verified arithmetic | Measured; no accuracy-qualified replacement benefit |

All three notebooks, the original simulation/training modules, and pre-existing numerical outputs remain untouched. A later separate `Baseline_Improved` notebook uses a different seven-interior-control family; its higher fidelity is **not substituted** into this report's three-interior-control comparison. Older quick/smoke results and older generated report drafts are not mixed with the frozen full run. [Publication audit](PUBLICATION_AUDIT.md) identifies the proposed public subset without deleting anything.

## 7. Reproducibility, assumptions, and remaining work

Use the exact paths in [SUBMISSION_FILES.txt](SUBMISSION_FILES.txt) for the proposed repository. The roughly 56 MB full reference dataset is synthetic and deliberately omitted from that lean public subset; it can be regenerated without retraining. All commands below run from the repository root and use **new output directories**. Do not run the old notebook/report generators over verified work.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v

# Regenerate only the full reference family (no neural training).
python prepare_reference_data.py --output-dir output/reproduced_references

# Check the frozen checkpoints, metrics, and figures against those references.
python verify_path_b.py --dataset output/reproduced_references/reference_dataset.npz \
  --output-dir output/reproduced_verification

# Reproduce the four-way experiment without replacing this submission's run.
python run_loss_ablation.py --source output/reproduced_references/reference_dataset.npz \
  --output-dir output/reproduced_ablation --steps 5000 --seed 598 --budget-seconds 300

# Optional: repeat the full baseline and two-model experiment in a fresh directory.
python run_project3.py --profile full --seed 598 --output-dir output/reproduced_full
python validate_controls.py output/reproduced_full/results.json

# Optional: reproduce both known-training-trajectory diagnostics safely.
python reproduce_legacy_diagnostics.py --output-dir output/reproduced_legacy
```

The 300 s ablation budget is machine dependent; a slower CPU may stop before completion. An interrupted run is not a matched result. Use another fresh directory and explicitly increase `--budget-seconds` if necessary; do not change only one condition's step count. PyTorch/FFT versions and hardware can change numerical values and timings. Dependencies are bounded in `requirements.txt`; the tested versions are recorded above. This code is ordinary NumPy/SciPy/PyTorch CPU Python and can run from Colab after dependencies and repository files are available, but **Colab itself and GPU execution were not tested in this review**.

Local checks: **13 tests passed**, zero failures/errors, in **@@TEST_SECONDS@@ s** wall time including interpreter startup; [actual test log](output/submission_checks/unit_tests.txt) and [structured result](output/submission_checks/unit_tests.json). These cover nonlinear stationary states, complex Fourier phase, norm conservation, time refinement, fidelity phase handling, fixed control endpoints, analytic residuals, split validation, and physical-time inference. Checkpoint/figure verification and exact regenerated-dataset comparisons are additional evidence, not extra unit-test counts. No new population-level accuracy or runtime claim is inferred from tests.

Remaining assumptions: dimensionless 1D mean-field GPE; harmonic trap; fixed $g$, initial state and final time; finite periodic box; limited control family and range; finite post-control observation window; no application-specific accepted accuracy target. There is no guarantee that the researcher would use this discretization, network, control range, objective weight, or query workload.

For the deadline, freeze this evidence and submit the bounded negative finding. Next scientific work, after submission, should repeat the ablation across seeds, sweep residual weights on training/validation controls, measure component gradient magnitudes/alignment, and test a selected intervention on fresh controls. Until an accurate operator survives numerical control verification, retain the conventional solver for decisions.

The team's own actions still required are to verify author/course metadata, resolve publication permission for named interview material, fill outreach details only if documented, request written-brief feedback, decide whether to send the memo, create/review the public GitHub repository from the allowlist, and submit the repository/report through the course's required channel. Researcher feedback and any memo delivery must be dated truthfully when they occur. No outreach or publication has been performed by this preparation step.

## 8. One-page recommendation memo — unsent draft

**To:** Dr. Jimmie Adriazola  
**From:** MAE 598 Project 3 team  
**Date:** October 20, 2026  
**Subject:** Retain the numerical GPE solver; investigate the current physics-loss tradeoff

We investigated your proposed comparison between conventional GPE control and a learned control-to-solution operator, using wave-packet squeezing. We recommend retaining the conventional solver for optimization decisions. The current physics-informed DeepONet is too inaccurate to support a replacement claim.

Our dimensionless 1D test uses a harmonic trap, fixed endpoints, three adjustable interior control values, and normalized nonlinear ground states. The conventional split-step Fourier method passed time/grid/domain checks. At final trap strength 100, local optimization improved fidelity from 0.8342 to 0.9450 and reduced the regularized objective from 0.2789 to 0.2530. A one-unit hold showed about 47% less width oscillation than the ramp, although the packet was not stationary.

The physics-informed operator's mean errors on held-out control functions were 27.24%, 56.09%, and 74.28% at final trap strengths 5, 25, and 100. A same-architecture supervised model performed better on the harder cases. The learned control candidate failed its numerical check: its verified objective was worse than the ramp. Fast evaluation alone therefore does not demonstrate useful acceleration.

A matched four-way ablation held the network, data, initialization, sampled points, optimizer, and 5,000-step budget fixed. Adding the PDE penalty increased the hardest-case mean error by about 18 percentage points, with or without the boundary penalty. Boundary effects were small and mixed. The current physics-loss formulation contributes to fitting failure at this budget, but we have not established whether weighting, optimization, representation, or conditioning explains that effect. One seed cannot establish a unique general cause.

Additional supervised training improved a known trajectory's error from 30.86% at 5,000 steps to 7.63% at 50,000, so the short run does not prove an absolute capacity limit. This was a fitting diagnostic, not successful generalization across controls.

We suggest first testing residual-weight choices and component gradients, then repeating across seeds and verifying any selected fix with fresh controls and the conventional solver. Before assessing practical speedup, we would value your actual query workload, runtime/hardware benchmark, and acceptable fidelity/stability tolerances; these remain unspecified. We also request corrections to our unreviewed written brief. We have not yet tested splitting.

**Delivery status:** draft only; no email sent and no response or written-brief approval claimed.
'''
    for key, value in replacements.items():
        report = report.replace('@@' + key + '@@', value)
    assert '@@' not in report
    destination.write_text(report)
    print(f"Built {destination} from saved measured evidence")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='REPORT.md')
    parser.add_argument('--replace', action='store_true')
    args = parser.parse_args()
    build(args.output, args.replace)
