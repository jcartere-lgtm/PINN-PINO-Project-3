# Contribution and provenance map

This maps **uploaded artifacts**, not unverified student identities. Team members should replace these labels with their real names *only after checking ownership*.

| Artifact / idea | Source | Role in integrated project |
|---|---|---|
| Nonlinear GPE solver, stationary states, exact constrained control objective | `project3_working 2.zip`: `project3_gpe.py` | Canonical baseline and P1 reference |
| Complex physics-informed DeepONet, raw loss histories, supervised counterpart | `project3_working 2.zip`: `project3_operator.py`, `run_project3.py` | Canonical P2–P5 and within-model comparator |
| Controlled PDE/BC ablation and saved figures | `project3_working 2.zip`: `run_loss_ablation.py`, `output/loss_ablation_4way/` | Path B controlled mechanism test |
| Standalone supervised Fourier-style terminal-state operator, sinusoidal control basis, basic wave-packet teaching plots | `DesignOptProj3(1).ipynb` | Alternative exploratory design; documented, **not** substituted into common metrics |
| RMS wave-packet width diagnostic | Inspired by standalone Colab; newly implemented on the primary solver in `integration_validation.py` | Comparable supplemental physical plot and measured widths |
| Cross-check, portability fixes, curated consolidated documentation | Integration performed October 28, 2026 | Separate reproducible submission; originals unchanged |

Earlier copies of the modular project and the PDF screenshots are preserved as source uploads, not treated as additional independent experiments. See [INTEGRATION_AUDIT.md](INTEGRATION_AUDIT.md).
