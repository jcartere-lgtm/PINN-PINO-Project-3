# Project 3 requirement checklist

| Assignment requirement | Status | Where / caveat |
|---|---|---|
| Researcher identified / interviewed | **Completed, team-reported** | `REPORT.md` §1 and `RESEARCHER_CONTEXT.md`, interview Oct 16 |
| Outreach log / researcher reservation evidence | **Partial** | Initial contact, reservation evidence not supplied; do not invent |
| Problem brief, PDE, computational bottleneck, practical value | **Completed with unknowns** | `REPORT.md` §2; researcher's exact runtime/query count/accuracy needs unknown |
| Researcher review of written brief | **Missing** | Researcher suggested the experiment orally but has not approved brief |
| Minimal 1-D case, dimensionality, IC/BC, control family, difficulty knob | **Completed** | `REPORT.md` §2; `u_T` in 5, 25, 100 |
| P1: accurate conventional solver, time/mesh/box checks and wall time | **Completed** | `REPORT.md` §3, P1 JSON, independent integration check |
| P2: separate PDE / BC / IC / data losses | **Completed** | `REPORT.md` §4, `output/project3/p2_loss_components.png`, CSVs |
| P3: held-out complex relative L2 vs knob | **Completed** | `REPORT.md` §4, `output/project3/p3_error_vs_knob.png` |
| P4: space–time pointwise complex error and Fourier error spectrum | **Completed** | `REPORT.md` §4, `output/project3/p4_error_field_and_spectrum.png` |
| P5: training, data-generation, inference, hardware and break-even | **Completed, with caveat** | `REPORT.md` §4, stored full run; mathematical break-even **not accuracy-qualified** |
| Path B: diagnosed limitation, matched controlled experiment | **Completed at tested setting** | Four-way data+IC ± PDE ± BC ablation; one seed, not unique low-level mechanism |
| Path C remedy | **Not claimed** | No proven fix; Path B appropriate |
| Public reproducible code, fixed seeds, tests | **Completed locally** | `python -m unittest discover -s tests -v`; `integration_validation.py` |
| Single GitHub-renderable report and figures | **Prepared, not hosted/visually verified on GitHub** | `REPORT.md`, tested local images/links |
| One-page recommendation memo addressed to researcher | **Draft prepared; delivery missing** | `REPORT.md` §8; send and record only real response |
| Researcher attribution/publication consent | **Unconfirmed** | Confirm or anonymize interview material **before public release** |
| Final GitHub repository URL / Canvas submission | **Missing** | Requires team's action; no remote actions executed |

The course rubric gives Path B full base-credit potential but only if its diagnosis is supported by controlled evidence. The numerical evidence and caveats are preserved in the single report; a negative result is not a failure of the assignment.
