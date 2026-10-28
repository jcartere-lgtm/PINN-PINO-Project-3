# Publication audit — integrated review (October 28, 2026)

**Integration update:** The original 65-file list described below was superseded by the new `SUBMISSION_FILES.txt` generated from this separate integrated directory. New files are `INTEGRATION_AUDIT.md`, `CONTRIBUTIONS.md`, `ASSIGNMENT_CHECKLIST.md`, `integration_validation.py`, and `integration_evidence/*`. No original upload was modified; publication permission is still unconfirmed. The old statements about 65 files apply to the *source package*, not the integrated folder.


**No files have been deleted, moved out of the project, or published.** All existing notebooks and verified numerical outputs remain local and unchanged. `REPORT.md` is the single final report. `SUBMISSION_FILES.txt` is the exact proposed public allowlist (65 files); it includes the original simulation/training modules, tests, preserved full-run evidence, prior diagnostics, and the new controlled ablation. It does not mean researcher permission has been obtained.

## Include in the proposed public submission

- Root report and README; documented researcher context with unknowns explicit; requirements and tested versions.
- GPE/DeepONet runners and tests, verification, safe reference regeneration, safe legacy-diagnostic reproduction, and report/audit scripts.
- Frozen full-run JSON/CSVs, two small checkpoints, and five verified figures.
- Known-trajectory diagnostic summaries, losses, and figures.
- All four ablation checkpoints, losses, metrics, summary, and figures.
- Numerical control validation with only its absolute account path replaced; provenance hashes record that single transformation.
- Actual unit-test log, metric/figure checks, regeneration comparison, preservation snapshot, and final checks.

The allowlist is authoritative. A public repository assembled from it can regenerate references without uploading raw synthetic datasets. `audit_publication.py` produces local-only inventory files; its local inventory/audit are excluded from publication. README's earlier text is retained as `output/submission_checks/README_before_path_b.md`; only the entry-point documentation was revised.

## Proposed exclusions — leave these on disk

| Local item | Reason to omit from the lean public subset |
|---|---|
| All three `DesignOptProj3*.ipynb` files | Preserved development history, duplicated embedded code/results, and possible cached account paths. The root Markdown report plus Python sources form this submission. No notebook is modified. |
| `output/project3/reference_dataset.npz` | About 55.8 MB of regenerable synthetic reference fields; not necessary to commit. |
| `output/quick_pilot/`, `output/smoke/` | Older pilot/smoke experiments; not the canonical P1–P5 evidence. Quick references can be regenerated for diagnostics. |
| `output/baseline_improved/` and `baseline_*`, `build_baseline_improved.py`, `README_Baseline_Improved.md` | Separate control-family experiment, preserved locally; prevent mixing its fidelity with the three-interior-control comparison. |
| `Project3_ReviewReady.zip` | Stale duplicate archive that predates the final ablation/report; do not upload alongside a second version of the evidence. |
| Old generated `EXPERIMENT_REPORT.md`, `RECOMMENDATION_DRAFT.md`, notebook builders/executors, `finalize_report.py` | Historical drafts and generators, superseded as submission entry points. Old commands can replace verified outputs. |
| Original `output/project3/control_validation.json` | Contains an absolute account path. Publish the numerically identical relative-path copy in `submission_evidence/` instead. |
| Regenerated NPZs, recreated figure copies, README backup, `.DS_Store`, `.mplconfig`, `__pycache__`, `.ipynb_checkpoints`, local inventory/audit | Temporary or duplicate review artifacts. Keep locally; do not stage. |

Largest excluded files in the current inventory:

| File | Size (decimal) |
|---|---:|
| `output/project3/reference_dataset.npz` | 55.82 MB |
| `output/submission_checks/full_reference_regeneration/reference_dataset.npz` | 55.82 MB |
| `output/quick_pilot/reference_dataset.npz` | 8.42 MB |
| `output/smoke/reference_dataset.npz` | 8.42 MB |
| `output/submission_checks/quick_reference_regeneration/output/quick_pilot/reference_dataset.npz` | 8.42 MB |
| `output/baseline_improved/trajectories.npz` | 5.96 MB |
| `Project3_ReviewReady.zip` | 3.55 MB |
| `DesignOptProj3.ipynb` | 1.44 MB |
| `DesignOptProj3_Baseline_Improved.ipynb` | 0.87 MB |
| `output/baseline_improved/results.json` | 0.26 MB |
| `DesignOptProj3_original.ipynb` | 0.24 MB |
| `output/submission_checks/verification/recreated_figures/post_control_breathing.png` | 0.23 MB |

## Sensitive-material and duplication review

We scanned text, JSON, CSV, Python and notebook files for absolute account paths, temporary paths, email-like strings, private-key headers and credential assignments. Matching locations below require review; a pattern match is not proof of a secret. The scanner does not establish publication consent or legal rights.

| Relative file | Pattern counts | Proposed action |
|---|---|---|
| `DesignOptProj3.ipynb` | absolute_account_path: 1 | exclude |
| `execute_corrected_notebook.py` | temporary_absolute_path: 1 | exclude |
| `output/project3/control_validation.json` | absolute_account_path: 1 | exclude |

The original control-validation path is not changed in place. Its sanitized copy retains every numerical field and records both hashes. Excluded recreated validation output may also contain local absolute paths. The archive and notebooks should remain excluded even if a simple text scan misses embedded or archived private material. No claim of a comprehensive forensic security audit is made.

Byte-identical local files include `baseline_gpe.py` and `project3_gpe.py`; reference-regeneration checks create additional intentional duplicates. The machine-readable local audit lists all duplicate groups and archive sizes without removing them. Retain the canonical `project3_gpe.py` in the public subset. Small ablation A/D checkpoints reproduce the original model metrics, but are retained as separate experiment provenance.

## What the team must do

1. Read `REPORT.md` and confirm team names, course metadata, and the October 16 interview account. Fill unknown outreach details only from actual records.
2. Confirm permission/attribution for public named interview material; if unavailable, decide how to anonymize it before publication. Written-brief approval, memo delivery, and researcher response remain unclaimed.
3. Review the exact allowlist. Create a clean staging folder or Git repository from **only** those paths; do not upload the working folder or old ZIP wholesale. No Git repository was initialized by this preparation step.
4. The existing `.gitignore` ignores full-run checkpoints. After repository initialization, an explicit allowlist command can stage the intended small checkpoint files: `git add -f --pathspec-from-file=SUBMISSION_FILES.txt`. Inspect `git diff --cached --stat` and the staged contents before committing. This command is a team action; it has not been run here.
5. Create/push the GitHub repository, verify images render, and submit the required repository/report link through the course channel. Check any presentation or other course deliverables yourself. No email, repository publication, or course submission has been performed.

Any later cleanup/deletion needs a separate decision. This audit proposes publication exclusions only.
