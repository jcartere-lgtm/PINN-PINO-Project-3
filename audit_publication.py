"""Create a publication allowlist and inventory; never delete project files."""
import csv
import hashlib
import json
from pathlib import Path
import re
import zipfile


ROOT = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit():
    evidence = ROOT / "submission_evidence"
    evidence.mkdir(exist_ok=True)
    original = ROOT / "output/project3/control_validation.json"
    control = json.loads(original.read_text())
    safe_control = dict(control, source_results="output/project3/results.json")
    safe_path = evidence / "control_validation.json"
    safe_path.write_text(json.dumps(safe_control, indent=2) + "\n")
    checked = json.loads(safe_path.read_text())
    assert {k:v for k,v in checked.items() if k!='source_results'} == {
        k:v for k,v in control.items() if k!='source_results'}
    (evidence / "control_validation_provenance.json").write_text(json.dumps(dict(
        original_file="output/project3/control_validation.json", original_sha256=digest(original),
        public_file="submission_evidence/control_validation.json", public_sha256=digest(safe_path),
        transformation="Only source_results changed from an absolute account path to a relative repository path. No numerical field changed."), indent=2) + "\n")

    public = [
        ".gitignore", "README.md", "REPORT.md", "PUBLICATION_AUDIT.md", "SUBMISSION_FILES.txt",
        "RESEARCHER_CONTEXT.md", "requirements.txt", "requirements-tested.txt",
        "project3_gpe.py", "project3_operator.py", "run_project3.py", "validate_controls.py",
        "diagnostic_operator.py", "capacity_diagnostic.py", "run_loss_ablation.py",
        "prepare_reference_data.py", "reproduce_legacy_diagnostics.py", "verify_path_b.py",
        "build_path_b_report.py", "audit_publication.py",
        "tests/test_project3.py", "tests/test_operator.py",
        "submission_evidence/control_validation.json", "submission_evidence/control_validation_provenance.json",
        "output/project3/results.json", "output/project3/p1_reference_validation.json",
        "output/project3/reference_controls.csv", "output/project3/physics_loss_history.csv",
        "output/project3/supervised_loss_history.csv", "output/project3/p3_p5_heldout_errors.csv",
        "output/project3/physics_operator.pt", "output/project3/supervised_operator.pt",
        "output/project3/p2_loss_components.png", "output/project3/p3_error_vs_knob.png",
        "output/project3/p4_error_field_and_spectrum.png", "output/project3/control_comparison.png",
        "output/project3/post_control_breathing.png",
        "output/diagnostic/summary.json", "output/diagnostic/capacity_summary.json",
        "output/diagnostic/physics_loss_history.csv", "output/diagnostic/supervised_loss_history.csv",
        "output/diagnostic/single_training_control_diagnostic.png", "output/diagnostic/capacity_learning_curve.png",
        "output/loss_ablation_4way/results.json", "output/loss_ablation_4way/ablation_errors.png",
        "output/loss_ablation_4way/ablation_effects.png", "output/loss_ablation_4way/ablation_loss_components.png",
        "output/submission_checks/preserved_before_path_b.json",
        "output/submission_checks/verification/verification.json",
        "output/submission_checks/dataset_regeneration.json",
        "output/submission_checks/unit_tests.json", "output/submission_checks/unit_tests.txt",
        "output/submission_checks/final_checks.json",
    ]
    for name in "ABCD":
        public.extend(f"output/loss_ablation_4way/{name}_{suffix}" for suffix in
                      ["loss_history.csv", "metrics.csv", "model.pt"])
    public = sorted(public)
    (ROOT / "SUBMISSION_FILES.txt").write_text("\n".join(public) + "\n")
    listed = set(public)
    # Inventory only. Cache contents, archives and datasets remain on disk.
    files = sorted(p for p in ROOT.rglob('*') if p.is_file() and '.git' not in p.parts)
    rows, hashes = [], {}
    patterns = {
        'absolute_account_path': re.compile('/' + 'Users' + r'/[A-Za-z0-9._-]+/'),
        'temporary_absolute_path': re.compile(r'/private/(?:tmp|var)/'),
        'email_address': re.compile(r'[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}', re.I),
        'private_key_header': re.compile(r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----'),
        'credential_assignment': re.compile(r'''(?:api[_-]?key|access[_-]?token|password)\s*[=:]\s*["'][A-Za-z0-9_\-]{16,}["']''', re.I),
    }
    sensitive_hits = []
    suffixes = {'.py','.md','.json','.csv','.txt','.ipynb'}
    for path in files:
        relative = path.relative_to(ROOT).as_posix()
        size = path.stat().st_size
        sha = digest(path)
        hashes.setdefault(sha, []).append(relative)
        hits = {}
        if path.suffix in suffixes:
            content = path.read_text(errors='replace')
            for key, pattern in patterns.items():
                count = len(pattern.findall(content))
                if count:
                    hits[key] = count
            if hits:
                sensitive_hits.append(dict(file=relative, hits=hits, proposed_public=relative in listed))
        rows.append(dict(file=relative, bytes=size, sha256=sha,
                         proposed_public=relative in listed, pattern_counts=json.dumps(hits)))
    inventory = evidence / 'local_inventory.csv'
    with inventory.open('w',newline='') as stream:
        writer=csv.DictWriter(stream, fieldnames=['file','bytes','sha256','proposed_public','pattern_counts'])
        writer.writeheader(); writer.writerows(rows)
    duplicates = [paths for paths in hashes.values() if len(paths)>1]
    archives=[]
    for p in files:
        if p.suffix=='.zip':
            with zipfile.ZipFile(p) as archive:
                archives.append(dict(file=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,
                                     member_count=len(archive.infolist()),
                                     uncompressed_bytes=sum(x.file_size for x in archive.infolist())))
    public_hits=[h for h in sensitive_hits if h['proposed_public']]
    # The scanner definitions in this file are not matches for real account paths.
    scan = dict(deleted_files=[],inventory_files=len(rows),proposed_public_files=len(public),
                public_present_bytes=sum((ROOT/p).stat().st_size for p in public if (ROOT/p).is_file()),
                potential_sensitive_patterns=sensitive_hits, public_pattern_hits=public_hits,
                duplicate_groups=duplicates, archives=archives,
                caution="Pattern scan is not a guarantee of privacy, consent or publication rights. No files were deleted.")
    (evidence/'local_audit.json').write_text(json.dumps(scan,indent=2)+'\n')
    largest=sorted([r for r in rows if not r['proposed_public']],key=lambda r:r['bytes'],reverse=True)[:12]
    table='\n'.join(f"| `{r['file']}` | {r['bytes']/1e6:.2f} MB |" for r in largest)
    # Never reproduce potentially private strings; report only relative locations and counts.
    hit_table='\n'.join(f"| `{h['file']}` | {', '.join(k+': '+str(v) for k,v in h['hits'].items())} | {'include; review' if h['proposed_public'] else 'exclude'} |" for h in sensitive_hits
                        if 'recreated_figures' not in h['file'])
    report=f'''# Publication audit — proposed changes only

**No files have been deleted, moved out of the project, or published.** All existing notebooks and verified numerical outputs remain local and unchanged. `REPORT.md` is the single final report. `SUBMISSION_FILES.txt` is the exact proposed public allowlist ({len(public)} files); it includes the original simulation/training modules, tests, preserved full-run evidence, prior diagnostics, and the new controlled ablation. It does not mean researcher permission has been obtained.

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
{table}

## Sensitive-material and duplication review

We scanned text, JSON, CSV, Python and notebook files for absolute account paths, temporary paths, email-like strings, private-key headers and credential assignments. Matching locations below require review; a pattern match is not proof of a secret. The scanner does not establish publication consent or legal rights.

| Relative file | Pattern counts | Proposed action |
|---|---|---|
{hit_table or '| None | No matching patterns | — |'}

The original control-validation path is not changed in place. Its sanitized copy retains every numerical field and records both hashes. Excluded recreated validation output may also contain local absolute paths. The archive and notebooks should remain excluded even if a simple text scan misses embedded or archived private material. No claim of a comprehensive forensic security audit is made.

Byte-identical local files include `baseline_gpe.py` and `project3_gpe.py`; reference-regeneration checks create additional intentional duplicates. The machine-readable local audit lists all duplicate groups and archive sizes without removing them. Retain the canonical `project3_gpe.py` in the public subset. Small ablation A/D checkpoints reproduce the original model metrics, but are retained as separate experiment provenance.

## What the team must do

1. Read `REPORT.md` and confirm team names, course metadata, and the October 16 interview account. Fill unknown outreach details only from actual records.
2. Confirm permission/attribution for public named interview material; if unavailable, decide how to anonymize it before publication. Written-brief approval, memo delivery, and researcher response remain unclaimed.
3. Review the exact allowlist. Create a clean staging folder or Git repository from **only** those paths; do not upload the working folder or old ZIP wholesale. No Git repository was initialized by this preparation step.
4. The existing `.gitignore` ignores full-run checkpoints. After repository initialization, an explicit allowlist command can stage the intended small checkpoint files: `git add -f --pathspec-from-file=SUBMISSION_FILES.txt`. Inspect `git diff --cached --stat` and the staged contents before committing. This command is a team action; it has not been run here.
5. Create/push the GitHub repository, verify images render, and submit the required repository/report link through the course channel. Check any presentation or other course deliverables yourself. No email, repository publication, or course submission has been performed.

Any later cleanup/deletion needs a separate decision. This audit proposes publication exclusions only.
'''
    (ROOT/'PUBLICATION_AUDIT.md').write_text(report)
    print(json.dumps(dict(public_files=len(public),public_pattern_hits=public_hits,
                          inventory_files=len(rows),deleted_files=[]),indent=2))


if __name__=='__main__':
    audit()
