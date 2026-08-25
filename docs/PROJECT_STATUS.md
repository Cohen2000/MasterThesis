# Project status and repository guide

Last updated: **2026-08-25**

## Short version

This is an active research repository, not a polished software package. The
current thesis track is the V2 estimator benchmark plus the V2.1/final-panel LLM
evaluation. Earlier pilots and exploratory branches are preserved because they
document how the design changed and because some later artifacts depend on
their schemas.

If you are new to the project:

1. read the root [`README.md`](../README.md);
2. use [`TARGET_EVALUATION_FREEZE.md`](TARGET_EVALUATION_FREEZE.md) for the
   scientific target and scoring rules;
3. use [`README_V2.md`](README_V2.md) for the estimator benchmark;
4. use [`LLM_EVIDENCE.md`](LLM_EVIDENCE.md) for measured LLM-related evidence;
5. treat everything else as supporting detail unless linked from those files.

## Stable or frozen components

- Temporal-network generators, walk/retrieval mechanisms, feature extraction,
  and leakage-aware benchmark evaluation under `src/`.
- V2 benchmark presets and dataset registry under `config/`.
- The 32-graph target-panel selection.
- The dyadic active-window survival profile and its primary ProfileMAE scoring
  rule.
- The invalid-output policy: raw LLM predictions are never clipped, sorted,
  imputed, or silently repaired.
- Historical prompt/answer artifacts explicitly described as frozen in the
  design documents.

“Frozen” does not mean the complete thesis experiment is finished. The target
document lists the remaining gates that must close before the final comparison
is immutable.

## Active work

- Analysis of the six-cell one-factor-at-a-time LLM input ablation across Codex,
  Gemini, DeepSeek V4 Flash, and Qwen3.6-27B. Generation is finished; the
  paired comparison is computed but not yet written up.
- Validation of response noise, input sensitivity, and model/harness effects.
- Synthesis of classical estimators, supervised baselines, and LLM results on
  matched cases.
- Deciding which exploratory non-walk findings belong in the thesis body versus
  an outlook section.

Active answer files are append-only and may be incomplete while jobs are
running. Counts in runbooks and logs are operational snapshots, not final
reported sample sizes.

## Historical and legacy material

The following are intentionally retained but are not the recommended starting
point:

- early `main_run1`, pilot, and Phase-3 artifacts;
- older model/API runners superseded by `src/run_llm_v2.py`;
- obsolete or optional Slurm files for model configurations that were dropped;
- `backup_before_v21/` and `resume_fix.patch`;
- historical `.log`/`.pid` files already tracked before the current ignore
  policy;
- environment-specific commands containing old local or cluster paths.

Legacy does not necessarily mean incorrect. It means the file belongs to an
earlier experiment, a provenance record, or an optional branch and should not
be composed into the current pipeline without checking its assumptions.

Nothing in this classification is a cleanup request. Files are kept until an
archival release can separate current code, reproducibility artifacts, and
historical provenance without losing information.

## Generated data and results

- Raw third-party datasets under `data/raw/` and the local literature PDFs under
  `literature/papers/` are excluded from the public repository. They may still
  exist in a researcher's local working tree.
- `results/` is ignored because model answers, case tables, and evaluation
  artifacts can be large or actively changing.
- Compact historical reports and summaries already selected for provenance may
  remain versioned; large generated tables and answer files do not.
- A fresh clone therefore does not necessarily contain every local artifact
  referenced by an operational runbook.
- Secrets belong in environment variables. No API credential should be stored
  in a prompt, answer record, script, or committed environment file.

## Which command should I run?

| Goal | Command or entry point |
|---|---|
| run unit/invariant tests | `PYTHONPATH=src python -m unittest discover -s tests` |
| run the synthetic smoke pipeline | `bash scripts/run_benchmark_v2_smoke.sh` |
| inspect required real datasets | `python src/check_real_data.py --preset v2` |
| run the resumable V2 laptop benchmark | `SHARDS=12 JOBS=1 RESET=0 bash scripts/run_benchmark_v2_local.sh` |
| understand current LLM evidence | [`LLM_EVIDENCE.md`](LLM_EVIDENCE.md) |
| inspect the active OFAT workflow | [`LLM_OFAT_RUNBOOK.md`](LLM_OFAT_RUNBOOK.md) |
| inspect cluster jobs | the relevant runbook plus `slurm/`; do not run model inference on a login node |

Do not launch API or cluster jobs merely to test installation. Unit tests and
the synthetic benchmark smoke use no external model API.

## Known public-release gaps

- The thesis does not yet have an archival DOI; repository-level citation
  metadata is provided in [`../CITATION.cff`](../CITATION.cff).
- Raw third-party datasets and literature PDFs are not distributed. Source
  terms still need to be checked by anyone obtaining them independently; see
  [`THIRD_PARTY.md`](THIRD_PARTY.md).
- Some historical documentation is in German or contains machine-specific
  examples.
- There is no promise of a stable Python API or semantic versioning before the
  thesis release.
- Generated result bundles are not yet organized into a single archival
  release.

These gaps are documented so that public visibility is not mistaken for a
finished reproducibility package.
