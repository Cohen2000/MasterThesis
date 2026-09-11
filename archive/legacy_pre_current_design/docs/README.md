# Documentation index

This page is the public navigation layer for the repository. Documents preserve
the state in which decisions and runs were made, so an older runbook can be
historically correct without being the recommended entry point today.

For the current boundary between stable, active, and legacy work, read
[`PROJECT_STATUS.md`](PROJECT_STATUS.md) first.

## Core design and benchmark

| Document | Role | Status |
|---|---|---|
| [`README_V2.md`](README_V2.md) | expanded estimator benchmark, access mechanisms, targets, and scale | current benchmark reference |
| [`README_BENCHMARK.md`](README_BENCHMARK.md) | original benchmark rationale and execution | retained background; partly predates V2 |
| [`TARGET_EVALUATION_FREEZE.md`](TARGET_EVALUATION_FREEZE.md) | target hierarchy, scoring, invalid-output policy, and remaining freeze gates | authoritative for evaluation |
| [`DESIGN_BASELINES.md`](DESIGN_BASELINES.md) | design-aware analytical and supervised comparisons | current supplementary methods |
| [`V2_VALIDATION.md`](V2_VALIDATION.md) | recorded V2 validation and regression checks | historical validation snapshot |
| [`CHANGES_V2.md`](CHANGES_V2.md) | concise implementation delta from the earlier benchmark | historical change log |

## Evidence and experimental design

| Document | Role | Status |
|---|---|---|
| [`SEED_VS_GRAPH_EVIDENCE.md`](SEED_VS_GRAPH_EVIDENCE.md) | measured value of more graphs versus more walk seeds | completed design study |
| [`LLM_EVIDENCE.md`](LLM_EVIDENCE.md) | consolidated input/model evidence and its limits | main evidence map; evolves with analysis |
| [`LLM_NOISE_PROBE.md`](LLM_NOISE_PROBE.md) | preregistered response-noise versus input-noise design | design record |
| [`LLM_NOISE_RESULT.md`](LLM_NOISE_RESULT.md) | first completed Gemini noise result | dated result snapshot, not the final cross-model summary |
| [`NONWALK_QWEN36_SCREEN.md`](NONWALK_QWEN36_SCREEN.md) | bounded Qwen screen of non-walk access mechanisms | exploratory |
| [`NONWALK_SCREEN_PREREG_2026-08-20.md`](NONWALK_SCREEN_PREREG_2026-08-20.md) | target-blind selection and analysis commitments for that screen | preregistration record |

## Model runs and operational runbooks

| Document | Role | Status |
|---|---|---|
| [`LLM_V21_RUNS.md`](LLM_V21_RUNS.md) | model matrix, sampling settings, escalation ladder, and execution history | mixed plan/log; consult dated status notes |
| [`LLM_OFAT_RUNBOOK.md`](LLM_OFAT_RUNBOOK.md) | six-cell LLM input ablation and resumable commands | generation complete; analysis open; German |
| [`RUN_V2_LINUX.md`](RUN_V2_LINUX.md) | full V2 laptop workflow | environment-specific example |
| [`../slurm/README_cluster.md`](../slurm/README_cluster.md) | early cluster pilot workflow | historical; not the current cluster entry point |

## Maintenance notes

| Document | Role | Status |
|---|---|---|
| [`MERGE_NOTES.md`](MERGE_NOTES.md) | provenance from an earlier code/result merge | maintainer history |
| [`THIRD_PARTY.md`](THIRD_PARTY.md) | data/PDF exclusion, source metadata, and licensing boundary | current public-release notice |
| [`../CLAUDE.md`](../CLAUDE.md) | detailed maintainer/agent constraints and frozen-artifact rules | internal working context, not a scientific overview |

## Naming caveats

- `v1`, `v2`, and `v2.1` refer to experiment generations, not packaged software
  releases.
- “Frozen” can apply to a target definition or artifact while the complete
  thesis experiment still has open gates.
- A filename containing `final` usually records the role intended at creation;
  use the dated status documents rather than the filename alone to infer
  current authority.
- Result reports are descriptive unless they explicitly state a prespecified
  inferential claim.
