# Offline acceptance report — main experiment

Canonical run: `results/main_experiment/frozen_20260916`
Branch: `experiment/offline-freeze-20260916` · Master seed: `20260916` · Date: 2026-09-16

**Scope of this report: the offline pipeline only.** No LLM inference was performed, no batch
job was submitted, no credential was read or printed, and no priced API call was made at any
point. `execution_policy.json` carries `dispatch_enabled: false` and `inference_authorized: false`,
and `smoke_tests_executed: 0`.

---

## 1. What was built and what was actually checked

The pipeline covers the full chain from raw sources to dispatch-ready prompts: canonicalisation
and ground truth, budget definition and walk calibration, the four observation mechanisms
(R/S/H/B), the input block and prompt serialisation, the offline baselines (plug-in, mechanism
correctors, frozen training median, ExtraTrees), the leave-one-source-out training folds, the
request manifest, and the strict response evaluator. Everything after that — the LLM calls
themselves — is prepared but deliberately not executed.

Checks that were actually run (not merely implemented):

| Check | Command / artifact | Result |
| --- | --- | --- |
| Focused frozen tests | `unittest discover -s tests/frozen_main` | 14 tests, OK |
| Full test suite | `unittest discover -s tests` | 66 tests, OK |
| Structural verification | `scripts/verify_main_offline.py` → `verification.json` | `verified: true`, 1417 checksums, 33 862 seeds, 0 collisions |
| Resume idempotence | second identical invocation → `resume_verification.json` | 1418 artifacts identical, `changed_artifacts: []` |
| **From-scratch reproduction** | independent second run into a fresh directory, artifact-by-artifact hash comparison | 1 414 artifacts compared, 73 differ, and every difference is a wall-clock measurement (`seconds`, `upper_bound_seconds_linear`); graphs, observations, models, baselines, the request manifest, every CSV header and all prompt hashes are bit-identical |
| Walk kernel correctness | `weighted_walk_expectation.json` | one-step expectation 4.5963 vs. exact 4.60185, inside 6 SE (0.1269) |
| Generator ground truth vs. closed form | DAR α=0 has independent windows, so ρ_k is analytic | both replicates match Pr{Bin(5,0.2) ≥ k}/[1−0.8⁵] at all four k; worst absolute deviation 0.0029 |
| Mock/real separation | `scripts/check_main_evaluation_mocks.py` → `mock_check.log` | 2688 rows checked, `mock_only: true`, `mock_as_real_rejected: true`, `inference_calls: 0` |
| Evaluator on an empty response set | `scripts/evaluate_main_responses.py` → `no_llm_execution/` | `started: 0`, `not_started: 2688`, `complete_main_result: false` |
| CNS source integrity | MD5 against the published release | `98892459f73e774cf79e7977edfeee3e` matched; legacy export equality holds |

Three reproducibility defects were found during this acceptance pass and fixed. The first two
share a root cause: a checkpoint reloaded from disk carries the sorted key order that `write_json`
imposes, while a freshly computed dictionary carries its literal order, so outputs that inherited
that order depended on whether a stage had just run or was resumed. The third is a separate
resume defect.

1. `requests.jsonl` was written with insertion-order keys, so the 2 688-row manifest came out
   byte-different between a fresh run and a resumed one. It is now written with sorted keys.
2. The calibration result reached the `budget_summary.csv` header with provenance-dependent key
   order, so the CSV column order varied the same way. `calibrate` now rebuilds that dictionary
   in a fixed order, and raises if an unexpected key appears rather than silently dropping it.
3. The walk timing pilot re-ran on every resume and rewrote `timing.json` even when calibration
   was already complete. It is now skipped once calibration exists.

None of this touched any research decision or any content-addressed identity: `prompt_sha256`,
`block_sha256` and the model and feature hashes are computed with `digest`, which sorts keys, and
were verified identical across all 2 688 rows before and after. The canonical run was regenerated
from scratch with the corrected code; superseded directories are preserved, not deleted.

## 2. Commands

```bash
# Offline pipeline (full run; also the resume command — it is idempotent)
.venv/bin/python scripts/run_main_offline.py --out results/main_experiment/frozen_20260916

# Structural verification
.venv/bin/python scripts/verify_main_offline.py --run results/main_experiment/frozen_20260916

# Compact acceptance evidence (derived from the finished run; no sampling, no training)
.venv/bin/python scripts/build_main_acceptance_evidence.py \
    --run results/main_experiment/frozen_20260916 \
    --out results/main_experiment/evidence

# Tests
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests/frozen_main -v

# Evaluation against a response file (none exists yet; with no file it reports "not started")
.venv/bin/python scripts/evaluate_main_responses.py \
    --run results/main_experiment/frozen_20260916 \
    --out results/main_experiment/no_llm_execution
```

Resume semantics: re-running the pipeline command reuses every completed checkpoint. Completed
graphs, observations, models and calibration files are hash-verified before reuse and a mismatch
aborts the run. A resume rewrites no result: of 1 418 artifacts, none changes. The only files
that differ afterwards are `report.json` and `environment.json`, which record the invocation's
own elapsed time, and the derived `checksums.json` that covers them. Changing the source code
changes the recorded input hashes, which the guard reports as `resume dependency changed` —
that is intended, and requires a fresh output directory.

Cluster: a CPU batch script is prepared at `scripts/run_main_offline_cpu.sbatch`. **It was not
submitted.** The complete run takes about 200–260 s on one local core, so a scheduler round-trip
buys nothing. The probe of the environment (SLURM, account `tu`, partition `cpu`, 192 CPUs/node,
max walltime `3-00:00:00`) and the decision are recorded in `cluster_probe.json`, with
`cluster_jobs_submitted: 0` and no compute on login nodes.

## 3. Manifests and compact reports

Written to `results/main_experiment/evidence/` (tracked in git; the bulk artifacts stay local):

| File | Contents |
| --- | --- |
| `index.json` | run report, execution policy, headline counts, aggregate tolerances |
| `data_and_ground_truth.json` | per graph: N, D, M, B, events per window, true ρ₂…ρ₅, source hashes |
| `budget_deviations.json` | per graph: B, p, n_panel and its budget error, C, L, calibration and validation means, relative error, MCSE/B, `budget_matched` |
| `observation_status.json` | status/arm breakdown, planned vs. actual calls, empty samples listed individually |
| `baseline_results.json` | all 80 stratum × arm × method rows (5 strata × 4 arms × 4 methods) with MAE₂, MCSE, ProfileMAE, valid/replacement/empty fractions |
| `training_folds.json` | the 7 folds with held-out source, row counts, feature count, frozen median, hyperparameters, model hashes |
| `prompt_sizes.json` | min/max/mean per tokenizer, headroom against the 4096 limit, template verification flags |

**Data volume and ground truth.** 24 graphs: 16 real sources (of which 6 are the held-out test
stratum) and 8 synthetic ones (DAR α∈{0, 0.8} and Activity-Driven memoryless/memory, two paired
replicates each). Real sources range from `sp_hypertext2009` (N=113, M=20 818) to `jodie_lastfm`
(N=1 980, M=1 293 103) and `copenhagen_bluetooth` (N=692, M=2 426 279). True ρ₂ spans
0.003 (`nr_digg_reply`) to 0.789 (`ad_memory_r1`).

**Budget deviations.** All 24 graphs report `budget_matched: true`, with no
`search_limit_reached_without_budget`. Independent walk validation used 1024 fresh walks
everywhere (no extension to 4096 was needed). Worst relative deviation of the validation mean
from B: **0.696 %** (acceptance threshold 5 %). Worst MCSE/B: **0.484 %** (threshold 1 %). The
node-panel budget error, which is bounded by the integer choice of n, stays within ±0.90 %.

**Empty samples.** **Zero.** No observation has D_obs = 0, so the frozen-training-median
replacement path is implemented and unit-tested but is not exercised by any planned call.

**Baseline results (real test stratum, MAE₂ / ProfileMAE).** `valid_fraction` is 1.0 and
`replacement_fraction` is 0.0 for every row.

| Arm | plug-in | corrector | ExtraTrees | median |
| --- | --- | --- | --- | --- |
| R (node panel) | 0.00588 / 0.00380 | *(= plug-in by design)* | 0.11663 / 0.05500 | 0.22885 / 0.11163 |
| S (random walk) | 0.28424 / 0.21831 | **0.02830 / 0.01181** | 0.11233 / 0.06162 | 0.22885 / 0.11163 |
| H (time suffix) | 0.07958 / 0.06850 | 0.15889 / 0.06354 | 0.11446 / 0.05333 | 0.22885 / 0.11163 |
| B (event sampling) | **0.02238 / 0.01153** | 0.06974 / 0.04289 | 0.10864 / 0.05208 | 0.22885 / 0.11163 |

Two observations worth recording, neither of which changes any research decision:

- The **walk corrector earns its keep**: on arm S it cuts MAE₂ from 0.284 to 0.028, a tenfold
  reduction. On arms H and B the corrector is *worse* than the raw plug-in on MAE₂ (though better
  on ProfileMAE for H), which is a real result about those estimators, not a defect.
- **ExtraTrees is a weak learned baseline.** On the real stratum it sits at ≈0.11 MAE₂, an order
  of magnitude behind the plug-in. On the high-persistence synthetic strata it is *worse than
  simply predicting the frozen training median* (dar_a08: 0.428 vs. 0.416; ad_memory: 0.438 vs.
  0.428) — the expected signature of a model trained on 15–16 real sources being applied to
  targets far outside its training range (true ρ₂ ≈ 0.77–0.79 vs. training median 0.354). Flagged
  here so the eventual LLM comparison is read against a correctly understood reference point.

**Training folds.** 7 ExtraTrees models: 6 leave-one-source-out folds for the real test sources
(240 rows, 15 independent sources each) and one all-16-source model for the synthetic tests
(256 rows). 88 features, 4 targets, scikit-learn 1.7.2, hyperparameters frozen as specified
(`n_estimators=500`, `criterion=squared_error`, `max_features=1.0`, `min_samples_split=5`,
`random_state=0`, `bootstrap=False`, `n_jobs=1`), sample weights 1/(G·4·n_a).

**Prompt sizes.** All 224 prompts, against the hard limit of 4096 tokens including the chat
template:

| Counter | min | max | headroom at max | over limit |
| --- | --- | --- | --- | --- |
| Qwen thinking (exact template) | 873 | 1393 | 2703 | 0 |
| Qwen non-thinking (exact template) | 875 | 1395 | 2701 | 0 |
| DeepSeek (message texts) | 823 | 1081 | 3015 | 0 |
| Sol (o200k proxy) | 802 | 1061 | 3035 | 0 |

The Qwen counts are verified against the explicitly rendered chat template, token id by token id.
DeepSeek and Sol counts are *proxies* — the provider's exact framing overhead is not publicly
pinned and must be confirmed at release time. Given ≈2.7 k tokens of headroom, no plausible
framing overhead approaches the limit, and no truncation is required anywhere.

## 4. Status of the 224 planned observations and 2 688 planned calls

| Quantity | Planned | Prepared | Started | Empty |
| --- | --- | --- | --- | --- |
| Main observations | 224 | **224** | — | **0** |
| Logical LLM calls | 2 688 | 2 688 manifest rows | **0** | — |

All 224 observations carry status `prepared_not_started`. The arm split is 5/5/1/5 samples per
graph (R/S/H/B) across 14 graphs of the evaluated set, 16 observations per source; H has a single
sample because the time-suffix mechanism is deterministic — which is also why every H row reports
MCSE exactly 0. Each observation maps to 12 logical calls (4 configurations × 3 repeats),
giving 224 × 12 = 2 688. `actual_calls` is 0 for every row and `started_calls` is 0
in both `report.json` and `verification.json`.

**Empty samples are reported separately and there are none**: `empty_observations: 0`,
`empty_observation_ids: []`, `empty_fraction: 0.0` in all 80 baseline summary rows. No observation
therefore takes the "no LLM call, frozen median" path, and no observation is in the
"not started, no estimate at all" category beyond the global fact that the study has not run.

Training observations (256) are a separate, disjoint set used only to fit the baselines; they are
not part of the 224 and generate no LLM calls.

## 5. Implemented / executed / blocked

**Implemented and successfully executed offline:** source ingestion and canonicalisation with
integrity checks; ground-truth persistence profiles; budget definition and walk calibration with
independent validation; all four observation mechanisms; input-block construction, serialisation
and the strict parser; all four baseline families; the 7 training folds; the 2 688-row request
manifest; prompt token accounting; the response evaluator including the replacement rule; seed
derivation with collision checking; checkpointing, resume and immutability guards; and the
from-scratch reproduction of the entire run.

**Implemented but deliberately not executed:** every LLM call. The client code, payload
construction per configuration, retry/reserve/watchdog policy and batch handling exist and are
exercised against mocks only. `dispatch_enabled` and `inference_authorized` are both false.

**Blocked, pending authorisation and credentials** (not blocked by this codebase): the four
configurations' actual dispatch, and with it any real result for MAE₂.

**Known limitations, none of which block the offline pipeline:** DeepSeek and Sol prompt sizes are
proxies pending provider confirmation (§3); ExtraTrees extrapolates poorly to high-persistence
synthetic strata (§3); the cluster path is prepared but unexercised (§2).

## 6. Technical start conditions for the later LLM calls

1. API authentication plus current prices and reserve limits for both paid providers.
2. Provider framing confirmed: exact DeepSeek and Sol input-size verification against the real
   tokenisation, replacing the proxies used here.
3. Qwen3.6-35B-A3B served under vLLM 0.20.1 in BF16 on 2 × H100 80 GB at the pinned revision,
   with reasoning and JSON output separated by the `qwen3` reasoning parser.
4. Authorised technical inference tests covering streaming, usage accounting and transport
   recovery — none of which have been run.

---

## Conclusion

**The offline pipeline is ready.** It runs end to end, reproduces itself bit-for-bit from
scratch, resumes idempotently, passes all 66 tests and all structural verifications, and has
prepared all 224 observations and all 2 688 planned calls with zero empty samples.

This states nothing about the LLM study, which **has not been conducted**: zero calls were
started, zero model responses exist, and no real result for the primary metric exists yet.
