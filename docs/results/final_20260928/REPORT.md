# Final panel: 12 real sources, 12 surrogates, 8 synthetic graphs (2026-09-28)

## Design as evaluated

- **Arms:** R, S (with crawl log), H and B. S_obs is historical ([v11 results](../panel888_v11_main_20260923/MAIN_RESULTS.md)) and not part of current runs or tables.
- **Estimators per arm:** plugin, training median, shared MLE and ExtraTrees (ET), plus Qwen and the paid APIs. The reference is plugin for R and the MLE for S, H and B. ET for S anchors on the MLE (four-estimator rule).
- **Graphs:** 12 real sources: the eight v11 sources plus reality_mining, lkml_reply, sp_malawi and nr_radoslaw_email (own leave-one-source-out fold). Each real source has one P[w,t] surrogate. The four new surrogates follow the existing procedure: one pre-fixed shuffle, all invariants audited, own 10% calibration and the parent's sampler streams. There are also 8 synthetic graphs.
- **Evaluation:** equal-source MAE_2 over the 12 real sources is primary, ProfileMAE secondary. Surrogates and synthetic graphs are separate blocks. Originals and surrogates are compared as pairs (12 families), never as independent sources.

Tables: [MAIN_RESULTS.md](MAIN_RESULTS.md), [VARIABILITY.md](VARIABILITY.md), [HISTORY.md](HISTORY.md), [WALK.md](WALK.md). Long format: `PREDICTIONS.csv` (observation × method × ET fit or LLM repeat).

## What ran and what was reused

- **Reused unchanged:** all v11 observations and answers, the new-source preparation, the 11 ET fits per arm and fold (523 selections, 550 fits), all Qwen answers of the four added originals, and the v11 API answers.
- **Newly computed:**
  - the four surrogates with 48 R/S/H/B observations;
  - ET predictions for the surrogate rows of the two affected folds: 88 tasks, of which 3 use the pickled v11 production models and 85 are refits with the stored choices and seeds. Every refit reproduces the existing predictions of its fold to within 1e-12;
  - 288 Qwen answers on H100 under the unchanged v11 protocol;
  - the history decomposition and the walk diagnostics.
- **Checks:**
  - Replicate 0 of R/H/B equals the v11 ET predictions exactly.
  - Leakage audit over 528 fits: no test family (original or surrogate) appears in the real training rows or the inner selection sources of its fit. This includes the nr_radoslaw_email fold. The three reality/lkml/malawi families are not training sources at all.
  - All eight added graphs are budget-matched within 5% in every arm, and no H panel is saturated.
- **Qwen on the added graphs:** 576 answers, 574 valid. The 2 invalid answers are non-thinking S answers with non-monotone profiles.
- **Wall time on uc3:** the surrogate additions ran 05:30–07:11 UTC, including two resubmissions. See `WALL_TIMES.csv`.

## Diagnostic findings

- **History (H):** truncation does not generally compensate.
  - For k = 2 the numerator loss exceeds the denominator loss: real 0.51 vs 0.40 for the last 60%, 0.36 vs 0.21 for the first 60%.
  - The population-level truncated plugin therefore underestimates rho_2 (real signed error −0.052, surrogates −0.098). Only 1 of 12 real sources is within 0.01.
  - For k = 4 and 5 the truncated plugin is 0 by construction.
  - Node sampling adds almost nothing on top of truncation (real plugin MAE_2 0.065 at population level vs 0.064 with the H draws; MLE 0.069 vs 0.069).
  - First vs last 60%: the real MAE_2 is 0.059 vs 0.065 for plugin and 0.060 vs 0.069 for the MLE.
- **Walks (added graphs):**
  - The unchanged gate criterion is applicable to all eight added graphs and passes only for lkml_reply and nr_radoslaw_email. reality_mining, sp_malawi and four surrogates fail; in v11 only sp_hospital__pwt failed. The gate judges the design ratio, which is no longer a reported estimator. The S reference is now the MLE, whose walk bias and SD are listed separately.
  - **sp_malawi** is the extreme case:
    - weight ESS 6.4, revisit rate 0.89, and only about 20 of 347 dyads discovered per walk;
    - the walk touches 18% of the 21 communities;
    - MLE bias +0.186 with SD 0.209 (plugin bias +0.163, SD 0.291).
  - On sp_malawi__pwt the production MLE rejects 2 of 1000 walk histograms: these walks discovered only dyads active in all five windows. They are counted as failures, not repaired. This case never occurred in the actual test draws.
- **Variability:** training variability (11 ET fits), sampling variability (3 sampler draws) and LLM answer variability (3 repeats) are reported separately and not compared. The real ET MAE_2 SD across fits is 0.001–0.003 per arm. The deterministic plugin/median/MLE outputs have sampling variability, which is not zero.

## Remaining issues

- **API rows are pending** for the four added real sources and their surrogates (96 observations); the API rows in the tables currently cover the v11 graphs only. No ranking is formed from different source sets.
- The walk-gate failures above apply to the S arm of the listed graphs, whose S results stay in the tables. As in v11, the flags are reported and nothing is excluded.

## API extension (started 2026-09-28 on request)

- **Frozen set:** 96 observations (4 added originals + 4 surrogates × R/S/H/B × 3 draws), hash manifest `API_FREEZE_EXT.json`. The original observations are the unchanged test draws.
- **Answers:** DeepSeek 96 (one each), GPT-6 Sol 288 and GPT-6 Sol + Python 288 (three repeats each, Batch), 672 in total. The configurations are those of v11.
- **Run directories:** new ones (`api_runs/*_ext`). The technical smoke/pilot records were taken over from the v11 runs, so no new smoke or pilot was paid.
- **Expected cost** from the v11 spend per answer: DeepSeek ≈ USD 2.2, GPT ≈ 6.6, GPT + Python ≈ 23.4, together ≈ USD 32.
- **Hard caps:**
  - DeepSeek: USD 3.10 (the account balance).
  - GPT: the two runs split the remaining USD 108 under the USD 200 total, at most about USD 53 each, including the Batch worst-case reserve.
- **DeepSeek** starts only off-peak: weekdays outside 01:00–04:00 and 06:00–10:00 UTC, with no start within 60 minutes of a peak.
- **Commands** (resumable; the runner skips collected IDs and never relaunches a started request):

```bash
M=~/.local/share/masterthesis; R=$M/api_runs; C="--observation-set ext --observations $M/v12_api_ext_observations"
SH="--shared-budget-dir $R/openai_v11 --shared-budget-dir $R/openai_tools_v11"
scripts/api_cycle.sh openai $R/openai_ext $C --repeats 3 --budget-usd 145 --batch-size 72 $SH
scripts/api_cycle.sh openai $R/openai_tools_ext $C --tools --repeats 3 --budget-usd 145 --batch-size 96 $SH
scripts/api_cycle.sh deepseek $R/deepseek_ext $C --repeats 1 --budget-usd 3.3694   # 3.10 + imported pilot spend
# joint evaluation once all three are complete
python scripts/evaluate_api.py --observations v11=$M/v11_api_observations ext=$M/v12_api_ext_observations \
  --deepseek v11=$R/deepseek_v11 ext=$R/deepseek_ext --openai v11=$R/openai_v11 ext=$R/openai_ext \
  --openai-tools v11=$R/openai_tools_v11 ext=$R/openai_tools_ext --out docs/results/final_20260928/api
```
