# v11 scientific protocol

Design `panel888-access-v11-20260923` estimates full-archive persistence `rho_k = Pr(K_e >= k | K_e >= 1)` for `k=2..5` across `W=5` windows. `rho_2` is primary. The panel contains eight real sources, eight matched P[w,t] surrogates and eight synthetic instances. Primary evaluation is equal-source real MAE_2; ProfileMAE and the two other strata are secondary. The 10% expected-discovered-active-dyad-window calibration and all three sampler draws per arm remain those of v10.

## Final access contract

Each mechanism releases its sampling-control information to the estimator/model:

| Arm | Released design information |
|---|---|
| R | Actual sampled `n_panel`; complete histories of retrieved panel dyads. |
| H | Actual sampled `n_panel` and `Temporal_access`; only accessible history. The main access pattern is h=.60, without an extra numeric h field. |
| S | Crawl log and traversal counts; `L` is reconstructible. Complete histories of traversed dyads. |
| B | Bernoulli event-retention probability `p`. |

`S_obs` uses the same S draws without the crawl log. It remains a Qwen-only information ablation and is excluded from paid API main comparisons. Full-archive population totals N/D/M, full-archive coverage, the calibration target, sampling fractions requiring hidden full N, target truth and other unreleased full-archive statistics are not released to the estimator/model. The R/H prompt phrase “full vertex count ... unknown” means unknown **to the model under this access contract**; it is not a claim about an operator's knowledge. The completed R/H prompt text is unchanged.

The v11 Qwen main result composes released R/H with unchanged v10 S/S_obs/B. Plugin, median and MLE definitions remain unchanged. R/H ExtraTrees adds only `log1p_n_panel` and uses the completed nested leave-one-real-training-source-out selection; S/S_obs/B ExtraTrees remains unchanged. No sampler, Qwen inference, MLE fit or ET model was rerun to compose v11.

An LLM answer is valid only if its final text is one JSON object containing exactly finite, non-increasing `rho_2..rho_5` in [0,1]. No clipping or repair is allowed. Accuracy is conditional on validity. Qwen has three model repeats. Executed paid runs (2026-09-23): DeepSeek Flash one repeat (budget-limited), GPT-6 Sol three repeats, and, added on request after repeat 1, a GPT-6 Sol variant with the hosted Python tool (code interpreter, at most 10 tool calls, otherwise identical request, three repeats). All use reasoning effort high; generation caps are the provider maxima (DeepSeek 384k, GPT 128k). Provider reasoning objects differ: Qwen generated reasoning blocks, DeepSeek raw provider reasoning, and OpenAI provider reasoning summaries. Only final answers enter cross-provider strategy/accuracy comparisons.

The [sealed walk gate](results/panel888_v10_walk_gate_20260923/WALK_GATE.md) remains applicable. Workplace retains its current windows and surrogate; the empty-window/calendar-cycle issue is a limitation rather than a new experiment.

## Amendment 2026-09-28: four estimators and extension

Every sampler is evaluated with exactly four estimators: plugin, training-median, the shared MLE and ExtraTrees (plus the LLMs). The design estimator is no longer reported, and the reference is plugin for R and the MLE for S, S_obs, H and B. ExtraTrees for S and S_obs therefore anchors on the MLE (target baseline and the four anchor features) instead of the design estimator; all S/S_obs folds were reselected (same nested leave-one-real-training-source-out grid) and refitted with the production seeds. R, H and B ExtraTrees, all observations, prompts, Qwen and API answers are unchanged. The MLE fallback never triggered (0 of 420 observations).

The evaluation adds four real test sources (reality_mining, lkml_reply, sp_malawi and nr_radoslaw_email, the latter with its own leave-one-source-out fold) and one P[w,t] surrogate for each (existing procedure, own calibration, parent sampler streams) through the same pipeline and Qwen protocol. The main analysis weights the twelve real sources equally; surrogates (12) and synthetic graphs (8) are separate blocks and originals and surrogates are compared as pairs. The arms are R, S, H and B; S_obs is historical. ExtraTrees variability is reported over 11 replicates (production plus 10 with new training draws, new forest seeds and rerun selection). Details: [final results](results/final_20260928/REPORT.md).
