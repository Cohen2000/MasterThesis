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

An LLM answer is valid only if its final text is one JSON object containing exactly finite, non-increasing `rho_2..rho_5` in [0,1]. No clipping or repair is allowed. Accuracy is conditional on validity. Qwen has three model repeats. Paid providers initially have one repeat each; repeat-based uncertainty is not comparable until further repeats are completed. Provider reasoning objects differ: Qwen generated reasoning blocks, DeepSeek raw provider reasoning, and OpenAI provider reasoning summaries. Only final answers enter cross-provider strategy/accuracy comparisons.

The [sealed walk gate](results/panel888_v10_walk_gate_20260923/WALK_GATE.md) remains applicable. Workplace retains its current windows and surrogate; the empty-window/calendar-cycle issue is a limitation rather than a new experiment.
