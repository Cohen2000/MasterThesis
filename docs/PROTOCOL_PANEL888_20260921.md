# v10 scientific protocol

Design `panel888-access-v10-20260923` estimates full-archive persistence `rho_k = Pr(K_e >= k | K_e >= 1)` for `k=2..5` across `W=5` windows. The primary target is `rho_2`. The panel contains eight real sources, their eight matched P[w,t] timestamp-shuffled surrogates, and eight synthetic instances. The primary evaluation is equal-source MAE_2 over the real sources; ProfileMAE is secondary. Surrogates and synthetic instances remain separate strata.

## Observation mechanisms

R samples a uniform node panel with complete history. H uses a uniform node panel with the last 60% of elapsed time. B independently retains events at released probability `p`. S starts uniformly on the full vertex set and takes exactly `L` interaction-following steps without burn-in or restart. Each step chooses uniformly from full-archive event records incident to the current vertex. S releases each traversed dyad once with complete history and a crawl log of traversal counts. S_obs uses the same S draws but withholds the log; it is a completed Qwen information ablation and is excluded from the new cross-provider API main comparison. Parent and P[w,t] surrogate share sampler streams. All arms calibrate expected discovered active dyad-windows to 10% of the full total, with 5% validation tolerance.

Blocks contain only released observations and operator-known design information. S_obs adds pattern-level `inv_events`; S also adds `traversals` and `traversals_per_event`. S traversals sum to `L`. Blocks exclude full-archive sizes, calibration target, coverage fraction and truth. Floats use 12 significant digits. The system prompt, user prefix, arm rules and frozen observation blocks are unchanged across providers. The API main arms are R, S, H and B on all 24 graphs, with three sampler draws each. DeepSeek has one model repeat; GPT has three; completed Qwen keeps three.

## Estimation and evaluation

All non-LLM estimates use the serialized block. Plugin estimates the observed pattern profile. The median is trained on real sources within each leave-one-real-source-out fold. Design is defined for S and S_obs as the pattern ratio weighted by `traversals_per_event` and `inv_events`, respectively. S uses the plain Hájek/Hansen-Hurwitz ratio, without finite-sample bias correction. Shared zero-truncated Beta-Binomial MLE uses arm-specific observation models and the lowest-objective valid converged start; it falls back to the homogeneous model only if no start converges. References are R plugin, S/S_obs design and H/B MLE.

Arm-specific pooled ExtraTrees predicts a residual on plugin or reference from released-evidence features. In each outer fold, anchor and `min_samples_leaf` {1, 5, 20} × `max_features` {0.5, 1.0} are selected by leave-one-real-training-source-out CV among the remaining real sources, using the final pooled composition and block weights. Final training uses only 10% training draws and excludes the test real source. Synthetic ET results are marked in-distribution.

An LLM answer is valid only if its final text is one JSON object with exactly `rho_2..rho_5`, finite numeric values in [0,1], non-increasing; one whole-answer code fence may be removed. No estimate is clipped, repaired or imputed. Accuracy is conditional on valid answers and validity is reported. Source-level paired comparisons use exact sign-flip inference and leave-one-source-out ranges; draw-clustered MCSE is secondary. The [sealed tables](results/panel888_v10_main_20260923/MAIN_RESULTS.md) give the numerical record.

## Walk gate

The 24-graph audit used 1,000 walks per graph. Its post-hoc amended gate applies where the interaction stationary shift exceeds 0.05 and requires |S design bias| ≤ 0.1 |plugin bias| and S design RMSE ≤ 0.5 plugin RMSE. Failures remain in all results, flagged “not correctable at this budget.” The [sealed gate](results/panel888_v10_walk_gate_20260923/WALK_GATE.md) records 13 of 14 applicable sources passing, with `sp_hospital__pwt` retained as the failure. Sealed v9 null, window, history and mixture diagnostics are in `docs/results/panel888_offline/`.
