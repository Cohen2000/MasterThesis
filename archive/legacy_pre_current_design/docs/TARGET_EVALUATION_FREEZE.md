# Target and evaluation freeze candidate

Target status: **FROZEN**. The complete main experiment is not yet fully
frozen: the input contract, exact requested-output schema, model
configurations, final 96 sampled cases with their walk budget, and the
immutable prompt hash still have to be recorded. See the gate list at the end.

## Prespecified main-study panel

The panel contains exactly 32 temporal graph instances:

- all 8 available empirical networks;
- 8 literature-based mechanistic networks: 4 homogeneous DAR(1) instances
  (two sizes crossed with low/high memory) and 4 activity-driven/memory
  instances (two sizes crossed with memoryless/high-memory dynamics);
- 16 controlled variants: one low-persistence (`rho_target=0.15`) and one
  high-persistence (`rho_target=0.55`) timing variant for every empirical
  backbone, using the same bursty/contiguous construction and replicate 0.

Each controlled pair preserves nodes, collapsed topology and per-edge event
counts. Only event timing changes. The exact IDs are emitted by
`src/select_target_panel.py`; selections are parameter-design based and do not
depend on LLM or estimator performance.

Three access strategies yield 96 graph-walk cases. Eight prespecified LLM
configurations yield 768 responses.

## Population and target definitions

The population is every undirected dyad with at least one event in the complete
stream over the normalized horizon `[0,1]`. With `W=5` equal-width windows,
`K_e` is the number of distinct active windows of dyad `e`.

Primary target:

`rho_k = P(K_e >= k)` for `k=2,3,4,5`, jointly called the **dyadic
active-window survival profile**.

Primary scalar metric:

`ProfileMAE = mean_k(|rho_hat_k-rho_k|), k=2..5`.

Headline component:

`rho_2`, called the **multi-window dyad share**.

Secondary target:

`C_one_step = P(A[e,w+1]=1 | A[e,w]=1)`, pooled over dyads and `w=1..4`,
called **pooled adjacent-window dyad retention**.

Derived descriptive target:

`mean_occupancy = E[K_e/5] = (1+rho_2+rho_3+rho_4+rho_5)/5`. It is computed
from the submitted profile for the primary analysis; an independently supplied
value is a consistency diagnostic only.

Two observations on the historical 84-case panel bear on that choice, without
settling it. Across all twelve evaluated runs the error of the *stated*
`mean_occupancy` and of the value *implied by that run's own profile* are
practically identical, so the separate field carried no additional information
there. As a diagnostic it does separate runs: the share of answers satisfying
the identity ranges from 24% to 100%, and rises with reasoning effort within
matched model pairs. Whether that is worth a required output field on the
final panel is a design question, not a measured one. Numbers in
`docs/LLM_EVIDENCE.md`.

Robustness target:

`lifetime_mean_over_T = mean_e((t_last(e)-t_first(e))/T)`, with singleton
dyads contributing zero.

## Scoring

- Report ProfileMAE, component MAE/bias for every `rho_k`, and `rho_2` MAE.
- Interpret ProfileMAE as
  `sum_{k=2}^5 |rho_hat_k-rho_k| / 4`; for two valid monotone profiles this is
  the normalized one-dimensional Wasserstein-1 distance between their
  active-window-count distributions.
- Report `C_one_step` MAE as secondary.
- Report profile-implied mean-occupancy MAE and lifetime MAE as robustness.
- Report `Skill = 1 - MAE / MAE_mean_floor` per target/component.
- Group-macro results are primary; pooled case MAE is secondary.
- Report parse rate, 90% interval coverage/width when requested, and raw
  monotonicity-violation rate.
- Report a rank statistic (Spearman against truth) beside the error. Level and
  ordering are separable failure modes here, and on the historical suite they
  dissociate sharply: the naive read-off ranks better than any model while its
  level is far off, and some models rank poorly while sitting near the correct
  level. An error value alone hides this.
- A constant predictor at the training mean is a useful reference row. Because
  the target distribution is wide, that constant is hard to beat, and a method
  only earns its variance by ranking well enough to pay for it.
- Note that `mean_floor` is the mean of training-set truth
  (`src/evaluate_benchmark.py`), so it is the simplest supervised predictor
  rather than an information-free one. Which comparisons that affects is worth
  thinking through per condition; it matters most where labelled information
  also reaches the competitor.
- The supervised ceiling needs a stated training protocol. Fitting it on the
  96 panel cases alone is thin: in the budget probe it saturates and is matched
  or beaten by the analytical mask MLE at larger budgets. Training on the
  frozen `benchmark_v2` case set (about 90k cases, 38 groups) and predicting
  onto the panel with the panel backbones held out gives it the headroom the
  name implies.

## Invalid or missing output

- Parse only the final JSON object.
- A requested missing, non-numeric, non-finite, or out-of-range value is
  invalid for that target.
- Do not clip, impute, reorder, or monotonically project raw LLM outputs.
- Report both complete-case target metrics and the corresponding validity rate.
- For primary ranking, also report failure-penalized loss: each missing,
  non-finite, out-of-range, or otherwise invalid requested component receives
  absolute loss 1 (the maximum on the `[0,1]` target scale). This prevents
  selective non-response from improving complete-case MAE.
- Classical shape-projected predictions may be reported as a separately named
  baseline, never as a repair of an LLM response.

## Prespecified slices

- access strategy: `time_agnostic_t`, `time_respecting`,
  `recent_history_k20`;
- coverage: `<.01`, `.01-.05`, `.05-.20`, `>.20`;
- graph category: empirical, literature-based synthetic, controlled variants;
- additionally group-macro by complete source/family.

On this panel, coverage tracks graph size far more than walk budget, so the
coverage slice is partly a size slice. At budget 800 the `<.01` band contains
only the MathOverflow backbone, i.e. one independent group. Merging the two
lowest bands, or treating coverage as a continuous covariate alongside the
bands, keeps that slice interpretable. Note also that raw error grows with
coverage here because the high-coverage graphs are the small dense ones with
large target values; skill against the floor tells a different story than MAE.

## Walk budget evidence

A read-only probe ran the three access strategies over all 32 panel graphs at
nested budgets 200/400/800/1600/3200 via
`bash scripts/run_panel_budget_probe.sh` (walks plus
`src/report_panel_budget.py`). Because standard strategies draw one trajectory
at the largest budget and every smaller budget is an exact prefix, the whole
ladder costs one pass and the budgets are perfectly paired. The pass takes well
under a minute on a laptop; the manifest is
`results/final_target_panel/panel32_final.csv`, which already carries every
column `src/run_benchmark_walks.py` needs. The probe derives its walk plan into
a temporary config, so `config/` is untouched.

At budget 800 the 96 cases land at median coverage 0.056, mean 0.105, with the
middle 80% between 0.017 and 0.265 and extremes at 0.003 and 0.498. All four
coverage bands are populated (9/32/43/12). Models see about 520 dyads in the
median case, and the mask histogram stays compact enough that prompt length is
not a binding constraint.

Budget 800 also spreads the estimator ladder widest: floor 0.130, naive
read-off 0.141, occupancy MLE 0.084, mask MLE 0.079, supervised profile model
0.064 in ProfileMAE. Smaller budgets leave the high-coverage band nearly
empty, larger ones drain the low band and let the analytical estimators catch
up with the supervised one.

Two side observations from the same run are worth carrying into the analysis.
The naive read-off is worse than the floor and carries a large negative bias,
which is the same failure mode the language models show. And the oracle
diagnostics separate the two error sources: selection bias pushes estimates up,
window censoring pushes them down much harder, and the net is the observed
underestimation. Those `oracle__` columns only exist on newly generated case
files, not on the frozen shards.

## Remaining gates before the complete experiment is frozen

Completed: the exact 32-row panel is materialized, the truth-only
`W in {4,5,8}` robustness analysis has been run, and the input evidence is
summarized in `docs/LLM_EVIDENCE.md`. Remaining gates are:

1. record the final input contract;
2. record the exact requested-output schema consistent with this target
   hierarchy (the historical V2.1 nine-key schema is not automatically final);
3. record the final model configurations, and how non-pinnable product screens
   are labelled if they are included;
4. generate and validate the 96 panel-by-walk cases without changing the
   32-graph panel, and fix the walk budget while doing so;
5. generate the final prompt and record its immutable hash.

The point of the gates is that the main comparison stops moving once they
close. Adding targets, tasks, graph families, input variants, or model
families afterwards changes what the numbers mean, so later ideas are better
placed in clearly labelled appendix work.

A prompt-isolation experiment is optional model-specific appendix work. It is
not a prerequisite for choosing the scientific targets or freezing this panel.

Status 2026-08-02: dropped for the thesis scope. The scaffolding exists
(`src/make_target_prompt_isolation.py`, `src/eval_target_prompt_isolation.py`,
`slurm/target_prompt_isolation_qwen.sbatch`) but was never run: no isolation
prompts were ever generated and no answers exist, neither locally nor in the
cluster workspace. The files stay in place as optional appendix work; nothing
in the frozen panel, the target hierarchy, or the open gates depends on them.
