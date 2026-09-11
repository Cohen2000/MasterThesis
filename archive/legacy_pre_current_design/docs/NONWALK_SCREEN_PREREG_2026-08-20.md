# Non-walk screen: dated decision record

Date fixed: **2026-08-20**.  This is an exploratory screening study adjacent
to, and not a silent modification of, the previously frozen walk target and
32-instance panel.  Results from this screen may select access conditions for
the later confirmatory LLM comparison; they do not change the target panel or
its truth values.

## Question and target

The screen asks which low-budget observation design produces which selection
and temporal-censoring bias, and whether prompt-observable information leaves
predictable residual error beyond correctly specified classical baselines.
The primary target and metric remain the dyadic active-window survival profile
`rho_W5_k2..k5` and group-macro ProfileMAE.  `rho_W5_k2` is the headline
component; `C_one_step` and normalized mean lifetime are secondary.

Time-prefix and random-window cases are labelled **full-horizon
extrapolation**, not ordinary within-horizon bias correction.  Their truth is
defined on the complete `[0,1]` stream.  No unbiasedness claim is made without
a stationarity/model assumption.

## Frozen screen design

- Graphs: the 32 rows in `results/final_target_panel/panel32_final.csv`.
- Event budgets: `100, 400, 800, 1600, 3200`.
- Independent sample seeds per graph/stochastic strategy: four.  The
  chronological prefix is deterministic and is stored once per graph/budget;
  nominal duplicate seeds would be pseudo-replication.
- Budgets within a seed are repeated/nested observations, not independent
  replicates.  Uncertainty is clustered at least by graph group and sample
  seed; budget trajectories are analysed as paired trajectories.
- Main non-walk designs: uniform fixed-size event reservoir, chronological
  event-count prefix, randomly anchored contiguous event-count window, and
  end-time ego retrieval at `k in {1,5,20}`.
- Diagnostic references: node-induced full-history panel from budget 400
  upward and `ego_recent_kall`.  They are not automatically eligible as
  budget-matched headline arms.
- Reduced time resolution is excluded because it can alter the window-defined
  target rather than only the observation process.

For ego retrieval all queries occur at `T_end`; event ids are deduplicated.
Only the newest returned event expands the FIFO frontier.  Thus query-node
order is k-invariant at equal query count.  Query and unique-event budgets are
both retained.  An exact event-budget cutoff keeps the newest part of the last
response and marks it `partial_response`; no likelihood may treat that response
as complete.

Node-panel size is selected from the design expectation
`M*n*(n-1)/(N*(N-1))`, never by searching for a panel whose realized event
count happens to approach the target.  Realized counts are reported.  Budget
100 is omitted because small panels can be empty and extremely unstable.

## Required diagnostics before interpretation

1. Ten-bin event-rate CV and trend, late/early event-rate ratio, and node/dyad
   arrival in the first half for every graph.
2. Realized full-degree distribution of selected/query nodes versus all active
   nodes: mean-degree ratio and two-sample KS distance.  Walk and snowball bias
   will be compared empirically on the same 32 graphs rather than inferred
   solely from an asymptotic random-graph result.
3. Requested versus realized event count, query count, temporal-window width,
   duplicate removal, restart count, and partial-response incidence.
4. Coverage and performance by graph category and complete source/family.

## Classical and learned baselines

Every eligible arm receives plug-in, occupancy MLE, position-aware mask MLE,
and supervised group-held-out models.  Two ExtraTrees variants must be named
separately:

- `ET_prompt_parity`: only quantities mechanically recoverable from the exact
  prompt supplied in that LLM condition;
- `ET_feature_oracle`: all observable engineered features, reported only as a
  ceiling and never called a fair LLM competitor.

For the event reservoir the screen additionally requires:

- Chao1 and Good-Turing as richness/coverage diagnostics, not rho estimators;
- exact fixed-size inclusion probability
  `1-C(M-m_e,B)/C(M,B)`;
- `oracle_reservoir_ht_true_label`, using full dyad multiplicity and label only
  to separate discovery/selection error from label censoring;
- an observable empirical-Bayes/deconvolution baseline fitted from the
  per-window frequency-of-frequencies.  It must estimate a joint temporal
  activity mixture, not merely a one-dimensional rate mixture that cannot
  identify persistence.  Its precise likelihood, regularization/grid, and
  training/tuning protocol must be frozen before reservoir results are used to
  claim LLM value.  If a credible version cannot be validated, the thesis will
  state that limitation rather than treating Chao1 as a substitute.

Implementation fixed before the Panel32 results were inspected:

- name: `reservoir_temporal_mixture_eb`;
- input: exact frequency table of sampled five-window count vectors, also
  disclosed in the corresponding LLM prompt;
- latent components: each of the 31 non-empty active-window masks crossed with
  28 per-active-window rate values;
- active-window full counts: independent zero-truncated Poisson variables with
  a common component rate; inactive-window counts are zero;
- rate grid: 28 geometric points from `0.02` to
  `min(1e5, max(20, 2*(max_observed_window_count+1)/p))`;
- observation model: independent Bernoulli thinning with `p=B/M`, explicitly
  reported as an approximation to fixed-size reservoir sampling;
- likelihood: conditional on a dyad being detected at least once; detected
  component weights are converted to population weights by inverse component
  detection probability;
- fitting: deterministic EM, no truth-based tuning;
- readout: the population weight on masks with at least `k` active windows.

This is a serious observable competitor but not an exact design-based
reservoir estimator.  Convergence rate and sensitivity to sampling fraction
must be reported beside its errors.

Technical amendment after the first baseline execution and before LLM-arm
selection: all 640 fits reached the originally specified 300-iteration limit
because weights continued to move among near-equivalent rate-grid points.  The
unmodified first output is retained in `baselines_initial_em300`.  Since the
estimand depends on aggregate mask mass rather than identification of every
rate-grid weight, the stopping rule is now five consecutive iterations with a
maximum change below `1e-7` in `rho_k2..rho_k5`, after at least 20 and at most
500 iterations.  This amendment was triggered by the convergence diagnostic,
not selected using target errors; final profile and raw-weight deltas are both
saved.

Second technical amendment before arm selection: the unrestricted mask-rate
mixture remained practically non-identified, with target profiles drifting
toward boundary solutions as iterations increased.  It is retained only as a
diagnostic attempt.  The scored observable baseline is therefore
`reservoir_factorized_temporal_eb`: population weights factor into a 31-mask
distribution and a shared 28-point rate distribution.  These two distributions
maximize the detected-dyad conditional likelihood directly with deterministic
L-BFGS.  This adds the explicit assumption that mask and event rate are
independent in the population, but yields an identified, reproducible
persistence readout.  The change addresses failed identification and was made
before applying the LLM-arm selection rule.

Convergence handling: the primary L-BFGS limit is 500 iterations.  The first
factorized run converged in 605/640 cases; all 35 remaining cases stopped only
because they hit that limit.  Before arm selection, a deterministic retry up
to 2000 iterations was therefore added for limit-hit cases.  Retry status and
both limits are saved; no target error is used in this rule.

## Prespecified LLM control and arm-selection rule

Every selected LLM arm has two paired inputs:

1. the full sampled input;
2. `metadata_only_no_sample`, containing the same non-event metadata as (1)
   — `W`, requested/realized budget, disclosed strategy, and declared node
   count if and only if it is also present in (1) — but no events, histograms,
   observed-node/edge counts, or sample-derived diagnostics.

This control is mandatory even if classical baselines already perform well.
It tests whether predictions arise mainly from dataset/scale priors rather
than correction of the supplied sample.

Two distinct non-walk conditions at budget 800 will enter the first API LLM
comparison.  Node panel and `ego_recent_kall` are ineligible references.

- **Easy condition:** eligible condition with the smallest group-macro
  ProfileMAE of the best feasible, non-oracle classical baseline.
- **Correction-potential condition:** among the remaining conditions, the one
  with the largest positive difference
  `ProfileMAE(mask_MLE) - ProfileMAE(ET_prompt_parity)`.  If no difference is
  positive, choose the condition with the largest absolute difference and
  treat the anticipated result as a negative control.
- Ties within `0.002` ProfileMAE are broken in this fixed order:
  `uniform_event_reservoir`, `ego_recent_k20`, `ego_recent_k5`,
  `time_random_window_events`, `time_prefix_events`, `ego_recent_k1`.

The complete table used by this deterministic rule will be saved before any
LLM responses for these arms are obtained.  Both selected arms are run even if
one looks classically solved; this prevents a screen that only hunts positive
LLM results.
