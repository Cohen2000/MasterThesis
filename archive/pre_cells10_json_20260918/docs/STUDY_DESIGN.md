> Historical pre-freeze documentation. The 2026-09-16 main experiment is defined by
> [the current runbook](MAIN_EXPERIMENT_RUNBOOK.md) and its linked freeze.

# Current study design

The research question is whether an LLM can estimate full-network persistence
from a partially observable temporal graph, and how estimation changes when
observation affects relationship selection, temporal history, or both.

## Population, target and score

Normalize the complete temporal interaction stream to `[0,1)` and partition it
into five equal non-overlapping windows. Let `E_full` be all undirected dyads
with one or more events in the complete stream. For each such dyad,

```text
A_e(w) = 1 iff dyad e has at least one event in window w
K_e = sum_w A_e(w)
rho_k = |{e in E_full : K_e >= k}| / |E_full|, k = 2,3,4,5
```

Every dyad has equal weight. The denominator is fixed by the complete graph.
Multiple events within a window do not add active windows. Predict the four
components jointly, with `1 >= rho_2 >= rho_3 >= rho_4 >= rho_5 >= 0`.

The primary error is `ProfileMAE = sum_{k=2}^5 |rho_hat_k-rho_k| / 4`.
Mean occupancy is derived: `E[K/5]=(1+rho_2+rho_3+rho_4+rho_5)/5`.
It is not requested as a separate main prediction.

Raw responses are immutable. Monotonicity and bounds violations are recorded
without repairing the prediction. The current scoring helper computes the
stated arithmetic MAE on finite numeric components, even when they violate
bounds or monotonicity, and exposes the violation flags. Parse failures have no
invented numerical score. An aggregate missing-output penalty is not specified
by this design and has not been imported from an earlier experiment.

## Observation mechanisms

1. Reference: a uniform random node/participant panel with full history.
2. Selection dominated: one Simple Random Walk local crawl with full histories
   of discovered relations; this is the bridge to EstGraph.
3. History loss dominated: recency/time truncation, preserving the same
   conceptual population access while exposing only recent history.
4. Both: a sampled event stream, affecting edge discovery and visible history.

These are the four main mechanisms. Their semantics and operational choices
are recorded in [SAMPLING.md](SAMPLING.md).

## Dataset strategy

The empirical panel is selected after a census of all candidate inputs. Keep
the complete registry until that decision is made.

Matched timing variants preserve nodes, collapsed topology and per-edge event
counts. Only temporal assignment changes. Low/high `rho_2` targets near
0.15/0.55 are candidates whose feasibility must be measured. Record achieved
profiles and allocation deviations; do not equate nominal and achieved targets.

Mechanistic families are DAR(1) temporal dynamics on DCSBM substrates, with
intended `n=500/1500` crossed with low/high temporal memory, and activity-driven
networks in memoryless and tie-reinforcement modes. The generic implementations
are preserved. Numerical memory settings and the final empirical panel remain
open design choices.

## Prompt and replication

Use one generic zero-shot architecture: TARGET / DEFINITIONS; OBSERVATION
MECHANISM with separate selection/access and history-access descriptions;
OBSERVED DATA; TASK; and the final JSON result. The prompt has no solved
examples, persona, stated bias direction, or correction formula. Free-form
reasoning is allowed. The last non-empty line is one JSON object containing
only `rho_2`, `rho_3`, `rho_4`, `rho_5`. State the monotonicity constraint.

For each graph × observation mechanism, use four independent sampler seeds.
On each identical sampled observation, obtain three LLM repeats. Retain every
raw response and identify the sampler seed and repeat separately. Median across
the three repeats may later be a robust headline prediction; it does not
replace the raw repeats needed for `SD_LLM`, `SD_sampler`, monotonicity-violation
rates and round-number diagnostics. No mixed-effects analysis is required now.
