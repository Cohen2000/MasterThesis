# Design-aware supplementary baselines

This module adds only methods that answer a question not already covered by
Ridge, Random Forest, ExtraTrees, histogram gradient boosting, occupancy MLE,
and mask MLE. It writes to a separate result directory and does not overwrite
the frozen benchmark-v2 results.

## Included

### Model-assisted Hansen--Hurwitz

For `time_agnostic_t`, collapsed-graph simple-random-walk traversals converge
to the uniform distribution over directed edges. For every sampled dyad, the
existing mask model supplies the posterior label probability
`P(K >= k | n, mask)`. Averaging that probability over traversals (equivalently,
weighting each unique dyad by its traversal count `n`) gives the
model-assisted Hansen--Hurwitz estimate.

This is not reported for `time_respecting`, `recency_biased`, or
`recent_history`: their draw probabilities depend on time and path history and
are not identified by a single stored trace.

### Shape projection

Every four-vector is projected by PAVA onto
`1 >= rho_2 >= rho_3 >= rho_4 >= rho_5 >= 0`. Base and projected estimates are
both retained, so the effect is paired and auditable. Analytical occupancy
profiles are normally monotone already; this is mainly informative for learned
multi-output profiles.

### Discovery diagnostics

Chao1 and Good--Turing sample coverage are emitted as discovery diagnostics.
They are not called persistence estimators: neither identifies how persistent
the never-observed dyads were, and within-walk recaptures are dependent.

### Optional structured ML reference

`--fit-extra-trees` fits exactly one learned profile model: ExtraTrees on the
existing `combined` observable feature set, with GroupKFold by source/family.
It predicts all four `rho_k` jointly and is evaluated before and after PAVA.
This is the only additional learned comparison; no extra booster zoo is added.

## Run

Fast smoke:

```bash
source .venv/bin/activate
python src/evaluate_design_baselines.py \
  --cases results/benchmark_v2/results/cases_shard_000.csv.gz \
  --strategies time_agnostic_t --max-cases 200 \
  --out-dir results/benchmark_v2/design_baselines_smoke
```

All analytical/design-aware additions:

```bash
python src/evaluate_design_baselines.py \
  --cases 'results/benchmark_v2/results/cases_shard_*.csv.gz' \
  --out-dir results/benchmark_v2/design_baselines
```

Add the single structured ExtraTrees reference:

```bash
python src/evaluate_design_baselines.py \
  --cases 'results/benchmark_v2/results/cases_shard_*.csv.gz' \
  --out-dir results/benchmark_v2/design_baselines \
  --fit-extra-trees --jobs -1
```

The shell quotes around the shard glob are intentional: the runner expands and
deduplicates the shards itself. Outputs are `SUMMARY.md`, `metrics.csv`,
`rankings.csv`, `predictions.csv.gz`, and `discovery_diagnostics.csv.gz`.

## Oracle-label extension

Current feature extraction also emits two `oracle__` diagnostics on newly
generated case files:

* true labels on the uniquely seen dyads (label error removed, selection bias
  retained);
* true labels averaged over traversals (both errors removed for stationary
  `time_agnostic_t`).

The frozen case shards predate these columns and cannot reconstruct dyad
identities from their aggregated histograms. They must not be fabricated.
Regenerate cases from event files if this oracle decomposition is required.

An empirical-inclusion oracle for the directed temporal walks would additionally
require many repeated walks on every complete benchmark graph. It is a useful
separate experiment, but not a cheap baseline and is intentionally not hidden
inside this evaluator.
