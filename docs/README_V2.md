# Literature Benchmark v2

This repository keeps the original `smoke` and `full` presets intact and adds
independent `v2_smoke` and `v2` presets. V2 never overwrites v1 outputs:

- v1 data/results: `data/benchmark_full`, `results/benchmark_full`
- v2 data/results: `data/benchmark_v2`, `results/benchmark_v2`
- v2 result bundle: `benchmark_v2_results_to_share.zip`

No LLM or external model API is called by either v2 runner. The pipeline uses
only graph/event generators, temporal crawls, analytical estimators, classical
scikit-learn models, and evaluation scripts.

## What v2 adds

### Data and controlled processes

- disjoint temporal chunks for sufficiently long real streams; every chunk keeps
  the source-level `group_id` to prevent leakage;
- four temporal surrogates:
  - global timestamp permutation;
  - within-window timestamp permutation preserving every edge-window count;
  - edge-lifetime-constrained resampling preserving per-edge counts and first/last time;
  - degree-preserving topology rewiring while retaining the multiset of edge event series;
- controlled real and synthetic twins with bursty handout and both legacy and
  contiguous active-window layouts;
- lower-persistence homogeneous DAR(1) cases (`chi=0.05`);
- heterogeneous edge-specific DAR persistence;
- community-correlated DAR persistence (`alpha_within != alpha_between`);
- renewal streams separating edge lifetime from within-lifetime burstiness;
- an optional broader real-data registry. Six original v1 data files remain
  sufficient to run v2; MathOverflow and Bitcoin-OTC can be downloaded with the
  included helper.

### Access mechanisms

- full timed history (`time_agnostic_t`);
- forward-in-time crawl (`time_respecting`);
- recency-biased crawl;
- recent-history access with `k=5` and `k=20`;
- three pooled forward walks at the same total budget;
- a small static `time_agnostic` sentinel.

Budgets are `100, 400, 800, 1600, 3200`; every main condition uses four walk
seeds. The final 100 observed anonymized `(u,v,t)` tuples are saved per case for
later LLM input ablations without rerunning the crawls.

### Targets and estimators

Targets:

- fraction of pairs active in at least 2, 3, 4, or 5 windows;
- one-step persistence `C`;
- mean normalized active span;
- event-weighted persistence.

Analytical baselines additionally include conditional persistence and a
leave-one-family-out beta correction calibrated globally and per data block.
Classical inputs are occupancy, active-window patterns, crawl diagnostics,
combined features, and combined features stacked with analytical estimates.
The supervised screen includes Ridge, Random Forest, Extra Trees, and histogram
gradient boosting; the more expensive transfer protocols retain the two strongest
tree families.

### Evaluation

- leakage-safe source/family `GroupKFold`;
- strategy-blind training without a strategy label;
- leave-one-data-block-and-overlapping-group-out evaluation;
- synthetic-to-real transfer;
- group-macro, worst-group, p90-group, bias, RMSE and rank correlation;
- finer coverage bands and seed-to-seed noise diagnostics;
- EdgeBank-style frequency and recency AUC diagnostics.

To keep a laptop run memory-safe, metrics are produced for all seven targets,
but the large row-level `predictions.csv.gz` stores only the central `rho_W5_k2`
target by default. This can be changed in `config/benchmark.yaml` under
`presets.v2.evaluation.prediction_targets`.

## Expected scale

With only the six v1 real datasets, v2 is designed for roughly 700 instances
and about 85,000 cases. Optional real datasets increase this count. Exact scale
is printed after data construction and recorded in the manifest.

## Quick start on Linux

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# Copy/download raw real datasets into data/raw first.
python src/check_real_data.py --preset v2

bash run_benchmark_v2_smoke.sh

# Full resumable laptop run; use RESET=1 only for a deliberately fresh rebuild.
SHARDS=12 JOBS=1 RESET=0 bash run_benchmark_v2_local.sh
```

The local runner limits BLAS/OpenMP libraries to one thread, executes crawl
shards serially, uses `nice -n 10`, validates intermediate artifacts, resumes
valid finished shards, performs all evaluations, creates diagnostic tables, and
finally tests the result ZIP.

## Frozen-band gate before new LLM prompts

V2 does not modify `src/walks.py`. Nevertheless, before generating new LLM
prompts, rerun the existing `make_pilot_cases_v2.py` workflow on the original
band machine and compare its reference-band summary with the archived v1 band.
That gate is separate from the estimator benchmark and should remain binding.

## Reproducibility notes

- v1 presets and output directories remain available.
- all generator and crawl seeds are deterministic and derived from the preset seed;
- variants and chunks of a real source share a source-level group;
- variants of one synthetic substrate family share a family-level group;
- raw real data are intentionally not bundled in this ZIP.
