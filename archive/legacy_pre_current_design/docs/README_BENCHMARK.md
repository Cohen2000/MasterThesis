> **Version 2 available:** The original `smoke` and `full` presets remain reproducible.
> The expanded laptop/cluster benchmark uses the separate `v2_smoke` and `v2`
> presets. See [`README_V2.md`](README_V2.md).

# Literature- and real-world-grounded estimator benchmark

This adds a new estimator-screening phase without deleting the earlier pilot or
LLM results. The screen is intentionally completed **before** new LLM calls are
selected.

## What is included

### Data blocks

1. Unchanged real temporal event streams from several domains.
2. Exact timestamp shuffling `P[w,t]`: fixed collapsed topology, per-edge event
   counts and global timestamp multiset.
3. Controlled persistence variants on real substrates. These are clearly
   labelled `steered_P[w,t]`, not presented as maximum-entropy null models.
4. DAR(1) link-state dynamics on DCSBM or LFR candidate topologies. `alpha`
   controls copying and `chi` controls marginal occupancy independently.
5. Activity-driven-with-memory streams as a mechanism-shift comparison.
6. Controlled DCSBM timing twins for broad ground-truth coverage.
7. ER and BA remain available as historical appendix controls but are disabled
   in the full preset by default.

T3Former is not used as a synthetic generator because its experiments derive
tasks from real datasets. Its relevant lesson here is the inclusion of real
data, not a nonexistent T3Former generation algorithm.

### Access models

- `time_agnostic_t`: clean historical-time reference;
- `time_respecting`: uniform causal forward walk;
- `recency_biased`: short-gap causal forward walk;
- `recent_history`: reverse-time walk over the k most recent prior events;
- `time_agnostic`: only a 10% static-only sentinel in the full run.

### Observable inputs

- exact sparse `(n,w)` histogram, retained as JSON for the later LLM phase;
- exact sparse `(n,window-mask)` histogram;
- fixed ML encodings of occupancy and five-window masks;
- first/last observed window, adjacent-window overlap and sampled inter-event
  gaps;
- restarts, revisits, discovery curve, collisions, hit counts and observed
  subgraph degrees.

True coverage, total graph size, true degrees, generator parameters and labels
are never model inputs. They are metadata only.

### Estimators and comparison models

- observed plug-in;
- count-only uniform occupancy MLE;
- new position-aware mask MLE over all 31 nonempty five-window masks;
- beta-MLE, calibrated only on other source/family groups;
- mean floor;
- Ridge regression;
- Random Forest;
- Extra Trees.
- Histogram Gradient Boosting.

All learned predictions use `GroupKFold` by complete real source or synthetic
family. Every variant of a held-out source is held out together.

Targets are the full persistence profile
`rho_W5_k2 ... rho_W5_k5` and consecutive-window persistence `C_one_step`.

## 1. Setup

Run from the project root:

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 2. Mandatory smoke test

```bash
bash run_benchmark_smoke.sh
```

Success ends with:

```text
SMOKE OK: results/benchmark_smoke/SCREEN_SUMMARY.md
```

The smoke data is synthetic and only validates implementation correctness. Do
not interpret its model ranking scientifically.

## 3. Prepare real datasets

Place downloaded raw files under `data/raw/`, using the exact filenames in
`config/datasets.yaml`. This command prints what is present and the official
dataset page for every missing file:

```bash
python src/check_real_data.py
```

The full preset requires at least four real datasets and includes ten when all
are available. Controlled/shuffled variants are limited to six manageable,
format-verified datasets.

## 4A. Full run on bwUniCluster

The existing `cpu` partition is used. First create the environment on the login
node:

```bash
bash slurm/setup_env.sh
```

Then submit the dependency chain:

```bash
bash slurm/benchmark_submit.sh
```

This runs one data job, a 12-task walk array, and one grouped evaluation job.

Monitor with:

```bash
squeue --me
tail -f benchmark_01_data_*.out
tail -f benchmark_03_eval_*.out
```

If the cluster uses a different partition, change only the `#SBATCH
--partition` lines in `slurm/benchmark_*.sbatch`.

## 4B. Full run without Slurm

```bash
python src/build_benchmark_data.py --preset full --overwrite
python src/validate_benchmark.py --manifest data/benchmark_full/manifest.csv
python src/run_benchmark_walks.py --preset full
python src/validate_benchmark.py --cases results/benchmark_full/cases.csv.gz
python src/evaluate_benchmark.py \
  --cases results/benchmark_full/cases.csv.gz \
  --out-dir results/benchmark_full
python src/collect_benchmark_results.py --preset full
```

## 5. What to send back

The Slurm evaluation job creates this automatically. Otherwise create it with:

```bash
python src/collect_benchmark_results.py --preset full
```

Send back:

```text
benchmark_full_results_to_share.zip
```

It contains the manifest, case summaries, out-of-fold predictions, metrics,
rankings, compact report and ranking figure. Raw real event files are not
included.

Those results are enough to decide:

1. which access models contain recoverable signal;
2. whether masks or crawl diagnostics improve over `(n,w)`;
3. where analytical MLEs fail but learned models succeed;
4. which real domains expose distribution shift;
5. which 50–100 representative cases and input conditions should go to the
   LLM benchmark next.

## Main configuration

All experimental dimensions are in `config/benchmark.yaml`. Run the provided
full preset unchanged once before expanding it. Parameter expansion should then
be motivated by observed failure regions rather than by a blind Cartesian grid.

## Literature mapping

- Gauvin et al., *Randomized reference models for temporal networks*:
  `P[w,t]` timestamp shuffling.
- Mazzarisi et al., DAR(1): edge-state copy/resample dynamics.
- Perra et al. and activity-driven-with-memory extensions: heterogeneous node
  activity and tie reinforcement.
- Nguyen et al., CTDNE: causal temporal walks.
- EstGraph: walk discovery, collision and revisit summaries.
- EdgeBank/TGB: recurrence, frequency and recency as strong temporal signals.
- CAW and temporal neighbourhood models: anonymous/recent temporal context.
- Longa et al.: binary temporal activity signatures.
