# Thesis pilot backbone (Phase 1) on bwUniCluster 3.0

Goal of this run: the **floor -> plug-in -> MLE -> oracle band per walk, as a
function of coverage**, for BOTH estimands (window persistence `rho_headline`
and mean occupancy `mean_span_frac`). This is the LLM-independent backbone. The
key result it produces: under forward-time sampling the analytical MLE falls
*below* the information-free floor while a supervised oracle still recovers the
target, and that gap grows as coverage drops.

What was changed vs your originals: `run_pilot_walks.py` now also writes two
columns `rho_mle`, `occ_mle` (the occupancy-MLE read off the (n,w) of each
checkpoint, via the new `corrected_estimator.py`); `pilot_eval.py` adds an `mle`
estimator tier and prints a `floor -> mle -> oracle` band line. Everything else
is your code untouched.

Run **everything from the project root** (`thesis_pilot/`). The cluster login
node has internet; compute nodes do not, so the one pip step is on the login node.

---

## 0. Upload and unzip (from your laptop, VPN on)

```bash
scp thesis_pilot_bundle.zip uc3:/home/tu/tu_tu/tu_zxokn55/
ssh uc3
cd ~ && unzip -o thesis_pilot_bundle.zip && cd thesis_pilot
```

## 1. One-time environment (on the login node, ~2-4 min)

```bash
bash setup_env.sh
```
Success ends with `ENV READY.` and `core deps OK`. (If it prints that xgboost is
absent, that is fine; the eval falls back to HistGradientBoosting.)

## 2. Smoke test FIRST (verifies the whole pipeline in ~3 min)

```bash
sbatch jobs/smoke.sh
```
Check it (replace JOBID with the number sbatch printed):
```bash
squeue --me
cat smoke_<JOBID>.out
```
**What success looks like:** the file lists `--- target: rho_headline ---` with an
`mle` row, a line `>> low -cov gap (plugin - xgb_full) = +...`, and ends with
`SMOKE DONE`. Files `summaries_smoke.csv`, `smoke_results.csv`,
`smoke_mae_vs_coverage.png` exist. (On this tiny grid the "mle captures +nan"
note is expected: too little data for the oracle to beat the floor. The real run
fixes that.)

If the smoke job fails, paste me `smoke_<JOBID>.out` and stop here.

## 3. The real run (3 jobs, in order)

Submit one, wait for it to finish (`squeue --me` empty for that job, or check the
`.out`), then submit the next. Or chain them automatically with dependencies:

```bash
JID1=$(sbatch --parsable jobs/01_make_data.sh)
JID2=$(sbatch --parsable --dependency=afterok:$JID1 jobs/02_run_walks.sh)
sbatch          --dependency=afterok:$JID2 jobs/03_eval.sh
```

Rough timings: make ~30-90 min, walks ~30-60 min, eval ~30-60 min.

Watch progress / sanity:
```bash
squeue --me
tail -f 01_make_*.out      # then 02_walks_*.out, 03_eval_*.out
```
`02_walks_*.out` ends by printing **mean coverage per size** (must DROP as n grows).
`03_eval_*.out` prints the **COVERAGE-GAP SUMMARY** with, per estimand and per
coverage band, a line `floor X -> mle Y -> oracle Z (mle captures F of recoverable)`.

## 4. Send me back these files

- `summaries.csv`            (the per-row data; lets me rebuild the band + bias curve + ranking dissociation)
- `data_grid/manifest.csv`   (ground truth per instance incl. C_one_step, for the occupancy-C check)
- `grid_results.csv`         (aggregated MAE per cell)
- `grid_mae_vs_coverage.png`
- the full text of `03_eval_<JOBID>.out`

If `summaries.csv` is large, gzip it first: `gzip -k summaries.csv` and send `summaries.csv.gz`.

To pull them to your laptop, from your laptop:
```bash
scp uc3:~/thesis_pilot/summaries.csv uc3:~/thesis_pilot/grid_results.csv \
    uc3:~/thesis_pilot/grid_mae_vs_coverage.png uc3:~/thesis_pilot/data_grid/manifest.csv ./
```

---

## Notes / knobs

- **Partition full?** `cpu` may show 0 idle. If jobs queue too long, change
  `--partition=cpu` to `--partition=cpu_il` in the job files (more idle nodes in
  your `sinfo`). Same time/mem work there.
- **No `--account` line** is needed (your working test jobs had none).
- **Escalate to 250k nodes** (only if the coverage range needs to go lower): in
  `jobs/01_make_data.sh` set `--sizes 400,2000,10000,50000,250000`,
  `--mem=180G`, `--time=12:00:00`, and consider `--partition=highmem`. Do this
  only after the 50k run looks right.
- **Reproducibility:** all seeds are derived deterministically from
  (substrate, n, family, target, rep), so reruns are identical.

## What comes after this run (not in this bundle)

- Phase 2: a bias-aware MLE added as another tier in the same band (decides
  classical-recoverable vs learning-only).
- Phase 3: the LLM rows (zero-shot + disclosed-mechanism) over the API, slotted
  into the same band; the earlier A/B becomes rows here.
- Optional Phase 0 for the C / representation question: run your existing
  `census.py` on the 17 real datasets and check whether `C_one_step` decouples
  from `mean_span_frac` there (on synthetic they are ~0.99 correlated, so C is
  redundant; real data may differ and is the honest place to test it).
