# Empirical dataset census

Evidence for choosing the empirical panel, the controlled timing targets and
the window count. **No panel, ranking or target decision is made here.**

Produced by

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/census_datasets.py --realized-twins --out-dir results/dataset_census
```

(Python 3.10.12, NumPy 2.2.6, pandas 2.3.3). The code is `src/dataset_census.py`,
and the parsing layout comes from `config/datasets.yaml` (`format` and
`census_audit`). Only local files were read, and nothing was downloaded.
Independent reruns reproduced every value, including the timing attempts,
under both this environment and NumPy 1.26 / pandas 2.1.

## Preprocessing conventions

- **Events.** Each valid data row is one event. Rows are not deduplicated,
  weights are not expanded, and there is no sampling, component restriction or
  filtering beyond the rules below.
- **Invalid rows.** Rows with the wrong field width, an empty endpoint, or a
  non-numeric or non-finite timestamp are removed (`invalid_rows_removed`).
  Comment, blank and registry header lines are counted separately.
- **Self-events** (`u == v` after namespacing) are removed
  (`self_events_removed`). They do not define the observation horizon.
- **Undirected dyads.** `(u,v)` and `(v,u)` map to the same canonical unordered
  dyad. `E_full` contains every dyad with at least one valid event.
- **Bipartite JODIE data.** Users are prefixed `u:` and items `i:` before dyads
  are built, so a user ID can never collide with an item ID.
- **Windows.** The horizon is `[t_min, t_max]` of the retained events. The
  normalized time is `x = (t - t_min)/(t_max - t_min)` and the window is
  `min(floor(W*x), W-1)`, so `t_max` falls in the final window. The shared helper
  `census.window_index` adds a 1e-9 guard in window units. On every dataset and
  every W = 2..20 its assignment matched exact arithmetic (integer, or long double
  for fractional timestamps) event by event.
- **Target.** `A_e(w)=1` iff dyad e has an event in window w, `K_e = sum_w A_e(w)`,
  `P_e = K_e/W`, and `rho_k = P(K_e >= k)` with equal weight per dyad.

**W = 5 is the primary target.** W = 2..20 is sensitivity evidence only.

## Files and columns

### `empirical_census.csv` — one row per registered dataset (18)

| Column | Meaning |
|---|---|
| `dataset`, `file`, `label`, `domain` | Registry identity |
| `bipartite`, `originally_directed` | `yes`/`no`/`unknown` (from `census_audit`) |
| `status` | `ok`, `absent` (registered, not local) or `error` |
| `n_nodes` | \|V\| of retained non-self events |
| `n_edges_full` | \|E_full\| (undirected dyads) |
| `n_events` | M, retained valid events |
| `events_per_edge` | M / \|E_full\| |
| `fraction_m_eq_1`, `p_m_ge_2..p_m_ge_5` | Per-dyad event counts m_e: P(m_e = 1), P(m_e >= k) |
| `rho_2..rho_5` | W=5 survival profile P(K_e >= k) |
| `q1..q5` | Exact W=5 distribution P(K_e = k). The sum is 1 to within 1e-15 on every row |
| `mean_occupancy_derived` | (1 + rho_2 + ... + rho_5)/5 = mean(P_e) |
| `raw_min_timestamp`, `raw_max_timestamp` | Over all parsed valid rows, in source units |
| `observation_start`, `observation_end`, `observation_span_source_units`, `window_duration_source_units` | Retained-event horizon and span/5, in source units |
| `timestamp_unit`, `time_unit_evidence`, `metadata_source` | Unit and its documentary basis |
| `physical_observation_span_seconds`, `physical_W5_window_seconds` | Only when the unit is documented as seconds; otherwise `UNKNOWN` |
| `data_rows`, `header_rows_skipped`, `comment_or_blank_rows_skipped`, `observed_row_widths` | Parser accounting |
| `invalid_rows_removed`, `self_events_removed` | Exclusions |
| `duplicate_dyad_timestamp_events` | Retained events that repeat an earlier event on the same dyad at the identical timestamp (kept; see warnings) |
| `parser_note`, `warnings` | Layout audit notes and parsing/unit warnings |
| `raw_sha256` | SHA-256 of the raw input file |

### `window_sensitivity.csv` — long format, 323 rows per dataset

Key columns: `dataset`, `W` (2..20) and `n_edges_full`. Each row also has a
`statistic`:

| `statistic` | `k` | `tau` | `value` | `event_count_upper_bound` |
|---|---|---|---|---|
| `rho` | 2..W | NA | P(K_e >= k) | P(m_e >= k) |
| `mean_P` | NA | NA | mean(K_e/W) | NA |
| `median_P` | NA | NA | median(K_e/W) | NA |
| `relative_survival` | ceil(tau·W) | 0.2, 0.4, 0.6, 0.8, 1.0 | P(P_e >= tau) = P(K_e >= ceil(tau·W)) | P(m_e >= ceil(tau·W)) |

`ceil(tau·W)` uses exact decimal fractions. A `rho` row with the same `k` does
**not** mean the same relative threshold at different W. Compare across W with
`mean_P`, `median_P` and `relative_survival`.

### `twin_feasibility.csv` — necessary event-count bound, one row per dataset

| Column | Meaning |
|---|---|
| `empirical_rho_2` | W=5 rho_2 of the original stream |
| `p_m_ge_2` | P(m_e >= 2), the upper bound on rho_2 for any count-preserving timing variant |
| `rho_k_upper_bound` (k=2..5) | P(m_e >= k) |
| `target_0_15_count_feasible`, `target_0_55_count_feasible` | P(m_e >= 2) >= target (exact rational comparison) |
| `target_0_15_margin`, `target_0_55_margin` | P(m_e >= 2) − target |
| `p_distinct_timestamps_ge_2` | Sensitivity only: the same bound if same-dyad same-timestamp duplicate rows counted as one event |

This bound is **necessary, not sufficient**. It does not show that the timestamp
allocator can reach a target.

### `twin_realized_feasibility.csv` — one deterministic attempt per dataset × target

The attempt calls `benchmark_generators.family_from_events` and then
`generator.make_instance(family, target, seed, hub_bias=False,
span_layout="contiguous")`. The family uses the empirical timestamp pool, i.e.
the multiset of normalized timestamps is reused, so each window's total event
count is also fixed. `round(target·|E_full|)` dyads, capped by the number with
m_e >= 2, are chosen uniformly as intended multi-window dyads. Timestamps are
then allocated under window capacity.

| Column | Meaning |
|---|---|
| `requested_rho_2` | 0.15 or 0.55 |
| `seed` | `crc32("20260911|census_twin|<dataset>|<target>")` |
| `W`, `span_layout`, `timestamp_mode`, `hub_bias` | Fixed settings (5, contiguous, empirical, False) |
| `status` | `ok` = the generator ran without error. **It does not mean the target was reached** |
| `achieved_rho_2`, `absolute_deviation` | rho_2 of the twin under the census convention, and its distance from the target |
| `nodes_preserved`, `topology_preserved`, `per_dyad_counts_preserved`, `timestamps_preserved` | Invariant checks against the original stream |
| `allocation_deviations` | Dyads whose realized number of windows differs from the intended span (capacity repairs) |
| `error` | Exception text if `status` = `error` |

No event streams were saved.

## Warnings and ambiguities

- **`snap_wikitalk`**: registered but absent locally, and not downloaded
  (`status=absent`).
- **`nr_enron_employees`**: 22,394 of 47,088 retained events (47.6%) repeat an
  earlier same-dyad same-timestamp row. The source of these duplicates is not
  documented, so they are kept as events, per the no-deduplication rule. This
  inflates M, `events_per_edge`, the P(m_e >= k) columns and the count bounds.
  It does not affect K or rho. See `p_distinct_timestamps_ge_2`. The original
  direction is also undocumented (`unknown`).
- **Other duplicate shares** (same-dyad same-timestamp): Email-EU 1.5%,
  Radoslaw 0.7%, CollegeMsg 0.07%, and under 0.02% for MathOverflow, Reddit and
  MOOC. Every other dataset has none.
- **Unknown physical units**: `nr_radoslaw_email`, `nr_enron_employees`,
  `nr_digg_reply`, and the four JODIE files. Their raw values fall in calendar-
  plausible ranges when read as Unix or relative seconds, but no documentation
  for these local exports establishes the unit. Physical durations are
  therefore `UNKNOWN`.
- **Self-events removed**: MathOverflow 116,109 (22.9% of rows), Enron 3,484,
  Digg 1,424, Radoslaw 51.
- **NetRepo layout**: all three files are `u v weight t`, with weight 1 on every
  row. The configured time column 3 is correct, so the Digg registry note
  "u v t" describes a different release.
- **Copenhagen Bluetooth**: a local export of `bt_symmetric.csv` with timestamps
  on a 300 s grid. It has no mirrored (`v,u,t`) or duplicate rows, so
  symmetrization did not double events.
- **No dataset failed to parse.** `invalid_rows_removed` is 0 everywhere.
