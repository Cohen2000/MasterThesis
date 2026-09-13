# Pre-main-experiment freeze checks

Deterministic diagnostics for the intended design (W=5, rho_2..rho_5, ProfileMAE). There are no LLM/API calls, downloads or config changes, and the census outputs are read only. These tables contain numbers only; no design decision is made here.

## 1. Command

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/pre_main_freeze_checks.py
```

Git HEAD `62648cff9c8674bdb80e250951450d40d056fbff (working tree has uncommitted changes)`; Python 3.10.12, NumPy 2.2.6, pandas 2.3.3; runtime 1.9 min. Outputs are written only to this directory, and existing outputs are not overwritten unless `--overwrite` is given. To reproduce elsewhere, add `--out-dir <new-dir>`. `/results/**` is git-ignored by the existing `.gitignore`.

## 2. Datasets analyzed

| Design name | Registry key | Status |
|---|---|---|
| SocioPatterns Hospital | `sp_hospital` | analyzed |
| SocioPatterns High School 2013 | `sp_highschool2013` | analyzed |
| Copenhagen Bluetooth | `copenhagen_bluetooth` | analyzed |
| CollegeMsg | `snap_collegemsg` | analyzed |
| Email-Eu | `snap_email_eu` | analyzed |
| MathOverflow | `snap_mathoverflow` | analyzed |
| Radoslaw Email | `nr_radoslaw_email` | analyzed |
| JODIE Reddit | `jodie_reddit` | analyzed |

## 3. Seeds and parameter rules

- Sampler seeds (diagnostic only; no final seeds exist): four slots 0-3. Each is `numpy.random.default_rng(crc32("pre_main_freeze_checks|<dataset>|<mechanism>|slot<j>"))`. The numbers are in every raw CSV row and carry no scientific meaning. The 25% and 50% observations of a seed are nested (prefix designs).
- Timing variants: one attempt per backbone x target, seed `crc32("20260911|census_twin|<dataset>|<target>")`. This is the census rule, so rho2_015 repeats the census attempt. The generator is `family_from_events` + `make_instance(hub_bias=False, span_layout="contiguous")` with the empirical timestamp pool.
- Reference: uniform node order (`default_rng(seed).permutation(N)`); panel size n = smallest n whose median event coverage over the 4 seeds is closest to the target; induced dyads with full histories.
- Selection: `walks.run_walk(time_agnostic)` on the collapsed graph. Start rule: uniform over all nodes (walks.run_walk time_agnostic placement); no restarts, so the walk stays in the start component. Walk steps = smallest step count whose median event coverage over the 4 seeds is closest to the target. Walks are simulated for 4|E| steps, doubled up to 64|E| until the median passes 50% (`walk_steps_simulated`). Each traversed dyad exposes its full history once.
- History loss: cutoff c = the distinct complete-axis time whose suffix {t >= c} has the event share closest to the target (ties go to the later c). Suffix events only.
- Both: `nonwalk_samplers.uniform_event_reservoir` with round(target x M) records.
- Plug-in: K from visible events on the complete-axis windows, with observed dyads (K>0) as the denominator. normalized_profile_error(W) = (1/W) sum_{k=2..W} |rho_full(W,k) - rho_hat(W,k)|, which is 0.8 x ProfileMAE at W=5.
- RW revisit_fraction = steps that arrive at an already visited node / steps. repeated_dyad_step_fraction = steps along an already discovered dyad / steps. Degrees are collapsed full-graph degrees.
- W summaries take the seed median within each dataset x mechanism x severity cell, then the median over cells.

## 4. Key results

**Panel and timing variants (W=5).** Achieved rho_2 per target; `inv` = all exact invariants hold.

| dataset | nodes | dyads | events | rho_2..rho_5 | P(m>=2) | rho2_015 | rho2_035 |
|---|---|---|---|---|---|---|---|
| sp_hospital | 75 | 1139 | 32424 | 0.445 / 0.164 / 0.046 / 0.014 | 0.857 | 0.148 (inv yes) | 0.350 (inv yes) |
| sp_highschool2013 | 327 | 5818 | 188508 | 0.486 / 0.267 / 0.128 / 0.052 | 0.693 | 0.149 (inv yes) | 0.349 (inv yes) |
| copenhagen_bluetooth | 692 | 79530 | 2426279 | 0.447 / 0.221 / 0.107 / 0.042 | 0.741 | 0.150 (inv yes) | 0.350 (inv yes) |
| snap_collegemsg | 1899 | 13838 | 59835 | 0.103 / 0.013 / 0.003 / 0.000 | 0.622 | 0.147 (inv yes) | 0.349 (inv yes) |
| snap_email_eu | 986 | 16064 | 332334 | 0.506 / 0.298 / 0.159 / 0.038 | 0.736 | 0.150 (inv yes) | 0.349 (inv yes) |
| snap_mathoverflow | 24759 | 187986 | 390441 | 0.084 / 0.019 / 0.005 / 0.001 | 0.396 | 0.142 (inv yes) | 0.331 (inv yes) |
| nr_radoslaw_email | 167 | 3250 | 82876 | 0.551 / 0.376 / 0.285 / 0.197 | 0.851 | 0.144 (inv yes) | 0.350 (inv yes) |
| jodie_reddit | 10984 | 78516 | 672447 | 0.419 / 0.248 / 0.161 / 0.099 | 0.541 | 0.149 (inv yes) | 0.349 (inv yes) |

Census cross-checks (raw SHA-256; recomputed counts, rho, m_e shares, duplicates, exclusions and horizon; W-grid rho) hold for 8 of 8 backbones.
JODIE Reddit identity check: PASS (users=10000; items=984; 672447 interactions, matching the published JODIE Reddit counts; header, id ranges, SHA-256 and file time order also checked; see `dataset_identity_check`).

**Severity (ProfileMAE at W=5, median over seeds; median event coverage in brackets).**

| dataset | target | reference (node) | selection (RW) | history loss (recency) | both (events) |
|---|---|---|---|---|---|
| sp_hospital | 0.25 | 0.030 [0.247] | 0.005 [0.250] | 0.126 [0.250] | 0.032 [0.250] |
| sp_hospital | 0.50 | 0.016 [0.519] | 0.010 [0.502] | 0.065 [0.500] | 0.017 [0.500] |
| sp_highschool2013 | 0.25 | 0.005 [0.249] | 0.006 [0.250] | 0.191 [0.250] | 0.042 [0.250] |
| sp_highschool2013 | 0.50 | 0.006 [0.500] | 0.004 [0.500] | 0.126 [0.500] | 0.020 [0.500] |
| copenhagen_bluetooth | 0.25 | 0.005 [0.250] | 0.001 [0.250] | 0.191 [0.250] | 0.016 [0.250] |
| copenhagen_bluetooth | 0.50 | 0.003 [0.500] | 0.001 [0.500] | 0.105 [0.500] | 0.011 [0.500] |
| snap_collegemsg | 0.25 | 0.001 [0.250] | 0.001 [0.250] | 0.010 [0.250] | 0.009 [0.250] |
| snap_collegemsg | 0.50 | 0.001 [0.500] | 0.000 [0.500] | 0.005 [0.500] | 0.005 [0.500] |
| snap_email_eu | 0.25 | 0.005 [0.251] | 0.003 [0.250] | 0.124 [0.250] | 0.038 [0.250] |
| snap_email_eu | 0.50 | 0.003 [0.501] | 0.002 [0.500] | 0.081 [0.500] | 0.017 [0.500] |
| snap_mathoverflow | 0.25 | 0.001 [0.250] | 0.002 [0.250] | 0.017 [0.250] | 0.009 [0.250] |
| snap_mathoverflow | 0.50 | 0.001 [0.500] | 0.001 [0.500] | 0.009 [0.500] | 0.004 [0.500] |
| nr_radoslaw_email | 0.25 | 0.028 [0.243] | 0.013 [0.250] | 0.260 [0.250] | 0.033 [0.250] |
| nr_radoslaw_email | 0.50 | 0.010 [0.500] | 0.011 [0.500] | 0.130 [0.500] | 0.010 [0.500] |
| jodie_reddit | 0.25 | 0.011 [0.250] | 0.011 [0.250] | 0.163 [0.250] | 0.025 [0.250] |
| jodie_reddit | 0.50 | 0.006 [0.500] | 0.011 [0.500] | 0.091 [0.500] | 0.010 [0.500] |

**Recency purity (W=5).** plug-in = observed suffix. roster = full dyad roster with truncated histories. survivor = surviving dyads with full histories.

| dataset | target | cutoff | W5 windows in suffix | vanished dyads | MAE plug-in | MAE roster | MAE survivor |
|---|---|---|---|---|---|---|---|
| sp_hospital | 0.25 | 0.737 | 2 | 0.558 | 0.126 | 0.149 | 0.077 |
| sp_hospital | 0.50 | 0.495 | 3 | 0.356 | 0.065 | 0.102 | 0.056 |
| sp_highschool2013 | 0.25 | 0.725 | 2 | 0.548 | 0.191 | 0.214 | 0.161 |
| sp_highschool2013 | 0.50 | 0.463 | 3 | 0.321 | 0.126 | 0.160 | 0.086 |
| copenhagen_bluetooth | 0.25 | 0.745 | 2 | 0.553 | 0.191 | 0.198 | 0.145 |
| copenhagen_bluetooth | 0.50 | 0.494 | 3 | 0.309 | 0.105 | 0.136 | 0.073 |
| snap_collegemsg | 0.25 | 0.260 | 4 | 0.742 | 0.010 | 0.020 | 0.044 |
| snap_collegemsg | 0.50 | 0.184 | 5 | 0.466 | 0.005 | 0.011 | 0.026 |
| snap_email_eu | 0.25 | 0.528 | 3 | 0.482 | 0.124 | 0.184 | 0.150 |
| snap_email_eu | 0.50 | 0.341 | 4 | 0.278 | 0.081 | 0.128 | 0.082 |
| snap_mathoverflow | 0.25 | 0.691 | 2 | 0.726 | 0.017 | 0.025 | 0.025 |
| snap_mathoverflow | 0.50 | 0.426 | 3 | 0.472 | 0.009 | 0.018 | 0.015 |
| nr_radoslaw_email | 0.25 | 0.754 | 2 | 0.532 | 0.260 | 0.309 | 0.287 |
| nr_radoslaw_email | 0.50 | 0.475 | 3 | 0.401 | 0.130 | 0.212 | 0.207 |
| jodie_reddit | 0.25 | 0.752 | 2 | 0.569 | 0.163 | 0.202 | 0.194 |
| jodie_reddit | 0.50 | 0.508 | 3 | 0.344 | 0.091 | 0.139 | 0.098 |

**Simple random walk at 50% target (median over seeds).**

| dataset | steps | steps/dyads | node cov | dyad cov | event cov | revisit | repeat-dyad steps | visited/full mean degree | LCC nodes |
|---|---|---|---|---|---|---|---|---|---|
| sp_hospital | 839 | 0.74 | 1.000 | 0.511 | 0.502 | 0.912 | 0.306 | 1.00 | 1.000 |
| sp_highschool2013 | 4130 | 0.71 | 0.995 | 0.495 | 0.500 | 0.921 | 0.303 | 1.00 | 1.000 |
| copenhagen_bluetooth | 56273 | 0.71 | 0.983 | 0.506 | 0.500 | 0.988 | 0.285 | 1.02 | 1.000 |
| snap_collegemsg | 10234 | 0.74 | 0.733 | 0.496 | 0.500 | 0.864 | 0.330 | 1.31 | 0.997 |
| snap_email_eu | 11337 | 0.71 | 0.873 | 0.496 | 0.500 | 0.924 | 0.297 | 1.14 | 1.000 |
| snap_mathoverflow | 137340 | 0.73 | 0.653 | 0.492 | 0.500 | 0.882 | 0.326 | 1.46 | 0.996 |
| nr_radoslaw_email | 2288 | 0.70 | 0.856 | 0.498 | 0.500 | 0.938 | 0.293 | 1.16 | 1.000 |
| jodie_reddit | 70508 | 0.90 | 0.770 | 0.564 | 0.500 | 0.880 | 0.372 | 1.26 | 1.000 |

Reddit sides at 0.25 (median): user coverage 0.561, item coverage 0.933, visited nodes that are items 0.141 (items are 0.090 of all nodes).
Reddit sides at 0.50 (median): user coverage 0.749, item coverage 0.983, visited nodes that are items 0.114 (items are 0.090 of all nodes).

**W robustness.** Mechanisms by median normalized error over cells, lowest first.

| W | ordering (all cells) | same as W=5 | same within 0.25 / 0.50 | dataset x severity cells same |
|---|---|---|---|---|
| 4 | srw < node_panel < sampled_event_stream < recency_truncation | yes | yes / yes | 0.69 |
| 5 | srw < node_panel < sampled_event_stream < recency_truncation | yes | yes / yes | 1.00 |
| 6 | srw < node_panel < sampled_event_stream < recency_truncation | yes | yes / yes | 0.75 |
| 8 | srw < node_panel < sampled_event_stream < recency_truncation | yes | NO / yes | 0.62 |
| 10 | srw < node_panel < sampled_event_stream < recency_truncation | yes | yes / yes | 0.75 |

**Grouped leave-one-backbone-out constant reference (ProfileMAE, W=5).**

| held-out backbone | natural | rho2_015 | rho2_035 |
|---|---|---|---|
| sp_hospital | 0.060 | 0.004 | 0.021 |
| sp_highschool2013 | 0.055 | 0.005 | 0.006 |
| copenhagen_bluetooth | 0.030 | 0.004 | 0.006 |
| snap_collegemsg | 0.180 | 0.003 | 0.008 |
| snap_email_eu | 0.082 | 0.003 | 0.003 |
| snap_mathoverflow | 0.182 | 0.009 | 0.025 |
| nr_radoslaw_email | 0.189 | 0.004 | 0.004 |
| jodie_reddit | 0.051 | 0.001 | 0.004 |

## 5. Hard failures and mismatches

**Hard failures:** none.

**Implementation/documentation mismatches (recorded, not fixed):**

- Timing targets: `config/study.yaml` lists [0.15, 0.55], `dataset_census.TARGETS` is ['0.15', '0.55'], and README.md, docs/STUDY_DESIGN.md and docs/DATASETS.md say 0.15/0.55. These checks use the intended 0.15/0.35.
- `config/study.yaml` controlled_timing.preserve = ['nodes', 'collapsed_topology', 'per_edge_event_counts'] omits the global timestamp multiset, which the empirical-pool generator preserves and which is checked here.
- `config/study.yaml` datasets.panel_status = 'pending_census'; the intended eight-backbone panel is not recorded in the config (left unchanged).
- `config/study.yaml` master_seed is null and no numeric sampler seeds exist; the diagnostic seeds in section 3 were defined for these checks only.
- Reference mechanism: docs/SAMPLING.md gives full histories of relations *incident* to selected participants (README.md and docs/STUDY_DESIGN.md say only "uniform random node/participant panel"). `nonwalk_samplers.node_panel_full_history` returns incident dyads with an adaptive whole-node event-budget stop. The intended mechanism is a fixed-size uniform node sample with the *induced* subgraph, which is what is implemented here. `nonwalk_samplers.node_panel_size` already uses the induced inclusion probability n(n-1)/(N(N-1)).
- Selection mechanism: no full-history random walk exists in src/. `walks.run_walk` (time_agnostic) records no timestamps. These checks reuse its transitions and add the full-history lookup. The start-node and component policy is not documented; the rule used is stated in the seeds section.
- History-loss mechanism: not implemented in src/. docs/SAMPLING.md says it keeps the "same conceptual population access as the reference" and leaves the roster and the zero-event dyads unspecified. Here it is a suffix of the complete stream, so dyads without suffix events vanish; the known-roster counterfactual is reported separately.
- Time axis: the design text says [0,1), but the census/generator map t_max to 1.0 in the last window with a 1e-9 guard (documented in docs/REPRODUCIBILITY.md). That convention is used unchanged, and observations are never rescaled.

## 6. Outputs

- `results/pre_main_freeze_checks/panel_integrity.csv`
- `results/pre_main_freeze_checks/timing_variant_check.csv`
- `results/pre_main_freeze_checks/timing_variant_invariants.csv`
- `results/pre_main_freeze_checks/severity_diagnostics_raw.csv`
- `results/pre_main_freeze_checks/severity_diagnostics_summary.csv`
- `results/pre_main_freeze_checks/recency_purity.csv`
- `results/pre_main_freeze_checks/rw_diagnostics.csv`
- `results/pre_main_freeze_checks/w_sensitivity_raw.csv`
- `results/pre_main_freeze_checks/w_sensitivity_summary.csv`
- `results/pre_main_freeze_checks/constant_baseline_grouped_loso.csv`
- `results/pre_main_freeze_checks/README.md`

Code: `scripts/pre_main_freeze_checks.py`; tests: `tests/diagnostics/test_pre_main_freeze_checks.py`.
