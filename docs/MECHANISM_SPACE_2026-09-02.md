# Point 5: do the five arms span the mechanism space, or were they selected?

Analytic. No model output, no new sampling: every sampler below was already run
against ground truth and stored in the repository.

## What was available

Seven sampler families exist in `src/nonwalk_samplers.py` and
`src/run_nonwalk_screen.py`. Eleven of their configurations already carry the
three columns the channel decomposition needs — `rho_W5_k2`,
`oracle__seen_label_rho_k2`, `est__plugin_rho_k2` — at five budgets each
(100, 400, 800, 1600, 3200):

- `results/nonwalk_screen/panel32_cases.csv.gz`: `uniform_event_reservoir`,
  `time_prefix_events`, `time_random_window_events`, `ego_recent_k1`,
  `ego_recent_k5`, `ego_recent_k20`, `ego_recent_kall`
- `results/nonwalk_crawl_screen/crawl_cases_shard_*.csv.gz`: `bfs_crawl_k5`,
  `bfs_crawl_k20`, `forest_fire_k5`, `forest_fire_k20`

Nothing had to be re-run. `activity_proportional_dyad_full_history` also exists
in the sampler module but has no screen output, so it is not plotted.

## The axes

- **x**, channel composition: `|selection| / (|selection| + |censoring|)`, from
  `docs/BIAS_CHANNELS_2026-09-02.md`. 0 = the bias is entirely censoring,
  1 = entirely selection.
- **y**, signed `rho_2` bias of the naive plug-in.

Figure: `docs/figures/mechanism_space.png`, table:
`results_summary/g3/mechanism_space.csv`.

## Result

| | channel composition | signed bias |
|---|---|---|
| **chosen five** | **0.029 – 1.000** | **−0.301 … +0.163** |
| all eleven alternatives, all five budgets | 0.221 – 0.415 | −0.341 … −0.007 |

Not one of the eleven alternatives, at any of its five budgets, leaves the band
0.22–0.42. They are mixed, censoring-dominated and downward-biased — near
duplicates of one another and of the two forward walks already in the design.

The chosen set holds both endpoints of the space:

- `time_agnostic_t` at 0.029 is the only near-pure censoring instrument
  available anywhere in the repository. The closest alternative is
  `time_prefix_events` at 0.23, eight times further along the axis.
- `event_sample_then_full_history` at 1.000 with **+0.163** is the only sampler
  of any kind whose bias points upward. Every alternative is negative at every
  budget.
- `node_panel_full_history` supplies the zero case at −0.002.

So the answer to "were the five picked for their results" is that three of them
occupy positions no alternative reaches, and the two that do sit inside the
alternatives' band — `recent_history_k20` and `time_respecting` — are the two
the design already labels as opposed-channel arms in (j). Adding any alternative
would add a fourth, fifth or sixth point to a cluster that is already
represented twice.

## Honest limits

**Coverage is not matched.** At a common budget the samplers realize different
dyad coverage (0.06–0.12 for the alternatives, 0.199 for
`event_sample_then_full_history`, 0.079 for `node_panel_full_history`), so
vertical distances mix mechanism with information. This is why every alternative
is drawn as its **trail across all five budgets** rather than as a single point:
the trails show that the horizontal position is a property of the mechanism and
barely moves with budget, while the vertical position does move. The claim being
made is about the x-axis.

**The alternatives were screened, not budget-calibrated.** They were run on a
budget grid, not put through the coverage-parity procedure the five arms went
through. Their y-values are therefore indicative rather than a like-for-like
bias comparison.

**Absence of an upward-selection sampler is a property of this repository, not
of the world.** No available sampler prefers quiet dyads. That gap is exactly
the one the sixth-arm proposal would fill, and until it is filled the figure
shows one populated corner and one empty one.

## Reproducing

```bash
python src/report_mechanism_space.py
```
