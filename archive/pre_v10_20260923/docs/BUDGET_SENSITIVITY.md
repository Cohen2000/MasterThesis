# Budget sensitivity (ancillary study)

How does estimation error change with the amount of observable dyad-window
information? Grid fixed before any 10% Qwen accuracy was seen:
**b ∈ {2.5, 5, 10, 20, 30, 40, 50}%**, T_g(b) = b · sum_e K_e. The main study stays at
10% and is the 10% point of this analysis (reused by identity, never regenerated);
nothing was selected from these results.

Everything except the budget is the main study: the 24 graphs, W = 5, rho_2..rho_5,
the five arms (R, S1, S2, H, B) with their calibration rules and 5% tolerance,
prompts, 3 test draws × 3 repeats, Qwen thinking/non-thinking, references, LOSO
folds (surrogates in their parent's fold), the independent synthetic training pool
(5 training draws, refitted ExtraTrees with the fixed hyperparameters and seed
per budget), equal-source evaluation and the strict parser. Sol and DeepSeek are
not part of it.

Each non-main budget has its own sampler identity (`R-p888-20260921-b025`, ...),
hence fresh random streams, observation and request IDs; at 10% nothing changes
(the main `requests.jsonl` is reproduced byte-identically by the parameterised
code). S1 and S2 share the identical walk draw at every budget; they differ only
in what the observation shows.

## Running

    python scripts/budget_sensitivity.py prepare <b>     # per budget; on uc3: cluster/budget_sensitivity_prepare.sbatch
    python scripts/budget_sensitivity.py feasibility
    python scripts/budget_sensitivity.py audit           # builds results/panel888_budget_sensitivity/run
    bash scripts/cluster_bundle.sh panel888_final_budget results/panel888_budget_sensitivity/run
    # on uc3: bash submit_production.sh panel888_final_budget 48 all <commit>
    bash scripts/integrate_budget_sensitivity.sh <commit>   # verify, collect, evaluate, plots

The per-budget preparation ran on uc3 CPU nodes in `venv_offline` with the local
package versions.

## Feasibility ([feasibility.csv](results/panel888_budget_sensitivity/feasibility.csv))

R, S1/S2 and B match every budget within 5% on all 24 graphs; no walk component
ceiling binds. H can only observe the last 60% of the archive, so its target is
structurally unreachable for a few graphs; these cells are reported, not re-tuned:

| Budget | H unmatched (expected coverage) | saturated H panels (one draw) |
|---|---|---|
| 20% | CollegeMsg (15.5%) | 1 |
| 30%, 40% | CollegeMsg (15.5%), CollegeMsg surrogate (26.8%) | 2 |
| 50% | + Workplace (44.7%) | 6 |

Observation counts per budget (5 arms): 2.5% 360; 5% 360; 20% 358; 30% 356;
40% 356; 50% 348 — 12,828 new Qwen requests in total (5 arms × 2 configs, minus
the H cells above). Audit ([audit.json](results/panel888_budget_sensitivity/audit.json)):
no seed collision, no request ID shared with the main study or between budgets.

## Qwen run and results

One production chain (`panel888_final_budget`, 48 shards, commit `9db974e`).
Where a request's ID, configuration, repeat, prompt hash, payload hash and seed
were byte-identical to one already answered in an earlier (superseded)
workspace, that answer was reused (`cluster/reuse_answers.py`; 2,592 of 12,828
reused this way, all R/H); the remaining 10,236 were generated fresh. Archive
verified (checksums, requests, observations, runner, engine, model revision),
collected strictly: 12,828 of 12,828, no duplicates, no missing answers, 0
output-limit hits; one answer was in flight at a job deadline (technical
failure, not regenerated), one further answer never closed its reasoning (no
final answer). An independent re-parse agrees with the evaluation on every row.

S1 and S2 are 100% strictly valid at every budget on the real sources (both
configurations). Real sources, Qwen thinking MAE2 (equal-source, valid answers):

| Arm | 2.5% | 5% | 10% (main) | 20% | 30% | 40% | 50% |
|---|---|---|---|---|---|---|---|
| R  | 0.029 | 0.025 | 0.015 | 0.012 | 0.013 | 0.010 | 0.006 |
| S1 | 0.084 | 0.068 | 0.060 | 0.053 | 0.047 | 0.038 | 0.031 |
| S2 | 0.066 | 0.076 | 0.053 | 0.048 | 0.055 | 0.031 | 0.043 |
| H  | 0.244 | 0.209 | 0.227 | 0.249 | 0.243 | 0.220 | 0.188 |
| B  | 0.283 | 0.272 | 0.205 | 0.145 | 0.215 | 0.164 | 0.134 |

The same-information design-aware baseline (`primary_corrector`) achieves, on
the identical observations:

| Arm | 2.5% | 5% | 10% (main) | 20% | 30% | 40% | 50% |
|---|---|---|---|---|---|---|---|
| S1 | 0.085 | 0.069 | 0.060 | 0.053 | 0.047 | 0.038 | 0.031 |
| S2 | 0.031 | 0.026 | 0.014 | 0.013 | 0.009 | 0.006 | 0.006 |

S1's Qwen-thinking MAE2 tracks its same-information plug-in baseline almost
exactly at every budget, as expected: S1 shows no walker information beyond
what the plug-in already uses. S2 gives the model the traversal counts and
inverse-degree weights the design-aware correction needs; Qwen-thinking's error
drops somewhat relative to S1 but stays far above the design-aware estimator
computed from the identical information, at every budget in the grid. The model
partially exploits the walker information but does not approach the
statistically optimal use of it, and this pattern is stable across the whole
coverage range, not an artefact of the 10% operating point.

Evidence in [results/panel888_budget_sensitivity/](results/panel888_budget_sensitivity/):
`tidy_results.csv.gz` (one row per budget x observation x estimator x repeat: budget,
expected and observed coverage, graph, evidence block, arm, method, status, validity,
AE2, ProfileAE, signed error), `summary.csv` (equal-source MAE2/ProfileMAE with
draw-clustered MCSE and valid fractions per budget x block x arm x method), plots
`MAE2_<block>.png` (primary: `MAE2_real.png`; surrogate and the four synthetic
conditions separately), `ProfileMAE_real.png`, `valid_fraction_real.png`. The
x-axis is the achieved expected dyad-window coverage; H is capped where its
target is unreachable. The dotted line marks the main study's fixed 10%; it was
not chosen from these curves.
