# Budget sensitivity (ancillary study)

How does estimation error change with the amount of observable dyad-window
information? Grid fixed before any 10% Qwen accuracy was seen:
**b ∈ {2.5, 5, 10, 20, 30, 40, 50}%**, T_g(b) = b · sum_e K_e. The main study stays at
10% and is the 10% point of this analysis (reused by identity, never regenerated);
nothing was selected from these results.

Everything except the budget is the main study: the 24 graphs, W = 5, rho_2..rho_5,
the four arms with their calibration rules and 5% tolerance, prompts, 3 test draws
× 3 repeats, Qwen thinking/non-thinking, references, LOSO folds (surrogates in their
parent's fold), the independent synthetic training pool (5 training draws, refitted
ExtraTrees with the fixed hyperparameters and seed per budget), equal-source
evaluation and the strict parser. Sol and DeepSeek are not part of it.

Each non-main budget has its own sampler identity (`R-p888-20260921-b025`, ...), hence
fresh random streams, observation and request IDs; at 10% nothing changes (the main
`requests.jsonl` is reproduced byte-identically by the parameterised code).

## Running

    python scripts/budget_sensitivity.py prepare <b>     # per budget; on uc3: cluster/budget_sensitivity_prepare.sbatch
    python scripts/budget_sensitivity.py feasibility
    python scripts/budget_sensitivity.py audit           # builds results/panel888_budget_sensitivity/run
    bash scripts/cluster_bundle.sh panel888_budget_sensitivity results/panel888_budget_sensitivity/run
    # on uc3: bash submit_production.sh panel888_budget_sensitivity 36 all <commit>
    bash scripts/integrate_budget_sensitivity.sh <commit>   # verify, collect, evaluate, plots

The per-budget preparation ran on uc3 CPU nodes (jobs 7094332 environment check,
7094415 budgets) in `venv_offline` with the local package versions; the environment
check reproduced 36 main observations and 24 ExtraTrees predictions exactly. The
fitted forests stay in `$WS/budget_sensitivity_offline` (hashes in the model manifests).

## Feasibility ([feasibility.csv](results/panel888_budget_sensitivity/feasibility.csv))

R, S and B match every budget within 5% on all 24 graphs; no SRW component ceiling
binds. H can only observe the last 60% of the archive, so its target is structurally
unreachable for a few graphs; these cells are reported, not re-tuned:

| Budget | H unmatched (expected coverage) | saturated H panels (one draw) |
|---|---|---|
| 20% | CollegeMsg (15.5%) | 1 |
| 30%, 40% | CollegeMsg (15.5%), CollegeMsg surrogate (26.8%) | 2 |
| 50% | + Workplace (44.7%) | 6 |

Observations / Qwen requests per budget: 2.5% 288/1,728; 5% 288/1,728; 20% 286/1,716;
30% 284/1,704; 40% 284/1,704; 50% 276/1,656 — 10,236 new Qwen requests in total
(5,118 per mode). Audit ([audit.json](results/panel888_budget_sensitivity/audit.json)):
no seed collision, no request ID shared with the main study or between budgets.

## Qwen run and results

One chain from commit 2ac8373 (jobs 7094845 round 1, 7094846/7094847 rounds 2/3,
7094848 archive). All 10,236 requests had a result after round 1; the no-op
rounds 2/3 (they only admit never-started requests) were then cancelled so the
archive job could run. Archive verified (checksums, requests, observations,
runner, engine, model revision), collected strictly: 10,236 of 10,236, no
duplicates, no missing answers, 0 output-limit hits; one thinking answer was in
flight at a job deadline (technical failure, not regenerated), two thinking
answers never closed their reasoning (no final answer). An independent re-parse
agrees with the evaluation on all 11,964 Qwen rows including the 10% anchor.

Valid answers (thinking / non-thinking): 2.5% 862/864, 864/864; 5% 862/864, 863/864;
10% (main) 863/864, 864/864; 20% 855/858, 858/858; 30% 851/852, 851/852;
40% 852/852, 852/852; 50% 828/828, 826/828.

Real sources, Qwen thinking MAE2 (equal-source, valid answers):

| Arm | 2.5% | 5% | 10% (main) | 20% | 30% | 40% | 50% |
|---|---|---|---|---|---|---|---|
| R | 0.029 | 0.025 | 0.015 | 0.012 | 0.013 | 0.010 | 0.006 |
| S | 0.030 | 0.019 | 0.014 | 0.009 | 0.010 | 0.006 | 0.005 |
| H | 0.226 | 0.242 | 0.213 | 0.216 | 0.211 | 0.237 | 0.220 |
| B | 0.318 | 0.266 | 0.215 | 0.194 | 0.146 | 0.162 | 0.147 |

Evidence in [results/panel888_budget_sensitivity/](results/panel888_budget_sensitivity/):
`tidy_results.csv.gz` (one row per budget x observation x estimator x repeat: budget,
expected and observed coverage, graph, evidence block, arm, method, status, validity,
AE2, ProfileAE, signed error), `summary.csv` (equal-source MAE2/ProfileMAE with
draw-clustered MCSE and valid fractions per budget x block x arm x method), plots
`MAE2_<block>.png` (primary: `MAE2_real.png`; surrogate and the four synthetic
conditions separately), `ProfileMAE_real.png`, `valid_fraction_real.png`, and
`RESULT_FREEZE.json` with all hashes. The x-axis is the achieved expected
dyad-window coverage; H is capped where its target is unreachable. The dotted
line marks the main study's fixed 10%; it was not chosen from these curves.
