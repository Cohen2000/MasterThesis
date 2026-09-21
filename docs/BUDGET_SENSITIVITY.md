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
