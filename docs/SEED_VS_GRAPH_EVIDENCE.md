# More graphs or more walk seeds? A measured answer

Status: measured 2026-08-20 on the frozen 32-graph panel. Read-only probe, no
frozen artifact touched.

## Question

The supervisor offered two ways to make the final picture robust: evaluate
more graphs, or evaluate the same graphs under several walk seeds. Rather than
argue the trade-off, it can be measured, because the panel design is nested:
group > graph within group > walk seed within graph. Only the bottom level is
what an extra seed buys down.

The primary statistic of the main comparison is the group-macro mean error, so
that is what gets a standard error:

```
SE^2(G, I, S) = s2_group / G  +  s2_inst / (G*I)  +  s2_seed / (G*I*S)
```

An extra seed shrinks only the last term, which is already divided by `G*I`.

## What was run

```bash
SEEDS=8 BUDGETS="[400, 800, 1600]" DETAIL_BUDGET=800 \
    OUT_DIR=results/panel_seed_probe bash scripts/run_panel_budget_probe.sh

PYTHONPATH=src python src/report_seed_variance.py \
    --cases results/panel_seed_probe/cases.csv.gz --budget 800
```

32 panel graphs x 3 access strategies x 8 walk seeds, nested budgets in one
pass. 768 cases at budget 800. Runtime 3m54s on a laptop; no SLURM needed.
Outputs: `results/panel_seed_probe/cases.csv.gz`, `REPORT.txt`,
`SEED_VARIANCE.txt`.

## Result

Walk-seed noise is small and, more importantly, sits at the level of the
design where averaging has already removed it.

Variance shares of the per-case ProfileMAE (budget 800, across strategies and
estimators): graph-within-group 65-100%, walk seed 0-35%, group ~0%. The group
level being near zero is a design consequence, not an anomaly: each controlled
twin pair deliberately spans the persistence range inside its own backbone's
group, so most heterogeneity is within-group by construction.

SE of the group-macro ProfileMAE at budget 800:

| method | S=1 | S=2 | S=4 | S=8 | 2x graphs, S=1 |
|---|---:|---:|---:|---:|---:|
| supervised ExtraTrees | 0.0162 | 0.0159 | 0.0158 | 0.0158 | 0.0114 |
| occupancy MLE | 0.0169 | 0.0166 | 0.0164 | 0.0163 | 0.0119 |
| mask MLE | 0.0190 | 0.0188 | 0.0187 | 0.0187 | 0.0134 |
| mean floor (LOGO) | 0.0264 | 0.0264 | 0.0264 | 0.0264 | 0.0186 |
| naive read-off | 0.0326 | 0.0325 | 0.0325 | 0.0325 | 0.0230 |

Going from 1 to 8 walk seeds reduces the SE by 1-4%. Going from 1 to
*infinitely many* seeds never reduces it by more than 4%. Doubling the number
of graphs reduces it by about 29%. No number of seeds reaches what twice the
panel would give.

The same table at budgets 400 and 1600 gives the same verdict (seeds 1-8%,
double graphs ~29%), so this is not an artifact of the chosen budget.

The mechanism is visible directly: the within-graph coverage CV across 8 seeds
has median 0.021 and p90 0.075. The same budget on the same graph sees
almost the same fraction of it every time. The mean floor and the naive
read-off show exactly 0% seed variance, which is the expected sanity anchor.

## Consequences for the design

1. **One walk seed per case in the final LLM run.** Walk-seed replication is
   not where the uncertainty is, and every seed multiplies LLM cost by the
   number of cases. This frees the entire replication budget for other axes.

2. **Panel size is the binding constraint on resolution.** With 32 graphs,
   12 groups and 1 seed, the smallest paired difference in group-macro
   ProfileMAE detectable at 80% power (two-sided 5%) is:

   | comparison | cases | MDD |
   |---|---:|---:|
   | two methods on the same sample, all strategies pooled | 96 | 0.023 - 0.051 |
   | two methods on the same sample, one strategy only | 32 | 0.017 - 0.064 |
   | two access strategies (independent samples) | 96 | 0.025 - 0.044 |

   The range reflects how correlated the two compared methods are: closely
   related methods have the smaller MDD.

3. **The input ablation needs all three strategies, not a 32-case subset.**
   The historical within-model spread across input variants is about 0.027
   (`docs/LLM_EVIDENCE.md`), which sits at or below the 32-case detection
   threshold. Running the one-factor-at-a-time cells over all 96 cases shrinks
   the SE by sqrt(3) and moves the MDD to roughly 0.010-0.019, which resolves
   an effect of that size. On 32 cases the honest reportable result would be
   an equivalence statement, not a comparison.

4. **The open replication question is on the model side, not the walk side.**
   This probe covers sampling noise in the *input*. It says nothing about an
   LLM's own response noise under repeated sampling, which the historical
   evidence suggests is large: the ratio of within-case prediction spread
   across input formats to between-case spread runs 0.33-1.23 across runs.
   If replication budget is spent anywhere, repeated LLM sampling on fixed
   prompts is the candidate with actual measured variance behind it. That
   has not been measured yet and is the natural follow-up probe.

## Limits

- Measured on classical estimators only. LLM errors could in principle load
  differently onto the seed level, though they consume the same sample.
- Variance components use balanced-design moment estimators floored at zero;
  a near-zero component is reported as zero rather than negative.
- The MDD figures assume approximate normality of group-level means over 12
  groups. They are planning numbers, not the inferential procedure of the
  final analysis.

## Follow-up: the same question for the LLM (prepared, not yet run)

Everything above is measured on classical estimators, so it does not settle
the question for a language model. A model consumes the same sample but adds a
noise source the estimators do not have: its own sampling at generation time.
`src/make_llm_noise_probe.py` and `src/report_llm_noise.py` measure both.

Two arms sharing one cell, on 32 panel graphs, `time_agnostic_t`, budget 800,
`disclosed`/`mask`, 5 repeats:

- **response arm** -- walk seed 0 fixed, the identical prompt generated 5
  times. Within-graph spread is `s2_resp`.
- **input arm** -- 5 different walk seeds, one generation each. Within-graph
  spread is `s2_resp + s2_input`.

288 generations; `results/llm_noise_probe/prompts.jsonl`.

The reported design table extends the formula by one level:

```
SE^2 = s2_group/G + s2_inst/(G*I) + s2_input/(G*I*S) + s2_resp/(G*I*S*R)
```

with `S` walk seeds and `R` generations per prompt, so walk replication and
generation replication are priced against each other and against panel size.

**One implementation detail this depends on.** `run_llm_v2.HFModel` reseeds
torch before every generation, so on the cluster backend two identical prompts
decoded with the same seed produce byte-identical text. A response-noise arm
run without varying that seed would report exactly zero noise as an artifact
of the runner, not a property of the model. Records may now carry `gen_seed`,
which the runner prefers over `--seed` and echoes into the answer record; the
probe sets it in both arms, so the arms differ only in whether the walk was
redrawn. API backends are stochastic at temperature > 0 regardless.

This probe reports variance components only. Its error levels are not an LLM
result for the final panel, which has not been frozen yet.
