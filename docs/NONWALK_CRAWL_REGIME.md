# Crawl-regime extension: is a walk the best local observer?

Added 2026-08-25.  Exploratory, like the non-walk screen it extends; it does
not touch the frozen panel, its truth values, or any frozen LLM artifact.

## Why

`docs/NONWALK_SCREEN_PREREG_2026-08-20.md` compared walks against eight
non-walk designs, but only one of those families obeys the same access rule as
a walk.  A walk may query a node and follow what that node returns; it never
sees the global event stream.  Under that rule the reservoir, the chronological
prefix and the random window are out of scope, and `node_panel_full_history`
and `ego_recent_kall` are labelled diagnostic references in the prereg itself.
That left `ego_recent_k1/k5/k20` as the entire fair comparison, and ego
retrieval expands its frontier along a single edge -- the endpoint of the
node's newest event.  A walk therefore beat *one* crawl geometry, not the
crawl family.

This extension adds the two standard crawl alternatives so the claim "the walk
is the better local observer" rests on a real contrast.  It also adds the two
walk arms the earlier comparison file left out -- the recency-biased walk and
`recent_history_k5` -- because quoting the walk side by its most idealized arm
alone would stack the same comparison the other way.

## What was added

`neighbourhood_crawl` in `src/nonwalk_samplers.py` keeps the ego access
primitive exactly -- a queried node returns its `k` newest unique incident
records at `T_end`, only unique events are charged against the budget -- and
changes the frontier rule alone:

| strategy | frontier rule |
|---|---|
| `ego_recent_k*` (existing) | only the endpoint of the single newest event |
| `bfs_crawl_k*` | every neighbour appearing in the response |
| `forest_fire_k*` | each of those neighbours with probability 0.35 |

At equal `k` the three differ in crawl geometry and in nothing else, so the
comparison isolates breadth against depth.  `k=5` and `k=20` were run to keep
the ego pairs matched.

The seed key is per family (`_strategy_seed_key`): the k sweep inside a family
shares its restart order, but the families do not share one, because their
frontier rules consume the generator differently and a shared key would buy no
comparability.

## Result: group-macro ProfileMAE at budget 800

Every row is the same panel, the same target and the same evaluator; only the
observation design differs.  Lower is better.

| access design | regime | ET prompt-parity | best analytical |
|---|---|---:|---:|
| walk, time-agnostic | walk | **0.0571** | **0.0504** (occ MLE) |
| BFS crawl k=20 | crawl | 0.0669 | 0.1219 (plug-in) |
| walk, recency-biased | walk | 0.0676 | 0.0798 (occ MLE) |
| forest fire k=20 | crawl | 0.0682 | 0.1213 (plug-in) |
| walk, recent-history k=20 | walk | 0.0687 | 0.0830 (mask MLE) |
| walk, time-respecting | walk | 0.0699 | 0.1146 (mask MLE) |
| ego retrieval k=20 | crawl | 0.0717 | 0.1179 (plug-in) |
| ego retrieval k=5 | crawl | 0.0761 | 0.1488 (mask MLE) |
| walk, recent-history k=5 | walk | 0.0771 | 0.0909 (mask MLE) |
| BFS crawl k=5 | crawl | 0.0812 | 0.1533 (mask MLE) |
| forest fire k=5 | crawl | 0.0855 | 0.1478 (mask MLE) |
| ego retrieval k=1 | crawl | 0.0970 | 0.1686 (plug-in) |

The no-sample control sits at 0.145--0.154 for every arm, so all of these do
read their sample.

Three readings, in decreasing confidence:

**1. The analytical gap is the finding that survives.**  Every walk arm lies at
0.050--0.115 with a simple estimator; every crawl arm lies at 0.118--0.169.
The regimes do not overlap, and the ranking inside each regime barely matters
next to the gap between them.  The mechanism is visible in which estimator
wins: a crawl queries at `T_end` and returns the newest records, so its sample
is temporally censored by construction and the occupancy likelihood is
misspecified on it.  `occ_mle` is the worst estimator on every crawl arm
(0.16--0.36) and the best one on the two walk arms that spread over the whole
horizon.

**2. On the learned side the regimes overlap, and the walk advantage is
carried by the idealized arm.**  The time-agnostic walk (0.0571) is the only
design that clearly leads.  Behind it, BFS crawl k=20 (0.0669), the
recency-biased walk (0.0676), forest fire k=20 (0.0682) and the recent-history
walk (0.0687) sit inside a range of 0.002 on twelve groups, which this design
cannot resolve.  Stated carefully: *a realistic biased walk and a good crawl
are indistinguishable here once a learned model reads the features*; what the
walk keeps is the analytical route in reading 1.

The time-agnostic walk earns its lead by ignoring the arrow of time -- it may
traverse an edge at any of its event times, so its sample is spread over the
horizon rather than pulled towards one end.  That is an access assumption, not
a free improvement, and it is the least realistic of the five walk arms.  A
comparison that quotes only this arm overstates how much the walk design buys.

**3. Frontier geometry moves the crawl arms less than depth does.**  At `k=20`,
BFS beats ego retrieval (0.0669 against 0.0717) and forest fire lands between
them.  At `k=5` it reverses: BFS (0.0812) and forest fire (0.0855) are worse
than ego retrieval (0.0761), because a wide frontier spends a shallow budget on
many nodes with little history each.  Across both, moving `k` from 5 to 20 buys
more than any change of frontier rule.  BFS does crawl a connected region
rather than a chain -- a median of 1 restart per case against 19 for ego
retrieval -- but on this target that pays only modestly.

## Selection bias, median over the panel at budget 800

| strategy | selected/full mean-degree ratio | degree KS | queried nodes |
|---|---:|---:|---:|
| ego k=1 | 1.004 | 0.005 | 688 |
| ego k=5 | 1.129 | 0.103 | 207 |
| forest fire k=5 | 1.166 | 0.142 | 203 |
| ego k=20 | 1.188 | 0.208 | 51 |
| bfs k=5 | 1.222 | 0.187 | 216 |
| forest fire k=20 | 1.284 | 0.266 | 49 |
| bfs k=20 | 1.364 | 0.270 | 49.5 |

BFS is the most hub-biased design in the set -- every response feeds the
frontier, and high-degree nodes appear in more responses.  Forest fire's
burning damps that, which is what it is for.  Note the direction of the finding:
BFS is simultaneously the most biased crawl and the most accurate one, so on
this panel the bias is not what limits crawl accuracy.

## Limits

- One budget (800) carries the headline; the other four budgets are in the case
  files but were not read into this table.
- The walk arms are the five configured in `config/benchmark.yaml`.  A
  degree-corrected or Metropolis-Hastings walk is not implemented and is
  therefore not in the comparison.
- `burn_prob = 0.35` is the classic forward-burning value and was not tuned.
  A burn sweep would be the next question, not a rerun of this one.
- The crawl arms are learned-model comparisons on 12 groups, same as the rest
  of the screen; differences of a few thousandths are not resolvable here.
- Nothing was run through an LLM.  This is a design comparison between access
  models, not an addition to the model matrix.

## Reproduce

```bash
source .venv/bin/activate
export PYTHONPATH=src
seq 0 7 | xargs -P 8 -I{} python src/run_nonwalk_screen.py \
    --config config/nonwalk_screen.yaml --preset crawl_extra \
    --num-shards 8 --shard-id {}

python src/run_benchmark_walks.py \
    --config config/walk_crawl_compare.yaml --preset panel32_b800 \
    --manifest results/final_target_panel/panel32_final.csv \
    --out results/nonwalk_crawl_screen/walk_cases_b800_extended.csv.gz

python src/evaluate_nonwalk_baselines.py \
    --cases results/nonwalk_screen/panel32_cases.csv.gz \
            'results/nonwalk_crawl_screen/crawl_cases_shard_*.csv.gz' \
            results/nonwalk_crawl_screen/walk_cases_b800_extended.csv.gz \
    --strategies bfs_crawl_k5 bfs_crawl_k20 forest_fire_k5 forest_fire_k20 \
                 ego_recent_k1 ego_recent_k5 ego_recent_k20 \
                 time_agnostic_t time_respecting recency_biased \
                 recent_history_k5 recent_history_k20 \
    --out-dir results/nonwalk_crawl_screen/baselines
```

Sampling takes about three minutes on sixteen cores, the walk arms about
another minute, the evaluation under a minute.

Two reproduction checks were run rather than assumed.  The `ego_recent_*` rows
reproduce the earlier screen exactly (0.0717 / 0.0761 / 0.0970).  The three
walk strategies shared with `results/walk_nonwalk_comparison/walk_cases_b800.csv.gz`
reproduce that file feature-for-feature, which is what pins `seed: 20260820`
and `recency_decay_scale: 0.10` in `config/walk_crawl_compare.yaml` as the
settings the earlier walk arm was built with.
