# Freeze 2026-09-16: implementation contract

> Current design: **`cells10-20260917`**, specified in the last section of this
> file. Earlier sections describe the designs it revises and remain valid for
> everything the revision does not change.

Authority: `MAIN_FREEZE_SOURCE.txt`, complete paragraph/table/math text extracted
from the supplied DOCX, plus the user's instructions. No applicable AGENTS.md
was found in this repository or its ancestors. Initial git status was clean.
Work branch: experiment/offline-freeze-20260916.

| Requirement | Existing code | Required change | Evidence |
|---|---|---|---|
| Canonical events, horizons | dataset_census.parse_audited | source-specific proximity deduplication, exact boundaries, CNS original comparison | data manifests, boundary fixtures |
| Four samplers, expected event volume | legacy walks/nonwalk_samplers | fixed panel, weighted walk, suffix, independent Bernoulli | exact fixtures, MC expectations |
| Walk calibration/validation | diagnostic-only scripts | 256 prefixes; independent 1024/4096; cap and flags | timings and per-source calibration files |
| Eight synthetic graphs | legacy DCSBM/event generators | frozen G(n,m) DAR and synchronous AD | algorithms below, paired latent hashes |
| Input and baselines | persistence_prompt/evaluation legacy | shared CSV block, 88 features, scalar correctors | roundtrips, numerical reference checks |
| Training | no current pipeline | 16 sources, seven fits, LOSO, equal weights | fold manifests, model hashes |
| Requests and evaluation | archived clients | offline manifests only; strict parser and outcome reducer | mock-only fixtures, 224/2688 inventory |
| Resume/reports | no current pipeline | atomic stage files, dependency hashes, locks | replay hashes and reports |
| Learned baseline on a diverse pool | ExtraTrees on 16 real sources | 400 training and 100 development synthetic graphs, 50/25/25 block weights, 133 features | frozen pool definition, fold manifests, development report |
| Two mixture correctors | homogeneous correctors only | zero-truncated Beta-Binomial suffix; Beta-mixed activity with the ZTP event layer | derivations, exactness tests, development table |
| Recent-cap arm H (2026-09-17) | suffix node panel, 3-to-5-window correctors | uniform dyad sample, five most recent events per dyad, `at_cap_dyads` column, bound-midpoint reference, 195 features | `tests/frozen_main/test_hrecent5.py`, `scripts/check_h_recent.py`, revision report |

## Generator algorithms (written before implementation)

DAR follows the binary copy-or-refresh recurrence in Williams et al., Eq. (1),
https://arxiv.org/pdf/1909.08134 (read Eq. 1 and backbone definition).
For each replicate draw a uniform 5000-element subset of the unordered pairs of
500 vertices: G(500,5000). There are exactly five states, one per archive window.
X[e,1] ~ Bernoulli(.2); for windows 2..5 copy the preceding state with
probability alpha, otherwise refresh with an independent Bernoulli(.2).
This is a stationary start; no burn-in. A potential edge/window receives
1+Poisson(1) events if active, placed independently uniformly inside that window.
This event layer is our operationalization, not a claim about the original model.
The paired alpha=0/.8 graphs share backbone, initial states, copy uniforms,
refresh uniforms, potential event counts and event positions, even when a state
is inactive in one member. The two replicates have independent graph streams.
No trimming of [0,1], and never-active substrate edges are outside E_full.

AD follows Perra et al., generative rules, https://arxiv.org/html/1203.5351,
and Karsai et al., section 0.2, https://arxiv.org/pdf/1303.5966 (both read).
Draw 500 activities by inverse CDF of a^-2.8 on [.01,1]. For rounds 1..1000,
activate each vertex independently with probability a_i (eta=1, Delta t=1).
Each active vertex chooses exactly one distinct partner. Memoryless choice is
uniform over the other 499 vertices. Memory choice uses the partner set at the
start of the round: choose a new partner with probability 1/(n_i+1), otherwise
an old partner uniformly. An empty old list forces a new partner; an exhausted
new list forces an old partner. Sorted candidate lists make selection stable.
All choices are simultaneous; at round end update the memories of both endpoints.
Mutual initiations collapse to one undirected contact per dyad/round, explicitly
confirmed by the user. Place that contact at (round-.5)/1000, so each window
contains exactly 200 rounds. These event times and synchronous updates are
our operationalization; original articles do not settle every implementation detail.
Paired modes share activities, activation uniforms, new/old decision uniforms,
and partner-quantile uniforms indexed by (round,vertex); differing candidate sets
can produce different partners. Replicates are independent. Horizon stays [0,1].

## Design revision: ten-percent budget (2026-10-01)

Design id `budget10-20261001`. The previous budget was the suffix event count,
about 56 % of the archive, and at that level three of the four arms observed more
than half of all active dyad-windows: the estimation problem was close to a
census. The budget is now a fixed **ten percent of the full event archive**, so
that the study measures inference under partial observation rather than counting.
W, the target population and the persistence definitions are unchanged.

Arm H changes from "every event in windows 3-5" to "a uniform node panel, then
every event in windows 3-5 inside that panel". This is what makes the lower budget
possible at all. The old H carried no free parameter -- it *was* the budget -- so
the only way to lower it was to shorten the suffix, and at two observed windows the
zero-truncated model has one free cell probability for two parameters and stops
being identifiable. Panel size is chosen as for R, from

    E[M_obs] = n(n-1) / (N(N-1)) * M_suffix,

so H needs the larger panel by the ratio M_full/M_suffix. Because the panel is
drawn over nodes without reference to any event, dyad inclusion is uniform and
independent of activity; conditional on inclusion a dyad's window pattern is
exactly what it was, so J | q ~ Bin(3,q) truncated at J >= 1 continues to hold and
both the homogeneous suffix corrector and the Beta-Binomial candidate remain valid
unchanged. Dyads sharing a node are included together, so dyads are *not*
independent; that affects the variance of the cell counts and therefore any
standard error computed as if they were, not the correctness of the likelihood.
The claim is checked empirically in
`tests/frozen_main/test_budget10_design.py::test_panel_leaves_the_window_count_distribution_alone`.

All four arms are now stochastic, so all four carry five samples: 280 main
observations, 3 360 planned calls, 1 680 of them Qwen. These sizes are derived in
`common.py` from the replication scheme instead of written as literals, so a change
to the scheme cannot leave a stale number behind. Arm B keeps each event with
probability exactly 0.10; R and H round their panel to the nearest integer and
report the resulting relative budget error.

Observed information actually achieved on the six real sources, as the median share
of active dyad-windows: R 9.8 %, S 2.4 %, H 10.7 %, B 37.1 %, against 57 / 20 / 55 /
78 % under the superseded design. Arm B still finds many dyads because one retained
event suffices to reveal a dyad; what it loses is which windows they were active in.

## Baseline revision 1 (2026-09-16)

Revision id `baseline-revision-1-20260916`. The learned reference was fitted on
sixteen real sources only and behaved accordingly outside their range, so it now
also trains on a frozen synthetic pool: 200 DAR and 200 activity-driven training
graphs plus 50 and 50 development graphs, drawn by a fixed stratified NumPy draw
whose admissible ranges are derived in `src/main_experiment/pool.py` from generator
semantics rather than from the known targets of the main-test instances. No new
generator family and no new LLM test graph is introduced. Each pool graph is drawn
on its own stream under the domain `pool`, disjoint from the main-test domain
`graph`, and is generated alone, so no latent quantity is shared between graphs.

Training weights are exactly 50 % real, 25 % DAR and 25 % activity-driven, equal
per graph inside a block, per arm inside a graph and per observation inside an arm,
so the far larger synthetic row count cannot outvote the real block. The
fold-specific real training median and the empty-sample replacement stay defined on
the real sources alone and are unchanged. Hyperparameters and the single shared
multi-output regressor are unchanged; no hyperparameter search was run.

The feature block grows from 88 to 133: 31 pattern dyad shares, 5 window event
shares, M_obs/D_obs, the 4 plug-in profile values and the 4 values of the existing
homogeneous corrector, the latter used as a fixed transform independently of the
candidate outcome. All of them are functions of the serialized observation input
only; no full-graph size, ground truth, realised coverage, source name, generator
parameter or generator family is a feature, and the LLM prompt is untouched. Empty
samples code every derived entry as zero rather than dividing by zero, and missing
windows stay distinguishable through the access indicators.

Two development-only correctors are implemented in `src/main_experiment/mixtures.py`,
which carries the likelihoods, the sufficiency arguments and the identifiability
limits. The suffix candidate is exactly saturated and is therefore reported as a
secondary estimate; the event candidate is over-identified and is recommended as
primary for its arm. Parameter bounds and tolerances were fixed before any
performance check. Results and their limits:
`results/baseline_revision_20260916/DEVELOPMENT_REPORT.md`.

## Numerical and engineering conventions

All empirical windows use source-time cutpoints lo + j*(hi-lo)/5 and searchsorted
with side=right; hi belongs to window 5. No epsilon shifts. Proximity duplicates
are collapsed only after direction collapse. Communication duplicates remain.
Node-panel integer ties choose the smaller n (allow 0..N); B must be positive.
Walk transitions use integer event weights and unbiased integer draws. SplitMix64
is the fixed walk PRNG; other streams use NumPy PCG64, with versions recorded.
Prefix calibration stores first-discovery volume increments, not event histories.
Simulation can stop early only when every edge in the start component has been
retrieved and only when traversal counts are not requested. Main walks always
execute all L transitions. Repeated doubling uses the same seeded paths; all
prefix means at the final bound are available for integer bisection.

Run outputs must not depend on whether a stage was just computed or reloaded from
a checkpoint. A reloaded artifact carries the sorted key order that `write_json`
imposes, while a freshly computed dictionary carries its literal order, so any
output that inherits that order has to be normalised. Three places needed it:
`requests.jsonl` is written with sorted keys; the calibration result is rebuilt in
a fixed key order before it reaches the `budget_summary.csv` columns; and the walk
timing pilot is skipped once calibration is complete, so a resume no longer
rewrites a finished checkpoint. Content-addressed identities (`prompt_sha256`,
`block_sha256`, model and feature hashes) were never affected, because `digest`
sorts keys. Two independent from-scratch runs are compared artifact by artifact,
and a resume must change no artifact at all.

## Design revision: recent-cap arm H (2026-09-17)

Design id `budget10-hrecent5-20260917`. **This revision was specified after the
results of `budget10-20261001`, including the Qwen answers, had been seen. It is a
documented revision, not a retroactive preregistration.** Everything not named
here is unchanged: W=5, the target rho_2..rho_5 over E_full, MAE2 as the main
metric, the ten-percent expected event volume, the sources, generators, labels,
the pool and its train/development split, arms R, S and B, the ExtraTrees
hyperparameters and the 50/25/25 block weights.

### Why arm H changed

Arm H is meant to isolate the effect of *bounded recent event history* under
uniform relation selection. The suffix node panel mixed two things: it hid whole
windows for every dyad, so its correctors had to extrapolate from three windows
to five, and it selected dyads through node pairs. The revision keeps selection
uniform over active dyads and bounds the history per dyad instead.

Literature, and what it does and does not cover:

* Gibbons (2001), *Distinct Sampling for Highly-Accurate Answers to Distinct
  Values Queries and Event Reports*, VLDB, 541–550: a uniform sample of distinct
  values with a limit on the rows kept per value. Above the limit, Gibbons keeps a
  reservoir sample of that value's rows and stores its exact occurrence count.
* Zhou et al. (2022), *TGL: A General Framework for Temporal GNN Training on
  Billion-Scale Graphs*, PVLDB 15(8), 1572–1580: temporal neighbour sampling is
  either uniform over past neighbours or restricted to the most recent ones, the
  latter being the usual choice for memory-based models.
* Vitter (1985), *Random Sampling with a Reservoir*, ACM TOMS 11(1), 37–57: a
  uniform fixed-size sample without replacement from a sequence. This is the
  within-dyad subset used by the mechanism comparison below.

**Own work.** The combination is our own: uniform active dyads as the distinct
values, the *most recent* events rather than Gibbons' reservoir, a cap of five,
no exact per-dyad count (only whether the cap was reached), the budget rule, and
the derivation of the per-dyad bounds and the midpoint reference. The literature
supports the building blocks, not this configuration or any expected ranking.

### Sampler

With m_e the full-archive event count of dyad e, D=|E_full|, B=round(0.10 M_full)
and C = sum_e min(5, m_e), the dyad count d is fixed before any draw as the
integer in 1..D whose expected volume d*C/D is closest to B, ties to the smaller
d. It never depends on a realised sample. A draw takes d dyads uniformly without
replacement from E_full (`numpy.random.Generator.choice`, stream
`(domain, graph, 'H-recent5', index)`), and each sampled dyad keeps its
min(5, m_e) most recent events. Under this design E[plug-in] equals the
population share of dyads whose *recent* window count reaches k, and the oracle
profile of the sampled dyads is a simple random sample of the true profile.

Four budget conditions are recorded separately in `budget.json` and never merged:
`h_relative_budget_error`, `h_target_unreachable` (C < B), `h_saturated` (d = D)
and `h_within_tolerance` (|error| <= 5 %). The cap is five for every source.
A saturated sample is deterministic, so the graph carries **one** H observation;
its three model answers are model repeats, not sampler repeats. Per-arm budget
matching is `budget_matched_by_arm`: R by its panel rounding, S by the walk
validation, H by its expected-volume tolerance, B exactly; `budget_matched` is
their conjunction. On the current data only `sp_highschool2013` is saturated
(C = 18 667 < B = 18 851, error −0.98 %, inside tolerance); no pool graph is.

**Computing the recent events.** Windows are ordered in time and the cut points
use `side='right'`, so every event of window j is later than every event of
window j−1 and ties never straddle a cut. The five most recent events therefore
fill the stored window counts from window 5 backwards (`sampling.recent_counts`).
`tests/frozen_main/test_hrecent5.py` checks this against an explicit timestamp
sort on random graphs with ties on cut points and repeated timestamps, and on
`sp_hospital` and `snap_collegemsg`; `verify_main_offline.py` re-derives every
stored H block and compares the retained counts with the backward fill.

### Input

All 31 non-empty five-window patterns, `Temporal_access=1,1,1,1,1`, all five
window counts, parameter line `n_dyads=d` (equal to D_obs by construction), and a
fourth table column `at_cap_dyads`: the listed dyads of that pattern with exactly
five retrieved events. That is *possibly truncated*, not *provably truncated*: a
dyad with exactly five events is at the cap without having lost anything. No
full-archive count and no true truncation flag reaches any estimator; a test
checks that two archives differing only before the retained events give
byte-identical blocks. The rule and the extra column are explained only in
`config/main_experiment/rule_H_recent5_v2.txt`; `system.txt`, `user_prefix.txt`
and the R, S and B rules are unchanged, so those prompts are byte-identical to the
previous design. The superseded rule text is kept verbatim as
`rule_H_suffix_panel_v1.txt`.

### Fixed H reference (own derivation)

For a listed dyad with fewer than five retrieved events the history is complete,
so K = J (observed active windows). For a dyad at the cap, every event later than
its earliest retained one was retained as well, so windows b+1..5 are complete,
where b is the earliest observed window; only windows 1..b−1 are unknown. Hence

    J <= K <= J + b − 1.

L_k and U_k are the shares of sampled dyads whose lower and upper bound reach k;
L_k equals the plug-in. The fixed reference is the midpoint (L_k + U_k)/2
(`baselines.h_midpoint`). The intervals bound the profile **of the sampled
dyads**; they are not guaranteed full-archive bounds. There is no validity rule
and no clipping of any estimate, LLM answers included, against them. The
reference is the primary H corrector and is also the value of the H
`corrector_rho_k` features.

The three-to-five-window extrapolation (homogeneous corrector and Beta-Binomial
candidate) is removed from the current arm H, including its derived features. It
survives only in the development variant `H_suffix_v1` (arm code outside `ARMS`),
which reproduces all 70 stored H blocks and prompts of `budget10-20261001` byte
for byte.

### Features and training

Feature version `features-v3-hrecent5-20260917`, 195 entries: the previous 88 base
entries with the H parameter slot renamed to `n_dyads`, plus 31 at-cap counts;
the previous 45 derived entries, plus 31 at-cap shares (at-cap count / D_obs).
At-cap entries are zero for R, S and B. Because the regressor is shared across
arms, all seven pooled folds and all seven real-only folds are refitted and all
arms are re-predicted. Hyperparameters and weights are unchanged; a saturated H
observation carries the whole H weight of its graph.

### Versioning and reuse

| Item | Previous | Current |
| --- | --- | --- |
| Offline run | `results/main_experiment/budget10_20261001` | `results/main_experiment/hrecent5_20260917` |
| Baseline revision | `results/baseline_revision_20261001` | `results/baseline_revision_hrecent5_20260917` |
| Training revision | `baseline-revision-2-budget10-20261001` | `baseline-revision-3-hrecent5-20260917` |
| H observation / request ids | `<graph>__H__s<i>` | `<graph>__H-recent5__s<i>` |
| H seed identifier | `H` | `H-recent5` (legacy variant keeps `H`) |
| Qwen answers | `qwen_archive_20261001.tgz` | R/S/B reused, H in `hrecent5_20260917_qwen/` |

Observation counts are derived from the calibrated budgets (`common.planned_sizes`);
no status, manifest or evaluation path carries a fixed H sample count. The current
run has 276 main observations (13×20 + 16), 316 training observations and 3 312
planned calls, 1 656 of them Qwen. R, S and B answers are reused only where
`scripts/check_qwen_reuse.py` finds identical request id, observation block,
prompt hash, request seed, stored-answer metadata, generation-relevant payload
and per-request sampling configuration. The payload fields that differ
(`response_format`, `stream`, `stream_options` versus `executed_*`) are the
documentation correction of commit 9c5b04b; `run_qwen_batch.py` never read them.
No previous H answer is used.

### Offline acceptance and the mechanism comparison

`scripts/check_h_recent.py` draws 20 samples per main graph on a separate stream
(one for the saturated graph) and reports the SRS check of the selection
component, the information actually lost over all dyads, signed and absolute
error components, coverage of the truth by [L,U], volume, at-cap share, width,
and the existing baselines with Monte-Carlo errors. On the same sampled dyads it
also keeps min(5, m_e) events drawn uniformly without replacement instead of the
most recent ones (identical volume by construction) and compares lost information
and plug-in error. That comparison uses no LLM call, fits no model, and does not
apply the recent-cap bounds to the uniform subsets.

### Evaluation additions

`scripts/evaluate_main_responses.py` keeps parser rule v2 as the main rule and
v3 as a separate sensitivity directory, and adds, all labelled secondary or
post-hoc: the fold's real training median as a visible reference, a fixed 50/50
shrinkage of each answer towards the plug-in and, separately, towards the training
median (the agreed "50/50 shrinkage" names no target, so both readings are
reported), the componentwise median of three answers, the spread of the three
valid answers, per-source rows, and the Monte-Carlo error split into its
sampler and model-repeat components (a deterministic draw has sampler component
zero). A cell whose answers were all replaced carries no numeric model value.

## Design revision: matching on active dyad-windows (cells10-20260917)

Design id `cells10-20260917`. **Specified after the results of
`budget10-hrecent5-20260917` had been seen; a documented revision, not a
retroactive preregistration.** It changes two things and nothing else: what the
arms are matched on, and how the final answer is produced.

### Why

The goal is the same amount of target-relevant observation under different
missingness mechanisms. The event budget did not deliver that: at ten percent of
the events the real sources showed R 9.8 %, S 2.4 %, H 45.8 % and B 37.1 % of their
active dyad-windows, because many events fall into the same active window of the
same dyad and add nothing to K_e. The target is built from exactly those units,
rho_k = |{e: K_e >= k}|/|E_full| with K_e the number of active windows of e.

### Matched quantity

Every arm is calibrated before any draw to the same expected number of observed
active dyad-windows, T = 0.10 * W with W = sum_e K_e. Only the experimenter uses W.

| Arm | Parameter | Rule |
| --- | --- | --- |
| R | n_panel | pi(n) W closest to T, pi(n) = n(n-1)/(N(N-1)); a panel keeps every cell of an included dyad |
| S | L | walk calibration as before (256 paths, doubling, integer bisection, 1 024/4 096 validation walks), but a first discovery adds K_e instead of m_e; transitions stay proportional to m_e |
| H | n_dyads | d sum_e J_e / D closest to T, J_e = windows of the five most recent events; cap 5 unchanged |
| B | p | the p with sum over active cells of 1-(1-p)^n_cell = T, solved by bisection |

Tolerance, flags and matching per arm are unchanged in form (|relative error| <=
5 %, walk MCSE <= 1 %, `budget_matched_by_arm`). Expected events and dyad shares are
reported as descriptors and are not matched; they now differ by design. A
feasibility check on the fourteen main graphs before implementation found every
arm reachable everywhere, no saturated H, and at ten percent of the cells event
shares of R 10 %, S 10-41 %, H 1-10 %, B 1-10 % and dyad shares of R 10 %, S 5-10 %,
H 10-15 %, B 10-27 %. The sampling rules in the prompt are unchanged; B's p and all
other parameters are graph-specific values, as before for R, S and H.

### Final answer

Both Qwen modes generate with vLLM's structured outputs, the final answer
constrained to `common.ANSWER_REGEX` (keys rho_2..rho_5 in order, values 0 or 1 or
a decimal in [0,1] with at most six places), with `reasoning_parser='qwen3'`. vLLM
applies the constraint once reasoning has ended: after the generated `</think>` in
thinking mode, and from the first generated token in non-thinking mode, whose
chat template closes the reasoning block in the prompt. Non-thinking is therefore
defined as a direct estimate without a derivation. In the previous design the
non-thinking mode wrote its derivation into the answer and never produced a
bare or fenced JSON object (0/828 under the main parser), which made its numbers
invisible behind the plug-in replacement. Parser v2 remains the main rule and
still checks monotonicity; a non-monotone answer keeps the plug-in replacement.
Sol and DeepSeek were already planned with JSON output. The sampling parameters of
the two modes stay those of the model card.

### Identity and reuse

Every arm carries the design tag in its seed identifier and in observation and
request ids (`R-c10`, `S-c10`, `H-recent5-c10`, `B-c10`). All observations,
baselines, models and all 1 680 Qwen answers are produced anew; nothing from an
earlier design is reused. The legacy suffix variant is unchanged.

### Unchanged

W=5, the target, MAE2 and its replacement rule, sources, generators, labels, the
pool definition and split, the four mechanisms and their prompt texts, the H cap,
the 195 features, ExtraTrees hyperparameters and block weights, the evaluation
additions, model, revision, template and runner.

### Known limits carried forward

The matching constant (ten percent of W) is not told to any estimator. ExtraTrees
can learn it from its training rows, the LLM cannot; this is part of the trained
reference's extra information. The two Qwen modes differ in sampling parameters as
well as in reasoning. Six real sources limit every real-source comparison.

### Execution

`scripts/run_cells10_offline.sh` runs the offline chain with resumable steps;
`scripts/cluster_bundle.sh` uploads; `cluster/submit_production.sh` submits the
generation, two resume rounds and the archive job as one dependency chain;
`scripts/finish_cells10.sh` collects and evaluates. `docs/HANDOFF_cells10.md`
records the state and the next command.
