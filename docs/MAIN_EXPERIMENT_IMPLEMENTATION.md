# Freeze 2026-09-16: implementation contract

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
