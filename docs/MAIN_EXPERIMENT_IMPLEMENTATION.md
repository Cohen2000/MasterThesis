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
