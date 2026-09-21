# Final study: panel888-pwt-srw-20260921

This is the sole final study panel: eight real human–human sources, eight matched
P[w,t] surrogates, and eight synthetic instances. Earlier runs are development
provenance only, archived in `archive/pre_panel888_20260921/`. This revision is
not a retroactive preregistration. Panel membership, the shuffle rule, three test
sampler draws and all changes were fixed for methodological reasons, without
selection on comparative new LLM accuracy. No old Qwen answer is reused.

Real sources: sp_hospital, sp_highschool2013, copenhagen_bluetooth, sp_workplace
(four physical/proximity), snap_email_eu, snap_collegemsg, snap_mathoverflow,
nr_digg_reply (four digital person–person). Each has one `__pwt` surrogate.
Synthetic: dar_a0_r1, dar_a08_r1, dar_a0_r2, dar_a08_r2,
ad_memoryless_r1, ad_memory_r1, ad_memoryless_r2, ad_memory_r2.
Generator parameters and within-r common-latent constructions are unchanged;
new versioned streams regenerate every graph and observation.

## Temporal control and random numbers

Uniformly permute the timestamp vector of already canonically cleaned parent
event records using PCG64 and a SHA256-derived, versioned deterministic seed.
Record index is stable event identity. Keep each record's dyad. Do not deduplicate
or reorder records after shuffling. The surrogate is a temporal event multiset:
new coincident dyad/time records retain multiplicity. Audit nodes, support,
per-dyad event counts, total events, exact timestamp multiset and archive bounds.
Only timestamp assignment is randomized; this can change many edge-local temporal
properties, not only persistence. Index 0 is the sole productive surrogate;
99 additional independent shuffles are offline null diagnostics, never candidates
for selection, never training data, never LLM input or additional sources.

Sample streams key on parent identity: R/H use the same random node permutation
within each arm, S uses the same start/transition stream and hence walk prefix,
B uses the same uniform random number per stable event index. Each graph is
calibrated separately to T=0.10*sum_e K_e, including each surrogate. Different
H panel sizes, walk lengths and Bernoulli probabilities are allowed.

## Fixed estimand, mechanisms and training

W=5, rho_2..rho_5, primary conditional MAE2, secondary conditional ProfileMAE.
R is uniform nodes with full histories. S is a single simple random walk with
uniform start and uniform distinct neighbor, no burn-in/restart; full histories
of traversed dyads. Walk_A occurs only for S and is raw traversal counts grouped
by K_e. H is uniform nodes with recent time suffix h=.60; B independently thins
event records. All arms match 10% expected active dyad-window cells. S calibration
uses 256 paths, validation 1024, extended to 4096 when relative MCSE exceeds .01;
5% matching tolerance and cap min(100*D,1000000). Report unreachable budgets.

Three test sampler draws per stochastic cell; saturated H has one deterministic
draw. Three model repeats per observation for every configuration. Five training
sampler draws are retained, including the independent synthetic training and
development pool (400/100 graphs, existing parameter grids). The 16 real training
sources are unchanged. Eight LOSO folds exclude the entire real parent source;
surrogates use that identical parent-excluded fold and never enter training.
Synthetic tests use the full real pool plus the independent synthetic training
pool for pooled ExtraTrees, and full real pool for real-only ExtraTrees. No main
synthetic test instance is training data. All models/medians/references refit fresh.
500 trees, existing 134 features and weights .50 real/.25 DAR/.25 AD unchanged.
Forest randomness is also newly versioned: SHA256-derived extratrees/all_folds
stream, modulo 2**32 for scikit-learn; the same forest seed is used in every fold.

Primary references remain R plugin-equivalent, S traversal frequency (stationary/
asymptotic within the component, not finite-walk unbiased), H homogeneous
zero-truncated Binomial (working model), B fixed beta-ZTP mixture with frozen
fallback. Plugin, median, pooled and real-only ExtraTrees remain reported.
Scientific prompt templates are byte-preserved from the last freeze. Fresh
observations, IDs, seeds, hashes and rendered requests are required for all arms.
Strict JSON with optional single whole-answer fence; no imputation/extraction.

## Evidence blocks and paired control

Real: equal weight for eight sources. Surrogate: eight parent–surrogate pairs,
reported separately, never sixteen independent real sources. Synthetic: four
conditions separately; r1/r2 are outer replicates, mode contrasts paired within r.
For matched arm, sampler index and model repeat, Delta rho = surrogate - parent
and Delta rhohat = predicted surrogate - predicted parent. AE_Delta_2 is primary;
mean absolute error across k=2..5 is secondary Delta_ProfileAE. Pairwise validity
requires both answers valid. Report coverage of valid pairs, source means and
equal-parent aggregation. If only one side is deterministic H, reuse its one
observation across counterpart draws without counting it as independent data.
Signed error, sign agreement and correlations are descriptive only.

## Offline analyses fixed before inference

See SENSITIVITY_INVENTORY_PANEL888.md for exact inherited grids and historical
exclusions. H .40/.60/.80, four-part oracle decomposition, cross-h predictions
from .60 models; SRW 32 independent common-prefix paths at L and min(4L,1000000),
component ceilings, stationary component profiles, finite-walk error, revisits,
traversal and hub concentration; W=2..20 including legacy {4,5,8}; calibration,
MCSE, budget-matched subsets, baseline fit diagnostics and error decomposition.
No diagnostic selects a primary parameter or changes production walk lengths.

## Execution

Qwen Thinking and Nonthinking are generated afresh only after offline audits and
commit/bundle verification, in one isolated production chain. Sol/DeepSeek are
prepared only and prohibited from dispatch. DeepSeek off-peak release is future
work. Durable job IDs and status prevent duplicate submission.
