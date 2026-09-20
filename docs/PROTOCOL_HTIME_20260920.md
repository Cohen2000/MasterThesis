# Protocol: cells10-htime60-20260920

Specified after the JSON-design Qwen results were known: an explicit prospective
H revision, not a retroactive preregistration. Primary h=0.60 is fixed by the user;
sensitivity h={0.40,0.60,0.80} cannot select a different primary value.

## Sampling and estimand

Graphs, archive horizons, W=5 targets, R/S/B samplers and seeds, pool definition
and graph seeds stay unchanged. H samples n vertices uniformly without replacement
from V_full. At common query time t_end it retrieves all events of induced dyads
with t >= t_start+(1-h)*(t_end-t_start), including cutoff and end timestamps.
No event-count cap or per-dyad query/horizon is used. Sampling is defined on
timestamps independently of W. The three h values align with 2, 3 or 4 full
target windows; the timestamp filter also supports nonaligned cutoffs.

Let J_e count visible active windows before node selection. Choose integer n in
0..N minimizing |pi(n)*sum J_e - 0.10*sum K_e|, pi(n)=n(n-1)/(N(N-1)); ties
choose smaller n. An empty suffix uses the full panel and flags infeasibility.
Report unreachable target, saturation and existing 5% matching tolerance separately.
Never adapt h. At saturation use one deterministic observation, otherwise five.
The full sizes, coverage target and truth are never estimator inputs.

H seeds/IDs are new. Use a uniform node permutation per draw and its first n nodes;
common orderings pair the h sensitivities with nested panels. Training and sample
domains remain separate. Inputs state n_panel_history and History_fraction.
Inaccessible windows are ? and NA, not observed zero. N_obs counts endpoints of
retrieved events; suffix-invisible panel dyads are absent. R/S/B blocks and prompt
texts remain byte-identical.

## Statistical reference and training

Primary H reference: homogeneous zero-truncated Binomial working model.
For m visible full windows, J|J>0 ~ Binomial(m,q) truncated at zero. Estimate q
by m*q/[1-(1-q)^m]=mean(J), the conditional maximum-likelihood equation.
Boundary mean(J)=1 gives limiting q=0; mean(J)=m gives q=1.
Predict rho_k=P(Binomial(5,q)>=k | K>0), k=2..5. Empty inputs have no fitted q;
the existing flagged training-median fallback for statistical references remains.
LLM failures receive no reference replacement. Homogeneous independent stationary
window activity is a working assumption, not an unbiasedness/correctness claim.
No new H mixture, Markov model or tuning is introduced.

R plug-in, S walk ratio and B's fixed mixture/fallback stay primary. Plugin,
real-fold training median, pooled and real-only ExtraTrees remain visible.
Remove at-cap counts/shares and add history fraction and the H node panel size:
134 observed-only features. Refit seven pooled and seven real-only folds, plus
the offline pipeline's seven control fits. LOSO exclusions, 500 trees, parameters,
50/25/25 weights and balanced 500-graph pool stay fixed. Pool graphs/labels and
R/S/B observations are reproduced and verified against the archived run.

## Offline sensitivity and structural diagnostics

For each main graph/h recalibrate n to the same cell target; evaluate plugin,
H extrapolator, median and primary-h trained ExtraTrees. Applying the latter
at h=0.40/0.80 is explicitly cross-h sensitivity, not an independently trained
reference. Qwen is generated only for primary h=0.60.

Compute profiles T (full truth), P (full histories of all full-archive active dyads
induced by the selected panel), Q (full histories of its suffix-visible dyads),
C (their censored profile). C-T=(P-T)+(Q-P)+(C-Q). Node selection is P-T;
net history effect is C-P, including disappearance of entire dyads. Q-P isolates
that disappearance, C-Q within-dyad window loss. Report lost cells/dyads and
undefined empty profiles explicitly. These oracle quantities never enter training
features or estimator selection.

Report signed and absolute rho_2/profile components per source and h, then equal
source averages separately for real/synthetic strata. Descriptive history share
is mean(|C-P|)/(mean(|P-T|)+mean(|C-P|)); profiles use coordinatewise absolute
means. Above 1/2 means the average net history component exceeds node selection
on this stratum, not an additive share of total error or universal dominance.
Also report source shares, cancellation and paired MCSE. A time restriction alone
does not establish history domination. No diagnostic changes h or references.

## Generation, reuse and evaluation

New H requests have new prompts, sampling identity, seeds, IDs and payloads.
Separate workspace cells10_htime60_20260920 on uc3; dispatch H only. Model revision,
template, generic JSON, mode parameters, one-attempt policy and parser remain
fixed. Offline token/template/shard checks precede generation. No GPT/Sol or
DeepSeek calls. Old H responses are excluded.

R/S/B generation identity deliberately remains cells10-json-20260918: complete
payload including version, request ID, seed, prompt and block must match archived
originals exactly. Only the containing study envelope has the new design version.
Reuse evidence binds original answer/attempt hashes, runner/model/generation
bindings and new requests. Copy original answer bytes without rewriting them.
New ExtraTrees references require reevaluation of all arms, not regeneration of
identical inputs. Any mismatch blocks reuse.

Reliability, conditional MAE2/ProfileMAE, equal source weighting, hierarchical
MCSE, matched reference differences, strict JSON and no LLM imputation remain
unchanged. No post-hoc shrinkage. Previous code/docs/evidence and local bulk
outputs are retained under archive/pre_h_time_20260920. New run directories bind
code, this protocol, data, observations, references and responses by hashes.
