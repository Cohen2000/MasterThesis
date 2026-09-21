# Study protocol (design identifier panel888-pwt-srw-20260921)

The study panel has eight real human–human sources, eight matched P[w,t]
surrogates and eight synthetic instances. Earlier development runs are archived
under `archive/` and are not part of the study. This protocol is not a
retroactive preregistration; panel membership, the shuffle rule, the mechanisms,
draws and references were fixed without selection on LLM accuracy.

Real sources: sp_hospital, sp_highschool2013, copenhagen_bluetooth, sp_workplace
(four physical/proximity), snap_email_eu, snap_collegemsg, snap_mathoverflow,
nr_digg_reply (four digital person–person). Each has one `__pwt` surrogate.
Synthetic: dar_a0_r1, dar_a08_r1, dar_a0_r2, dar_a08_r2,
ad_memoryless_r1, ad_memory_r1, ad_memoryless_r2, ad_memory_r2; the two modes of
one family and replicate share all generator latents.

## Temporal control and random numbers

Uniformly permute the timestamp vector of the canonically cleaned parent event
records using PCG64 and a SHA256-derived, versioned deterministic seed. Record
index is stable event identity; each record keeps its dyad; records are not
deduplicated or reordered after shuffling, so new coincident dyad/time records
retain multiplicity. Audit nodes, support, per-dyad event counts, total events,
exact timestamp multiset and archive bounds. Only timestamp assignment is
randomized; this can change many edge-local temporal properties, not only
persistence. Index 0 is the sole productive surrogate; 99 further shuffles are
offline null diagnostics, never selected, trained on or sent to a model.

Sampler streams key on the parent: R/H use the same node permutation, S the same
start/transition stream (hence the same walk prefix on the identical support), B
the same uniform per stable event index. Each graph, including each surrogate, is
calibrated separately to T = 0.10 * sum_e K_e.

## Estimand, mechanisms and references

W = 5, rho_2..rho_5; primary conditional MAE2, secondary conditional ProfileMAE.
All arms match the expected number of observed active dyad-windows to T within
5%; unreachable targets are reported, never re-tuned.

- R: uniform node panel; complete histories of all dyads inside the panel.
- S: one degree-biased random walk (alpha = 1): uniform start vertex; from u the
  next vertex v has probability d_v / sum_{x in N(u)} d_x, with d the number of
  distinct neighbours in the full-archive support; event multiplicities do not
  affect transitions; exactly L transitions, no burn-in or restart. Every dyad
  traversed at least once is observed once with its complete history. L is
  calibrated on unique active dyad-window discovery: 256 calibration walks,
  1024 validation walks (4096 if the relative MCSE exceeds .01), cap
  min(100 D, 10^6).
- H: uniform node panel with elapsed-time history h = .60 (events with
  t >= t_start + .4 (t_end - t_start)).
- B: independent Bernoulli thinning of event records.

The model-side observation of every arm is the same kind of table: distinct
observed dyads grouped by window pattern with dyad and event counts, window event
totals, the arm's design parameter (n_panel, L, n_panel_history with h, p) and
the arm's sampling rule. For S nothing about traversal counts, revisits or
degrees is shown. The LLM, the plug-in and the learned references use only this
observation: 129 features computed from the block (counts, shares, design
parameter, plug-in profile and the arm corrector computed from the block).

Primary references: R plug-in-equivalent corrector; S the stationary
inverse-traversal-weight (Hájek / Hansen–Hurwitz type) estimator on the full
traversal sequence including repeats,
rho_k = sum_t I(K_{e_t} >= k)/(d_u_t d_v_t) / sum_t 1/(d_u_t d_v_t),
computed internally from the traversal log and degrees (consistent for the
walk's component-mixture target; not finite-sample unbiased); H homogeneous
zero-truncated Binomial working model; B beta-ZTP mixture with fixed fallback.
Plug-in, arm corrector, training median and pooled / real-only ExtraTrees are
reported for every arm.

## Draws, training and evaluation

Three test sampler draws per stochastic cell; a saturated H panel has one
deterministic draw. Three model repeats per observation and configuration
(Qwen thinking, Qwen non-thinking; Sol and DeepSeek requests are prepared but
not dispatched). Five training draws per arm for the 16 real training sources
and the independent synthetic pool (400 training / 100 development graphs).
Eight LOSO folds remove the entire real parent source; its surrogate uses the
same fold and never enters training; synthetic test instances use the
all-real fold. ExtraTrees: 500 trees, block weights .50 real / .25 DAR / .25 AD,
one versioned forest seed (SHA256 of extratrees/all_folds, modulo 2^32).
Answers are parsed strictly (one JSON object, optional single whole-answer
fence); invalid answers are never imputed.

## Evidence blocks and paired control

Real: eight sources weighted equally. Surrogates: eight parent–surrogate pairs,
reported separately. Synthetic: four conditions separately; r1/r2 are outer
replicates. For matched arm, sampler index and repeat, Delta rho = surrogate -
parent and Delta rhohat = predicted surrogate - predicted parent; AE_Delta_2 is
primary, Delta_ProfileAE secondary; a pair needs both answers valid. If only one
side is a deterministic H draw, its single observation is reused across the other
side's draws without counting as independent data. Signed error, sign agreement
and correlations are descriptive only.

## Offline analyses

Listed in SENSITIVITY_INVENTORY_PANEL888.md: H at h = .40/.60/.80 with the
oracle decomposition; the walk construct-validity check (1000 walks per graph at
L and 4L: plug-in and design-reference bias/variance, component and degree
selection targets, coverage, revisits, concentration); W = 2..20 incl. {4,5,8};
calibration and MCSE; the P[w,t] null; B mixture bounds; error decomposition.
The budget sensitivity (BUDGET_SENSITIVITY.md) repeats the study at 2.5–50%
coverage as an ancillary analysis. No diagnostic selects a design parameter.

## Execution

Qwen answers are generated on the cluster from the committed, sealed prompts in
isolated production chains with durable job IDs; a chain never regenerates an
admitted request. Each answer is bound to its request by ID, prompt hash,
payload hash and seed, and verified against the runner, engine and model identity.
