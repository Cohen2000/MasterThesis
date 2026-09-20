# Protocol: cells10-srw-htime60-20260920

Prospective S revision requested after the preceding Qwen results. Prior state:
`archive/pre_srw_20260920/`, source commit `96214c0`. It is not a retroactive
preregistration. Primary H remains h=0.60; no H parameter is selected using new
Qwen results. Earlier H jobs continue in their own immutable workspace.

## Simple random walk and reference

The graph is the undirected simple full-archive active-dyad graph (no isolates
in V_full). Draw the start vertex uniformly over V_full. At every transition,
select one distinct current neighbor uniformly, irrespective of event counts,
times or K_e. Discovering a dyad reveals its entire archived event history.
Exactly L steps, no burn-in, no restart, no teleportation, no volume-based stop.
Previously visited edges can be traversed again. No crossing between components.

Choose one integer L per graph by the existing 256-path common-prefix calibration
to T=0.10*sum K_e, with cap min(100*D,1000000). Validate with 1024 independent
paths, extended to 4096 when relative MCSE exceeds 1%. Retain the fixed 5%
matching tolerance. New S seeds and observation/request identities; no previous
weighted-walk checkpoint, S observation or S response may be reused.
Volume-only calibration may stop computation once the whole starting component
has been discovered, since all remaining prefix volumes are exactly constant.
Actual sampling and all traversal diagnostics execute all L transitions.

Walk_A[j]=sum_{e:K_e=j} r_e, raw counts, sum_j Walk_A[j]=L. Primary S reference:
rhohat_k=sum_{j>=k}Walk_A[j]/L. On a connected undirected graph, the stationary
directed-edge distribution is uniform, so traversal frequencies motivate the
uniform-active-dyad target. This is a stationary/asymptotic working reference,
not an unbiased finite-L estimator. No division by event multiplicity is used.
On disconnected graphs a path only estimates its starting component. Uniform
node starts induce component weights N_c/N, generally different from D_c/D.
Bipartite periodicity is retained; time averages, not convergence of a single
time marginal, motivate the reference. No laziness or burn-in is added.

Report component counts/sizes, the exact expected coverage ceiling
sum_c (N_c/N)*sum_{e in c}K_e, unreachable-target flags, calibrated/validated
coverage, repeated traversals, distinct-dyad coverage and finite-walk reference
bias. Offline diagnostics compare L and min(4L,1000000) with 32 fixed independent
paths, plus analytic stationary component profiles. Longer walks are diagnostics
only; no length adjustment or model/reference selection follows these results.

## Unchanged H and other arms

R/B mechanisms and identities remain from cells10-json-20260918. H remains
uniform node sampling at common query time t_end with all events satisfying
t >= t_start+(1-h)*(t_end-t_start). h=0.60, sensitivities 0.40/0.60/0.80.
Calibrate node panel separately to the same T. The timestamp-based mechanism is
independent of W; the selected fractions align with 2/3/4 complete W=5 windows.
Primary H reference remains homogeneous ZT-Binomial extrapolation from
J|J>0 ~ Binomial(m,q), solving m*q/(1-(1-q)^m)=mean(J), then predicting
P(Binomial(5,q)>=k | K>0). It is a working model, not a correctness guarantee.

Preserve H oracle profiles T (full truth), P (full histories of all panel dyads),
Q (full histories of suffix-visible panel dyads), C (censored profile).
C-T=(P-T)+(Q-P)+(C-Q). Node selection=P-T; net history=C-P includes entirely
invisible dyads. Report signed/absolute terms, loss, cancellation and descriptive
history shares separately for each source, real/synthetic stratum and h.
History share uses absolute component means, not a fraction of absolute total
error. The h=0.60 trained ExtraTrees model is applied to h=0.40/0.80 only as
explicitly labeled cross-h sensitivity, not optimized separately.

h=0.60 is retained conservatively: a meaningful time restriction with three
visible windows, between the stronger model-dependence at h=0.40 and weaker
history restriction at h=0.80. This is a design compromise, not a universally
optimal h or one selected for lowest estimation error. Diagnosis must still
determine whether history effects exceed node selection on the fixed panel.

## Pipeline, provenance and evaluation

Same graphs, labels, 500 balanced pool definitions/seeds, training weights,
LOSO folds, hyperparameters and 500 trees. All S observations and calibrations
are new. Reproduce and verify unchanged R/B/H blocks and prompts. Feature count
remains 134 but S's A_j and corrector features change meaning; bump feature and
training versions and refit all 7 pipeline control and 14 reference models.
Reevaluate every arm with these references. Statistical plugin/median and
pooled/real-only ExtraTrees remain visible. B's fixed mixture/fallback unchanged.

New S Qwen generation only, both modes, three repeats, unchanged model revision,
generic JSON, generation limits and single-attempt policy. R/B original JSON
requests and H time-suffix requests may be reused only after full block/prompt,
payload, ID, seed, raw answer, runner and model-binding verification. Keep their
original generation-version fields and raw bytes; the containing study envelope
is this new version. Never use earlier S answers. No GPT/Sol/DeepSeek calls.

Conditional MAE2/ProfileMAE, valid fraction, equal source weights, hierarchical
MCSE conditional on the panel, strict JSON with one optional whole-answer fence,
matched reference comparisons, and no LLM imputation are unchanged. No trailing
JSON extraction, replacement, shrinkage or result-driven baseline changes.
`config/study.yaml` is checked against executable constants. The shared frozen
non-S prompt retains its inapplicable old Walk_A definition to preserve exact
input identity; S substitutes the raw-count definition and explicitly explains
the new transition rule. In non-S inputs Walk_A is always NA.
