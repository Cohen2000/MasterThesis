# SRW revision: scientific assessment and result audit

Design: `cells10-srw-htime60-20260920`. Pre-generation source freeze:
`ff59f9340d11e0c4bfe6f8f8fd7fffb146cdef0b`. This is a prospective revision
after earlier results, not an original preregistration.

## Scope and provenance

The weighted-S implementation, configuration, tests, documentation, observations,
models and results are retained under `archive/pre_srw_20260920/`. The snapshot
comes from `96214c0`; larger artifacts remain local, as before. Earlier JSON and
recent-event H history remains under `archive/pre_h_time_20260920/`.

Only verified graphs are copied. SRW calibration, S observations and features are
new, as are all 7 control fits and 14 pooled/real-only reference fits. Refit all
arms because the training pool includes the changed S features. Hyperparameters,
labels, folds, weighting, statistical baselines for R/B/H and evaluation rules
are unchanged. Sensitivity and both structural diagnostics are rerun.

R/B original JSON answers and time-suffix H answers retain their original IDs,
seeds, generation-version fields and raw bytes. Reuse requires matching blocks,
messages, truth, payloads, rendered prompts, token counts, attempts, model files,
runner and generation configuration, plus full source-archive checksums. S IDs
and prompts are new; no weighted-S answers enter this revision.

`config/study.yaml` now specifies SRW, time-based H and the current 134 features.
The misleading current-revision comment in `common.py` was replaced. The supplied
September-16 freeze text is explicitly historical in the documentation index and
retained as a hash-bound source dependency, not the current protocol. Legacy
suffix/Beta-mixture and recent-event helper prose is not a definition of current
H. Likewise the shared non-S prompt's inapplicable Walk_A text is retained for
exact input reuse; non-S Walk_A is NA, and S receives the corrected raw-count text.

## Does S represent local/selection exploration?

Yes: its transition rule is purely topological and local, while each discovered
dyad releases its full history. There is no time censoring, restart, burn-in,
event-weighted transition or stopping at a realized volume. The chosen L executes
fully, including revisits. C++ paths match an independent Python SRW; event
multiplicity changes do not change paths. Uniform-start first-edge probabilities
and the stationary directed-edge distribution are tested independently.

All 14 main graphs meet the unchanged coverage tolerance: validated coverage
is 9.942%–10.064% of full active dyad-windows; L ranges from 124 to 21,290.
Twelve graphs are connected; CollegeMsg has 4 and MathOverflow 45 components.
The two disconnected graphs have node-vs-edge component-weight TV distances
0.00294 and 0.00343, respectively. Their analytical component-mixture ProfileAE
is approximately 0.00009 each. Every main target is structurally reachable.

The 32 fixed diagnostic paths per graph show 5.5%–21.8% repeated steps at L.
Increasing diagnostic length to 4L reduces mean ProfileAE relative to the
starting component's stationary profile on every main graph. This is evidence
of finite-walk effects, not proof of complete mixing at L or a pure separation
of initialization bias from sampling variance. Production L is not changed.

The raw traversal-frequency reference is stationary/asymptotically motivated:
stationary mass deg(v)/(2D) times transition 1/deg(v) gives 1/(2D) per directed
edge. Uniform node starts are not stationary in general. Disconnected components
and bipartite periodicity also prohibit a blanket finite-walk unbiasedness claim.
Plugin, training median and ExtraTrees remain separate comparisons.

## H assessment and h choice

I retain **h=0.60 as the main design choice**, not as an empirically proven optimum.
It leaves three full target windows visible and two censored, with common query
time at archive end. h=0.40 strengthens restriction but leaves only two visible
windows and increases reliance on the extrapolation model. h=0.80 weakens the
history contrast to R. No result-dependent change to H or its baseline is made.

The homogeneous zero-truncated Binomial extrapolator remains H's primary working
reference. Uniform panel sampling does not remove the bias from time-dependent
activity or heterogeneous dyad probabilities; neither correctness nor unbiasedness
is asserted. ExtraTrees sensitivity at h=0.40/0.80 uses models trained at h=0.60
and is explicitly cross-h evaluation, not separately optimized models.

History decomposition keeps full truth T, full-history panel P, full histories
of visible panel dyads Q, and censored profile C. Net history C−P includes both
dyad disappearance Q−P and lost windows C−Q. Compare it with panel selection P−T;
do not interpret ratios of these absolute components as fractions of total error,
because signed effects can cancel. Results apply to the fixed sources and draws.

## Execution and numerical results

Final numerical evidence is generated in `docs/results/srw_20260920/` by
`scripts/audit_srw_results.py` after full collection and evaluation. S production
uses jobs `7065982/7065983/7065984/7065985`; H uses the completed independent chain
`7065832/7065833/7065834/7065835`. No GPT/Sol or DeepSeek requests are admitted.
The all-provider completion flag therefore remains false even when Qwen completes.
