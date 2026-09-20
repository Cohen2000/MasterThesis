# Final study audit — cells10-final-20260920

Freeze `aaaac491006aa6981aeebc95b913b8847da76120`. Final chain
`7066386 → 7066387 → 7066388 → 7066389`: all tasks COMPLETED, exit 0.
Archive readback and local verification: 3,678 listed files, zero mismatches.
The archive's "answers=1681" includes engine_inputs.json; actual request answers
are exactly 1,680 with 1,680 unique attempts, not 1,681 model responses.

## Results and integrity

- Planned/found/valid/invalid/missing: **1680 / 1680 / 1678 / 2 / 0**.
- Nonthinking: 840/840 valid. Thinking: 838/840 valid.
- R/S: 420 valid each. H/B: 419 valid plus one invalid each.
- Invalid outputs are exactly `{}`: H/ad_memoryless_r1/draw2/repeat2 and
  B/ad_memory_r1/draw2/repeat1, both Thinking. No repairs or regeneration.
- Zero duplicates, prompt/payload mismatches, tokenlimit hits, technical failures,
  empty final texts or unclosed reasoning. Maximum output: 58,394 tokens;
  total: 8,001,266. Generic JSON generation does not guarantee the required keys.
- All answers use the final generation version. No earlier R/S/H/B answers enter
  collection. Original raw bytes, attempts, runner, model revision and file hashes,
  generation parameters, rendered prompts and tokenizer/engine counts are checked.
- Frozen code and evaluator match the freeze commit. All 600 observation blocks
  are independently rederived. 153 tests pass. Forty Qwen metric cells and their
  hierarchical MCSEs are independently recomputed from raw final answers.

See [full arm × mode × stratum results](RESULTS_FINAL_20260920.md),
[source-level results](results/final_20260920/source_results.csv), and
[machine-readable audit](results/final_20260920/audit.json).

## Technical/documentary exceptions, conservatively resolved

1. The preparation manifest's protocol hash is exactly the SRW protocol at
   `ff59f93`, whereas the freeze contains the final prompt appendix. This predates
   this audit. All executable/configuration/prompt preparation hashes match; the
   final protocol matches `aaaac49`. The exception is recorded, not erased or
   silently rebound. The older sections describe revision history; where they
   conflict about answer reuse, the final appendix and final request manifest
   govern: **all arms generated anew**. The frozen protocol itself is unchanged.
2. The reference file still carried the SRW design/block hashes. A separate final
   reference binding is generated only after proving equal parsed observations,
   truths and all 134 features for all 600 main/training observations. All 560
   pooled/real-only learned predictions are recomputed from the 14 saved models,
   with model hashes and LOSO/development exclusion checked. Only design and
   block-binding metadata change; no prediction, fit, baseline or rule changes.
3. Stale README/runbook/results current-state claims were replaced by final-only
   instructions. Their originals remain in `archive/pre_final_audit_20260920/`.
   Frozen development dependencies stay at their bound paths; older runs/imports
   are historical, not final-answer sources. Bulk raw artifacts remain Git-ignored.

## Scientific review

R uniformly samples a fixed node panel, retaining induced dyads with full histories.
S starts uniformly over V_full and chooses each distinct neighbor uniformly for
exactly L steps; traversed dyads release full histories. H uses a uniform panel
and a common elapsed-time suffix at archive end (h=.60). B independently thins
event records. All are calibrated to expected 10% of sum K_e, not event count or
realized sample size. Main SRW coverage is 9.942%–10.064%; all main budgets pass.

Raw SRW traversal frequencies are a stationary/asymptotic working reference,
not a finite-walk unbiased estimator. Components, nonstationary starts and finite
mixing remain relevant. The H homogeneous zero-truncated Binomial extrapolator
assumes exchangeable homogeneous window activity and is not guaranteed correct.
No claim of unbiasedness is warranted for either. H remains h=.60 as a conservative
three-visible-window compromise, not an error-minimizing selected parameter.

H's panel/history decomposition remains available. At h=.60 the descriptive
net-history shares are 70.5% (rho2) / 81.3% (profile) for real sources, 86.4% /
91.6% for synthetic sources. History exceeds panel selection for rho2 in 5/6 real
and 8/8 synthetic sources; this is not universal source-wise history dominance.
Signed effects can cancel, so these shares are not fractions of total absolute error.

Final prompts are exactly reconstructed from observation blocks and the frozen
common/arm-specific templates. Only S includes Walk_A. No truth, source name,
full graph sizes, oracle profile, budget target or reference formula is supplied.
There are 5 sampler draws × 3 repeats for each graph/arm/configuration, with unique
final request IDs and seeds separated from development requests. Accessible zeros
are correctly distinguished from missing/inaccessible data in each arm's text.

## Thesis limitations and remaining release

Accuracy is conditional on valid answers, with no imputation. Invalid fractions
must accompany accuracy. Equal-source aggregation and draw-cluster MCSE condition
on the fixed sources, fitted models and calibrated lengths; they do not quantify
generalization across all possible networks or training sets. Synthetic families
have only two main instances each. Working-model misspecification, source/domain
shift, component/mixing effects and H time heterogeneity remain substantive risks.
The experimenter uses oracle totals for budget calibration, unlike deployment.
Prompt revisions followed development work; do not describe the final freeze as
an untouched original preregistration or claim causal attribution from this audit.

Sol/DeepSeek: all 840 requests per provider are prepared, audited and unstarted;
the final API ledger is empty. The offline study is ready for a separately
authorized technical release, **not yet authorized for dispatch**. Provider smoke
checks, accepted model IDs, prices/budget and explicit release approval remain
necessary. No API calls were made during this audit.
