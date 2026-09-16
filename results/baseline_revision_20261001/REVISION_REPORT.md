# Design revision `budget10-20261001` — report

Branch `experiment/offline-freeze-20260916`. Superseded designs are kept:
`frozen_20260916` (suffix budget, 88 features) and `baseline_revision_20260916`
(pooled training at that budget) remain on disk as the development history.

## 1. What changed, and why

**The budget.** Previously every arm was calibrated to an expected observed volume
equal to the suffix event count — about 56 % of the archive. At that level three of
the four arms saw more than half of all active dyad-windows, so the task was closer
to counting than to inference. The budget is now a fixed **10 % of the full event
archive**. W, the target population and the persistence definitions are unchanged.

**Arm H.** It changes from "every event in windows 3–5" to "a uniform node panel,
then every event in windows 3–5 inside that panel". This is what makes the lower
budget possible at all: the old H had no free parameter — it *was* the budget — so
the only way to lower it was to shorten the suffix, and at two observed windows the
zero-truncated model has one free cell probability for two parameters and stops
being identifiable. The panel size follows the same rule as R, from
E[M_obs] = n(n−1)/(N(N−1)) · M_suffix, so H needs the larger panel by the factor
M_full/M_suffix.

*What the panel does and does not preserve.* The panel is drawn over nodes without
reference to any event, so a dyad is included with probability n(n−1)/(N(N−1))
independently of its activity, and conditional on inclusion its window pattern is
untouched. **Under the model's own assumptions** this preserves the marginal law of
the observed window count — J | q ~ Bin(3, q) truncated at J ≥ 1 — so the
homogeneous suffix corrector and the Beta-Binomial candidate keep the same working
likelihood. That is a statement about the marginal, not a proof that either
estimator is correct or unbiased. Uniform dyad inclusion does **not** make a ratio
estimator exactly unbiased: the ratio of two random totals is biased in finite
samples regardless of how uniformly the numerator and denominator were drawn.
Because a panel includes all dyads among the selected nodes, dyads sharing a node
enter together, so the per-dyad contributions are dependent and the likelihood is a
composite one in the sense of Varin, Reid & Firth (2011): consistent as an
estimating equation, but with standard errors that an independence assumption
understates. Checked empirically: over 30 draws the observed window-count
distribution moves by at most 0.005 from the full-suffix reference
(`test_panel_leaves_the_window_count_distribution_alone`). Dyads sharing a node are
included together, so dyads are **not** independent; that inflates the variance of
the cell counts and therefore any standard error computed as if they were, but it
does not affect the correctness of the likelihood.

**Replication.** All four arms are now stochastic, so all four get five samples:
280 main observations, 3 360 planned calls, 1 680 of them Qwen. These sizes are
derived from the replication scheme in `common.py`, not written as literals.
Running the chain end to end exposed three places that still hard-coded the old
5/5/1/5 scheme — the evaluator's cell shape, the mock check and the structural
verification — all now derived.

## 2. How much information is actually observed

Median over the six real main sources; the middle column is the one that determines
ρ, and absolute counts are given because shares alone hide how thin arm S is.

| Arm | dyads | **active dyad-windows** | events | D_obs median (min) |
| --- | --- | --- | --- | --- |
| R | 9.7 % | 9.8 % | 9.6 % | 1 472 (96) |
| S | 1.6 % | **2.4 %** | 10.2 % | 276 (19) |
| H | 12.7 % | 10.7 % | 10.4 % | 1 882 (110) |
| B | 43.4 % | 37.1 % | 10.0 % | 5 371 (607) |

Against the superseded design (57 / 20 / 55 / 78 % of active dyad-windows) this is
a five- to eightfold reduction. Every arm observes about 10 % of events by
construction; they differ in how those events are spread. Arm B still finds many
dyads, because one retained event reveals a dyad — what it loses is *which* windows
they were active in.

**Budget attainment.** Arm B keeps each event with probability exactly 0.10. The
two panel arms round to an integer panel and report the resulting error: across the
24 graphs the worst is 0.63 % for R and 2.58 % for H, and the worst walk-validation
deviation is 1.87 %, all inside the unchanged 5 % tolerance. All 24 graphs report
`budget_matched: true`, so unlike the previous pool run no case had to be carried
as unmatched.
No observation is empty (`empty_observations: 0`), so the frozen-median replacement
path is implemented and tested but never exercised.

## 3. Where the error comes from

An offline decomposition splits the plug-in error into the part caused by *which*
dyads a mechanism reaches and the part caused by what it loses about them. Full
histories are used for this evaluation only and never feed an estimator.

**The formula, stated exactly.** With `oracle` the profile over the observed dyads
computed from their complete histories,

    selection = oracle − truth,  history = plugin − oracle,  total = selection + history

is an exact identity. The share below is |history| / (|selection| + |history|): a
share of the **summed absolute components, not an additive share of |total|**. On
H and B the two components carry opposite signs and largely cancel, so their sum of
absolute values exceeds the net error several times over; `cancellation` reports
(|selection| + |history|) / |total|.

An earlier version of this report quoted 72 % and 80 % as if they were general.
They are the **development-graph** values. On the main observations, real and
synthetic separately:

| stratum | Arm | selection | history | net | history share | cancellation |
| --- | --- | --- | --- | --- | --- | --- |
| real | R | −0.0019 | 0 | −0.0019 | 0 % | 1.0 |
| real | S | +0.3222 | 0 | +0.3222 | 0 % | 1.0 |
| real | H | +0.127 | −0.192 | −0.065 | 60 % | **4.4** |
| real | B | +0.185 | −0.268 | −0.082 | 59 % | **5.5** |
| synthetic | H | +0.055 | −0.148 | −0.094 | 72 % | 2.2 |
| synthetic | B | +0.120 | −0.483 | −0.363 | 80 % | 1.7 |

R and S retrieve complete histories, so their entire error is selection and the
identity is trivial there. On H and B both components are large and of opposite
sign; on the real sources they are four to five times the net error. That is the
evidence for the design split — simple methods on R and S, mixture models on H and
B — but it also means the net error understates how much each mechanism actually
distorts.

## 4. Which baselines work, and under which conditions

### Development graphs (100 held-out synthetic, 2 000 observations)

| Arm | plug-in | homogeneous corrector | candidate | ExtraTrees pooled |
| --- | --- | --- | --- | --- |
| R | **0.0176** | = plug-in | — | 0.0185 |
| S | 0.1369 | 0.0318 | — | **0.0213** |
| H | 0.1010 | 0.1199 | 0.0380 | **0.0246** |
| B | 0.3839 | 0.1164 | 0.0719 | **0.0305** |

### Six real main sources (MAE₂, and the paired difference with its standard error)

| Arm | plug-in | corrector | candidate | ET pooled | what is actually distinguishable |
| --- | --- | --- | --- | --- | --- |
| R | **0.0197** | 0.0197 | — | 0.0299 | nothing beats the plug-in |
| S | 0.3222 | 0.0972 | — | 0.0615 | corrector −0.225 ± 0.063 (t 3.6): clearly needed |
| H | 0.0721 | 0.1705 | 0.0779 | 0.0590 | homogeneous corrector **+0.098 ± 0.023 (t 4.3): clearly harmful**; candidate repairs that (−0.093 ± 0.027, t 3.4) but only reaches parity with the plug-in (+0.006 ± 0.027, t 0.2) |
| B | 0.0823 | 0.0555 | 0.0585 | 0.0524 | everything within noise of everything else (all t < 2) |

**The honest reading.** On synthetic development data both candidates beat both
references by large margins. On the six real sources that advantage largely
disappears: the H candidate does **not** beat the plug-in, and on arm B no method
is distinguishable from any other at n = 6. What the real sources do show clearly is
that the *homogeneous* suffix corrector is actively harmful and that the walk
corrector on arm S is essential. Six sources is the binding constraint on
statistical power, and no amount of modelling fixes that.

The corrector choice was written down in `CORRECTOR_DECISION.md` **before** the real
sources were evaluated, and it is not revised in the light of the table above.

### Numerical behaviour of the candidates

`not_converged` never occurred in 1 000 development fits. Identifiability is
visibly weaker at this budget than at the old one: only 179/500 arm-B fits and
344/500 arm-H fits are `converged`, with 184 and 56 flagged `weakly_identified`.
The pre-registered fallback (unreliable fit → homogeneous corrector) fired on 3 %
of H and 16 % of B fits. Widening every parameter bound by two decades moves the
objective by ~1e−12 and the predicted profile by ~2e−8, so no fit is pinned by an
artificial bound.

Three documentation-versus-code defects were found and fixed: the optimiser
accepted any finite objective including the invalid-region penalty; the thinned
positive-Poisson closed form was stated for k ≥ 0 when it holds only for k ≥ 1
(the k = 0 term differs by 1/(e^λ − 1), and no computed quantity used it); and the
claim of "no clipping anywhere" contradicted a clamp in `predict_profile`, which
now raises beyond last-bit noise.

## 5. Serving Qwen on bwUniCluster

Everything below was verified on the cluster rather than assumed.

**Model.** `Qwen/Qwen3.6-35B-A3B` at the pinned revision
`995ad96eacd98c81ed38be0c5b274b04031597b0`, 26 shards, 71.9 GB in BF16, checked
against the shard index. The archived Qwen3.6-27B in the same workspace is *not*
used. Architecture `Qwen3_5MoeForConditionalGeneration`: 40 layers, 256 experts
with 8 active, 16 attention heads over 2 KV heads, native context 262 144.

**Stack.** vLLM **0.29.0**, transformers 5.17.0, torch 2.13.0+cu130, pinned in
`requirements.pinned.txt`. vLLM 0.11.0 — the newest release that resolves against
Python 3.9 — does not know this architecture; the model card asks for >= 0.19.0.
Sampling follows the card exactly: thinking `temperature 1.0, top_p 0.95, top_k 20,
presence_penalty 1.5`, non-thinking `0.7 / 0.80 / 20 / 1.5`.

**Measured configuration** (one H100, TP=1, `max_model_len` 262 144, max output
258 048, text-only):

| max_num_seqs | prompts | wall | output tokens | tok/s | s per answer | finish reasons |
| --- | --- | --- | --- | --- | --- | --- |
| 4 | 4 | 46 s | 19 767 | 428 | 11.5 | 4 × stop |
| **16** | 16 | 182 s | 149 348 | **819** | 11.4 | 16 × stop |

Model load 124 s, KV cache 851 196 tokens. Median output 9 546 tokens, maximum
14 224 — every generation ended on a regular end-of-sequence and **not one hit the
output limit**, so the 258 048 allowance is never the binding constraint. The
sixteen-sequence setting is the one used; it is measured rather than extrapolated.

**Three cluster problems that cost real time**, recorded because they will recur:

1. The modulefiles are Lmod `.lua`. A non-interactive shell loads classic Tcl
   modules, fails with `Magic cookie '#%Module' missing`, and silently leaves the
   system Python 3.9 in place — which then resolves cp39 wheels and an obsolete
   vLLM. Every script starts with `#!/bin/bash -l`.
2. The Qwen Triton kernels are JIT-compiled and need `nvcc`. The cluster modules
   stop at CUDA 12.8 while torch here is cu130; the matching 13.4 toolchain ships
   inside the venv as `nvidia-cuda-nvcc`, so the jobs point `CUDA_HOME` at it.
3. flashinfer 0.6.18 bundles its own CCCL headers and JIT-compiles against them.
   They are incompatible with that nvcc, so both its all-reduce kernel (reached
   only at TP > 1) and its sampler kernel kill the engine at startup.
   `VLLM_USE_FLASHINFER_SAMPLER=0` and TP=1 avoid both paths; vLLM's native
   sampler needs no compilation. TP=1 is possible at all because one H100 here has
   94 GB and the weights need about 66 GB.

**A correctness bug caught before any production run.** The probe reported zero
closed thinking blocks. The chat template opens the block in the *prompt*: with
thinking enabled the generation prompt ends with `<think>\n`, so the output
contains only the closing `</think>`; with thinking disabled the template writes
`<think>\n\n</think>\n\n` into the prompt and the output carries no marker at
all. The original split required a matched pair and would have handed the entire
reasoning text to the strict answer parser for every thinking answer. The split now
mirrors the template's own parsing and treats a missing closing tag in thinking
mode as "stopped inside the reasoning, no final answer exists".

## 6. Qwen results

1 680 of 1 680 planned Qwen calls completed: two modes x three repeats x 280
observations, no empty observations, nothing missing. Every answer ended on a
regular end-of-sequence; **no output-limit hit, no technical failure, no unclosed
reasoning block, no empty answer.** Output length 1 785 to 79 129 tokens, median
7 896, 13.8 million tokens in total, about 5.5 GPU-hours on H100s. The longest
answer used 31 % of the 258 048 allowance, so the limit never bound.

### The answer-format problem, and how it is handled

The two modes differ in where they put their derivation, and this is a property of
the chat template rather than of the models:

* **Thinking mode** gets a reasoning channel. The template ends the prompt with
  `<think>`, the model reasons and closes with `</think>`, and the final answer
  field contains the JSON alone. **840 of 840 answers are valid under the frozen
  rule** (606 bare, 234 after removing a whole-answer markdown fence).
* **Non-thinking mode** has the block closed in the prompt: the template writes
  `<think>\n\n</think>` before generation starts. The model then derives *inside the
  answer field* and appends the JSON at the end. **0 of 840 answers are valid under
  the frozen rule.**

  An earlier version of this report presented the closed channel as the *cause*.
  That over-claimed. The archived rendered prompts show the two modes differ only
  in those four tokens, and the system instruction — "Do not include explanations,
  additional keys, or Markdown in the final answer" — is byte-identical in both.
  Nothing forces a derivation into the answer field; the non-thinking mode simply
  does not follow the format instruction where the thinking mode does. This is an
  instruction-following difference under an otherwise identical prompt, which is
  why it is reported with a strict and a relaxed count in the manner of IFEval
  (arXiv:2311.07911) rather than resolved by choosing one number.

Under the frozen parser rule the non-thinking mode never produces a parseable
answer, so all 840 of its estimates become plug-in replacements and its MAE₂ is
identical to the plug-in by construction. That is a real statement about
instruction following, but it measures the template's asymmetry rather than the
model's estimation.

Both readings are therefore reported, and the second is labelled for what it is:

| | frozen rule | trailing-JSON extraction |
| --- | --- | --- |
| thinking | 840/840 valid | 840/840 (unchanged) |
| non-thinking | 0/840 valid | 785 valid, 51 with no JSON at the end, 4 non-monotone |

**The extraction rule was decided after seeing that the frozen rule yields zero
valid non-thinking answers.** It is a secondary analysis, kept in its own output
directory, and it changes nothing about the thinking mode. The frozen rule itself
was not modified.

### MAE₂ on the six real sources, against the offline baselines

| Arm | Qwen thinking | Qwen non-thinking* | plug-in | primary corrector | ExtraTrees pooled |
| --- | --- | --- | --- | --- | --- |
| R | 0.0199 | 0.0198 | **0.0197** | = plug-in | 0.0299 |
| S | 0.1384 | 0.1626 | 0.3222 | **0.0972** | 0.0615 |
| H | 0.2083 | 0.2250 | 0.0721 | 0.0779 (candidate) | **0.0590** |
| B | 0.1214 | 0.1145 | 0.0823 | 0.0585 (candidate) | **0.0524** |

\* sensitivity analysis under rule v3.

Paired differences against the **decided** references (CORRECTOR_DECISION.md:
corrector for R and S, the mixture candidate for H and B; pooled ExtraTrees as the
trained reference), Qwen thinking on the real sources:

| Arm | vs primary corrector | vs plug-in | vs ExtraTrees pooled |
| --- | --- | --- | --- |
| R | +0.0001 | +0.0001 | **−0.0100** |
| S | +0.0413 | **−0.1838** | +0.0770 |
| H | +0.1229 | +0.1362 | +0.1493 |
| B | +0.0623 | +0.0391 | +0.0690 |

An earlier version of this report compared against `run/baselines/*.json`, which
holds the *homogeneous* corrector and the real-only ExtraTrees — both development
references. On arm H that understated the gap by a factor of three: the reported
+0.038 was against the homogeneous corrector (0.171); against the decided candidate
(0.078) it is **+0.123**. Arm R is the one place the model beats the pooled learned
baseline, by 0.010.

**Two uncertainties, kept apart.** With the six sources held fixed, the
Monte-Carlo error of the source mean (variance over the five sampler draws' repeat
means, per Morris, White & Crowther 2019) is 0.004 / 0.018 / 0.025 / 0.013 for
R / S / H / B. The between-source standard error at n = 6 is 0.008 / 0.043 / 0.007 /
0.012. Neither dominates uniformly, they answer different questions, and no
equivalence is claimed anywhere from a difference that fails to reach significance
— per-source results are in `source_results.csv`.

The pattern is the same in both modes and it is the substantive result:

* **Arm R: the model matches the plug-in** (0.0199 against 0.0197). The decomposition
  showed arm R has no history loss and a selection error of −0.0002, so the correct
  behaviour is to report what one sees. The model does exactly that and does not
  invent a correction it does not need.
* **Arm S: the model corrects a large bias, but not as well as a dedicated estimator.**
  It takes 0.322 down to 0.138 — a real correction of the walk's over-sampling —
  while the ratio corrector reaches 0.097 and the learned baseline 0.062.
* **Arm H: genuine over-correction.** The signed error flips sign — the plug-in
  underestimates by −0.065, the model overestimates by +0.128 — so it does not
  merely miss, it corrects past the target.
* **Arm B: not over-correction.** An earlier version of this report claimed it was.
  The signed error does **not** flip: the plug-in underestimates by −0.082, the
  model by −0.017, so the bias is reduced by about 80 %. MAE₂ nevertheless rises
  from 0.082 to 0.121 because the dispersion across draws is 0.168. Higher MAE₂
  alone never establishes over-correction, and here it does not.

**The ranking is not the same on real and synthetic data.** That claim, also in an
earlier version, is false. Plug-in against Qwen thinking, by family:

| family | R | S | H | B |
| --- | --- | --- | --- | --- |
| real | plug-in | **Qwen** | plug-in | plug-in |
| dar_a0 | plug-in | **Qwen** | plug-in | **Qwen** |
| dar_a08 | Qwen | **Qwen** | plug-in | **Qwen** (0.523 → 0.370) |
| ad_memoryless | Qwen | **Qwen** | plug-in | plug-in |
| ad_memory | Qwen | **Qwen** | **Qwen** | **Qwen** |

Arm S is the only one where the model wins everywhere. On arm B it wins exactly
where the plug-in error is large (0.31–0.57) and loses where it is small (0.04–0.08).
Arm H is the only arm where the model loses almost everywhere. So the useful summary
is not "the model is worse on H and B" but: **the model helps when the naive reading
is badly wrong and hurts when it is already close**, with arm H the exception where
it hurts regardless.

## 7. Limits

* Six real test sources. Most differences on them are not statistically resolvable.
* The development graphs come from the same two generator families as the training
  pool, so ExtraTrees' large margin there is within-family generalisation. Its real-
  source margin is much smaller and mostly inside the noise.
* Arm H's fit remains exactly saturated on the aggregated window counts; the
  extrapolation into the two unobserved windows is a model assumption the data
  cannot check. The seven visible time patterns support descriptive exchangeability
  diagnostics, but because dyads sharing a node are dependent these are not turned
  into independent multinomial significance tests.
* The walk corrector on arm S is not claimed to be unbiased at finite, short walk
  lengths; at this budget the walk is short (median 276 discovered dyads).

## 8. Acceptance round of 2026-09-17

A review of commit `830dfb4` raised eight points. Six were confirmed defects, two
were claims of mine that the evidence does not support. Nothing below changes the
design, the estimator choice or any numerical rule in a direction that improves a
main-test result; the two changes that move a number move it the wrong way.

### Confirmed defects and what they changed

| # | Defect | Effect on results |
| --- | --- | --- |
| 1 | The response evaluator compared against `run/baselines/*.json`, the homogeneous corrector, not the corrector the decision record designates. | Arm H's paired difference was **+0.038**, against the decided candidate it is **+0.123**. Arm B: +0.066 → +0.062. |
| 2 | `_classify` returned the first matching label, so a boundary hit hid a start disagreement, and the fallback rule — which keys on the disagreement — never fired for those fits. | 17 of 1000 development fits changed prediction; fallbacks rose 94 → 111 (H 14 → 19, B 80 → 92). Development MAE₂ moved by +0.0004 (B) and +0.00002 (H), i.e. slightly **worse**. |
| 3 | `_flatness` accepted any finite profile objective, including the invalid-region penalty, and ignored `success`. A failed profile optimisation could therefore be read as a reliable identifiability diagnosis. | A `flatness_unavailable` flag now marks the absent diagnosis instead of it passing as good identifiability. |
| 4 | The `valid_only` path still used `range(1, 2 if arm=='H' else 6)`, evaluating arm H on its first sample only. | The conditional side analysis now uses all five samples for every arm. |
| 5 | The manifest payload declared `response_format: json_object` and streaming; the runner used the offline batch API with free generation and no streaming. | Payload now records what was executed and what was planned but unused. |
| 6 | The scratch workspace expires 2026-10-24 and held the only copy of the answers. | Archived; see below. |

### Claims of mine that the evidence does not support

* **"72 % / 80 % of the error comes from lost history."** Those are development-graph
  values and they are shares of the summed absolute components, not additive shares
  of the net error. On the real sources the shares are 60 % and 59 %, and the
  components are four to five times the net error because they cancel. Corrected in §3.
* **"The model over-corrects on H and B."** Only on H. On B the signed error does not
  flip; the bias falls from −0.082 to −0.017 and MAE₂ rises through dispersion.
* **"The ranking is the same on real and synthetic data."** False; arm B on dar_a08
  is a counterexample (plug-in 0.523 → Qwen 0.370). Corrected in §6.
* **"A closed thinking channel forces the derivation into the answer field."**
  Over-claimed. The rendered prompts differ only in four tokens and carry the same
  system instruction; this is an instruction-following difference, not a mechanism.
* **"Uniform dyad inclusion keeps the model valid."** Restated in §1 as preservation
  of the marginal law under the model's assumptions, with the composite-likelihood
  dependence and the finite-sample ratio bias named.

### Parser rules, versioned

| rule | fixed | thinking | non-thinking |
| --- | --- | --- | --- |
| v1 bare JSON | original contract | 606/840 | 0/840 |
| **v2 one whole-answer fence** | **before the run — main result** | **840/840** | **0/840** |
| v3 trailing JSON object | after the run — sensitivity | 840/840 | 785/840 |

v3 is applied identically to both configurations, not only to the one that prompted
it. Invalid answers keep the frozen replacement rule; the valid-only analysis is
reported separately and labelled as conditional. Mechanically selected examples —
first request ids in sort order, never chosen by outcome — with prompt tail, output
head and output tail are in `parser_rule_audit.json`.

### Archive

`results/main_experiment/qwen_archive/` and `$HOME/mainexp_archive_20261001` on the
cluster, outside the expiring workspace: **2 078 files, 94.2 MB**, containing the
1 680 raw answers, the 560 prompts as the tokenizer actually rendered them per mode,
the request manifest, all observation blocks, the model/tokenizer/template and all
26 weight-shard hashes, the pinned environment and 104 job logs. Every file is
listed in `CHECKSUMS.json`; read-back verification reports **zero mismatches** both
on the cluster and after transfer (tarball SHA-256 identical at both ends).

### Does any of this require a new Qwen run?

**No.** The 1 680 stored answers remain fully usable. Evidence: the model path,
architecture and dtype in every job log are the pinned ones; all 1 680 raw texts are
distinct and all 560 repeat-triples differ, so the per-request seeds took effect;
199 of 200 sampled answers quote their own `D_obs`; generated output tokens in the
job logs equal stored output tokens exactly (14 082 293), so nothing was generated
and discarded into the result set; and of 104 job attempts, 48 completed a shard,
44 found their shard already complete, and 12 produced no completed chunk at all —
**no shard generated in two attempts**, so no request was computed twice. Every
defect above is in the offline evaluation, not in the generation.
