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

*Why the panel does not disturb the model.* The panel is drawn over nodes without
reference to any event, so a dyad is included with probability n(n−1)/(N(N−1))
independently of its activity. Conditional on inclusion its window pattern is
exactly what it was, so J | q ~ Bin(3, q) truncated at J ≥ 1 continues to hold and
both the homogeneous suffix corrector and the Beta-Binomial candidate remain valid
unchanged. Checked empirically: over 30 draws the observed window-count
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

| Arm | selection | lost history | share of absolute error from lost history |
| --- | --- | --- | --- |
| R | −0.0002 | 0 | 0 % |
| S | **+0.134** | 0 | 0 % |
| H | +0.060 | **−0.161** | 72 % |
| B | +0.133 | **−0.517** | 80 % |

R and S retrieve complete histories, so their entire error is selection; for H and
B the lost windows dominate. This is the evidence for the split in the design:
simple methods on R and S, mixture models that address history loss on H and B.

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
* **Non-thinking mode** has no such channel: the template writes
  `<think>\n\n</think>` into the prompt, so the block is closed before generation
  starts. The model therefore derives *inside the answer field* and appends the
  JSON at the end. **0 of 840 answers are valid under the frozen rule.**

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

| Arm | Qwen thinking | Qwen non-thinking* | plug-in | best corrector | ExtraTrees pooled |
| --- | --- | --- | --- | --- | --- |
| R | 0.0199 | 0.0198 | **0.0197** | = plug-in | 0.0299 |
| S | 0.1384 | 0.1626 | 0.3222 | 0.0972 | **0.0615** |
| H | 0.2083 | 0.2250 | 0.0721 | 0.0779 (candidate) | **0.0590** |
| B | 0.1214 | 0.1145 | 0.0823 | 0.0555 | **0.0524** |

\* secondary analysis. Monte-Carlo standard errors are 0.001–0.025.

The pattern is the same in both modes and it is the substantive result:

* **Arm R: the model matches the plug-in** (0.0199 against 0.0197). The decomposition
  showed arm R has no history loss and a selection error of −0.0002, so the correct
  behaviour is to report what one sees. The model does exactly that and does not
  invent a correction it does not need.
* **Arm S: the model corrects a large bias, but not as well as a dedicated estimator.**
  It takes 0.322 down to 0.138 — a real correction of the walk's over-sampling —
  while the ratio corrector reaches 0.097 and the learned baseline 0.062.
* **Arms H and B: the model over-corrects and ends up worse than doing nothing.**
  On H it is three times the plug-in error, on B one and a half times. These are
  exactly the arms where the decomposition attributes 72–80 % of the error to lost
  history, so they require a model of what was lost, and the LLM's implicit model
  is worse than both the plug-in and the fitted ones.

On the synthetic strata the same ordering holds, with arm B on the high-persistence
families the worst case (thinking 0.37 on dar_a08 and 0.46 on ad_memory).

So on this task the LLM is useful exactly where the naive reading fails badly and
harmful where the naive reading is already decent. It never beats a purpose-built
estimator on any arm.

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
