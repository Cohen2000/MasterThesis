# Design `cells10-20260917` — report

Branch `experiment/offline-freeze-20260916`. Specification: last section of
[`docs/MAIN_EXPERIMENT_IMPLEMENTATION.md`](../../docs/MAIN_EXPERIMENT_IMPLEMENTATION.md),
fixed in commit `d225bff` (label fix `fix: training revision label for cells10`)
before any data or answer of this design existed. State and continuation:
[`docs/HANDOFF_cells10.md`](../../docs/HANDOFF_cells10.md).

**Status.** Specified after the results of `budget10-hrecent5-20260917` had been
seen; a documented revision, not a retroactive preregistration. Earlier designs and
their artifacts are unchanged.

## 1. What changed, and why

* **Matched quantity.** The goal is the same amount of target-relevant observation
  under different missingness mechanisms. Matching event volume did not deliver
  it: the real sources showed R 9.8 %, S 2.4 %, H 45.8 % and B 37.1 % of their
  active dyad-windows. Every arm is now calibrated to the same expected number of
  observed active dyad-windows, ten percent of sum_e K_e — the units the target is
  made of. Events and dyad shares are reported, not matched.
* **Final answer.** Qwen's final answer is constrained to the rho_2..rho_5 object
  after reasoning has ended (vLLM structured outputs, `reasoning_parser=qwen3`).
  Thinking reasons freely first; non-thinking becomes a direct estimate. In the
  previous design non-thinking wrote its derivation into the answer and had no
  valid answer under the main parser.

Everything else is unchanged: W=5, target, MAE₂ and replacement rule, sources,
generators, labels, pool and split, mechanisms and prompt texts, H cap, features,
ExtraTrees settings, evaluation, model and runner.

## 2. Produced anew, reused

Everything of this design is new: observations (main, training, pool), budgets and
walk calibrations, all 14 fitted folds, all baselines, the development check, the
decompositions, the H check and all 1 680 Qwen answers. Nothing is reused
(`check_qwen_reuse.py`: 0 of 1 680). Graphs, labels and the pool definition are
the same as before; the pool rebuild reproduced all 500 labels. The first offline
pass carried a stale training label; it was discarded and rerun before generation.

## 3. Matching and coverage

Largest absolute deviation of the expected matched quantity from its target over
the 24 graphs: R 0.58 %, S 1.12 % (walk validation), H 0.21 %, B exact (p solved,
0.0079–0.0992). All graphs are matched on every arm; no H sample is saturated, so
every arm has five draws: 280 main and 320 training observations, 3 360 planned
calls, 1 680 of them Qwen. Prompts are at most 1 429 tokens.

Median shares on the main panel, production draws:

| stratum | arm | active dyad-windows | events | dyads | D_obs median (min) | previous design: windows / events |
|---|---|---|---|---|---|---|
| real | R | 10.0 % | 9.6 % | 10.0 % | 1490 (95) | 9.8 % / 9.6 % |
| real | S | 10.0 % | 33.9 % | 5.9 % | 1043 (68) | 2.4 % / 10.2 % |
| real | H | 10.0 % | 2.1 % | 13.8 % | 1828 (156) | 45.8 % / 10.0 % |
| real | B | 10.0 % | 1.2 % | 13.2 % | 1920 (155) | 37.1 % / 10.0 % |
| synthetic | R | 10.0 % | 10.0 % | 10.0 % | 260 (144) | 10.0 % / 10.1 % |
| synthetic | S | 10.0 % | 10.8 % | 8.5 % | 224 (116) | 9.3 % / 10.0 % |
| synthetic | H | 10.0 % | 9.4 % | 10.9 % | 290 (227) | 10.6 % / 10.0 % |
| synthetic | B | 10.0 % | 5.5 % | 19.0 % | 480 (420) | 17.9 % / 10.0 % |

The active dyad-windows now agree across arms (10.0 %). Event volume and dyad
coverage differ by mechanism. On the real sources S retrieves a third of all
events: its event-weighted walk favours event-heavy dyads, whose full histories
carry many events per active window. H and B retrieve 1–2 %: on event-dense
sources an active window holds many events, and one retrieved event already
reveals it.

## 4. Offline acceptance

* `verify_main_offline.py`: 1 537 checksums, all 600 blocks re-derived from their
  seeds, matched quantities recomputed independently (H by an explicit timestamp
  sort), 3 360 requests with the answer constraint in every Qwen payload.
* 115 unit tests pass.
* H check (20 draws per graph, `results/h_recent5_check_cells10_20260917/`): the
  population history loss is unchanged by the new budget (it depends on the cap
  only). Two single-graph extremes in the 20-draw check (|z| = 2.25 on CollegeMsg,
  SD ratio 1.43 on AD memoryless r1) were rechecked with 400 draws: means within
  1.4 MCSE of zero, SD ratios 0.95–1.04.

| stratum | plug-in | midpoint | ET pooled | median | uniform-subset plug-in | expected ρ₂ loss recent / uniform | covers truth | window ratio |
|---|---|---|---|---|---|---|---|---|
| real | 0.1410 | 0.0369 | 0.0432 | 0.2288 | 0.0395 | -0.139 / -0.038 | 1.00 | 0.997 |
| dar_a0 | 0.0224 | 0.0233 | 0.0201 | 0.0357 | 0.0238 | -0.008 / -0.001 | 0.12 | 1.004 |
| dar_a08 | 0.0204 | 0.0184 | 0.0189 | 0.4158 | 0.0175 | -0.014 / -0.001 | 0.20 | 1.003 |
| ad_memoryless | 0.0062 | 0.0062 | 0.0052 | 0.3082 | 0.0062 | +0.000 / +0.000 | 0.00 | 1.002 |
| ad_memory | 0.0189 | 0.0178 | 0.0075 | 0.4276 | 0.0179 | -0.010 / -0.001 | 0.17 | 0.997 |

## 5. Baselines

### Development graphs (100 held-out synthetic, 2 000 observations, MAE₂)

| arm | plug-in | fixed reference | B mixture | ET pooled | ET real-only | median |
|---|---|---|---|---|---|---|
| R | 0.0182 | 0.0182 | — | 0.0178 | 0.1531 | 0.3055 |
| S | 0.1412 | 0.0309 | — | 0.0194 | 0.1564 | 0.3055 |
| H | 0.0510 | 0.0231 | — | 0.0149 | 0.1620 | 0.3055 |
| B | 0.4753 | 0.1222 | 0.0950 | 0.0378 | 0.1718 | 0.3055 |

The B mixture is weakly identified more often at the smaller retention
probabilities of this design (303 of 500 fits `weakly_identified`, 68 fall back to
the homogeneous corrector).

### Main panel, five production draws (MAE₂ ± between-source SE)

**real**

| arm | plug-in | fixed reference | B mixture | ET pooled | ET real-only | median |
|---|---|---|---|---|---|---|
| R | 0.0175 ± 0.0061 | 0.0175 ± 0.0061 | — | 0.0314 ± 0.0082 | 0.0359 ± 0.0058 | 0.2288 ± 0.0320 |
| S | 0.3127 ± 0.0526 | 0.0308 ± 0.0142 | — | 0.0444 ± 0.0094 | 0.0310 ± 0.0064 | 0.2288 ± 0.0320 |
| H | 0.1334 ± 0.0415 | 0.0345 ± 0.0085 | — | 0.0488 ± 0.0174 | 0.0438 ± 0.0114 | 0.2288 ± 0.0320 |
| B | 0.1812 ± 0.0397 | 0.0873 ± 0.0252 | 0.0765 ± 0.0222 | 0.0690 ± 0.0259 | 0.0847 ± 0.0221 | 0.2288 ± 0.0320 |

**synthetic**

| arm | plug-in | fixed reference | B mixture | ET pooled | ET real-only | median |
|---|---|---|---|---|---|---|
| R | 0.0176 ± 0.0027 | 0.0176 ± 0.0027 | — | 0.0214 ± 0.0048 | 0.1431 ± 0.0365 | 0.2968 ± 0.0597 |
| S | 0.1284 ± 0.0200 | 0.0296 ± 0.0062 | — | 0.0215 ± 0.0059 | 0.1430 ± 0.0366 | 0.2968 ± 0.0597 |
| H | 0.0186 ± 0.0042 | 0.0185 ± 0.0041 | — | 0.0133 ± 0.0034 | 0.1469 ± 0.0415 | 0.2968 ± 0.0597 |
| B | 0.4185 ± 0.0938 | 0.1109 ± 0.0281 | 0.1026 ± 0.0223 | 0.0282 ± 0.0072 | 0.1584 ± 0.0397 | 0.2968 ± 0.0597 |

"fixed reference" is the plug-in for R, the walk ratio for S, the bound midpoint
for H and the homogeneous corrector for B; the primary reference for B is the
mixture.

### Error decomposition on the main draws (ρ₂)

| stratum | arm | selection | history | net | history share | cancellation |
|---|---|---|---|---|---|---|
| real | R | -0.0024 | +0.0000 | -0.0024 | 0 % | 1.00 |
| real | S | +0.3127 | +0.0000 | +0.3127 | 0 % | 1.00 |
| real | H | +0.0070 | -0.1404 | -0.1334 | 92 % | 1.15 |
| real | B | +0.2875 | -0.4687 | -0.1812 | 62 % | 4.17 |
| synthetic | R | +0.0006 | +0.0000 | +0.0006 | 0 % | 1.00 |
| synthetic | S | +0.1284 | +0.0000 | +0.1284 | 0 % | 1.00 |
| synthetic | H | -0.0006 | -0.0077 | -0.0083 | 29 % | 1.44 |
| synthetic | B | +0.1288 | -0.5473 | -0.4185 | 81 % | 1.62 |

## 6. Qwen3.6-35B-A3B

### Execution

Spec commit `d225bff` (label fix and offline results `84ae6d1`), bundle verified on
the cluster, same model revision, weights, tokenizer, template and environment as
the earlier runs. One production chain (`cluster/submit_production.sh`): round 1
with six shards finished everything (non-thinking shards 7 min each, thinking
shards 50–55 min each, about 2.9 GPU-hours), rounds 2 and 3 found nothing to do,
the archive job built and verified the archive.

* **1 680 / 1 680 answers**, all regular ends, no output-limit hit, no unclosed
  reasoning, no empty answer; engine and runner prompt-token counts agree for all.
* **Answer constraint worked as specified.** Every final text is the rho object.
  Non-thinking: 840 answers of 44–54 tokens. Thinking: median 8 491 output tokens
  (max 80 801), the object right after `</think>`.
* **Parser.** v2 (main): thinking 839/840, non-thinking 840/840. The one invalid
  answer (`snap_mathoverflow__B-c10__s5__qwen_thinking__r1`) is non-monotone
  (rho_4 0.002 < rho_5 0.008) and keeps the fixed plug-in replacement. v1 and v3
  give the same counts, so the parser rule no longer matters.
* **Archive:** `results/main_experiment/cells10_qwen_archive.tgz` (SHA-256 file
  next to it; copy in `$HOME/cells10_archive` on the cluster): 1 990 files, all
  answers, the 560 rendered prompts, observations, requests, model identity, job
  and pinned environments, logs; zero read-back mismatches.
* A separate probe (`$WS/cells10_probe`, 8 requests on observations of the previous
  design) checked the constraint before any production data existed; it is not
  evaluated.

### Main result on the six real sources (MAE₂, parser v2)

| Arm | Qwen thinking | Qwen non-thinking | plug-in | primary reference | ET pooled | training median |
|---|---|---|---|---|---|---|
| R | 0.0176 | 0.4959 | 0.0175 | 0.0175 | 0.0314 | 0.2288 |
| S | 0.0799 | 0.5071 | 0.3127 | 0.0308 | 0.0444 | 0.2288 |
| H | 0.1373 | 0.4324 | 0.1334 | 0.0345 | 0.0488 | 0.2288 |
| B | 0.2009 | 0.5479 | 0.1812 | 0.0765 | 0.0690 | 0.2288 |

Paired differences (mean ± between-source SE) and Monte-Carlo error split into
model-repeat and sampler parts. Qwen thinking:

| Arm | vs primary | vs plug-in | vs ET pooled | vs median | MCSE (model / sampler) | signed ρ₂ | valid |
|---|---|---|---|---|---|---|---|
| R | +0.0001 ± 0.0001 | +0.0001 ± 0.0001 | -0.0138 ± 0.0090 | -0.2112 ± 0.0361 | 0.0020 (0.0001 / 0.0020) | -0.0025 | 1.000 |
| S | +0.0491 ± 0.0190 | -0.2328 ± 0.0469 | +0.0355 ± 0.0208 | -0.1489 ± 0.0483 | 0.0143 (0.0118 / 0.0091) | +0.0676 | 1.000 |
| H | +0.1028 ± 0.0324 | +0.0039 ± 0.0114 | +0.0884 ± 0.0444 | -0.0916 ± 0.0669 | 0.0055 (0.0070 / 0.0009) | -0.0711 | 1.000 |
| B | +0.1243 ± 0.0306 | +0.0197 ± 0.0245 | +0.1319 ± 0.0417 | -0.0280 ± 0.0487 | 0.0158 (0.0154 / 0.0066) | -0.1060 | 0.989 |

Qwen non-thinking (direct estimate):

| Arm | vs primary | vs plug-in | vs ET pooled | vs median | MCSE (model / sampler) | signed ρ₂ | valid |
|---|---|---|---|---|---|---|---|
| R | +0.4784 ± 0.0540 | +0.4784 ± 0.0540 | +0.4645 ± 0.0527 | +0.2671 ± 0.0388 | 0.0127 (0.0119 / 0.0046) | +0.4959 | 1.000 |
| S | +0.4764 ± 0.0608 | +0.1945 ± 0.0954 | +0.4627 ± 0.0561 | +0.2783 ± 0.0423 | 0.0087 (0.0136 / 0.0024) | +0.5071 | 1.000 |
| H | +0.3979 ± 0.0806 | +0.2990 ± 0.0984 | +0.3835 ± 0.0831 | +0.2035 ± 0.0680 | 0.0133 (0.0121 / 0.0065) | +0.4324 | 1.000 |
| B | +0.4714 ± 0.0612 | +0.3667 ± 0.0968 | +0.4789 ± 0.0366 | +0.3190 ± 0.0322 | 0.0094 (0.0123 / 0.0013) | +0.5479 | 1.000 |

Plug-in signed error on the real sources: R −0.002, S +0.313, H −0.133, B −0.181.

### Reading

* **Thinking** reproduces the plug-in on R, corrects most of the walk's selection
  bias on S (0.313 → 0.080) but stays behind the walk ratio (0.031) and ExtraTrees
  (0.044), stays at the plug-in on H (69 % of its valid real answers equal the lower
  bound, 21 % lie above the largest value the sampled dyads allow), and is worse
  than the plug-in on B (0.201 against 0.181) while the B mixture reaches 0.077.
* **Non-thinking**, now a direct estimate without a derivation, barely uses the
  observation: 73 distinct rho_2 values over 840 answers, 0.85 alone in 239 of
  them, correlation with the truth 0.50 (thinking 0.78, plug-in 0.69). Its answers
  are high regardless of the data (signed error +0.43 to +0.55 on the real
  sources), so it is worse than the constant training median on every arm.
* The earlier design could not show this: there, non-thinking derived in the
  answer field and every one of its answers was replaced by the plug-in.

### By stratum (MAE₂: thinking / non-thinking / plug-in / primary reference / ExtraTrees pooled)

| stratum | R | S | H | B |
|---|---|---|---|---|
| real | 0.018 / 0.496 / 0.018 / 0.018 / 0.031 | 0.080 / 0.507 / 0.313 / 0.031 / 0.044 | 0.137 / 0.432 / 0.133 / 0.034 / 0.049 | 0.201 / 0.548 / 0.181 / 0.077 / 0.069 |
| dar_a0 | 0.017 / 0.488 / 0.016 / 0.016 / 0.027 | 0.080 / 0.483 / 0.189 / 0.030 / 0.025 | 0.026 / 0.459 / 0.019 / 0.019 / 0.023 | 0.289 / 0.529 / 0.352 / 0.072 / 0.052 |
| dar_a08 | 0.019 / 0.126 / 0.019 / 0.019 / 0.038 | 0.063 / 0.117 / 0.139 / 0.048 / 0.045 | 0.032 / 0.133 / 0.032 / 0.033 / 0.020 | 0.444 / 0.162 / 0.634 / 0.163 / 0.023 |
| ad_memoryless | 0.009 / 0.391 / 0.009 / 0.009 / 0.007 | 0.018 / 0.423 / 0.045 / 0.004 / 0.003 | 0.004 / 0.204 / 0.004 / 0.004 / 0.003 | 0.060 / 0.709 / 0.040 / 0.025 / 0.013 |
| ad_memory | 0.026 / 0.123 / 0.026 / 0.026 / 0.014 | 0.054 / 0.136 / 0.140 / 0.036 / 0.013 | 0.021 / 0.124 / 0.019 / 0.018 / 0.007 | 0.459 / 0.155 / 0.648 / 0.150 / 0.024 |

### Arm H: where valid answers lie relative to the sample bounds (ρ₂, tolerance 5·10⁻⁴)

| stratum | config | below L | = L (plug-in) | inside | = U | above U |
|---|---|---|---|---|---|---|
| real | qwen_thinking | 7 % | 69 % | 2 % | 1 % | 21 % |
| real | qwen_nonthinking | 0 % | 0 % | 6 % | 0 % | 94 % |
| dar_a0 | qwen_thinking | 13 % | 67 % | 3 % | 3 % | 13 % |
| dar_a0 | qwen_nonthinking | 0 % | 0 % | 0 % | 0 % | 100 % |
| dar_a08 | qwen_thinking | 13 % | 80 % | 0 % | 7 % | 0 % |
| dar_a08 | qwen_nonthinking | 0 % | 0 % | 3 % | 0 % | 97 % |
| ad_memoryless | qwen_thinking | 10 % | 90 % | 0 % | 0 % | 0 % |
| ad_memoryless | qwen_nonthinking | 50 % | 0 % | 0 % | 0 % | 50 % |
| ad_memory | qwen_thinking | 7 % | 73 % | 3 % | 7 % | 10 % |
| ad_memory | qwen_nonthinking | 0 % | 0 % | 3 % | 0 % | 97 % |

### Per real source (AE2: thinking / non-thinking / plug-in / primary reference)

| source | R | S | H | B |
|---|---|---|---|---|
| sp_hospital | 0.042 / 0.434 / 0.042 / 0.042 | 0.120 / 0.456 / 0.337 / 0.095 | 0.240 / 0.438 / 0.237 / 0.021 | 0.251 / 0.468 / 0.293 / 0.128 |
| sp_highschool2013 | 0.025 / 0.460 / 0.025 / 0.025 | 0.171 / 0.467 / 0.396 / 0.045 | 0.151 / 0.368 / 0.185 / 0.064 | 0.203 / 0.451 / 0.233 / 0.113 |
| copenhagen_bluetooth | 0.006 / 0.480 / 0.006 / 0.006 | 0.016 / 0.479 / 0.422 / 0.016 | 0.218 / 0.347 / 0.229 / 0.050 | 0.199 / 0.475 / 0.200 / 0.004 |
| snap_email_eu | 0.022 / 0.399 / 0.022 / 0.022 | 0.101 / 0.411 / 0.418 / 0.020 | 0.133 / 0.396 / 0.122 / 0.039 | 0.247 / 0.438 / 0.238 / 0.068 |
| snap_collegemsg | 0.006 / 0.460 / 0.006 / 0.006 | 0.050 / 0.457 / 0.148 / 0.007 | 0.069 / 0.258 / 0.018 / 0.026 | 0.186 / 0.675 / 0.070 / 0.021 |
| snap_mathoverflow | 0.005 / 0.742 / 0.005 / 0.005 | 0.022 / 0.773 / 0.155 / 0.002 | 0.012 / 0.788 / 0.009 / 0.007 | 0.118 / 0.780 / 0.054 / 0.125 |

### Secondary, post hoc (real sources)

| config | arm | MAE₂ | median of three | 50/50 plug-in | 50/50 median | answer SD |
|---|---|---|---|---|---|---|
| qwen_thinking | R | 0.0176 | 0.0175 | 0.0175 | 0.1147 | 0.000 |
| qwen_thinking | S | 0.0799 | 0.0618 | 0.1902 | 0.1159 | 0.069 |
| qwen_thinking | H | 0.1373 | 0.1289 | 0.1176 | 0.1640 | 0.088 |
| qwen_thinking | B | 0.2009 | 0.1834 | 0.1761 | 0.1907 | 0.130 |
| qwen_nonthinking | R | 0.4959 | 0.4972 | 0.2468 | 0.2419 | 0.077 |
| qwen_nonthinking | S | 0.5071 | 0.5107 | 0.4099 | 0.2475 | 0.086 |
| qwen_nonthinking | H | 0.4324 | 0.4232 | 0.1503 | 0.2105 | 0.087 |
| qwen_nonthinking | B | 0.5479 | 0.5708 | 0.1836 | 0.2679 | 0.080 |

None of these is a main result. The median of three helps thinking where its
answers scatter (S 0.080 → 0.062, B 0.201 → 0.183). The 50/50 mixtures mostly show
how far the non-thinking answers are from the data.



## 7. Limits

* Six real sources; most real-source differences are within a few between-source
  standard errors.
* The matching constant is not told to any estimator; ExtraTrees can learn it from
  its training rows, the LLM cannot.
* The Qwen modes differ in sampling parameters as well as in reasoning, and
  non-thinking is now a direct estimate by construction.
* The H bound midpoint bounds the sampled dyads, not the archive; H loses history
  mainly on four event-dense real sources.
