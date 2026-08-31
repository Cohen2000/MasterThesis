# LLM noise probe

Six arms complete as of 2026-08-25: Gemini 3.1 Flash Lite, DeepSeek V4 Flash,
DeepSeek V4 Pro non-thinking, Qwen3.6-27B in both modes, and Codex
gpt-5.6-sol. 32 graphs, 12 groups, `time_agnostic_t`, budget 800,
`disclosed`/`mask`, response rate 1.00 and validity 1.00 throughout.

A seventh arm, DeepSeek V4 Pro with reasoning effort `high`, stopped at 14 of
its 60 planned prompts when the API account returned `HTTP 402 Insufficient
Balance`. It is listed below at the user's request, marked as partial. It is
not comparable to the full arms and no ranking may be read off it.

The cross-model result is in the next section and **revises the single-model
conclusion** that the rest of this document was written around. Sections below
it are the original Gemini reading, kept because the argument they make is
still the argument, and because the limitation they name is the one the other
five arms went on to confirm.

## All six arms

Reproducibility is the mean absolute gap between two generations of the
identical prompt, divided by the spread between graphs. At 1.00, asking twice
moves the answer as much as changing the network.

| model | ProfileMAE | reproducibility | pearson | share of within-graph noise from redrawing the walk |
|---|---:|---:|---:|---:|
| Codex gpt-5.6-sol | 0.053 | **0.07** | 0.993 | 68% |
| Qwen3.6-27B thinking | 0.073 | 0.31 | 0.783 | 71% |
| Qwen3.6-27B non-thinking | 0.082 | 0.53 | 0.613 | 70% |
| DeepSeek V4 Flash | 0.102 | 0.91 | 0.220 | 0% |
| Gemini 3.1 Flash Lite | 0.112 | 0.92 | 0.391 | 0% |
| DeepSeek V4 Pro non-thinking | 0.116 | 0.83 | 0.404 | 45% |
| *DeepSeek V4 Pro thinking (partial, 14/60)* | *0.026* | *--* | *--* | *--* |

**The thinking row is not a result.** It is 14 answers on 12 graphs, ended by a
billing failure rather than by the design, and it is the only row whose two
scoring rules disagree: complete-case ProfileMAE 0.026, failure-penalized 0.270.
The gap is the 32% of prompts the account could not pay for, not a property of
the model. No graph received two generations of the same prompt, so
reproducibility, pearson and the walk-noise share cannot be computed at all --
the probe's whole mechanism is missing from this arm. What the 0.026 does say,
weakly, is that the answers that did arrive were the most accurate in the
suite; whether that survives the other 46 prompts is exactly what was not
measured.

Reading the six complete arms:

**The sixth arm breaks the pattern the first five suggested.** With five arms
the split looked clean: the two models whose repeated answers barely agree
showed no input noise at all, the three whose answers do agree got 68-71% of
their within-graph noise from redrawing the walk.  DeepSeek V4 Pro
non-thinking sits in neither camp -- its repeated answers barely agree (0.83)
*and* 45% of its within-graph noise comes from the walk redraw.

So reproducibility does not predict input sensitivity, and the earlier reading
was an artifact of five points.  What survives is the weaker statement the
Limits section already made: a loud model can mask an input effect, but it does
not have to, and only the measurement decides.  The design consequence is
unchanged, and now rests on four of six arms rather than three of five.

It is also the weakest arm in the table despite being the largest DeepSeek
model, which is worth stating plainly: on this task, without thinking enabled,
model size bought nothing.

**`s2_input` = 0 is a property of Gemini and DeepSeek V4 Flash, not of the
probe.** The "one walk seed per case" rule below was derived from Gemini alone
and does not generalise. For Qwen, five walk seeds cut the standard error by
21% while five repeated generations cut it by 6%; for Codex the same comparison
is 9% against 3%; for DeepSeek V4 Pro non-thinking it is 13% against 7%. Where
the model reads the sample, seeds buy more than repeats.

Doubling the panel remains -29% for every arm, which is still the largest
single lever and is the one conclusion the extra arms left untouched.

---

## Original single-model reading (Gemini 3.1 Flash Lite, 2026-08-20)

288/288 generations, response rate 1.00, validity 1.00.

## The question

`docs/SEED_VS_GRAPH_EVIDENCE.md` found walk-seed replication nearly worthless
for classical estimators. That did not transfer to a language model by
assumption, because a model adds a noise source the estimators do not have:
its own sampling. The probe measures both, with two arms sharing one cell.

## Result 1: walk-seed noise is zero here too

| component | sd |
|---|---:|
| response (identical prompt, regenerated) | **0.1024** |
| input (redrawn walk) | **0.0000** |

The input arm was not noisier than the response arm at all. Redrawing the walk
adds nothing measurable on top of what the model already contributes by itself.
The classical conclusion carries over: **one walk seed per case.**

This also collapses a distinction the design worried about. When `s2_input` is
zero, an extra walk seed and an extra generation buy exactly the same thing --
both are simply one more draw -- and the generation is the cheaper and simpler
of the two.

## Result 2: the model barely reproduces its own answer

Two generations of the *identical* prompt, across 32 graphs:

- Pearson **0.391**, Spearman **0.346**
- mean absolute difference **0.1020**, against a between-graph sd of **0.1110**

**Ratio 0.92.** Asking the same model the same question twice moves `rho_k2`
about as much as changing to a different network. For comparison, the same
suite's spread between *models* is 0.120 and between *input formats* 0.027.

## Result 3: the instability is not why the model is wrong

For `rho_k2`, the error splits almost exactly in half:

| | value | squared |
|---|---:|---:|
| systematic bias | -0.1052 | 0.0111 |
| response noise (sd) | 0.1024 | 0.0105 |

Averaging generations can only remove the second half. Measured on
ProfileMAE, group-macro, response arm:

| R | error of each generation, averaged | error of the averaged prediction | gain |
|---:|---:|---:|---:|
| 1 | 0.1181 | 0.1181 | — |
| 2 | 0.1219 | 0.1190 | 0.0029 |
| 3 | 0.1240 | 0.1182 | 0.0058 |
| 5 | 0.1228 | 0.1167 | **0.0061** |

Five generations buy 0.006 of ProfileMAE. The bias survives untouched.

Rank correlation to truth does not improve either: Spearman 0.467 / 0.244 /
0.307 / 0.385 for R = 1/2/3/5. With 32 graphs the standard error of a Spearman
correlation is roughly 0.18, so these are indistinguishable from one another;
the honest statement is that averaging does not visibly help the ranking, not
that it hurts.

*Reading:* the wobble is real and large, but it sits on top of a bias that
dominates the verdict. A model that is unreliable *and* biased is not fixed by
asking it more often.

## Consequences for the design

| decision | evidence |
|---|---|
| ~~**1 walk seed per case**~~ — holds for Gemini and DeepSeek only, see above | `s2_input` = 0 for the classical estimators and for this model |
| **Panel size is still the lever** | doubling graphs: -29% SE; five repeats: -2% on the error metric |
| **Repeats where a cell must resolve a small effect** | see below |

The one place repetition earns its cost is the input ablation, where the
effect being tested is small. With the measured within-graph ProfileMAE sd of
0.0374, a paired comparison of two input cells over the ablation's 36 cases:

| generations per cell | SE of the paired difference | smallest detectable effect |
|---:|---:|---:|
| 1 | 0.0088 | **0.0247** |
| 2 | 0.0062 | **0.0175** |
| 3 | 0.0051 | **0.0143** |

The historical within-model input spread is 0.027, i.e. *right at* the
single-generation threshold. **The OFAT ablation should run 2-3 generations
per cell**, or its result is an equivalence statement by construction rather
than by finding. This is exactly the number the probe was run to obtain.

## Limits

- One model, and a small one: Flash Lite's calibration slope on the frozen
  suite is 0.09-0.17, so it responds weakly to the data. A model that barely
  reads the sample is the least likely to show input noise, which makes
  `s2_input` = 0 less surprising here than it would be for a model that does
  use the sample. The DeepSeek arm, and ideally a Qwen arm, are what would
  make this robust.
  **Resolved 2026-08-25:** both arms ran. DeepSeek behaved like Gemini,
  Qwen and Codex did not, and the split follows how well each model reproduces
  its own answer. See "All five arms" above.
- Variance components are balanced-design moment estimators floored at zero.
- Detection thresholds assume approximate normality over 12 groups; they are
  planning numbers, not the final inferential procedure.
- The probe reports variance, not standing. Its error levels are not an LLM
  result for the final panel, which is not frozen.
