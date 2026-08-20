# LLM noise probe: first result (Gemini 3.1 Flash Lite)

Measured 2026-08-20. 288/288 generations, response rate 1.00, validity 1.00,
32 graphs, 12 groups, `time_agnostic_t`, budget 800, `disclosed`/`mask`, no
output cap. Second model (DeepSeek V4 Flash) still running; these numbers are
one model and should be read as such.

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
| **1 walk seed per case** | `s2_input` = 0 for both classical estimators and this model |
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
- Variance components are balanced-design moment estimators floored at zero.
- Detection thresholds assume approximate normality over 12 groups; they are
  planning numbers, not the final inferential procedure.
- The probe reports variance, not standing. Its error levels are not an LLM
  result for the final panel, which is not frozen.
