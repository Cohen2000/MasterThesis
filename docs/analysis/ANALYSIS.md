# Estimating persistence from a sample

**Question:** a temporal network is only seen through a sample. Can language models estimate how persistent it is?

**Short answer:**
- Where a sampling rule has a simple textbook answer (R, S), GPT and DeepSeek give it. They are as good as the formula, and in S the formula is worse than MLE and ExtraTrees.
- Where there is none (H, B), they improvise. GPT stays close to the statistical model; DeepSeek and Qwen correct about right in at most a fifth of the cases.
- Python does not help, and in B it hurts.
- A network is hard when its sample holds few pairs (R, S) or when it is highly persistent (B).

## 0 · The task

![A toy network and what each arm shows of it](figures/fig0_toy.png)

**ρ₂** is the share of pairs active in at least 2 of 5 time windows (ρ₃ to ρ₅: at least 3 to 5). The real input looks like this, here shortened for the Hospital network in arm R:

```
Sampling rule: 24 random nodes; every contact between two of them is seen.
Seen: 132 pairs, 3,172 events

pattern (windows 1–5)   pairs   events      1 = active in that window
0 0 0 0 1                  12       86
0 0 1 0 0                  23      332
0 1 1 1 0                   5      283
1 1 1 0 0                   4      490
…                           …        …      (31 patterns in total)

Task: estimate ρ₂ … ρ₅ of the full network.
```

The naive share here is 56 of 132 pairs = 42 %, against a true ρ₂ of 45 %.

| Arm | Sample | Also told |
|---|---|---|
| R · random nodes | random nodes and all contacts among them | number of nodes |
| S · random walk | a walk along contacts, which meets busy pairs more often | visits per pair* |
| H · late time only | random nodes, only the last 60 % of the time visible | number of nodes |
| B · event loss | each event kept with a small probability p | p |

Each sample sees about 10 % of the network's activity, with 3 samples per network and arm. **Error** is the absolute error of ρ₂ in percentage points (pp), averaged over the 12 real networks.
\*The visit counts allow a textbook answer, the **simple reweighting**: each visit counts 1 / the pair's number of events.

| Method | What it is |
|---|---|
| Naive share | the share in the sample, uncorrected |
| Training median | a constant guess |
| MLE | statistical model of the sampling, no training |
| ExtraTrees | trained on 16 other real and 400 synthetic networks with known answers; a supervised reference, not a fair competitor |
| GPT | OpenAI `gpt-6-sol`, reasoning effort high |
| GPT + Python | the same, with OpenAI's hosted Python tool (up to 10 calls) |
| DeepSeek | `deepseek-flash`, reasoning effort high |
| Qwen thinking / no thinking | `Qwen3.6-35B-A3B`, run locally, with or without its thinking phase (temperature 1.0 / 0.7) |

The language models are not trained for the task; each answers every sample 3 times. The GPT and DeepSeek answers were collected on 23–29 Sep 2026.

## 1 · The 12 networks

| Network | Contacts | Nodes | Pairs | Events per pair | True ρ₂ |
|---|---|---:|---:|---:|---:|
| Digg replies | online replies | 30,360 | 85,155 | 1.0 | 0.3 % |
| MathOverflow | Q&A | 24,759 | 187,986 | 2.1 | 8 % |
| College messages | online messages | 1,899 | 13,838 | 4.3 | 10 % |
| Linux mailing list | mailing list | 26,885 | 159,996 | 6.4 | 15 % |
| Workplace | face-to-face | 217 | 4,274 | 18.3 | 29 % |
| Hospital | face-to-face | 75 | 1,139 | 28.5 | 45 % |
| Copenhagen | Bluetooth | 692 | 79,530 | 30.5 | 45 % |
| High school | face-to-face | 327 | 5,818 | 32.4 | 49 % |
| Email EU | email | 986 | 16,064 | 20.7 | 51 % |
| Malawi | face-to-face | 86 | 347 | 294.8 | 51 % |
| Radoslaw | email | 167 | 3,250 | 25.5 | 55 % |
| Reality Mining | Bluetooth | 96 | 2,539 | 92.5 | 61 % |

## 2 · Samples distort persistence, except in R

![Naive share minus truth](figures/fig1_sample.png)

S overstates persistence, while H and B understate it.

## 3 · Correction works in S and B, not in H

![Ranking per arm](figures/fig2_ranking.png)

- **S:** MLE, ExtraTrees and every thinking model beat the naive share in 11 of 12 networks.
- **B:** MLE, ExtraTrees and GPT do so in 8 or 9 networks.
- **H:** no method does so reliably on ρ₂, though over ρ₂ to ρ₅ correction helps (naive 7.2 pp, GPT 3.3 pp).
- **Qwen no thinking:** in R, S and B it answers 85–95 % almost regardless of the sample, and it is worse than the constant guess.

## 4 · The models recite the textbook, and improvise elsewhere

![Near which simple estimate the answers lie](figures/fig3_answer_types.png)

In S, 90 % of GPT's answers are the simple reweighting, and GPT's error (8.4 pp) is the formula's own (8.8 pp).

![How far the models correct](figures/fig4_correction.png)

This counts only samples that are off by at least 5 pp. GPT and DeepSeek correct about right in S. In H and B, only GPT (with or without Python) is about right in roughly half the cases. DeepSeek and Qwen are about right in at most a fifth and miss in both directions.

| Median reasoning tokens per answer | R | S | H | B |
|---|---:|---:|---:|---:|
| GPT | 489 | 3,051 | 4,764 | 8,966 |
| GPT + Python | 309 | 2,080 | 3,948 | 8,998 |
| DeepSeek | 9,069 | 35,483 | 50,226 | 57,932 |

All models think longest where they must improvise. DeepSeek thinks 6–19 times longer than GPT and is still worse. Its full reasoning says "guess" in 82 % of H and 58 % of B answers but in 2 % of R answers, for example: *"We have no data on windows 1-2, so any extrapolation is a guess."* GPT only reveals short summaries, and Qwen's reasoning is not analysed here.

## 5 · Asking twice gives different answers

![Spread for the same sample and for a new sample](figures/fig5_stability.png)

Asked again, the language models change their answer where they improvise: GPT by 5.1 pp in B, DeepSeek and Qwen thinking by 10–22 pp in H and B. A new sample moves every method by about 2 pp in R and by up to 6 pp in S. In H and B the samples of a network barely differ (MLE: ≤ 1.3 pp), so the models' own noise is far larger than any effect of the data.

## 6 · Python costs 3.4 times as much and does not help

| Arm | GPT | GPT + Python | Python better / worse in |
|---|---:|---:|---:|
| R | 2.7 | 2.7 | 0 / 1 |
| S | 8.4 | 8.1 | 2 / 0 |
| H | 5.4 | 6.1 | 3 / 7 |
| B | 10.8 | 15.1 | 2 / 8 |

The last column counts networks that differ by more than 0.5 pp. In B, Python's code fits a self-made numerical model in 92 % of answers. The answers then spread more (12.2 against 5.1 pp) and overshoot (+6.4 against +0.5 pp).

## 7 · What makes a network hard: few pairs and high persistence

Real networks differ in size, persistence and timing all at once. Two extra sets of networks tell these apart:
- **Time-shuffled copies:** each real network with random contact times. Nodes, pairs and event counts stay the same, while ρ₂ rises (mean 35 % → 62 %).
- **Synthetic networks:** 8 networks with 500 nodes each from two standard generators, each run without and with memory:

| Generator | How contacts arise | Without memory | With memory |
|---|---|---:|---:|
| DAR | each pair switches on or off in every window; with memory it keeps its last state 80 % of the time | ρ₂ 39–40 % | ρ₂ 78 % |
| Activity-driven | active nodes contact partners; with memory they prefer known ones | ρ₂ 5–6 % | ρ₂ 78–80 % |

### 7.1 Few pairs make R and S hard

![Error against the number of pairs in the sample](figures/fig6_sample_size.png)

With 10 % of the activity, a small network yields few pairs (Malawi 18–51, Digg about 8,500). In R and S, fewer pairs mean a larger error for every method, because the information is simply not in the sample.

### 7.2 High persistence makes B hard

![Error against true persistence, all 32 networks](figures/fig7_persistence.png)

Only in B does the error climb steadily with ρ₂, in real, shuffled and synthetic networks alike. Losing events hides more of a persistent network, so more has to be corrected, and DeepSeek and Qwen miss the most.

### 7.3 The real timing of contacts does not matter

| Median error of six methods (pp) | R | S | H | B |
|---|---:|---:|---:|---:|
| Real networks | 2.7 | 8.2 | 6.5 | 12.8 |
| Time-shuffled copies | 1.6 | 9.5 | 6.3 | 16.2 |

In R, S and H the shuffled copies are about as hard as the originals. In B they are harder, as their higher ρ₂ predicts (7.2). So no method depends on real timing patterns such as bursts of contacts.

### 7.4 Every network in detail

![Error per network, arm and method](figures/fig8_networks.png)

---

**More:** full results are in [MAIN_RESULTS.md](../results/final/MAIN_RESULTS.md), robustness checks in [VARIABILITY.md](../results/final/VARIABILITY.md), [W_SENSITIVITY.md](../results/final/W_SENSITIVITY.md) and [WALK.md](../results/final/WALK.md). Redraw the figures with `python scripts/analysis_figures.py`; `--inputs` also rebuilds [`data/`](data) and needs `data/raw` and `~/.local/share/masterthesis`.
