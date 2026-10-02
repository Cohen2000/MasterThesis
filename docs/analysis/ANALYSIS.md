# Estimating persistence from a sample

**Question:** a temporal network is only seen through a sample. Can language models estimate how persistent it is?

## 0 · The task

![A toy network and what each arm shows of it](figures/fig0_toy.png)

- **ρ₂ (persistence):** share of pairs active in ≥ 2 of 5 time windows. **Naive share:** the same share in the sample.
- The models also estimate ρ₃ to ρ₅ (≥ 3 to 5 windows). These behave like ρ₂ (section 3), so the analysis shows ρ₂.
- Errors in percentage points (pp).

<details>
<summary><h3>The four samples (arms)</h3></summary>

| Arm | What the sample shows | Also told |
|---|---|---|
| R · random nodes | random nodes and all events among them | number of nodes |
| S · random walk | a walk along events; busy pairs are met more often | visits per pair |
| H · late time only | random nodes, only the last 60 % of the time (windows 3–5) | number of nodes |
| B · event loss | each event kept with a small probability p | p |

Each sample shows about 10 % of the active pair-windows; 3 samples per network and arm.

</details>

<details>
<summary><h3>The methods</h3></summary>

| Method | What it is |
|---|---|
| Naive share | ρ₂ of the sample, uncorrected |
| Training median | a constant guess from the training networks |
| MLE | statistical model of the sampling, no training |
| ExtraTrees | trained on 16 other real and 400 synthetic networks; a supervised reference, not a fair competitor |
| GPT | OpenAI `gpt-6-sol`, reasoning effort high |
| GPT + Python | the same, with OpenAI's hosted Python tool (up to 10 calls) |
| DeepSeek | `deepseek-flash`, reasoning effort high |
| Qwen thinking / no thinking | `Qwen3.6-35B-A3B`, run locally, with or without thinking (temperature 1.0 / 0.7) |

Each language model answers each sample 3 times.

</details>

<details>
<summary><h3>One real input</h3></summary>

```
Sampling rule: 24 random nodes; every event between two of them is seen.   (Hospital, arm R)
Seen: 132 pairs, 3,172 events

pattern (windows 1–5)   pairs   events      1 = active in that window
0 0 0 0 1                  12       86
0 1 1 1 0                   5      283
1 1 1 0 0                   4      490
…                           …        …      (31 patterns in total)

Task: estimate ρ₂ … ρ₅ of the full network.        Naive share 56 / 132 = 42 %; true ρ₂ 45 %.
```

</details>

## 1 · What the networks look like

![Share of pairs active in each time window, per network](figures/fig0b_active.png)

- Radoslaw and Malawi rest on very persistent pairs (28–29 % are active in 4–5 windows). In Linux, College messages, MathOverflow and Digg, 85–100 % of the pairs are active in one window only.
- The same ρ₂ can hide different shapes: Malawi and Email EU both have 51 %, but 28 % against 16 % of their pairs are active in 4–5 windows.
- College messages, Reality Mining and Email EU fade out. Workplace is empty in window 3, so none of its pairs can be active in all 5 windows.

<details>
<summary><h3>Does the number of time windows matter?</h3></summary>

![True ρ₂ for other numbers of time windows](figures/fig14_windows.png)

- More windows raise ρ₂ a little (mean 30 % with 3, 35 % with 5, 41 % with 20 windows), but the networks keep their order (rank correlation with 5 windows ≥ 0.97 from 3 windows on).

</details>

## 2 · Samples distort persistence, except in R

![Naive share minus truth, per arm](figures/fig1_sample.png)

- The walk (S) overstates persistence; hiding the early time (H) and losing events (B) understate it.
- In H, pairs lose active windows, but whole pairs also disappear; the effects can cancel. In College messages, the time cut alone hides 85 % of pairs, yet the naive share is only 2 pp too high. A nearly correct share can hide a large loss of information.

## 3 · Correction pays where the sample is far off

![Error per arm, methods sorted](figures/fig2_ranking.png)

- Correction pays off clearly in S and B, only partly in H; in R there is nothing to correct. No method brings S, H or B down to R's 2.7 pp (best: 5.0, 3.9 and 7.6 pp).
- GPT is the best language model but has no consistent edge over MLE.
- Qwen no thinking is worse than a constant guess: in R, S and B it answers 85–95 % almost regardless of the sample.

![Error at every level of the profile, per arm](figures/fig2b_levels.png)

- ρ₃ to ρ₅ behave like ρ₂: the ranking hardly changes along the profile.
- In S and H, correction helps even more at the higher levels. In B, only MLE and ExtraTrees stay clearly ahead of the naive share up to ρ₅.

![How much each method corrects, per arm](figures/fig2c_amount.png)

- In H, every method corrects too much: the sample needs +5 pp, they add +7 to +15.
- In S and B the amount is about right on average; Qwen thinking makes only half of it in S.

![Error of each method against the error of the naive share, one dot per network and arm](figures/fig2d_breakeven.png)

- Every method leaves part of the sample's error and adds some of its own, so correction pays only where the sample is far off.
- Naive share off by 10 pp or more (22 cases): MLE, ExtraTrees and GPT are better in 21, DeepSeek in 17, Qwen thinking in 13. Off by less than 5 pp (18 cases): no method is better in more than 1.
- Of a large error, MLE and ExtraTrees leave about a fifth, GPT and DeepSeek half, Qwen thinking most (median 23, 17, 47, 51 and 85 %).

![Per network and method: better or worse than the naive share](figures/fig2e_networks.png)

- **S:** every method gains on every network but Digg, where there is nothing to correct.
- **H:** MLE, ExtraTrees and GPT gain on the four networks where the sample is off most (Reality Mining, Email EU, High school, Copenhagen). On Radoslaw, Linux and College messages every method loses; Qwen thinking gains nowhere.
- **B:** MLE, ExtraTrees and GPT gain on 8 or 9 networks, DeepSeek on 4, Qwen thinking on 2. On Malawi every method loses.

## 4 · Where a textbook answer exists, the models give it

![Answers equal to the textbook answer](figures/fig3_textbook.png)

- The textbook answer is the naive share in R and the **simple reweighting** in S: each visit counts 1 / the pair's number of events. A match means within 0.5 pp.
- In S, GPT's error (8.4 pp) is the formula's own (8.8 pp).

![Per network: answers equal to the simple reweighting](figures/fig3b_textbook_networks.png)

- GPT gives the formula on every network except Malawi (44 %); DeepSeek on some, Qwen thinking on few (Digg, Linux, MathOverflow).

## 5 · Elsewhere they improvise, and GPT does it best

![Answers that correct by about the right amount](figures/fig4_correction.png)

- "About right" means 50–150 % of the correction the sample needs.
- GPT is the best language model; MLE is about right more often in B (76 %).

![Per network: answers that correct by about the right amount](figures/fig4b_correction_networks.png)

- MLE and GPT are about right on most networks but miss different ones: MLE on Malawi, Hospital and MathOverflow, GPT on Malawi, Radoslaw and Workplace. DeepSeek and Qwen rarely are (one exception: DeepSeek on College messages).

![Estimates of ρ₂ to ρ₅ against the truth, per arm](figures/fig4c_profile.png)

- MLE, GPT and DeepSeek recover the whole profile on average, even ρ₄ and ρ₅ in H, which three visible windows cannot show (the naive share is 0 there). Qwen thinking does not.

![Reasoning length against error, one dot per model and arm](figures/fig4d_thinking.png)

- Every model thinks longest under event loss, but longer thinking does not make a model better: DeepSeek writes 6–19 times as much as GPT and is more accurate in no arm.
- Qwen thinking writes as little in H as in R: in 70 % of its H answers it returns the naive share.

<details>
<summary><h3>Reasoning tokens, and what the models write</h3></summary>

| Median reasoning tokens | R | S | H | B |
|---|---:|---:|---:|---:|
| GPT | 489 | 3,051 | 4,764 | 8,966 |
| DeepSeek | 9,069 | 35,483 | 50,226 | 57,932 |
| Qwen thinking | 6,068 | 9,589 | 6,402 | 13,173 |

- The two models whose reasoning is visible say when they improvise: "guess" appears in DeepSeek's reasoning in 82 % of H and 58 % of B answers (R: 2 %), in Qwen's in 40 % and 54 % (R: 4 %). DeepSeek, for example: *"We have no data on windows 1-2, so any extrapolation is a guess."*

</details>

## 6 · Improvised answers vary; the errors are systematic

![Spread of three answers to the same sample](figures/fig5_stability.png)

- Asked again with the same sample, a model repeats itself where it gives a textbook answer (R; in S GPT and DeepSeek) and answers differently where it improvises (H, B; in S Qwen thinking).

![Per network: spread of three answers, H and B together](figures/fig5b_noise_networks.png)

- GPT's and DeepSeek's answers vary mostly on persistent networks and hardly at all on Digg and MathOverflow. Qwen varies on almost every network.

![Per network: MLE estimates from three independent samples against the truth](figures/fig5c_sample_noise.png)

- A new sample barely moves the estimate on most networks. It moves more on the smallest networks (Malawi, Hospital) and, under the walk, on Linux, Reality Mining and High school.
- The gap to the truth is often larger than the spread between samples: the errors are systematic, not chance.

## 7 · Python does not help

![Error of GPT with and without Python](figures/fig6_python.png)

![Per network: change in GPT's error with Python](figures/fig6b_python_networks.png)

- In R and S, Python changes at most 2 pp on any network. In B it makes 8 of 12 networks worse, Malawi by 23 pp.
- In B on real networks, Python also makes repeated answers less stable: their median spread rises from 5.1 to 12.2 pp.

![Error of GPT with and without Python, twins and synthetic networks](figures/fig6c_python_groups.png)

- Python hurts only on real networks: on the twins B stays the same (13.3 → 12.8 pp), on synthetic networks it improves a little (12.9 → 10.7 pp).
- Arithmetic was not what GPT lacked: without Python it already matches the textbook answers to 0.5 pp (section 4).

## 8 · What makes a network hard

### 8.1 With random nodes or a random walk, the same networks are hard for every method

![Error of MLE against GPT, one dot per network](figures/fig7_agreement.png)

- **R, S:** the dots sit on the diagonal, so difficulty is a property of the network.
- **H, B:** the dots scatter, so it depends on the method. Copenhagen in B is easy for MLE (0.5 pp) and hard for GPT (17.5 pp).
- All six methods together rank the networks alike in R and S (mean rank correlation 0.93 and 0.89), not in H and B (0.42 and 0.38).

*Typical error* (used below): median error of MLE, ExtraTrees, GPT, GPT + Python, DeepSeek and Qwen thinking. It sums up R and S well, H and B only roughly.

### 8.2 Not the number of events, but how many pairs carry them

![Typical error against the number of events](figures/fig8_events.png)

- More events do not help in any arm.

*Pairs that carry the events:* as many equally busy pairs would hold the same events.

![Typical error against the pairs that carry the events](figures/fig8b_pairs.png)

- The more pairs carry the events, the easier R and S get; H and B only a little. Malawi's 102k events sit on effectively 55 of its 347 pairs.

### 8.3 Every network on its own

![Typical error per arm, one panel per network](figures/fig11_cards.png)

- No arm is the hardest everywhere: event loss on 5 networks, the walk on 4, late time on 2.
- **Digg:** easy everywhere; almost every pair has only one event.
- **Malawi:** hardest in S and B; effectively 55 of its 347 pairs carry the events.
- **Copenhagen:** easy except under event loss.
- **Linux mailing list:** low persistence, yet hard under the walk (11 pp): its samples differ strongly (section 6).
- **College messages:** easy in R and S, harder in H: 85 % of its pairs are active only in the hidden early time.

### 8.4 Time-shuffled twins: the thinking models read the timing; event loss gets harder

Each real network has a **twin**: the same pairs and events per pair, but the times shuffled among all events. Its true ρ₂ is 27 pp higher. A method that ignores timing would give both the same answer.

![Share of pairs active in each time window, per twin](figures/fig10b_twin_active.png)

- A twin keeps the rhythm of its network, but its pairs are active in more windows: those active in 4–5 windows rise from 10 % to 34 % of all pairs on average.

![Change in the estimate from each network to its twin](figures/fig10_twins.png)

- All methods that read timing raise their estimate; in R they follow the truth exactly.
- Qwen no thinking does not (−12 to +3 pp): it ignores the timing.
- Under event loss (B), all follow only part of the change; GPT the most.

![Error on the real networks and on their twins, per method](figures/fig10c_twin_methods.png)

- Under event loss the twins are harder for every method: shuffling makes the networks more persistent, and persistence is what event loss hides (8.6). The naive share's error doubles (16 → 33 pp); DeepSeek and Qwen thinking lose 12 pp, MLE, ExtraTrees and GPT 3–6 pp.
- In R, S and H the errors of the five correcting methods change by less than 4 pp.

![Per network and method: twin harder or easier than its network](figures/fig10d_twin_networks.png)

- **B:** the naive share loses most where shuffling raises ρ₂ most (rank correlation 0.90), and DeepSeek follows it (0.73). Malawi's twin is easier for MLE, ExtraTrees and GPT.
- **S:** the Linux twin is the only one that gets much harder for every method (17–22 pp); Malawi's gets easier for all.

### 8.5 Synthetic networks: memory makes event loss harder

The synthetic networks change one thing only, **memory**: each generator runs without and with it on the same random numbers (500 nodes, about 10k events).

- **DAR:** each pair is on or off in each window; with memory, it keeps its last state 80 % of the time.
- **Activity-driven:** active nodes create events with a partner; with memory, mostly a known one.

<details>
<summary><h3>DAR step by step (animation)</h3></summary>

![DAR on a small example, without and with memory](figures/gif_dar.gif)

- **Models** links that come back: an active pair tends to stay active (Williams et al. 2022). Without memory, only chance decides.
- **Here:** 5,000 possible pairs; on with chance 0.2; kept with chance 0.8 (memory) or never; 1 + Poisson(1) events per active window.

</details>

<details>
<summary><h3>Activity-driven step by step (animation)</h3></summary>

![Activity-driven on a small example, without and with memory](figures/gif_activity.gif)

- **Models** people who differ strongly in how active they are; the original model has no memory (Perra et al. 2012). Memory: people mostly call people they already know (Karsai et al. 2014).
- **Here:** 1,000 rounds; with memory, a new partner with chance 1 / (n + 1), n = partners known.

</details>

![Share of pairs active in each time window, synthetic networks](figures/fig12_synthetic_active.png)

- Memory puts about the same number of events on fewer pairs, which return in window after window: ρ₂ 39 → 78 % (DAR), 6 → 79 % (activity-driven).

![Error without and with memory, per method](figures/fig13_synthetic_arms.png)

- Under event loss, memory costs the naive share 30–60 pp and the language models up to 43 pp (activity-driven); MLE loses 3–6 pp. Under the walk it costs MLE, ExtraTrees, GPT and DeepSeek 1–5 pp.
- DAR is persistent even without memory (ρ₂ 39 %) and already hard under event loss (GPT 16 pp); memory adds less there.
- These are the two drivers seen in the real networks: few pairs carrying the events (S, 8.2) and high persistence (B, 8.6).

<details>
<summary><h3>Instances, and why ExtraTrees is so good here</h3></summary>

- 2 instances per generator; within an instance, both variants share all random numbers, so only memory differs.
- ExtraTrees was trained on other networks from the same two generators (B with memory: 2–3 pp, against 7–19 pp for MLE and GPT).

</details>

### 8.6 All 32 networks: under event loss, persistent networks are hard

![Typical error against true persistence, per arm, all 32 networks](figures/fig9_persistence.png)

- Under event loss (B), every network with ρ₂ above 30 % is hard (10–24 pp): real networks, twins and synthetic networks alike.
- Under the walk (S), only some of them are (1–33 pp).

![Error on networks with ρ₂ below and above 30 %, per method](figures/fig9b_methods.png)

- The same for every method: persistent networks are harder in S and B, most under event loss, where the error grows by a factor of 2 to 5 (naive share: 5 → 37 pp).
- A persistent network has more to lose: under event loss a pair must be seen in two windows to count.

<details>
<summary><h3>The same for ρ₃, ρ₄ and ρ₅</h3></summary>

![Typical error of ρ₃ to ρ₅ against the true value, per arm, all 32 networks](figures/fig9c_levels.png)

- Under event loss the error grows with the true value at every level, at ρ₅ almost on a line.
- Among the real networks, the same ones are hard at every level in R, S and B; in H they change (rank correlation between the errors at ρ₂ and ρ₅: 0.77, 0.76 and 0.93 against 0.35).

</details>

<details>
<summary><h2>Key findings</h2></summary>

**In one sentence:** how a sample is drawn decides how wrong it is; language models correct it as well as the textbook answer they recall, and not consistently better than a statistical model of the sampling.

1. **The sampling design decides how wrong the sample is.** Random nodes keep persistence (naive share off by 3 pp), the walk overstates it (30 pp), late time and event loss understate it (6 and 16 pp). No method brings S, H or B back to R's accuracy. (2, 3)
2. **Correction pays where the sample is far off.** Every method leaves part of the sample's error and adds some of its own: off by 10 pp or more, MLE, ExtraTrees and GPT beat the naive share in 21 of 22 cases; off by less than 5 pp, in at most 1 of 18. (3)
3. **Language models are as good as the textbook answer they recall.** Where one exists (R: naive share; S: simple reweighting), GPT and DeepSeek give it and inherit its error. (4)
4. **Where none exists (H, B), they improvise.** Only GPT is about right in half of its answers; the answers vary from run to run, while the errors are systematic. (5, 6)
5. **No language model beats MLE consistently.** MLE and ExtraTrees leave about a fifth of a large error, GPT and DeepSeek half, Qwen thinking most; Qwen no thinking does not read the sample. (3)
6. **More thinking and more tools do not help.** DeepSeek thinks 6–19 times as long as GPT and is more accurate in no arm; Python makes GPT worse under event loss on real networks. (5, 7)
7. **In R and S, difficulty is a property of the network** (few pairs carrying the events); in H and B, of the method. (8.1, 8.2)
8. **Persistence makes event loss hard,** however it arises: across the real networks, through time-shuffling (twins) and through memory (synthetic networks). (8.4, 8.5, 8.6)
9. **The whole profile behaves like ρ₂, and five windows are not a special choice.** (1, 3, 8.6)
10. **All thinking models read the timing;** Qwen no thinking does not. (8.4)

</details>

<details>
<summary><h2>Definitions, sources and all numbers</h2></summary>

- **ρ₂ … ρ₅:** share of pairs active in ≥ 2 … 5 of the 5 windows, among all pairs with an event. **Naive share:** the same in the sample.
- **Error:** |estimate − true ρ₂| in pp, averaged per network over samples and answers, then over networks.
- **Typical error:** median error of MLE, ExtraTrees, GPT, GPT + Python, DeepSeek and Qwen thinking.
- **Spread:** standard deviation of the three answers to the same sample.
- **Reasoning tokens:** as reported by the provider (GPT, DeepSeek); for Qwen all output tokens, that is the thinking plus an answer of about 55 tokens. The tokenizers differ, so only the order of magnitude compares across models.
- **Pairs that carry the events** (effective number of pairs): 1 / Σ (events of a pair / all events)², the inverse Simpson index. Malawi 55, Copenhagen 4,590, Digg 82,900.
- **Generators:** Williams, Mazzarisi, Lillo & Latora, *Phys. Rev. E* 105, 034301 (2022); Perra, Gonçalves, Pastor-Satorras & Vespignani, *Sci. Rep.* 2, 469 (2012); Karsai, Perra & Vespignani, *Sci. Rep.* 4, 4001 (2014).
- All errors per network and method: [figure](figures/fig_networks_detail.png), [MAIN_RESULTS.md](../results/final/MAIN_RESULTS.md). Robustness: [VARIABILITY.md](../results/final/VARIABILITY.md), [W_SENSITIVITY.md](../results/final/W_SENSITIVITY.md), [WALK.md](../results/final/WALK.md).
- Redraw: `python scripts/analysis_figures.py`; `--inputs` also rebuilds [`data/`](data) (needs `data/raw` and `~/.local/share/masterthesis`).

</details>
