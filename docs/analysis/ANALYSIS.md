# Estimating persistence from a sample

**Question:** a temporal network is only seen through a sample. Can language models estimate how persistent it is?

**Short answer**
- Where the sampling has a simple textbook correction, GPT and DeepSeek give it, and do no better than the formula.
- Where there is none, they improvise. GPT does best; DeepSeek and Qwen rarely get it right and answer differently each time.
- A new sample barely moves a good estimate, so the estimates are not chance results. A Python tool does not help.
- A network is hard when few pairs share its contacts, and, when contacts are lost, when it is highly persistent. All thinking models read the timing of contacts.

## 0 · The task

![A toy network and what each arm shows of it](figures/fig0_toy.png)

- **ρ₂:** share of pairs active in ≥ 2 of 5 time windows. **Naive share:** the same share in the sample.
- **Error:** absolute error of ρ₂ in percentage points (pp), mean over 12 real networks.

| Arm | Sample | Also told |
|---|---|---|
| R · random nodes | random nodes and all contacts among them | number of nodes |
| S · random walk | a walk along contacts, which meets busy pairs more often | visits per pair* |
| H · late time only | random nodes, only the last 60 % of the time visible | number of nodes |
| B · event loss | each event kept with a small probability p | p |

\*Allows the textbook answer, the **simple reweighting**: each visit counts 1 / the pair's number of events.

**Methods:** naive share, training median (a constant guess), MLE (statistical model), ExtraTrees (trained model, a supervised reference), and the language models GPT, GPT + Python, DeepSeek, Qwen thinking and Qwen no thinking.

<details>
<summary>More: real input, model versions, the 12 networks</summary>

Each sample sees about 10 % of the activity, with 3 samples per network and arm. ρ₃ to ρ₅ count pairs active in ≥ 3 to 5 windows.

```
Sampling rule: 24 random nodes; every contact between two of them is seen.   (Hospital, arm R)
Seen: 132 pairs, 3,172 events

pattern (windows 1–5)   pairs   events      1 = active in that window
0 0 0 0 1                  12       86
0 1 1 1 0                   5      283
1 1 1 0 0                   4      490
…                           …        …      (31 patterns in total)

Task: estimate ρ₂ … ρ₅ of the full network.        Naive share 56 / 132 = 42 %; true ρ₂ 45 %.
```

| Method | Details |
|---|---|
| MLE | statistical model of the sampling, no training |
| ExtraTrees | trained on 16 other real and 400 synthetic networks with known answers; not a fair competitor |
| GPT | OpenAI `gpt-6-sol`, reasoning effort high |
| GPT + Python | the same, with OpenAI's hosted Python tool (up to 10 calls) |
| DeepSeek | `deepseek-flash`, reasoning effort high |
| Qwen thinking / no thinking | `Qwen3.6-35B-A3B`, run locally, with or without its thinking phase (temperature 1.0 / 0.7) |

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

</details>

## 1 · What the networks look like

![Share of pairs by number of active windows](figures/fig0b_portrait.png)

- Meeting often is not the same as being persistent. In every network but Digg, 21–52 % of pairs meet several times, but only within one window.
- The same ρ₂ can hide different shapes. Malawi and Email EU both have 51 %, but 23 % of Malawi's pairs are active in all 5 windows, against 4 % in Email EU.

## 2 · Samples distort persistence, except in R

![Naive share minus truth](figures/fig1_sample.png)

- The walk (S) overstates persistence; hiding early time (H) and losing events (B) understate it.

## 3 · Who corrects it

![Error per arm, sorted](figures/fig2_ranking.png)

- Correction pays off clearly in S and B, only partly in H. In R there is nothing to correct.
- GPT is the best language model but has no consistent edge over MLE. Qwen no thinking is worse than a constant guess.

<details>
<summary>Counts behind this</summary>

- **S:** MLE, ExtraTrees and all thinking models beat the naive share by more than 0.5 pp in 11 of 12 networks. **B:** MLE, ExtraTrees and GPT do so in 8 or 9. **H:** at most 8 of 12.
- **GPT against MLE:** better in 2, 6, 4 and worse in 6, 4, 6 networks (S, H, B).
- **Qwen no thinking** answers 85–95 % in R, S and B almost regardless of the sample.

</details>

## 4 · Where a textbook answer exists, the models give it

![Answers equal to the textbook answer](figures/fig3_textbook.png)

- In R the textbook answer is the naive share, in S the simple reweighting. A match means within 0.5 pp. In S, GPT's error (8.4 pp) is the formula's own (8.8 pp).

![Per network: answers equal to the simple reweighting](figures/fig3b_textbook_networks.png)

- GPT gives the formula on every network except Malawi (44 %). DeepSeek gives it on some networks, Qwen thinking only on a few (Digg, Linux, MathOverflow).

## 5 · Elsewhere they improvise, and GPT does it best

![Answers that correct by about the right amount](figures/fig4_correction.png)

- "About right" means 50–150 % of the correction the sample needs.

![Per network: answers that correct by about the right amount](figures/fig4b_correction_networks.png)

- GPT is about right on most networks, but not on Malawi, Radoslaw and Workplace. DeepSeek and Qwen rarely are (one exception: DeepSeek on College messages).

![Estimates of ρ₂ to ρ₅ against the truth, per arm](figures/fig4c_profile.png)

- GPT recovers the whole profile in every arm, even ρ₄ and ρ₅ in H, which three visible windows cannot show (the naive share is 0 there).

<details>
<summary>Reasoning length and what DeepSeek writes</summary>

| Median reasoning tokens | R | S | H | B |
|---|---:|---:|---:|---:|
| GPT | 489 | 3,051 | 4,764 | 8,966 |
| DeepSeek | 9,069 | 35,483 | 50,226 | 57,932 |

- Both write the longest reasoning in H and B. DeepSeek writes 6–19 times as much as GPT and is still less accurate.
- "Guess" appears in DeepSeek's full reasoning in 82 % of H and 58 % of B answers (R: 2 %), e.g. *"We have no data on windows 1-2, so any extrapolation is a guess."*

</details>

## 6 · How much the estimates move

![Spread of three answers to the same sample](figures/fig5_stability.png)

- Asked again with the same sample, the language models answer differently where they improvise (H, B).

![Per network: spread of three answers](figures/fig5b_noise_networks.png)

- GPT's and DeepSeek's answers vary mostly on persistent networks and hardly at all on Digg and MathOverflow. Qwen varies on almost every network.

![Per network: MLE estimates from three independent samples against the truth](figures/fig5c_sample_noise.png)

- A new sample barely moves the estimate on most networks. It moves more on the smallest networks (Malawi, Hospital) and, under the walk, on Linux, Reality Mining and High school.
- The gap to the truth is often larger than the spread between samples: the errors are systematic, not chance.

## 7 · Python does not help

![Error of GPT with and without Python](figures/fig6_python.png)

![Per network: change in GPT's error with Python](figures/fig6b_python_networks.png)

- In R and S, Python changes nothing on any network. In B it makes 8 of 12 networks worse, Malawi by 23 pp.

## 8 · Which networks are hard

### 8.1 With random nodes or a random walk, the same networks are hard for every method

![Error of MLE against GPT, one dot per network](figures/fig7_agreement.png)

- **R, S:** the dots sit on the diagonal, so difficulty is a property of the network.
- **H, B:** the dots scatter, so it depends on the method. Copenhagen in B is easy for MLE (0.5 pp) and hard for GPT (17.5 pp).

### 8.2 Not the number of contacts, but how many pairs share them

![Error against contacts and against pairs that share them, per arm](figures/fig8_structure.png)

- The number of contacts shows no clear pattern in any arm.
- Contacts spread over more pairs make R and S clearly easier, and H and B a little. Malawi's 102,293 contacts sit on effectively 55 pairs.

### 8.3 With lost events, persistent networks are hard

![Error against true persistence, per arm](figures/fig9_persistence.png)

- Only in B does the error grow with ρ₂.

### 8.4 Time-shuffled twins: do the methods read the timing?

Each real network has a **twin**: the same nodes, pairs and contact counts, but every contact at a random time. A method that used only this static information would give the same answer for both twins. Yet the true ρ₂ of the twin is 27 pp higher.

![Change in the estimate from each network to its twin](figures/fig10_twins.png)

- All methods that read the time patterns raise their estimate, and in R they follow the truth exactly.
- Only Qwen no thinking does not follow (−12 to +3 pp). It does not read the timing.
- With lost events (B), every method follows only part of the change; GPT follows best.

### 8.5 Every network on its own

![Typical error per arm, one panel per network](figures/fig11_cards.png)

- **Digg:** easy everywhere, because almost every pair meets once.
- **Malawi:** hardest in S and B. Few pairs, and a few household pairs carry most contacts.
- **Copenhagen:** easy except under event loss.
- **Linux mailing list:** low persistence, yet hard under the walk (11 pp), because its samples differ strongly (section 6).
- **College messages:** easy in R and S, harder in H, where 85 % of its pairs are active only in the hidden early time.

<details>
<summary>Definitions and all numbers</summary>

- **Pairs that share the contacts** (effective number of pairs): how many equally busy pairs would hold the contacts (inverse Simpson index of the contact shares). Malawi 55, Copenhagen 4,590, Digg 82,900.
- **Typical error:** median error of MLE, ExtraTrees, GPT, GPT + Python, DeepSeek and Qwen thinking.
- **Synthetic networks** (in 8.3; 500 nodes each): DAR (pairs switch on and off per window; with memory a pair keeps its last state 80 % of the time) and activity-driven (active nodes contact partners; with memory they prefer known ones). ρ₂ ranges from 5 % to 80 %.
- All errors per network and method: [figure](figures/fig_networks_detail.png), [MAIN_RESULTS.md](../results/final/MAIN_RESULTS.md). Robustness: [VARIABILITY.md](../results/final/VARIABILITY.md), [W_SENSITIVITY.md](../results/final/W_SENSITIVITY.md), [WALK.md](../results/final/WALK.md).
- Redraw: `python scripts/analysis_figures.py`; `--inputs` also rebuilds [`data/`](data) (needs `data/raw` and `~/.local/share/masterthesis`).

</details>
