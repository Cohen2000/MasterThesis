# Persistence under sampling bias

**How does sampling bias affect persistence estimates in temporal graphs—and how do LLMs compare with conventional methods?**

## 0 · The challenge: sampling changes apparent persistence

<details>
<summary>Definitions</summary>

| Term | Meaning |
|---|---|
| Node | Person or account |
| Event | One interaction between two nodes |
| Pair | Two nodes that interact; direction ignored |
| Window | One of 5 equal time intervals |
| **Persistence · ρ₂** | **Share of interacting pairs active in ≥ 2 windows; our main target** |
| Profile · ρ₃ … ρ₅ | Shares active in ≥ 3 … 5 windows |
| Naive share | Persistence counted directly in the sample |
| Error · pp | Percentage points from truth; 40 % vs 50 % → 10 pp |
| Spread · SD | Standard deviation: how much estimates vary |
| Typical error | Median error across six main methods; excludes Qwen no thinking |

</details>

### 0a · One graph, four samples: different shares of returning pairs

![Each returning pair counts once; the four samples produce different persistence shares](figures/fig0_toy.png)

| Sampler | What it shows | What can go wrong |
|---|---|---|
| **R · random nodes** | Full histories between sampled nodes | Equal pair inclusion, but samples vary |
| **S · random walk** | Full histories of visited pairs; visit weights | Busy pairs are overrepresented |
| **H · late time** | Sampled nodes; windows 3–5 only | Early returns are hidden |
| **B · event loss** | Each event retained with probability p | Returns—and whole pairs—disappear |

### 0b · Real samples: walks overstate persistence; missing time and events usually understate it

![Naive persistence errors across the four samplers and 12 real networks](figures/fig1_sample.png)

## 1 · Twelve networks: persistence from 0.3 % to 61 %

### 1a · Twelve real networks, ordered by true ρ₂

<!-- table:network_specs -->
| Network | Interactions | Nodes | Pairs | Events | Events/pair | True ρ₂ |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Reality Mining | Phone proximity | 96 | 2,539 | 234,757 | 92.5 | 60.7 % |
| Radoslaw | Email | 167 | 3,250 | 82,876 | 25.5 | 55.1 % |
| Malawi | Face-to-face | 86 | 347 | 102,293 | 294.8 | 50.7 % |
| Email EU | Email | 986 | 16,064 | 332,334 | 20.7 | 50.6 % |
| High school | Face-to-face | 327 | 5,818 | 188,508 | 32.4 | 48.6 % |
| Copenhagen | Phone proximity | 692 | 79,530 | 2,426,279 | 30.5 | 44.7 % |
| Hospital | Face-to-face | 75 | 1,139 | 32,424 | 28.5 | 44.5 % |
| Workplace | Face-to-face | 217 | 4,274 | 78,249 | 18.3 | 29.0 % |
| Linux mailing list | Email replies | 26,885 | 159,996 | 1,028,233 | 6.4 | 15.3 % |
| College messages | Messages | 1,899 | 13,838 | 59,835 | 4.3 | 10.3 % |
| MathOverflow | Online replies | 24,759 | 187,986 | 390,441 | 2.1 | 8.4 % |
| Digg replies | Online replies | 30,360 | 85,155 | 86,203 | 1.0 | 0.3 % |
<!-- /table:network_specs -->

### 1b · When pairs return: the same ρ₂ can hide different activity patterns

![Pair activity per window, split by the number of windows each pair is active in](figures/fig0b_active.png)

### 1c · 3–20 windows: values change modestly, ordering even less

![Persistence for other time resolutions; the networks retain nearly the same order](figures/fig14_windows.png)

- Mean ρ₂ stays within **6 pp** of the five-window value; rank correlation **≥ 0.97**.

## 2 · The comparison: the same sample for every method

| Specification | Setting |
|---|---|
| Coverage | About 10 % of active pair–window cells |
| Samples | 3 seeds per network and sampler |
| LLM answers | 3 independent repeats per sample |
| Scoring | Mean absolute error; each network counts equally |
| **LLMs** | GPT (`gpt-6-sol`), GPT + Python, DeepSeek (`deepseek-flash`), Qwen3.6 thinking / no thinking |
| **Statistical estimators** | Naive share; MLE (fits a distribution; no training) |
| **Supervised models** | ExtraTrees (16 other real + 400 synthetic training graphs); training-median baseline |

<details>
<summary>ExtraTrees · the 174 input features</summary>

| Feature group | Count | Content |
|---|---:|---|
| Time patterns · pairs | 31 | Share of seen pairs per on/off pattern over the 5 windows |
| Time patterns · events | 31 | Share of seen events per pattern |
| Events per window | 5 | Share of seen events in each window |
| Size and density | 4 | Nodes, pairs and events seen (log); events per pair |
| Starting estimate | 4 | Simple ρ₂ … ρ₅: naive share (R), reweighting (S), MLE (H), thinning model (B) |
| Sampler | 6 | Sampler flags; B's keep rate p |
| Walk weights · S only | 93 | The 31 pattern shares, weighted three ways by walk visits |

- All computed from the text the LLMs see. ExtraTrees learns a correction to the starting estimate.

</details>

### Example input · Hospital, random nodes

- Sampling rule, sizes and time patterns; S also has weights. No truth, names or node IDs.

```text
Rule: 24 random nodes; all events between them are seen.
Seen: 132 pairs, 3,172 events
pattern (windows 1–5)   pairs   events
0 0 0 0 1                  12       86
0 1 1 1 0                   5      283
1 1 1 0 0                   4      490
…                           …        …
Task: estimate the full graph's ρ₂ … ρ₅.
```

## 3 · Estimation: correction helps in S and B, not reliably in H

<a id="ranking-real"></a>

### 3a · Mean error: no LLM consistently beats MLE in S/H/B

![Mean absolute error by method and sampler](figures/fig2_ranking.png)

- GPT is the strongest LLM; B is hardest overall. [Network-specific difficulty → 8b](#network-difficulty).
- **R:** nothing to correct—the naive share is unbiased; MLE's model fit costs 1 pp.
- **H:** no method reliably beats the naive share (6.4 pp); ExtraTrees comes closest (better on 8 of 12 networks). The time cut makes 51 % of returning pairs look non-returning but also hides 40 % of all pairs—the two nearly cancel ([HISTORY.md](../results/final/HISTORY.md)).

### 3b · Persistence profile: the broad method ranking changes little

![Errors at rho2 through rho5; broad method performance remains similar](figures/fig2b_levels.png)

- H: MLE improves at higher levels; the visible 3 windows cannot directly reveal ρ₄ or ρ₅.

### 3c · Corrected estimates: H/S tend above truth; B's errors cancel

![Mean signed error and middle 80 percent of individual estimates](figures/fig2c_amount.png)

- DeepSeek in B: mean −2.6 pp, absolute error **17.2 pp**—a good mean hides large errors.

## 4 · Correction: formulas help where information survives

### 4a · R/S: LLM answers often match the direct formula

![Language-model answers matching the naive share in R or reweighting in S](figures/fig3_textbook.png)

- R: count returning pairs. S: undo the walk's preference for busy pairs with weights.
- MLE also reweights S **before fitting**; its fitted estimate need not equal the direct formula.
- S: the formula alone gives 8.8 pp (GPT: 8.4). MLE (6.9) and ExtraTrees (5.0) improve on it.

<details>
<summary>4a detail · Formula matches on individual networks</summary>

### Walk estimates · formula matches vary by network

![Language-model estimates matching walk reweighting on each network](figures/fig3b_textbook_networks.png)

</details>

### 4b · H/B estimates: GPT comes closest among LLMs; MLE remains competitive

![Share of estimates within the target band in H and B](figures/fig4_correction.png)

- Target band: 50–150 % of the needed adjustment; samples ≥ 5 pp off truth only.

<details>
<summary>4b detail · Estimation performance on individual networks</summary>

### H/B estimates · reaching the target band depends on the network

![MLE and language-model estimates making about the needed adjustment on each network](figures/fig4b_correction_networks.png)

</details>

## 5 · Reasoning: needed, but more tokens do not guarantee lower error

### Tokens and error · B uses the most reasoning

<!-- table:reasoning -->
<table>
<thead>
<tr><th rowspan="2" scope="col">LLM</th><th colspan="4" scope="colgroup">Median reasoning tokens / error (pp)</th></tr>
<tr><th scope="col">R</th><th scope="col">S</th><th scope="col">H</th><th scope="col">B</th></tr>
</thead>
<tbody>
<tr><th scope="row">GPT</th><td>489 / 2.7</td><td>3,051 / 8.4</td><td>4,764 / 5.4</td><td>8,966 / 10.8</td></tr>
<tr><th scope="row">GPT + Python</th><td>309 / 2.7</td><td>2,080 / 8.1</td><td>3,948 / 6.1</td><td>8,998 / 15.1</td></tr>
<tr><th scope="row">DeepSeek</th><td>9,069 / 2.7</td><td>35,483 / 9.0</td><td>50,226 / 13.3</td><td>57,932 / 17.2</td></tr>
<tr><th scope="row">Qwen thinking</th><td>6,068 / 2.7</td><td>9,589 / 18.1</td><td>6,402 / 16.0</td><td>13,173 / 22.0</td></tr>
</tbody>
</table>
<!-- /table:reasoning -->

- DeepSeek uses **6–19×** GPT's reasoning tokens, with no accuracy gain.
- Qwen counts total output as a reasoning proxy.
- Qwen without thinking: 25–53 pp error, worse than the constant training median (23 pp); its answers barely track the truth (Spearman 0.25).

## 6 · Noise: new samples and repeated answers

### 6a · Noise from 3 sample redraws: highest in S

![Sample-redraw noise for MLE and fixed-fit ExtraTrees across the four samplers](figures/fig5_sample_variation.png)

- LLMs excluded: their redraw variation also contains model noise; pure sample noise is not directly separable.
- H/B: MLE's redraw SD is ≈ 1 pp, its error 7–8 pp—bias, not noise. More samples would not help.

<details>
<summary>6a detail · Which networks generate redraw noise?</summary>

### Per-network redraw noise · MLE and ExtraTrees, averaged over samplers

![Mean redraw SD per network for MLE and ExtraTrees](figures/fig5_redraw_networks.png)

</details>

<a id="answer-noise"></a>

### 6b · Noise from 3 LLM answers to the same sample: H/B dominate

![LLM answer-repeat noise across the four samplers, with retrained ExtraTrees for comparison](figures/fig5_stability.png)

- Low spread can still mean consistently wrong estimates.
- ExtraTrees gives one answer per sample; retrained 11× on new training samples, its estimates move ≤ 0.6 pp.

<details>
<summary>6b detail · Which networks generate answer-repeat noise?</summary>

### Per-network answer noise · GPT, DeepSeek and Qwen thinking, averaged over samplers

![Mean answer-repeat SD per network for GPT, DeepSeek and Qwen thinking](figures/fig5_answers_networks.png)

</details>

## 7 · Python: no consistent accuracy gain

### GPT with and without Python · event loss gets harder

![Mean error of GPT and GPT with Python on the real networks](figures/fig6_python.png)

- B: worse with Python on **8 of 12** networks; R/S change little.
- **Why no gain:** R/S need only arithmetic that GPT already does unaided. In B, GPT uses Python to fit its own statistical models (**8 code runs per answer**; R: 1). Its model choice varies between answers, so answer spread doubles (5.1 → 12.2 pp, [6b](#answer-noise)).
- Twins and synthetic networks: no B penalty ([figure](figures/fig6c_python_groups.png)). Cost: **3.4×** (USD 94.50 vs 27.59).

<details>
<summary>Python · Gains and losses on individual networks</summary>

### Python versus GPT · effects vary by network

![Change in GPT error with Python on individual networks](figures/fig6b_python_networks.png)

</details>

## 8 · Individual networks: average difficulty hides exceptions

### 8a · Difficulty: network-dependent in R/S, method-dependent in H/B

![MLE and GPT errors for each real network and sampler](figures/fig7_agreement.png)

- **H:** MLE assumes steady pair activity and adds 9–19 pp on 10 of 12 networks. Right where the sample is far too low (High school, Reality Mining); too much where it is already close (Linux, College messages)—there GPT adds only 2–3 pp.
- **B:** GPT wins on sparse networks (Linux, MathOverflow), where MLE overshoots by 9–10 pp. On dense networks GPT's answers scatter: Copenhagen 31–80 %, truth 45 %.
- Does MLE's assumption fail because of timing? [Twins → 9.1](#twins).

<details>
<summary>8a detail · On which networks does MLE win, on which GPT?</summary>

### H/B · GPT wins mostly on the sparse networks (bottom rows)

![MLE and GPT errors on each real network in H and B](figures/fig7b_method_networks.png)

</details>

### Event concentration · fewer effective pairs make R/S harder

![Typical error versus the effective number of pairs carrying the events](figures/fig8b_pairs.png)

- Effective pairs = equally busy pairs with the same event concentration. Malawi: **347 actual, 55 effective**.

<a id="network-difficulty"></a>

### 8b · Per-network error: B is hardest often, not always

![Typical error of the six main methods on each real network](figures/fig11_cards.png)

- Hardest sampler: **B on 6 networks · S on 4 · H on 2**.

## 9 · Controlled tests: do methods use timing and memory?

<a id="twins"></a>

### 9.1 · Twins: timing sensitivity in LLMs—except Qwen no thinking

| Kept | Changed | Test |
|---|---|---|
| Nodes, pairs, events per pair, global event times | Times reassigned to pairs; mean ρ₂ **+27 pp** | Timing sensitivity beyond node/pair/event totals |

![Change in estimated persistence after shuffling event times](figures/fig10_twins.png)

- Twins are resampled: in R/S/H the naive share itself rises by 18–28 pp, so following it proves little. **B** is the cleaner test: naive +11, MLE +17, GPT +24 of +27 pp.
- **H:** with timing shuffled, MLE's steady-activity assumption holds: 4.9 pp vs naive 9.4 (real networks: 7.1 vs 6.4).

<details>
<summary>Twins · Which methods estimate persistence best?</summary>

### Twin estimates · mean error and SD by method and sampler

![Mean error and between-network SD of each method on the twelve time-shuffled twins](figures/fig2_twins_ranking.png)

- vs [real networks (3a)](#ranking-real): R/S unchanged. H: MLE moves ahead of the naive share. B: all errors rise; the same three lead (ExtraTrees, MLE, GPT: 12–13 pp).

</details>

### 9.2 · Synthetic networks: two memory rules with real-world motivations

| Generator | Real-world motivation | Memory rule tested here |
|---|---|---|
| **DAR · link states** | Transport/contact links and diffusion ([Williams et al.](https://arxiv.org/abs/1909.08134)) | 80 % copy previous ON **or OFF**; otherwise redraw |
| **Activity-driven · partners** | Mobile calls and information spreading ([Karsai et al.](https://www.nature.com/articles/srep04001)); original model: epidemics ([Perra et al.](https://arxiv.org/abs/1203.5351)) | Known partner with probability **n/(n+1)**; n = partners known |

- Per generator: 2 networks without and 2 with memory. Same size (**500 nodes, ≈ 10k events**) and same random numbers, so only the memory rule differs.
- DAR only switches links ON or OFF; we add how many events an ON link has.
- The animations show tiny examples of the rule, not the tested networks.

### DAR · copying the last state makes links persist

- **Copy:** ON stays ON, OFF stays OFF. **New draw:** choose ON (20 %) or OFF again.

![DAR animation showing both ON and OFF states being copied](figures/gif_dar.gif)

### Activity-driven · remembered partners make contacts return

- Active nodes make one contact, lasting one round. With 3 known partners: **75 % known, 25 % new**.

![Activity-driven animation showing random versus remembered partner choices](figures/gif_activity.gif)

### 9.3 · Memory: similar event counts, much more persistence

![Pair activity in the synthetic networks with and without memory](figures/fig12_synthetic_active.png)

### 9.4 · Memory especially hurts event-loss estimates

![Typical error with and without memory in the two generators](figures/fig13_memory.png)

- The jump is largest where persistence starts low. Activity-driven: ρ₂ 6 → 79 %, B error 2 → 16 pp. DAR: 39 → 78 %, 13 → 16 pp.
- Two paired instances per generator: a mechanism check, not a broad synthetic benchmark.

<details>
<summary>Synthetic networks · Which methods estimate persistence best?</summary>

### Synthetic estimates · mean error and SD by method and sampler

![Mean error and between-network SD of each method on the eight synthetic networks](figures/fig2_synthetic_ranking.png)

- vs [real networks (3a)](#ranking-real): same leaders (ExtraTrees, MLE, GPT). H: all three, and DeepSeek, now beat the naive share (9.5 pp). B: ExtraTrees clearly first (3.6 vs MLE 7.5 pp).

</details>

### 9.5 · All 32 networks: persistence makes event loss hard; walks remain network-dependent

![Typical error versus true persistence across real, twin and synthetic networks](figures/fig9_persistence.png)

- **B:** all tested networks above ρ₂ = 30 % have **10–24 pp** typical error; hidden returns remain hard to recover.
- **S:** the same persistence range spans **1–33 pp** error; persistence alone does not determine difficulty.
- Twins and memory tests reproduce the B pattern. They also change pair histories or concentration: **30 % is not a universal threshold**.
- In pp only: a low ρ₂ leaves little room for error. Relative to ρ₂, B error shrinks as persistence rises (Spearman −0.72).

<details>
<summary>Key findings · the opening question answered</summary>

1. **Sampling bias shifts apparent persistence.** Walks overstate it (+28 pp); missing time and events understate it (−5, −16 pp); random nodes do not.
2. **Correction helps in S and B, not reliably in H**—and never reaches R's accuracy (2.7 pp). The remaining H/B error is bias, not noise.
3. **No LLM consistently beats the conventional methods.** MLE and ExtraTrees lead; GPT is the strongest LLM and on par with MLE in H. More reasoning or Python gives no consistent gain.
4. **Sampler, network and method interact.** R/S: a network is hard for all methods alike. H/B: the method's assumptions decide.
5. **Event loss on persistent networks stays hard.** Above ρ₂ ≈ 30 %, 10–24 pp error remain—on real, twin and synthetic networks.

</details>

<details>
<summary>Sources and all numbers</summary>

- Design, method settings and literature: [DESIGN.md](../DESIGN.md).
- Scores: [MAIN_RESULTS.md](../results/final/MAIN_RESULTS.md), [all network/method errors](figures/fig_networks_detail.png), [PREDICTIONS.csv](../results/final/PREDICTIONS.csv), [between-network error SDs](data/performance_spread.csv), [method-vs-method counts and sign-flip tests (descriptive)](data/paired_comparisons.csv).
- Noise: [VARIABILITY.md](../results/final/VARIABILITY.md), [per-network SDs and persistence profile](data/noise_by_network.csv), [estimated variance components](data/noise_components.csv).
- Correction: [mean residuals and absolute errors](data/correction_reliability.csv), [time cut in H](../results/final/HISTORY.md).
- LLMs: [reasoning tokens and Python code runs](data/under_the_hood.csv), [answers matching a formula](data/answer_types.csv).
- Controls: [twin activity](figures/fig10b_twin_active.png), [twin errors](figures/fig10c_twin_error.png), [Python on controls](figures/fig6c_python_groups.png), [higher persistence levels](figures/fig9b_levels.png).
- Robustness: [window sensitivity](../results/final/W_SENSITIVITY.md), [walk settings](../results/final/WALK.md).
- Redraw: `python scripts/analysis_figures.py`; `--inputs` rebuilds input summaries from locally stored raw data. Frozen results are unchanged.

</details>
