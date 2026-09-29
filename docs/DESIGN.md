# Study design

## In one paragraph

Many networks change over time: people email, meet or reply to each other on some days and not on others. This study asks how well a method can tell, from a small sample of such a network, how *persistent* its connections are, meaning how many of the pairs that interact at all keep interacting over time. Four ways of sampling a network, four statistical or machine-learning methods and five large language models are compared on 32 networks whose true answer is known. Terms in *italics* are explained in the [glossary](#glossary).

## Target

The time span of each network is cut into five equal *windows*. For every pair of nodes that interacts at least once, K is the number of windows in which it interacts. The study estimates `rho_k`, the share of interacting pairs with K >= k, for k = 2 to 5. The main target is `rho_2`: the share of interacting pairs that are active in at least two of the five windows.

## Networks

- **12 real networks** (face-to-face contact, phone proximity, email, message and reply networks from public sources, listed in `config/datasets.yaml` and `config/pipeline.yaml`). They are the main analysis and count equally.
- **12 time-shuffled copies** (*surrogates*), one per real network: the same pairs with the same number of interactions, but the time stamps are shuffled at random. They show how much a method relies on the real timing.
- **8 synthetic networks** from two random generators with known persistence (*DAR* and *activity-driven*), each with and without memory.

## Sampling arms

Each network is observed three times in each of four *sampling arms*. Every arm is tuned so that it sees, on average, the same amount of data: 10% of the (pair, window) cells in which something happens.

| Arm | How the sample is taken | What the methods are told about the sampling |
|---|---|---|
| R | Random nodes; every interaction between two sampled nodes is seen. | The number of sampled nodes. |
| S | A random walk on the network of all interactions, choosing the next pair in proportion to its number of events; every pair it passes is seen completely. | The walk's log (how often each pair was passed). |
| H | Random nodes, but only the last 60% of the time span is visible. | The number of sampled nodes and which windows are visible. |
| B | Every single interaction is kept with probability p, independently. | The probability p. |

All methods receive exactly the same sample and the same information. The true answer, the size of the full network and anything else not listed above is withheld.

## Methods

Four methods that need no language model (*offline methods*):

- **Observed share (plug-in)**: the share of observed pairs that are active in at least k observed windows. Under arm R every pair is equally likely to be sampled, so it has no selection bias there; the other arms favour some pairs and bias it. It is the *reference* for R.
- **Training median**: always predicts the median answer of the real training networks; a naive benchmark.
- **Statistical model (MLE)**: assumes each pair has its own activity probability, fits the distribution of these probabilities to the sample by maximum likelihood, and computes `rho_k` from the fit. Each arm changes only which data enter the fit. It is the *reference* for S, H and B.
- **ExtraTrees**: a machine-learning model (a random forest of randomised decision trees) that learns to correct a starting estimate from features of the sample. It is trained per arm on samples of 16 real training networks and 400 synthetic training networks, with settings chosen by *leave-one-network-out* validation, so a network is never used to train or tune the model that predicts it. The *production fit* is reported; ten further fits on newly drawn training samples show how much the result depends on the training data. Its output is limited to a *valid profile*.

Five language-model configurations receive the identical prompt: the sample as a small table plus a description of the sampling arm. They are Qwen (with and without a thinking phase), DeepSeek Flash, GPT-6 Sol, and GPT-6 Sol with a Python tool it may use for calculations. Each configuration answers every sample three times (*repeats*).

## Scoring

- An answer is **valid** when it is exactly one JSON object with the four values `rho_2..rho_5`, each between 0 and 1 and never increasing (one surrounding code block is allowed). Invalid answers are counted and left out of the error; language-model answers are never corrected or asked again.
- ExtraTrees output is always turned into a valid profile: each value is limited to [0, 1] and each `rho_k` to at most `rho_(k-1)`. This changes 88 of 4,224 predictions slightly and none of their `rho_2` values.
- **MAE_2** is the main measure: the mean absolute error of `rho_2`, first averaged over the samples and answers of a network, then over networks, so every network counts equally. **ProfileMAE** is the same over `rho_2` to `rho_5`.
- Methods are compared network by network; *exact sign-flip tests* give p values.

## How the results were computed

1. **Preparation** (computing cluster): networks, time-shuffled copies, sampling budgets, samples and prompts; the synthetic training networks; ExtraTrees; the Qwen answers (`scripts/prepare_study.py`, `scripts/build_training_pool.py`, `scripts/extratrees.py`, `cluster/`).
2. **Pipeline** (computing cluster): the remaining real networks and their copies, ExtraTrees for all networks, the checks of the random walk and of the time cut in arm H, the frozen samples for the API models, and the result table (`scripts/run_pipeline.py`, `src/pipeline/`).
3. **API models** (laptop): DeepSeek and GPT answers for the 384 frozen samples (`scripts/api_runner.py`, `scripts/api_cycle.sh`).
4. **Evaluation** (laptop): scoring and all final tables (`scripts/evaluate_api.py`, `pipeline.report.finalize`).

Every random number comes from one master seed combined with fixed text labels, for example the arm identities. These labels are data: changing their text would change the samples, so they are kept exactly as recorded.

The [final results](results/final/REPORT.md) contain all tables, the checks and the checksums of the raw data, the samples and the answers.

## Sources

The main building blocks follow published methods; the combination and the parameter choices are this study's own.

| Part of the study | Source | What is taken from it |
|---|---|---|
| Time-shuffled copies | Gauvin et al. (2022), *Randomized reference models for temporal networks* | The timestamp shuffle P[w,t] |
| DAR networks | Williams, Mazzarisi, Lillo & Latora (2022), *Non-Markovian temporal networks with auto- and cross-correlated link dynamics* | The DAR(1) on/off rule; the event counts are added here |
| Activity-driven networks | Perra et al. (2012), *Activity driven modeling of time varying networks*; Karsai, Perra & Vespignani (2014), *Time varying networks and the weakness of strong ties* | Active nodes contact others; memory rule c/(n + c) |
| Arms R, H, B | Rocha, Masuda & Holme (2017), *Sampling of temporal networks: Methods and biases* | Node, time and event sampling (H and B are variants) |
| Arm S | Masuda, Porter & Lambiotte (2017), *Random walks and diffusion on networks* | Random walk on a weighted graph |
| Active windows K | Lahiri & Berger-Wolf (2007), *Structure prediction in temporal networks using frequent subgraphs* | Temporal support of an edge; `rho_k` summarises it |
| Statistical model (MLE) | Dorazio & Royle (2003), *Mixture models for estimating the size of a closed population when capture rates vary among individuals* | Beta mixture of individual probabilities; the target and arm layers are new |
| Arm B model | Zeileis, Kleiber & Jackman (2008), *Regression models for count data in R* | Positive-count layer; the thinning likelihood is derived here |
| Walk weights | Hansen & Hurwitz (1943); Ribeiro & Towsley (2010); Pfeffermann (1993) | Inverse-probability weights and weighted pseudo-likelihood |
| ExtraTrees | Geurts, Ernst & Wehenkel (2006), *Extremely randomized trees* | The learning algorithm |
| Model selection | Cawley & Talbot (2010); Roberts et al. (2017) | Nested selection; folds grouped by network |
| Training median | Gneiting (2011), *Making and evaluating point forecasts* | Median = best constant forecast under absolute error |
| Paired generators | Glasserman & Yao (1992), *Some guidelines and guarantees for common random numbers* | Shared random numbers between compared variants |
| Sign-flip tests | Winkler et al. (2014), *Permutation inference for the general linear model* | Sign-flipping of paired differences |
| Variability | Bouthillier et al. (2021), *Accounting for variance in machine learning benchmarks* | Separate sources of variation |
| Language models on graphs | Maurya & Liu (2026), *Evaluating LLMs on large-scale graph property estimation via random walks* | Estimating a graph property from compact sample statistics |

Own choices without a standard in the literature: five windows (how the truth changes for 2 to 20 windows: [W_SENSITIVITY.md](results/final/W_SENSITIVITY.md)), the 10% budget, the last 60% for arm H, the calibration on active (pair, window) cells, the generator parameters, the ExtraTrees starting estimate and weights, and the exact information released to the methods.

## Glossary

- **Temporal network**: a list of interactions "node A with node B at time t".
- **Pair (dyad)**: two nodes that interact at least once. **Window**: one fifth of the time span.
- **Persistence, `rho_k`**: share of interacting pairs that are active in at least k of the 5 windows.
- **Sample / observation**: what a sampling arm sees of a network. Each network has 3 per arm.
- **Sampling arm**: one of the four ways of taking a sample (R, S, H, B above).
- **Budget**: how much a sample may see; here 10% of the active (pair, window) cells.
- **Surrogate (time-shuffled copy)**: the real network with randomly shuffled time stamps. Its name ends in `__pwt` (from the P[w,t] shuffling rule).
- **DAR / activity-driven**: two standard random network generators; DAR switches pairs on and off with a chosen memory, activity-driven lets active nodes contact others.
- **Offline method**: an estimator that runs without a language model.
- **Plug-in (observed share)**: the estimate you get by treating the sample as if it were the whole network.
- **MLE (maximum likelihood estimator)**: the parameter values that make the observed sample most probable under the model.
- **Reference**: the standard method of an arm, which the others are compared with.
- **ExtraTrees**: "extremely randomised trees", a random forest variant.
- **Production fit**: the one ExtraTrees model that is reported; the other ten fits only measure variability.
- **Leave-one-network-out**: a model that predicts network X never sees X (or its time-shuffled copy) in training or tuning.
- **Valid profile**: four values between 0 and 1 that never increase from `rho_2` to `rho_5`.
- **Repeat**: one of the three answers a language model gives to the same sample.
- **MAE (mean absolute error)**: the average size of the error, ignoring its sign. **Signed error**: the average error with its sign; positive means overestimation.
- **SD (standard deviation)**: the typical size of the variation between values.
- **Sign-flip test**: an exact test of whether a mean difference over networks could be chance.
- **API**: a paid online interface to a language model (DeepSeek, OpenAI).
