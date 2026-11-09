# The numbers of the thesis

Written by `scripts/thesis.py` from `data/`. "Thesis" is the number as printed in the thesis, "Repo" the number recomputed here. The tables of the thesis are in `thesis/tables/`, written from the same numbers.

| Where | What | Thesis | Repo | Match |
|---|---|---|---|---|
| Abstract | GPT-6 Sol answers that match the textbook correction (random walk) | 97 von 108 | 97 von 108 | yes |
| Abstract | random walk: error of the observed persistence and of GPT-6 Sol (pp) | 30,1 und 8,4 | 30,1 und 8,4 | yes |
| 1.4 | GPT-6 Sol answers that match the textbook correction (random walk) | 97 von 108 | 97 von 108 | yes |
| 1.4 | random walk: error of the observed persistence and of GPT-6 Sol (pp) | 30,1 und 8,4 | 30,1 und 8,4 | yes |
| 1.4 | node sampling, observed persistence vs. best method under any other sampler (pp) | 2,7 vs. 3,9 | 2,7 vs. 3,9 | yes |
| 1.1, 2.1 | repeat ratio of Wikipedia and Reddit (Yi et al., 2025, Table 5: 88.41 % and 88.32 %) | etwa 88 % | etwa 88 % | yes |
| 3.1 | pairs that interact only once: online platforms | 38 bis 99 % | 38 bis 99 % | yes |
| 3.1 | pairs that interact only once: other networks | 14 bis 33 % | 14 bis 33 % | yes |
| 3.1 | persistence of the online platforms | 0,3 % bis 15 % | 0,3 % bis 15 % | yes |
| 3.1 | persistence of the other networks | 29 % und 61 % | 29 % und 61 % | yes |
| 3.2 | shortest window: hospital (hours); longest: Linux mailing list (months) | 19 Stunden, 19 Monate | 19 Stunden, 19 Monate | yes |
| 3.2, 5.6 | with 3 to 20 windows: largest mean change of persistence (pp) | höchstens 6 | 6,0 | yes |
| 5.6 | with 3 to 20 windows: smallest rank correlation with W = 5 | mindestens 0,97 | 0,972 | yes |
| 3.3 | share of the pairs' active windows each sampler sees | etwa 10 % | 10 % | yes |
| 3.3 | share of all interactions seen, median over networks: walk, event sampling | 35 %, etwa 1 % | 35 %, etwa 1 % | yes |
| 3.3 | samples of the twelve networks | 144 | 144 | yes |
| 3.4 | mean persistence: networks, time-shuffled copies | 35 %, 61 % | 35 %, 61 % | yes |
| 3.4 | samples in total | 288 | 288 | yes |
| 3.5 | hospital sample: nodes, pairs, interactions, per window | 24, 132, 3.172, 371,1183,792,571,255 | 24, 132, 3.172, 371,1183,792,571,255 | yes |
| 3.5 | rows 00001, 00010, 01110, 11100 (pairs,interactions) | 12,86, 7,75, 5,283, 4,490 | 12,86, 7,75, 5,283, 4,490 | yes |
| 3.5 | persistent pairs, observed persistence, truth | 56 von 132, 42 %, 45 % | 56 von 132, 42 %, 45 % | yes |
| 4.3 | estimates in the study, invalid answers | 4.536, 2 | 4.536, 2 | yes |
| 4.3 | samplers of the invalid answers | beide beim Zufallsweg | beide beim Zufallsweg | yes |
| 4.4 | real training networks, of them test networks | 16, neun | 16, neun | yes |
| 4.4 | training samples per sampler of the learned reference | etwa 2.000 | 2.080 | yes |
| 4.5 | largest standard error of a mean error over networks (pp) | bis zu 4 | 4,0 | yes |
| 5.1, 5.7 | mean signed error of the observed persistence: node, walk, truncation, event (pp) | −0,9, +28,4, −5,3, −16,4 | −0,9, +28,4, −5,3, −16,4 | yes |
| 5.1 | random walk: networks where the observed persistence is too high by more than 5 pp | 10 von 12 | 10 von 12 | yes |
| 5.1, 6.2 | a crawl: Email EU looks this persistent (truth) | 92 % (51 %) | 92 % (51 %) | yes |
| 5.1 | time truncation: persistent pairs that lose their return; pairs hidden entirely (mean) | 51 %, 40 % | 51 %, 40 % | yes |
| 5.1 | event sampling: kept pairs persistent vs. overall vs. seen as persistent (mean) | 64 %, 35 %, 19 % | 64 %, 35 %, 19 % | yes |
| 5.2 | best method without training: walk, truncation, event | Modell, GPT-6 Sol, Modell | Modell, GPT-6 Sol, Modell | yes |
| 5.2 | cheaper LLMs worse than the observed persistence under truncation and event sampling | ja | ja | yes |
| 5.2 | cheaper LLMs err more than GPT-6 Sol whenever a correction is needed | ja | ja | yes |
| 5.2 | largest spread of the methods without LLM under truncation and event sampling (pp) | etwa 1 | 1,1 | yes |
| 5.2 | spread of GPT-6 Sol: truncation, event (pp) | 3,8 und 10,8 | 3,8 und 10,8 | yes |
| 5.2 | spread of DeepSeek V4.1 and Qwen3.6: truncation, event (pp) | 13 bis 25 | 13 bis 25 | yes |
| 5.3 | node sampling: LLM answers equal to the observed persistence | 318 von 324 | 318 von 324 | yes |
| 5.3 | node sampling: statistical model vs. observed persistence (pp) | 3,7 statt 2,7 | 3,7 statt 2,7 | yes |
| 5.3 | random walk: textbook correction itself, statistical model (pp) | 8,8, 6,9 | 8,8, 6,9 | yes |
| 5.3 | DeepSeek V4.1: answers that match the textbook correction (an invalid answer counts as not) | 73 von 108 | 73 von 108 | yes |
| 5.3 | Qwen3.6: answers that match the textbook correction (an invalid answer counts as not) | 22 von 108 | 22 von 108 | yes |
| 5.3 | GPT-6 Sol mit Python: answers that match the textbook correction (an invalid answer counts as not) | 100 von 108 | 100 von 108 | yes |
| 5.3 | Qwen3.6 answers equal to the uncorrected observed persistence (random walk) | 37 | 37 | yes |
| 5.3 | GPT-6 Sol answers that deviate from the textbook correction, and of them closer to the truth | 7 von 11 | 7 von 11 | yes |
| 5.3 | Village: textbook correction, GPT-6 Sol answers, truth | 100 %, 74 bis 86 %, 51 % | 100 %, 74 bis 86 %, 51 % | yes |
| 5.3 | GPT-6 Sol with Python, random walk (pp) | 8,1 | 8,1 | yes |
| 5.3 | random walk without the walk log: GPT-6 Sol (pp) | 10,7 | 10,7 | yes |
| 5.3, 5.7 | share of the gain that needs only the description of the walk | 90 % | 90 % | yes |
| 5.3 | random walk without the log: statistical model unweighted, with weights from the table alone (pp) | 29,9, 10,0 | 29,9, 10,0 | yes |
| 5.3 | time-shuffled copies: GPT-6 Sol answers that match the textbook correction | 91 von 108 | 91 von 108 | yes |
| 5.3 | time-shuffled copies: node sampling vs. best method under any other sampler (pp) | 1,6 gegenüber 4,3 | 1,6 gegenüber 4,3 | yes |
| 5.4 | event sampling: statistical model (pp) | 7,6 | 7,6 | yes |
| 5.4 | time truncation: statistical model signed error (pp) | 6,8 | 6,8 | yes |
| 5.4 | time truncation: statistical model vs. observed persistence (pp) | 7,1 gegenüber 6,4 | 7,1 gegenüber 6,4 | yes |
| 5.4 | time-shuffled copies, time truncation: statistical model vs. observed persistence (pp) | 4,9 gegenüber 9,4 | 4,9 gegenüber 9,4 | yes |
| 5.4 | time truncation: least error without training data | GPT-6 Sol, 5,4 | GPT-6 Sol, 5,4 | yes |
| 5.4 | event sampling: GPT-6 Sol (pp) | 10,8 | 10,8 | yes |
| 5.4 | networks where GPT-6 Sol errs less than the model: truncation, event, walk | 6, 6, 3 | 6, 6, 3 | yes |
| 5.4 | answer spread of GPT-6 Sol: event sampling, walk (pp) | 8,2, 0,2 | 8,2, 0,2 | yes |
| 6.1 | answer spread of GPT-6 Sol: zero under node sampling, growing with what the sampler hides | ja | ja | yes |
| 5.4 | event sampling: answer spread of DeepSeek and Qwen relative to GPT-6 Sol | zwei- bis dreimal | 2.5- bis 2.6-mal | yes |
| 5.4 | GPT-6 Sol, mean of its three answers: event, truncation (pp) | 8,2, 4,6 | 8,2, 4,6 | yes |
| 5.4 | with Python: truncation, event (pp) | 6,1, 15,1 | 6,1, 15,1 | yes |
| 5.5 | statistical model overshoots on the Linux mailing list (pp, sampler) | 11 (Zeitschnitt) | 11 (Zeitschnitt) | yes |
| 5.5 | online: platform-sampler combinations where GPT-6 Sol errs less than the model | 12 von 16 | 12 von 16 | yes |
| 5.5 | online: largest gap of GPT-6 Sol to the best method under any sampler (pp) | höchstens 0,7 | 0,7 | yes |
| 5.5 | online: every method other than GPT-6 Sol misses the best by at least 4 pp under some sampler | ja | ja | yes |
| 5.5 | properties tested | 99 | 99 | yes |
| 5.5 | Zufallsweg: Spearman of the median error with events_per_pair | +0,77 | +0,77 | yes |
| 5.5 | Zufallsweg: smallest |r| when one network is left out | über 0,7 | 0,70 | yes |
| 5.5 | Zeitschnitt: Spearman of the median error with abs_early_late_mismatch_2 | +0,78 | +0,78 | yes |
| 5.5 | Zeitschnitt: smallest |r| when one network is left out | über 0,7 | 0,72 | yes |
| 5.5 | Ereignisstichprobe: Spearman of the median error with rho2 | +0,85 | +0,85 | yes |
| 5.5 | Ereignisstichprobe: smallest |r| when one network is left out | über 0,7 | 0,80 | yes |
| 5.5 | CollegeMsg, statistical model: node, walk, event below 1 pp; truncation (pp) | unter 1; 11,0 | 0,8; 11,0 | yes |
| 5.6 | time-shuffled copies: signed error of the observed persistence under event sampling (pp) | −32,5 | −32,5 | yes |
| 5.6 | stricter thresholds: GPT-6 Sol level with or ahead of the model (walk, truncation), behind under event sampling | ja | ja | yes |
| 5.6 | with 2 windows: mean change (pp), rank correlation | 10,4, 0,88 | 10,4, 0,88 | yes |
