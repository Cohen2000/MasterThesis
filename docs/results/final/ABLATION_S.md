# Arm S without walk weights

The main S prompt supplies the walk log as well as the full event histories of visited pairs.
This ablation asks how GPT performs when that log is withheld.

The same 96 S samples are used: three draws from each of 12 real networks, their 12
time-shuffled twins and 8 synthetic networks. Only `inv_events`, `traversals` and
`traversals_per_event`, and their explanation in the sampling rule, are removed. The
rule itself, the walk length `L`, all other sample fields and the three-column
`pattern,dyads,events` table remain. GPT-6 Sol uses high reasoning, no Python tool
and three independent answers per sample. All 288 answers are valid.

Mean absolute error of rho_2, in percentage points; errors are averaged within each
network first, then equally across networks:

| Method | Real networks | Time-shuffled twins | Synthetic networks |
|---|---:|---:|---:|
| Naive share (`plugin`) | 30.06 | 24.56 | 12.71 |
| Aggregate inverse-event proxy (`proxy`) | 11.32 | 13.42 | 3.39 |
| Walk reweighting, with log (`hh`) | 8.80 | 9.96 | 2.84 |
| GPT, main prompt with log (`gpt_main`) | 8.41 | 9.23 | 2.95 |
| GPT, without log (`gpt_noweights`) | 10.68 | 9.47 | 3.05 |

The proxy uses only the retained table: each pattern receives weight `dyads²/events`
(zero for an empty row). The reweighted share uses `traversals_per_event` from the
original sample and is a reference; it is unavailable in the ablated prompt.

On real networks, removing the log raises GPT's error by 2.27 pp. Its answers still
correct much of the naive share's bias. The share within 0.5 pp of the walk
reweighting falls from 89.8% to 24.1%. The change in mean error is smaller on the
twins (+0.24 pp) and synthetic networks (+0.10 pp). These are descriptive results
for GPT only; no other language model was rerun in this ablation.

All values and answer counts: [ABLATION_S.csv](ABLATION_S.csv).

Reproduce the run with [scripts/ablation_s.py](../../../scripts/ablation_s.py):

```bash
python scripts/ablation_s.py             # dry run; no requests sent
python scripts/ablation_s.py --execute   # paid API requests; needs OPENAI_API_KEY
python scripts/ablation_s.py --evaluate  # results/ablation_s/summary.csv
```

The script reads `results/samples/`, generating the study's samples from the raw
networks if the directory is absent. Evaluation additionally needs the stored
`results/ablation_s/responses.jsonl`; raw responses remain outside the repository,
as in the main study. The main GPT estimates and ground truth come from the frozen
`PREDICTIONS.csv` and `TRUTH.json` in this directory.
