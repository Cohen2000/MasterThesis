# Production record

These programs produced the frozen table of all predictions, `docs/results/final/PREDICTIONS.csv`. They ran once, on the computing cluster (bwUniCluster, SLURM) and against the paid APIs. They are kept as a record of what was done and are not meant to be run again:

- The answers of the language models cannot be reproduced: the models sample at random, and the API runs were paid.
- The statistical model (MLE) is fitted by a numerical optimiser. On another machine its values agree with the frozen ones only up to about 0.0002, and ExtraTrees starts from these values, so a new run would change the numbers slightly. The observed share is reproduced exactly.

Everything that can be repeated exactly is in [`scripts/`](../scripts): the result tables and the analysis are rebuilt from the frozen predictions.

The code exactly as it ran is the Git tag `final-freeze`. Since then files were moved, import paths adjusted and a few unused helpers removed; no computation was changed. Job files that only submitted or monitored jobs were removed; they are in the tag.

## Steps in the order they ran

Stage 1 covers 24 networks (8 real, their 8 time-shuffled copies, 8 synthetic) and the 16 real training networks. Stage 2 adds 4 real networks and their copies, all further ExtraTrees fits and the checks.

| Step | Program | What it did | Ran on |
|---|---|---|---|
| 1 | `fetch_tokenizers.py` | Downloaded the tokenizers used to check the length of every prompt (`prompt_length.py`) | cluster |
| 2 | `prepare_study.py` | Built the networks and copies, tuned the sampling budgets, drew the samples, wrote the prompts and the list of Qwen requests (`model_requests.py`) | cluster, 4 CPUs, 96 GB |
| 3 | `build_training_pool.py` | Generated the 500 synthetic training networks of ExtraTrees and their samples | cluster, 4 CPUs, 96 GB |
| 4 | `extratrees.py cache`, `select`, `train` | Computed the features, chose the settings by leave-one-network-out validation, fitted and predicted | cluster, up to 8 CPUs, 45 jobs each |
| 5 | `run_qwen_engine.py` (job file `qwen_engine.sbatch`) | Qwen answers, with and without thinking, three per sample | cluster, 1 H100 GPU, 6 parallel jobs |
| 6 | `verify_qwen_answers.py`, `score_stage1.py` | Checked every Qwen answer against the request list; scored stage 1 | cluster |
| 7 | `check_random_walk.py` | Random-walk check with 1000 walks per network | cluster, 4 CPUs, 64 GB |
| 8 | `add_panel_size.py` | Added the number of sampled nodes to the R and H samples and rebuilt what depends on it (Qwen requests, ExtraTrees with one more feature); wrote the frozen samples for the API models | laptop and cluster |
| 9 | `run_pipeline.py --submit` (tasks in `pipeline/`, job file `pipeline_task.sbatch`) | Stage 2: the remaining networks, ExtraTrees for all networks with ten further fits, Qwen for the new samples, the checks of the walk and of the time cut, and the table of all predictions | cluster |
| 10 | `api_runner.py`, `api_cycle.sh` | DeepSeek and GPT answers for the 384 frozen samples, three per sample; safe to stop and resume (`run_guards.py`) | laptop |

## Stage 2 in `pipeline/`

| File | Contents |
|---|---|
| `task_graph.py`, `core.py` | The list of tasks, what each one needs, and the submission to SLURM; a finished task is never computed twice |
| `real_networks.py` | The four additional real networks: graph, budget, samples |
| `observe.py` | One sample: draw, text, features, MLE |
| `training_draws.py`, `extratrees_fits.py` | Training samples and the ExtraTrees fits (the reported fit and ten more) |
| `surrogates_and_checks.py` | Time-shuffled copies of the additional networks, the walk check, the check of the time cut in arm H, the frozen samples for the API models |
| `qwen.py` | Qwen requests of the new samples and the check of their answers |
| `report.py` | Collected every prediction into `PREDICTIONS.csv` and checked that no ExtraTrees fit saw the network it predicts |

Settings: [`config/pipeline.yaml`](../config/pipeline.yaml).

## Names in the code

- **Stage 1, stage 2**: the two rounds described above. Labels such as `v9`, `v10`, `v11` or `panel888` inside IDs are fixed text that enters the random seeds; changing them would change the samples.
- **Panel release**: the R and H samples that state the number of sampled nodes. Only these are reported; their IDs contain `-panel-release`.
- **`S_obs`**: a random-walk arm without the walk's log. It was drawn and fitted but is not reported. Its (always empty) column stays in the ExtraTrees features, because removing it would change the forests.
- **`__pwt`**: a time-shuffled copy of a network.
