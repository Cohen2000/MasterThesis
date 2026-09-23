# v10 runbook

On uc3, use the current checkout after `git pull --ff-only`. The sealed observation directory is `/pfs/work9/workspace/scratch/tu_zxokn55-llm_pilot/panel888_access_v10_main/mainexp/archive/observations/sample`. Completed offline stages are `cluster/audit_v10_walk.sbatch`, `cluster/confirm_v10_walk.sbatch`, `cluster/v10_prepare.sbatch`, `cluster/v10_pool.sbatch`, `cluster/v10_et_cache.sbatch`, `cluster/v10_et_select.sbatch`, `cluster/v10_et_train.sbatch` and `cluster/v10_finalize.sbatch`; the Qwen reproduction chain is `cluster/make_v10_production_bundle.sh` and `cluster/submit_production.sh`. Frozen results need no rerun.

Set the paths below on uc3. `check` and `cost` are offline. `smoke`, `submit`, `status` and `collect` contact a provider only with `--execute`. Keys can be read from the environment or `~/.config/masterthesis/api_keys.env` (directory mode 700, file mode 600).

```bash
WS=/pfs/work9/workspace/scratch/tu_zxokn55-llm_pilot
cd "$WS/panel888_v10_main"
source "$WS/venv_offline/bin/activate"
OBS=$WS/panel888_access_v10_main/mainexp/archive/observations/sample
python scripts/api_runner.py check --provider deepseek --observations "$OBS"
python scripts/api_runner.py check --provider openai --observations "$OBS"
python scripts/api_runner.py cost --provider deepseek --observations "$OBS"
python scripts/api_runner.py cost --provider openai --observations "$OBS"
```

Future one-request smoke commands, in fresh directories:

```bash
python scripts/api_runner.py smoke --provider deepseek --observations "$OBS" --budget-usd 10 --output "$WS/api_runs/deepseek_smoke" --execute
python scripts/api_runner.py smoke --provider openai --observations "$OBS" --budget-usd X --output "$WS/api_runs/openai_smoke" --execute
```

After inspecting the smoke outputs, future production commands:

```bash
python scripts/api_runner.py submit --provider deepseek --observations "$OBS" --budget-usd 10 --output "$WS/api_runs/deepseek_main" --execute
python scripts/api_runner.py submit --provider openai --observations "$OBS" --budget-usd X --output "$WS/api_runs/openai_main" --execute
python scripts/api_runner.py status --provider openai --output "$WS/api_runs/openai_main" --execute
python scripts/api_runner.py collect --provider openai --output "$WS/api_runs/openai_main" --execute
```

Replace `X` with an explicit USD budget at or above the printed conservative estimate. OpenAI production uploads one JSONL file and creates one 24-hour `/v1/responses` Batch. DeepSeek writes each result immediately and stops on provider error; reconcile partial output before starting a new run. No API run directory or Batch ID exists yet.
