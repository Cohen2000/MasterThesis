# v11 laptop runbook

These are future commands. No provider call has been made. The 288 sealed API observations and 12 training pilot draws already exist locally; `check` verifies the committed freeze hash. Qwen and offline computations are complete on uc3.

To rebuild only the compact v11 tables from completed local evidence, run:

```bash
cd /home/albert/Dokumente/MasterArbeit
.venv/bin/python scripts/build_v10_results.py --gate docs/results/panel888_v10_walk_gate_20260923/walk_gate.csv --v11-released "$HOME/.local/share/masterthesis/rh_panel_sensitivity/run" --out docs/results/panel888_v11_main_20260923
```

```bash
cd /home/albert/Dokumente/MasterArbeit
source .venv/bin/activate
OBS=$HOME/.local/share/masterthesis/v11_api_observations
PILOT=$HOME/.local/share/masterthesis/v11_token_pilot
SMOKE_OBS=$PILOT/jodie_reddit__R-p888-access-v9-20260922-panel-release__s1.json
DS_RUN=$HOME/.local/share/masterthesis/api_runs/deepseek_v11
GPT_RUN=$HOME/.local/share/masterthesis/api_runs/openai_v11
python scripts/api_runner.py check --provider deepseek --observations "$OBS" --repeats 1
python scripts/api_runner.py check --provider openai --observations "$OBS" --repeats 1
python scripts/api_runner.py cost --provider deepseek --observations "$OBS" --repeats 1 --output "$DS_RUN"
python scripts/api_runner.py cost --provider openai --observations "$OBS" --repeats 1 --output "$GPT_RUN"
python scripts/api_runner.py submit --provider deepseek --observations "$OBS" --repeats 1 --budget-usd 10 --max-concurrency 8 --output "$DS_RUN"
python scripts/api_runner.py submit --provider openai --observations "$OBS" --repeats 1 --budget-usd 200 --batch-size 96 --output "$GPT_RUN"
```

The runner reads laptop keys from `~/.config/masterthesis/api_keys.env` (directory 700, file 600). Every provider operation requires `--execute` and a positive explicit budget. The 128,000-token cap is per-request generation headroom; production cost projections require observed technical pilot usage. DeepSeek dispatch is UTC off-peak only, with a ten-minute pre-peak buffer.

## Future training smoke and token pilot

Each smoke uses one training R+N observation. Each 12-request pilot covers R/S/H/B and is technical/cost-planning only; do not tune answers, prompts or reasoning effort from accuracy.

```bash
python scripts/api_runner.py smoke --provider deepseek --observations "$OBS" --repeats 1 --smoke-observation "$SMOKE_OBS" --budget-usd 10 --output "$DS_RUN" --execute
python scripts/api_runner.py pilot --provider deepseek --observations "$OBS" --repeats 1 --pilot-observations "$PILOT" --budget-usd 10 --max-concurrency 8 --output "$DS_RUN" --execute
python scripts/api_runner.py smoke --provider openai --observations "$OBS" --repeats 1 --smoke-observation "$SMOKE_OBS" --budget-usd 200 --output "$GPT_RUN" --execute
python scripts/api_runner.py pilot --provider openai --observations "$OBS" --repeats 1 --pilot-observations "$PILOT" --budget-usd 200 --output "$GPT_RUN" --execute
python scripts/api_runner.py cost --provider deepseek --observations "$OBS" --repeats 1 --output "$DS_RUN"
python scripts/api_runner.py cost --provider openai --observations "$OBS" --repeats 1 --output "$GPT_RUN"
```

Review returned model, usage, reasoning exposure, final JSON and length flags before production. OpenAI may provide a reasoning summary but never raw chain-of-thought. Provider reasoning is retained separately; only final `rho_2..rho_5` answers enter cross-provider accuracy comparisons.

## Future repeat 1 production

Inspect the conservative cost projection before either submission. Repeat each OpenAI submit/status/collect cycle until all 288 IDs are collected; each Batch contains at most 96 requests. Both runners skip durably completed IDs and refuse uncertain launches.

```bash
python scripts/api_runner.py submit --provider deepseek --observations "$OBS" --repeats 1 --budget-usd 10 --max-concurrency 8 --output "$DS_RUN" --execute
python scripts/api_runner.py submit --provider openai --observations "$OBS" --repeats 1 --budget-usd 200 --batch-size 96 --output "$GPT_RUN" --execute
python scripts/api_runner.py status --provider openai --output "$GPT_RUN" --execute
python scripts/api_runner.py collect --provider openai --output "$GPT_RUN" --execute
```

## Optional later extension, only after repeat 1 spend review

`--repeats 3` targets 864 total IDs per provider and skips the 288 completed repeat-1 IDs. Each command rechecks actual plus conservatively projected total spend against the same approved budget. Run OpenAI submit/status/collect until all chunks are collected.

```bash
python scripts/api_runner.py cost --provider deepseek --observations "$OBS" --repeats 3 --output "$DS_RUN"
python scripts/api_runner.py cost --provider openai --observations "$OBS" --repeats 3 --output "$GPT_RUN"
python scripts/api_runner.py submit --provider deepseek --observations "$OBS" --repeats 3 --budget-usd 10 --max-concurrency 8 --output "$DS_RUN" --execute
python scripts/api_runner.py submit --provider openai --observations "$OBS" --repeats 3 --budget-usd 200 --batch-size 96 --output "$GPT_RUN" --execute
python scripts/api_runner.py status --provider openai --output "$GPT_RUN" --execute
python scripts/api_runner.py collect --provider openai --output "$GPT_RUN" --execute
```
