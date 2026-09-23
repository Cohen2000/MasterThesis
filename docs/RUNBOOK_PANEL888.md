# v10 runbook

The sealed Qwen and offline results need no rerun. GPT and DeepSeek run from the **laptop**. The 360 sealed main observations were copied once from the uc3 Qwen archive into the untracked local data directory below; the training smoke observation is from the v10 offline preparation. uc3 is only their source, not an API execution host.

## Local setup and offline checks

```bash
cd /home/albert/Dokumente/MasterArbeit
source .venv/bin/activate
OBS=$HOME/.local/share/masterthesis/v10_observations
SMOKE_OBS=$HOME/.local/share/masterthesis/v10_smoke_observation.json
DS_RUN=$HOME/.local/share/masterthesis/api_runs/deepseek_v10
GPT_RUN=$HOME/.local/share/masterthesis/api_runs/openai_v10
python scripts/api_runner.py check --provider deepseek --observations "$OBS"
python scripts/api_runner.py check --provider openai --observations "$OBS"
python scripts/api_runner.py cost --provider deepseek --observations "$OBS"
python scripts/api_runner.py cost --provider openai --observations "$OBS"
```

The local runner reads `~/.config/masterthesis/api_keys.env` (directory mode 700, file mode 600). `check`, `cost`, `prepare` and commands without `--execute` make no provider request. DeepSeek's report shows the official off-peak estimate ($0.15/1M input, $0.60/1M output at the token allowances) and the stricter $1/$3 planning ceiling with a 25% margin. Its UTC guard blocks weekday 01:00–04:00 and 06:00–10:00 UTC, with a 10-minute pre-peak buffer.

## Future technical smokes

Use only the training observation. Smoke checks auth, payload, format, parser, usage and returned model; do not select models from answer quality. The DeepSeek smoke and production commands share `DS_RUN` so its spend counts toward the same $10 budget.

```bash
python scripts/api_runner.py smoke --provider deepseek --observations "$OBS" --smoke-observation "$SMOKE_OBS" --budget-usd 10 --output "$DS_RUN" --execute
python scripts/api_runner.py smoke --provider openai --observations "$OBS" --smoke-observation "$SMOKE_OBS" --budget-usd 200 --output "$HOME/.local/share/masterthesis/api_runs/openai_smoke" --execute
```

GPT smoke is one synchronous `/v1/responses` request. Verify that `reasoning.summary=auto` is accepted and retain any provider summary, final output and reasoning-token usage separately. OpenAI does not expose raw chain-of-thought; an absent summary is not treated as hidden reasoning. DeepSeek retains raw `reasoning_content`. GPT production remains Batch. No smoke has been run.

## Future production and Batch collection

Inspect smoke outputs first. DeepSeek can resume with the same command and output directory; completed IDs are skipped. An attempted request without a durable response is flagged for reconciliation before retry. No new DeepSeek request starts during peak or the pre-peak buffer.

```bash
python scripts/api_runner.py submit --provider deepseek --observations "$OBS" --budget-usd 10 --max-concurrency 8 --output "$DS_RUN" --execute
python scripts/api_runner.py submit --provider openai --observations "$OBS" --budget-usd 200 --output "$GPT_RUN" --execute
python scripts/api_runner.py status --provider openai --output "$GPT_RUN" --execute
python scripts/api_runner.py collect --provider openai --output "$GPT_RUN" --execute
```

OpenAI submission uploads one JSONL file and creates one 24-hour `/v1/responses` Batch. No API run directory or Batch ID exists yet.
