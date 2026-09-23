# v10 runbook

GPT and DeepSeek run from the laptop. The sealed 360 main observations were copied from uc3 once; the technical inputs below are frozen v10 training observations. No API run has begun.

```bash
cd /home/albert/Dokumente/MasterArbeit
source .venv/bin/activate
OBS=$HOME/.local/share/masterthesis/v10_observations
SMOKE_OBS=$HOME/.local/share/masterthesis/v10_smoke_observation.json
PILOT=$HOME/.local/share/masterthesis/v10_token_pilot
DS_RUN=$HOME/.local/share/masterthesis/api_runs/deepseek_v10
GPT_RUN=$HOME/.local/share/masterthesis/api_runs/openai_v10
python scripts/api_runner.py check --provider deepseek --observations "$OBS"
python scripts/api_runner.py check --provider openai --observations "$OBS"
python scripts/api_runner.py cost --provider deepseek --observations "$OBS"
python scripts/api_runner.py cost --provider openai --observations "$OBS"
```

The local runner reads `~/.config/masterthesis/api_keys.env` (directory mode 700, file mode 600). All provider requests require `--execute` and an explicit budget. The 128,000-token generation cap is technical headroom, not expected usage. Before the pilot, `cost` reports the theoretical worst case separately and leaves the production projection unset.

## Later technical smoke and token pilot

Run each provider's one-request smoke first, then its 12-observation arm-stratified token pilot. Both use only training observations. Inspect auth, payload, returned model, parser, usage and length flags. The GPT smoke verifies `reasoning.summary=auto`; an absent summary is allowed. Do not score accuracy or alter prompts and reasoning effort from answers. DeepSeek requests run only in UTC off-peak windows, with a 10-minute pre-peak buffer.

```bash
python scripts/api_runner.py smoke --provider deepseek --observations "$OBS" --smoke-observation "$SMOKE_OBS" --budget-usd 10 --output "$DS_RUN" --execute
python scripts/api_runner.py pilot --provider deepseek --observations "$OBS" --pilot-observations "$PILOT" --budget-usd 10 --max-concurrency 8 --output "$DS_RUN" --execute
python scripts/api_runner.py smoke --provider openai --observations "$OBS" --smoke-observation "$SMOKE_OBS" --budget-usd 200 --output "$GPT_RUN" --execute
python scripts/api_runner.py pilot --provider openai --observations "$OBS" --pilot-observations "$PILOT" --budget-usd 200 --output "$GPT_RUN" --execute
python scripts/api_runner.py cost --provider deepseek --observations "$OBS" --output "$DS_RUN"
python scripts/api_runner.py cost --provider openai --observations "$OBS" --output "$GPT_RUN"
```

`cost` then reports pilot count, mean, median, p90, p95 and maximum generated tokens, reasoning tokens when exposed, and full-run projections. The conservative output projection uses `max(observed maximum, 1.25 × p95)`. Pilot spending counts within each provider's total budget.

## Later production

Inspect the pilot cost reports before execution. DeepSeek uses resumable waves of at most eight requests. Before each wave it rechecks UTC off-peak time, actual spend, remaining projected cost and the 128k worst case for requests in flight. An ambiguous launched request is never retried automatically.

```bash
python scripts/api_runner.py submit --provider deepseek --observations "$OBS" --budget-usd 10 --max-concurrency 8 --output "$DS_RUN" --execute
python scripts/api_runner.py submit --provider openai --observations "$OBS" --budget-usd 200 --batch-size 96 --output "$GPT_RUN" --execute
python scripts/api_runner.py status --provider openai --output "$GPT_RUN" --execute
python scripts/api_runner.py collect --provider openai --output "$GPT_RUN" --execute
```

Repeat the GPT `submit`, `status`, `collect` sequence for each subsequent chunk. The runner refuses a new chunk until the previous one is collected and its actual usage is included in the $200 guard. Production remains OpenAI Batch `/v1/responses`. Length-limited outputs and all provider-exposed reasoning are saved unchanged and flagged; no response is repaired.
