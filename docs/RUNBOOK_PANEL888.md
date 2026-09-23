# API runbook

The API comparison runs on the laptop from 288 frozen R/S/H/B observations (local copy, hash-checked against `docs/results/panel888_v11_main_20260923/API_FREEZE.json`). Keys live in `~/.config/masterthesis/api_keys.env` (dir 700, file 600). Every paid call needs `--execute` and a budget; `check` and `cost` are offline.

```bash
source .venv/bin/activate
M=$HOME/.local/share/masterthesis
OBS=$M/v11_api_observations; PILOT=$M/v11_token_pilot
SMOKE=$PILOT/jodie_reddit__R-p888-access-v9-20260922-panel-release__s1.json
DS=$M/api_runs/deepseek_v11; GPT=$M/api_runs/openai_v11; TOOLS=$M/api_runs/openai_tools_v11

# 1. One training smoke and a 12-request training pilot per configuration (cost planning only)
python scripts/api_runner.py smoke --provider deepseek --observations $OBS --smoke-observation $SMOKE --budget-usd 10 --output $DS --execute
python scripts/api_runner.py pilot --provider deepseek --observations $OBS --pilot-observations $PILOT --budget-usd 10 --max-concurrency 12 --output $DS --execute
python scripts/api_runner.py smoke --provider openai --observations $OBS --smoke-observation $SMOKE --budget-usd 200 --output $GPT --execute
python scripts/api_runner.py pilot --provider openai --observations $OBS --pilot-observations $PILOT --budget-usd 200 --output $GPT --execute

# 2. Production (DeepSeek off-peak; GPT as one Batch per repeat: submit, then status/collect until done)
python scripts/api_runner.py submit --provider deepseek --observations $OBS --repeats 1 --budget-usd 10 --max-concurrency 64 --output $DS --execute
python scripts/api_runner.py submit --provider openai --observations $OBS --repeats 3 --budget-usd 200 --batch-size 288 --output $GPT --execute
python scripts/api_runner.py status --provider openai --output $GPT --execute
python scripts/api_runner.py collect --provider openai --output $GPT --execute

# 3. GPT with the hosted Python tool: same commands with --tools, its own directory, shared budget
python scripts/api_runner.py submit --provider openai --tools --observations $OBS --repeats 3 --budget-usd 200 --batch-size 288 --shared-budget-dir $GPT --output $TOOLS --execute

# 4. Evaluation (per group of eight graphs and over all 24)
python scripts/evaluate_api.py --observations $OBS --truth-observations $M/v10_observations --deepseek $DS --openai $GPT --openai-tools $TOOLS --out docs/results/api_v11_20260923
```

Safety rules built into `scripts/api_runner.py`:

- **Budget:** a request or Batch starts only if the recorded spend plus the worst case of everything open stays within the budget. The worst case is the full generation cap (DeepSeek 384k, GPT 128k). For tool runs it is twice the costliest observed tool request. Spend is recorded as an upper bound: DeepSeek input at the cache-miss price, GPT input at the cache-write price, and USD 0.06 per Python container.
- **DeepSeek:** runs only off-peak, and no request starts within 60 minutes of a peak window.
- **Resuming:** every run resumes safely. Completed IDs are skipped, and launched GPT background responses are polled, not relaunched. HTTP 4xx rejections and Batch lines without usage are unbilled and are retried. Any other unclear outcome stops the runner until it is reconciled by hand.
