# LLM noise probe: response noise vs. sampling noise

Exploratory probe, not part of the final comparison. It reports **variance
components only**; its error levels are not an LLM result for the panel, which
is not frozen yet.

## Why

`docs/SEED_VS_GRAPH_EVIDENCE.md` showed that redrawing a walk barely moves a
*classical* estimator, so extra walk seeds buy almost nothing there. A language
model consumes the same sample but adds a second noise source the estimators do
not have: its own sampling at generation time. Before the final run fixes how
much to replicate, both have to be measured.

## Design

32 panel graphs, `time_agnostic_t`, budget 800, `disclosed`/`mask`, 5 repeats.
Two arms sharing one cell, so both are on the same scale:

| arm | walk | prompt text | measures |
|---|---|---|---|
| response | seed 0 fixed | identical 5x | `s2_resp` |
| input | seeds 0-4 | 5 different | `s2_resp + s2_input` |

288 generations. The difference identifies `s2_input`.

Both arms vary the per-record `gen_seed`. That is not cosmetic:
`run_llm_v2.HFModel` reseeds torch before every generation, so on the cluster
backend an identical prompt would otherwise decode byte-identically and the
response arm would report exactly zero noise as an artifact of the runner.

## 1. Generate the prompts (local, already done)

```bash
source .venv/bin/activate
PYTHONPATH=src python src/make_llm_noise_probe.py
```

Writes `results/llm_noise_probe/prompts.jsonl` (288 prompts, 32 graphs,
12 groups). It refuses to run if a `prompt_id` collides with the frozen suite.

## 2. Run it

### Option A -- DeepSeek V4 Flash over NIM (free, fast)

`deepseek-ai/deepseek-v4-pro` answers HTTP 410 -- end of life 2026-08-07, and
it is gone from the catalog. `deepseek-ai/deepseek-v4-flash-0731` is the
DeepSeek that remains, and its date suffix pins a version the `-pro` id never
did. The probe measures variance structure, not model quality, so the exact
sibling does not matter.

The catalog is public, so availability can be checked without a key:

```bash
curl -s https://integrate.api.nvidia.com/v1/models \
  | python3 -c 'import json,sys; [print(m["id"]) for m in sorted(json.load(sys.stdin)["data"], key=lambda x: x["id"])]'
```

It lists only live models -- `deepseek-v4-pro` is absent -- so it is a usable
availability signal rather than a catalogue of names. Generating still needs
`NVIDIA_API_KEY` in the environment. Generate one at build.nvidia.com and
export it **in your own terminal**; never paste it into a transcript, a file in
the repo, or a job script.

```bash
read -rsp 'NVIDIA API key: ' NVIDIA_API_KEY && export NVIDIA_API_KEY && echo
```

Smoke first (3 prompts, separate output file, does not block the full run):

```bash
NIM_MAXTOK=0 bash scripts/run_llm_v21_nim.sh dsv4-flash \
    --prompts results/llm_noise_probe/prompts.jsonl \
    --out results/llm_noise_probe/answers_dsv4flash_smoke.jsonl \
    --ids 0a8e5989118b,486603a64ee3,b4e772220a2a
```

`--smoke` is not usable here: it selects three fixed prompt ids from the frozen
suite, which do not exist in this file. The three ids above are three
generations of one *identical* prompt, so the smoke's first job is to show that
the answers differ at all -- three identical replies would mean the sampling is
effectively greedy and the response arm is dead.

Full run (about 1-1.5 h; rerun the same command to resume):

```bash
NIM_MAXTOK=0 bash scripts/run_llm_v21_nim.sh dsv4-flash \
    --prompts results/llm_noise_probe/prompts.jsonl \
    --out results/llm_noise_probe/answers_dsv4flash.jsonl
```

Sampling comes from the `dsv4-flash` mode block: temperature 1.0, top_p 1.0.
Temperature > 0 is required -- a greedy run would measure zero response noise
for the same reason the `gen_seed` fix exists.

`NIM_MAXTOK=0` drops the output cap (see the note below); `NIM_MAX_WAIT`
controls how long an HTTP 529 is sat out.

### Option B -- Gemini 3.1 Flash Lite over AI Studio (free, second character)

Worth running as well as Option A rather than instead of it: the probe's
question is how much *different* models are moved by a redrawn walk, and a
single model cannot answer that. The free tier gives `gemini-3.1-flash-lite`
500 requests/day at 15 RPM, and the probe needs 288 -- it fits in a single
day. `--temperature 1.0` is already set
in the runner, which the probe requires: a greedy run would measure zero
response noise by construction.

Pass the files through the environment, never as extra flags. The runner's
outer completion loop reads `$PROMPTS`/`$OUT` directly, so a `--prompts` passed
as an extra arg would leave it counting against the frozen suite and looping
until `GEMINI_MAX_PASSES`.

```bash
read -rsp 'Gemini API key: ' GEMINI_API_KEY && export GEMINI_API_KEY && echo
```

Smoke on three generations of **one identical prompt**, so the first thing it
proves is that the probe can measure anything at all -- three identical answers
would mean the sampling is effectively greedy and the response arm is dead:

```bash
PROMPTS_FILE=results/llm_noise_probe/prompts.jsonl \
OUT_FILE=results/llm_noise_probe/answers_gemini31fl_minimal_smoke.jsonl \
SMOKE_IDS=0a8e5989118b,486603a64ee3,b4e772220a2a \
GEMINI_MAXTOK=0 \
bash scripts/run_llm_v21_gemini.sh minimal --smoke
```

Then the full pass:

```bash
PROMPTS_FILE=results/llm_noise_probe/prompts.jsonl \
OUT_FILE=results/llm_noise_probe/answers_gemini31fl_minimal.jsonl \
GEMINI_MAXTOK=0 \
bash scripts/run_llm_v21_gemini.sh minimal
```

The outer loop reruns until every prompt has a complete record and is safe to
interrupt at any point. `--limit` now forces a single pass instead of entering
that loop, which would otherwise have produced `GEMINI_MAX_PASSES * limit`
records.

*Interpretation limit worth carrying:* Flash Lite's calibration slope on the
frozen suite is 0.09-0.17, i.e. it barely moves with the data. A model that
hardly reads the sample will show little input noise for a reason that is not
about walk stability, so this arm alone can tilt the answer toward "response
noise dominates". Confirm on a second model of different character before the
design decision is taken.

### Option C -- Qwen3.6-27B on the cluster (free, slow, different character)

Documented throughput is ~16.5 tok/s, so 288 generations are roughly 40 GPU
hours in one task. Shard it (8 shards -> ~5 h each) or cut the probe down
first with `--repeats 3` and a graph subset. Upload the probe prompts under
their own name; do **not** overwrite `$WS/llm_v21/prompts.jsonl`.

### Option D -- DeepSeek V4 Pro over the official API

The Pro arms use the same frozen 288 prompts as every other full arm. Smoke and
full run append to a mode-specific, resume-safe answer file. Thinking and
non-thinking costs share one global hard ceiling; the wrapper refuses a ceiling
above USD 0.84.

```bash
bash scripts/run_noise_deepseek_pro.sh smoke
bash scripts/run_noise_deepseek_pro.sh run
bash scripts/run_noise_deepseek_pro.sh status

DEEPSEEK_PRO_NOISE_MODE=think bash scripts/run_noise_deepseek_pro.sh smoke
DEEPSEEK_PRO_NOISE_MODE=think bash scripts/run_noise_deepseek_pro.sh run
DEEPSEEK_PRO_NOISE_MODE=think bash scripts/run_noise_deepseek_pro.sh status
```

For a fast mixed run, `run_noise_deepseek_pro_parallel.sh` uses one locked
budget across all processes. It runs the full 288-prompt non-thinking arm in
four shards and a 60-prompt thinking subset (12 graphs, three draws per noise
arm) in twelve shards. The default cap is $2.69 of new spend; an existing
ledger is resumed rather than reset:

```bash
bash scripts/run_noise_deepseek_pro_parallel.sh start
bash scripts/run_noise_deepseek_pro_parallel.sh status
```

Defaults: `deepseek-v4-pro`, non-thinking, official Chat Completions endpoint,
8192 output tokens, and the documented USD-per-million prices 0.003625 cache
hit / 0.435 cache miss / 0.87 output. Thinking uses effort `high`. Override the ceiling downward with
`DEEPSEEK_PRO_NOISE_MAX_USD`; the cumulative answer-file cost remains in force
after every resume.

## Output budget

`--max-tokens 0` (via `NIM_MAXTOK=0` / `GEMINI_MAXTOK=0`) omits the field from
the request so the model may generate as long as it needs. The motivation is
not generosity: on the frozen suite most of the spread in the
failure-penalized numbers came from non-responses rather than worse answers,
and a truncation rate that differs between arms would confound this probe's
variance comparison directly.

Two things to check rather than assume:

- **Omitting is not the same as unlimited.** The server's own default applies,
  and on some OpenAI-compatible backends that default is *smaller* than an
  explicit large value. Verify on the smoke that `finish_reason` is `stop`:

  ```bash
  python3 -c '
  import json,collections,glob
  c=collections.Counter()
  for f in glob.glob("results/llm_noise_probe/answers_*smoke*.jsonl"):
      for l in open(f): c[json.loads(l).get("finish_reason")]+=1
  print(c)'
  ```

  Any `length` means the cap is still binding and an explicit large value is
  the better instrument.

- **The HF/cluster backend is unaffected.** `--max-new-tokens` must be a
  number there, and an uncapped generation would collide with the SLURM wall
  clock. Cluster runs keep their explicit budgets.

The answer record stores `max_tokens: null` when no cap was requested, which is
a different run condition from any particular number and stays visible in the
evidence.

## 3. Evaluate

```bash
PYTHONPATH=src python src/report_llm_noise.py \
    --answers 'results/llm_noise_probe/answers_*.jsonl' \
    --metric penalized
```

`--metric rho_k2_pred` reads the raw prediction instead of the error, which
separates "the answer moves" from "the error moves". The report ends with a
design table pricing walk seeds (`S`) against repeated generations (`R`)
against panel size, plus the response/validity rate per arm -- if those differ
by more than 10 points between arms, the variance comparison is confounded with
non-response and the report says so.
