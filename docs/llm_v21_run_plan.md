# V2.1 LLM run plan (model round 2)

Status 2026-07-14: runner and scripts prepared; no jobs submitted, no API
calls made. Prompts stay the frozen `results/llm_v2/prompts.jsonl`
(84 cases x 5 conditions = 420 prompts, 3 access strategies). All new
answers and logs go to `results/llm_v21/` (local/NIM) or `$WS/llm_v21/`
(cluster), never into the frozen `results/llm_v2/` files.

## Model x mode matrix

| model | mode | backend | script | out file (tag) |
|---|---|---|---|---|
| Qwen/Qwen3.6-27B | thinking | hf (cluster) | `slurm/llm_v21_qwen36_27b_think.sbatch` | `answers_qwen36-27b_think.shard{0..7}.jsonl` |
| Qwen/Qwen3.6-27B | non-thinking (`enable_thinking=False`) | hf (cluster) | `slurm/llm_v21_qwen36_27b_nothink.sbatch` | `answers_qwen36-27b_nothink.jsonl` |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-32B | always-thinking (continuity baseline) | hf (cluster) | `slurm/llm_v21_r1_distill_32b.sbatch` | `answers_r1-distill-32b.shard{0..3}.jsonl` |
| deepseek-ai/DeepSeek-R1-0528-Qwen3-8B | always-thinking (optional) | hf (cluster) | `slurm/llm_v21_r1_0528_qwen3_8b.sbatch` | `answers_r1-0528-qwen3-8b.jsonl` |
| mistralai/mistral-small-4-119b-2603 | `reasoning_effort=none` | NIM api | `scripts/run_llm_v21_nim.sh mistral-none` | `answers_mistral-small-4_none.jsonl` |
| mistralai/mistral-small-4-119b-2603 | `reasoning_effort=high` | NIM api | `scripts/run_llm_v21_nim.sh mistral-high` | `answers_mistral-small-4_high.jsonl` |
| deepseek-ai/deepseek-v4-pro | thinking | NIM api | `scripts/run_llm_v21_nim.sh dsv4-think` | `answers_deepseek-v4-pro_think.jsonl` |
| deepseek-ai/deepseek-v4-pro | non-thinking | NIM api | `scripts/run_llm_v21_nim.sh dsv4-nothink` | `answers_deepseek-v4-pro_nothink.jsonl` |
| gemini-3-flash (default, override via `GEMINI_MODEL`) | `reasoning_effort=high` + thought summaries | Gemini api (free tier) | `scripts/run_llm_v21_gemini.sh think` | `answers_gemini-3-flash_think.jsonl` |
| gemini-3-flash (default, override via `GEMINI_MODEL`) | `reasoning_effort=minimal` | Gemini api (free tier) | `scripts/run_llm_v21_gemini.sh minimal` | `answers_gemini-3-flash_minimal.jsonl` |

Logs: SLURM writes `llmv21_*_<jobid>.out` per job/array task; the NIM
script tees to `results/llm_v21/logs/<tag>.log`. Every mode has its own
answers file and log; resume is per `prompt_id` (a record only counts as
done if it is not error/length-truncated and its final JSON parses with
all nine keys — see `is_complete_record` in `src/run_llm_v2.py` and
`tests/test_run_llm_v2.py`). Reasoning is stored separately: NIM
`reasoning_content` or the local `<think>...</think>` block land in
`"reasoning"`, the final reply in `"answer"`.

Sampling per official model guidance:

- Qwen3.6 thinking: temperature 1.0, top_p 0.95, top_k 20 (model card,
  general tasks; the card's "precise coding" preset would be 0.6).
- Qwen3.6 non-thinking: temperature 0.7, top_p 0.8, top_k 20.
- R1-Distill-32B and R1-0528-Qwen3-8B: temperature 0.6, top_p 0.95.
- Mistral Small 4: none -> 0.15 (Small-3.2-equivalent), high -> 0.7/0.95
  (Magistral-equivalent). To be confirmed, see open questions.
- DeepSeek V4 Pro (NIM): temperature 1.0, top_p 1.0; sampling parameters
  are reported to be ignored in thinking mode.

## Cluster preparation (login node, safe: no jobs, no model downloads)

Upload runner + frozen prompts into a fresh workspace subdir (from the
laptop/repo machine; resolve the workspace path on the cluster first):

```bash
WS=$(ssh uc3 'ws_find llm_pilot 2>/dev/null || echo $HOME/llm_pilot_ws')
ssh uc3 "mkdir -p '$WS/llm_v21'"
scp src/run_llm_v2.py results/llm_v2/prompts.jsonl uc3:"$WS/llm_v21/"
scp slurm/llm_v21_common.sh slurm/llm_v21_*.sbatch uc3:MasterArbeit/slurm/
```

(The sbatch files locate `llm_v21_common.sh` via the submit directory, so
submit them from the repo root or from `slurm/` on the cluster.)

Compatibility check (read-only, offline, allowed on the login node):

```bash
source "$WS/venv/bin/activate"
python3 - <<'PY'
import torch, transformers
print("python ok, torch", torch.__version__, "transformers", transformers.__version__)
# Qwen3/Qwen3.6 and R1-0528-Qwen3-8B need Qwen3 support (transformers >= 4.51;
# Qwen3.6 may need newer -- the pinned pilot venv has 4.46.3, see open questions)
from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES as M
print("qwen3 architecture registered:", "qwen3" in M)
PY
# what is already in the offline HF cache (models must be prefetched on login):
huggingface-cli scan-cache 2>/dev/null || ls "$HF_HOME/hub" 2>/dev/null
```

If transformers is too old, the venv needs an upgrade (login node,
internet available) before any Qwen3.6 / R1-0528 job — coordinate first;
do not touch the pinned pilot venv without approval since old sbatch
files still reference it. Model prefetch (login node, large download —
needs explicit go-ahead):

```bash
HF_HOME="$WS/hf_cache" huggingface-cli download Qwen/Qwen3.6-27B
HF_HOME="$WS/hf_cache" huggingface-cli download deepseek-ai/DeepSeek-R1-0528-Qwen3-8B
```

Chat-template check once the model is cached (offline, tokenizer only):

```bash
HF_HOME="$WS/hf_cache" HF_HUB_OFFLINE=1 python3 - <<'PY'
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
on  = tok.apply_chat_template([{"role":"user","content":"hi"}],
                              tokenize=False, add_generation_prompt=True,
                              enable_thinking=True)
off = tok.apply_chat_template([{"role":"user","content":"hi"}],
                              tokenize=False, add_generation_prompt=True,
                              enable_thinking=False)
print("templates differ:", on != off)   # must be True for the hybrid switch
PY
```

## Smoke runs (3 real prompts per mode — needs explicit go-ahead)

Fixed smoke prompt_ids, one per access strategy (frozen prompts):

- `0b394cf2d923` (time_agnostic_t)
- `3106eb7c74bb` (time_respecting)
- `90e26b753383` (recent_history_k20)

Cluster (writes `*_smoke.jsonl`, array tasks > 0 exit immediately):

```bash
SMOKE=1 sbatch slurm/llm_v21_qwen36_27b_think.sbatch
SMOKE=1 sbatch slurm/llm_v21_qwen36_27b_nothink.sbatch
SMOKE=1 sbatch slurm/llm_v21_r1_distill_32b.sbatch
SMOKE=1 sbatch slurm/llm_v21_r1_0528_qwen3_8b.sbatch   # optional model
```

NIM (needs NVIDIA_API_KEY in the env; writes `*_smoke.jsonl` + smoke log):

```bash
bash scripts/run_llm_v21_nim.sh mistral-none  --smoke
bash scripts/run_llm_v21_nim.sh mistral-high  --smoke
bash scripts/run_llm_v21_nim.sh dsv4-think    --smoke
bash scripts/run_llm_v21_nim.sh dsv4-nothink  --smoke
```

Smoke acceptance: 3 records each, `finish_reason` not error/length,
`reasoning` non-empty for thinking modes and empty/None for non-thinking
modes, `answer` contains a final JSON object with all nine keys
(`python3 -c "import sys; sys.path.insert(0,'src'); import run_llm_v2 as r, json; print([r.is_complete_record(json.loads(l)) for l in open(sys.argv[1])])" <file>`).

## NIM smoke results (2026-07-15) and decisions

- `mistral-none`: 3/3 stop, ~30-43 s, ~2.5-3.1k completion tokens, schema
  and metadata clean -> full run approved unchanged.
- `dsv4-nothink`: 3/3 stop, ~29-53 s, ~0.7-0.9k completion tokens, clean
  -> full run approved unchanged.
- `mistral-high`: 2/3 stop + 1/3 length at ~212-241 s. Reasoning alone
  consumed ~15.5k tokens even in the stop cases; the length record has an
  empty answer and stayed correctly retryable. Fix: budget 16384 -> 24576
  and `--stream` (runs sit close to the gateway timeout). Re-smoke before
  the full run (resume retries only the incomplete third prompt).
- `dsv4-think`: 0 records, repeated HTTP 504 with empty body -- the NIM
  gateway cuts non-streaming connections before long thinking generations
  finish (mistral-high at 241 s still passed, dsv4-think runs longer).
  Fix: `--stream` (SSE, reassembled client-side; new `read_stream` in the
  runner plus retryable mid-stream disconnects). Re-smoke required before
  the full run.
- Resume verified on the real files: none/nothink 3 done, high 2 done
  (retry = 1 prompt), think 0 done (retry = 3 prompts).

Full-run time estimates from the smoke latencies (420 prompts, serial,
1 s pacing): mistral-none ~4.3 h, dsv4-nothink ~4.9 h, mistral-high
~26-30 h, dsv4-think unknown until its re-smoke (plausibly 1.5-2.5 days
at 4-8 min/prompt). NIM allows this in parallel (per-request rate is ~1-2
per minute per run); check the remaining API credits on build.nvidia.com
before launching the two long runs (~840 requests plus retries).

## Gemini free-tier background run (laptop, multi-day)

Gemini exposes an OpenAI-compatible endpoint
(`https://generativelanguage.googleapis.com/v1beta/openai/`), so the
normal `api` backend is used. The free tier throttles per project
(roughly 10 requests/min and a per-model daily quota that resets at
midnight Pacific; exact numbers: https://aistudio.google.com/rate-limit).
`scripts/run_llm_v21_gemini.sh` is built for that: it paces requests
(`GEMINI_SLEEP`, default 7s), treats HTTP 429 as "wait, don't fail"
(`--rate-limit-max-wait`, doubling backoff capped at `GEMINI_MAX_WAIT`,
default 3600s), and loops runner passes until all 420 prompts have a
complete record. It is safe to interrupt, suspend, or restart at any
point; resume is per `prompt_id` and answers stay append-only. A Google
AI Pro subscription does not change API quotas (it covers the Gemini
app/AI Studio, not the API).

Start in the background (key from a password manager; `read -rs` keeps
it out of the shell history):

```bash
read -rs GEMINI_API_KEY && export GEMINI_API_KEY
bash scripts/run_llm_v21_gemini.sh think --smoke     # 3 prompts, check first
nohup bash scripts/run_llm_v21_gemini.sh think \
    > results/llm_v21/logs/nohup_gemini_think.out 2>&1 &
tail -f results/llm_v21/logs/gemini-3-flash_think.log
```

Stop/pause (restarting the same command resumes):

```bash
pkill -f run_llm_v21_gemini.sh || true
pkill -f 'api-key-env GEMINI_API_KEY' || true
```

Gemini-specific open points (verify in the smoke run):

- which models the project's free tier currently includes (check AI
  Studio and set `GEMINI_MODEL` accordingly; `gemini-3-flash` assumed);
- where the OpenAI-compat layer returns thought summaries — the runner
  stores `message.reasoning_content` or `message.reasoning`; if the
  smoke records show reasoning `null` in think mode, inspect one raw
  response and extend the field mapping;
- `reasoning_effort=none` is only honored by gemini-2.5 models; for
  gemini-3 models `minimal` is the lowest level, so the second arm is
  "minimal thinking", not "no thinking";
- do not attach Cloud Billing to the project mid-run: that switches the
  project to the paid tier and requests start costing money.

## Open compatibility questions

1. **transformers version on the cluster**: the pinned pilot venv
   (`slurm/pilot_env_setup.sh`) installs transformers 4.46.3 — too old for
   the Qwen3 architecture (needs >= 4.51; Qwen3.6 may require the current
   release). Check via the compat snippet; likely a new/updated venv is
   needed for the Qwen3.6 and R1-0528 jobs.
2. **Qwen3.6 non-thinking presence_penalty**: the model card recommends
   presence_penalty 1.5 in instruct mode; HF `generate()` has no such
   parameter (vLLM does). Current plan: run without it and watch for
   repetition loops in the smoke output; escalate to a vLLM-based runner
   only if needed.
3. **Mistral Small 4 sampling**: NIM docs confirm `reasoning_effort`
   `none|high` as top-level request field, but official temperature
   recommendations for the 2603 release were not verifiable (docs page
   returns 403); 0.15 / 0.7+top_p 0.95 are carried over from Small 3.2 /
   Magistral. Confirm against build.nvidia.com before the full run. Also
   note a known vLLM issue where `reasoning_effort` in the wrong place
   400s — we send it top-level, which is the hosted-NIM contract.
4. **DeepSeek V4 Pro thinking budget**: phase 3 needed 16384 tokens for
   the pro thinking mode; NIM reports sampling ignored in thinking mode.
   `--max-tokens 16384` is set; verify truncation rate in the smoke run.
5. **R1-0528-Qwen3-8B**: model card says run like Qwen3-8B but with
   DeepSeek repo configs; template auto-thinks (no `enable_thinking`).
   Only include in the matrix if the compat + smoke checks pass.

## Cleanup candidates (do NOT delete yet)

Superseded by this round (verify nothing else references them first):

- `slurm/llm_v2_qwen14b.sbatch`, `slurm/llm_v2_qwen32b.sbatch` (Qwen2.5)
- `slurm/llm_v2_r1_32b.sbatch` (replaced by `llm_v21_r1_distill_32b.sbatch`)
- `slurm/pilot_qwen14b_v2.sbatch`, `slurm/pilot_qwen32b.sbatch`,
  `slurm/pilot_r1_14b.sbatch`, `slurm/pilot_r1_32b.sbatch`,
  `slurm/pilot_r1_smoke.sbatch`, `slurm/pilot_env_setup.sh` (pilot phase;
  the env-setup pins the Qwen2.5-era venv — keep until the new venv story
  is decided)
- `src/run_llm_pilot_v3.py`, `src/make_pilot_cases_v2.py`,
  `src/make_pilot_data.py`, `src/run_pilot_walks.py`, `src/pilot_eval.py`
  (pilot pipeline)
- `src/run_deepseek_api.py`, `src/run_deepseek_api_8192.py`,
  `src/run_gemini_api.py` (phase-3 API one-offs; superseded by the
  `api` backend of `src/run_llm_v2.py`)
