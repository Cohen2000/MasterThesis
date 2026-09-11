# V2.1 LLM runs: plan and log

Both forward-looking and retrospective: the model matrix and the run recipes
below were written before the runs, the escalation ladder and its outcomes
were added afterwards. Results and their interpretation live in
`docs/LLM_EVIDENCE.md`; this file records what was executed and how.

Status 2026-07-14: runner and scripts prepared; no jobs submitted, no API
calls made. Prompts stay the frozen `results/llm_v2/prompts.jsonl`
(84 cases x 5 conditions = 420 prompts, 3 access strategies). All new
answers and logs go to `results/llm_v21/` (local/NIM) or `$WS/llm_v21/`
(cluster), never into the frozen `results/llm_v2/` files.

## Model x mode matrix

| model | mode | backend | script | out file (tag) |
|---|---|---|---|---|
| Qwen/Qwen3.6-27B | thinking | hf (cluster) | `slurm/llm_v21_qwen36_27b_think.sbatch` | `answers_qwen36-27b_think.shard{0..7}.jsonl` |
| Qwen/Qwen3.6-27B | non-thinking (`enable_thinking=False`) | hf (cluster) | `slurm/llm_v21_qwen36_27b_nothink.sbatch` (briefly dropped 2026-07-17, re-added same day; budget 3072 -> 8192 after the mistral-none verbosity lesson) | `answers_qwen36-27b_nothink.jsonl` |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-32B | always-thinking (continuity baseline) | hf (cluster) | `slurm/llm_v21_r1_distill_32b.sbatch` | `answers_r1-distill-32b.shard{0..3}.jsonl` |
| deepseek-ai/DeepSeek-R1-0528-Qwen3-8B | always-thinking (optional) | hf (cluster) | `slurm/llm_v21_r1_0528_qwen3_8b.sbatch` | `answers_r1-0528-qwen3-8b.jsonl` |
| mistralai/mistral-small-4-119b-2603 | `reasoning_effort=none` | NIM api | `scripts/run_llm_v21_nim.sh mistral-none` | `answers_mistral-small-4_none.jsonl` |
| mistralai/mistral-small-4-119b-2603 | `reasoning_effort=high` | NIM api | `scripts/run_llm_v21_nim.sh mistral-high` | `answers_mistral-small-4_high.jsonl` |
| deepseek-ai/deepseek-v4-pro | thinking | NIM api | DROPPED 2026-07-16: thinking generations run several minutes per prompt (smoke aborted); a 420-prompt run was judged too slow for the remaining time budget | -- |
| deepseek-ai/deepseek-v4-pro | non-thinking | NIM api | `scripts/run_llm_v21_nim.sh dsv4-nothink` | `answers_deepseek-v4-pro_nothink.jsonl` -- **endpoint retired 2026-08-07**, see below |
| gemini-3.1-flash-lite (default, override via `GEMINI_MODEL`) | `reasoning_effort=high` + thought summaries | Gemini api (free tier) | `scripts/run_llm_v21_gemini.sh think` | `answers_gemini-3.1-flash-lite_think.jsonl` |
| gemini-3.1-flash-lite (default, override via `GEMINI_MODEL`) | `reasoning_effort=minimal` | Gemini api (free tier) | `scripts/run_llm_v21_gemini.sh minimal` | `answers_gemini-3.1-flash-lite_minimal.jsonl` |

**Model availability, 2026-08-20.** `deepseek-ai/deepseek-v4-pro` now answers
HTTP 410 on NIM: *"has reached its end of life on 2026-08-07T09:00:00Z and is
no longer available."* The completed 420/420 run stays valid as recorded
evidence, but it can no longer be reproduced, extended, or given additional
prompt cells. Anything that would require new generations from this model --
the two new one-factor-at-a-time input cells among them -- is closed for it.

That is a finding, not just an inconvenience: a hosted API row in the model
matrix can disappear inside a single project, while the open-weights rows on
the cluster stay reproducible because the weights and sampling parameters are
held locally. Where a claim needs to survive to the thesis defence, the
cluster rows carry it more safely than the API rows.

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

2026-07-17 budget update ("no thinking cap" decision, matching the API
runs): qwen36-think 16384 -> 32768, r1-distill 12288 -> 32768 new tokens.
At this budget one 4 h array pass may not finish a shard; the full-run
chain therefore submits repeat passes via `--dependency=afterany` (resume
skips complete prompts, so extra passes are cheap no-ops):

```bash
cd ~/MasterArbeit
j1=$(sbatch --parsable slurm/llm_v21_qwen36_27b_think.sbatch)
j2=$(sbatch --parsable --dependency=afterany:$j1 slurm/llm_v21_qwen36_27b_think.sbatch)
j3=$(sbatch --parsable --dependency=afterany:$j2 slurm/llm_v21_qwen36_27b_think.sbatch)
j4=$(sbatch --parsable --dependency=afterany:$j3 slurm/llm_v21_qwen36_27b_nothink.sbatch)
j5=$(sbatch --parsable --dependency=afterany:$j4 slurm/llm_v21_qwen36_27b_nothink.sbatch)
j6=$(sbatch --parsable --dependency=afterany:$j5 slurm/llm_v21_r1_distill_32b.sbatch)
j7=$(sbatch --parsable --dependency=afterany:$j6 slurm/llm_v21_r1_distill_32b.sbatch)
```

`llm_v21_common.sh` activates `$WS/venv_v21` if present (new venv for
Qwen3.6-capable transformers), otherwise the pinned pilot venv.

Cluster smoke (writes `*_smoke.jsonl`, array tasks > 0 exit immediately):

```bash
SMOKE=1 sbatch slurm/llm_v21_qwen36_27b_think.sbatch
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
  and metadata clean -> full run approved unchanged. UPDATE full run
  2026-07-15: 59% of the first 175 records hit the 4096 cap (verbose
  edge/window enumeration before the JSON; stop records median 3210 /
  p90 3963) and temp 0.15 makes retries near-deterministic -> budget
  raised to 16384 + --stream, run restarted with resume (completed
  records under the old cap stay valid; caps do not alter sampling
  below them). Records now also carry a `max_tokens` field.
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
  the full run. UPDATE 2026-07-16: mode dropped entirely -- the streamed
  re-smoke was aborted because each thinking generation runs several
  minutes; a full run (extrapolated multi-day) does not fit the remaining
  time budget. dsv4-nothink stays in the matrix.
- Resume verified on the real files: none/nothink 3 done, high 2 done
  (retry = 1 prompt), think 0 done (retry = 3 prompts).

Full-run time estimates from the smoke latencies (420 prompts, serial,
1 s pacing): mistral-none ~4.3 h, dsv4-nothink ~4.9 h, mistral-high
~26-30 h, dsv4-think unknown until its re-smoke (plausibly 1.5-2.5 days
at 4-8 min/prompt). NIM allows this in parallel (per-request rate is ~1-2
per minute per run); check the remaining API credits on build.nvidia.com
before launching the two long runs (~840 requests plus retries).
Observed 2026-07-16: the real mistral-high pace is ~516 s/prompt
(~60 h total), i.e. about twice the smoke-based estimate; ~32% of
first attempts finish with `length` at 32768 and get their single
automatic retry from the chain's trailing pass.

### mistral-high 64k escalation pass (named ablation, 2026-07-19)

After the 32768-token run settled at 388/420, the 32 remaining prompts
were all `length`-bound: repeated attempts (up to 5) never produced a
final JSON. Token accounting on the successful answers explains why --
median ~21.8k, p90 ~31.4k, max 32,753 completion tokens, i.e. the
successful cases already press against the cap. Five prompts that had
also seen a local DNS outage were retried first; one converged
(`cbb86ee3123a`), the other four hit `length` again, so the outage was
not their cause.

The remaining 32 are therefore rerun once at `--max-tokens 65536`. This
is an explicit, named budget ablation, not a silent change: it splits
the population into "converges at 32k", "converges only at 64k", and
"does not converge", which is itself a reportable model property.

Records are distinguished by their `max_tokens` field, so the escalation
attempts stay separable from the 32k attempts in the append-only answer
file; nothing earlier is rewritten. Prompts that still finish with
`length` at 65536 stay open and are reported as non-converging.

Outcome (2026-07-19, 09:56-14:48): all 32 converged. Every escalation
attempt finished with `stop` and a complete nine-key JSON, so
mistral-high reaches 420/420. Completion tokens on those 32: min 25,504,
median 39,484, max 51,960 -- i.e. they needed 25-52k tokens, and 65,536
was never the binding constraint either.

The reportable finding is therefore that the `length` cases were a
budget artifact, not non-converging reasoning: capping mistral-high at
32,768 tokens silently drops ~8% of the suite, and plausibly the harder
cases rather than a random subset. Records carry `max_tokens` (32768 vs
65536), so an evaluation can condition on it.

## Open-weights escalation ladder (cluster, status 2026-08-02)

The three open-weights arms use the same adaptive design as the mistral-high
pass: a primary sweep, then a rerun of every incomplete case from scratch at a
larger budget, into its own answer files. `--max-length-attempts 1` gives each
case one attempt per tier and counts the `length` records already present in
the *output* file, so resubmitting a finished tier exits within seconds instead
of regenerating the same truncation. A larger budget therefore needs a new
output file; rerunning the old one is a no-op by design.

State after tier 2 (queue empty, all chains ran to completion; jobs
6103427-29 for think, 6103431-32 for r1):

| arm | primary | tier 2 | union | open |
|---|---|---|---|---:|
| Qwen3.6-27B NoThink | 8k: 222/420 | 16k: 186/198 | 408/420 | 12 |
| Qwen3.6-27B Think | 32k: 23/420 | 64k: 309/397 | 332/420 | 88 |
| R1-Distill-32B | 32k: 302/420 | 64k: 1/118 | 303/420 | 117 |

Primary and tier 2 are disjoint by construction, so the union is an addition.
Tier 2 covered every target (198 records for 198 targets, 397 for 397, 119 for
118), and every open case is `finish_reason=length` at exactly the cap, with
no errors and no OOMs anywhere. The remaining gap is budget, not breakage.

The elasticity separates the arms and drives the decision:

- **NoThink** is capped at 16384, far below what the thinking arm received,
  and only 12 cases remain -> third tier at 32768
  (`llm_v21_qwen36_27b_nothink_32k.sbatch`, 4 shards, 3 cases each).
- **Think** went from 23/420 at 32k to 309/397 at 64k, so the tail is still
  live -> third tier at 126976
  (`llm_v21_qwen36_27b_think_128k.sbatch`, 16 shards, ~6 cases each).
- **R1-Distill** resolved 1 of 118 at 64k. That is a flat curve; no further
  tier is planned and the arm is reported at 303/420 under failure-penalized
  loss, with the non-convergence itself as the finding.

Sampling is identical across an arm's tiers, so the budget is the only
variable; with the same seed a tier-3 generation reproduces the truncated
tier-2 prefix and continues past it. Escalation files beyond tier 2 must pass
*every* earlier tier to `select_llm_escalation.py` — passing only the primary
would reselect the cases tier 2 already resolved.

### Preflight (login node, read-only, offline)

The 128k budget has to fit together with the prompt inside the model's
position limit; the frozen prompts reach ~2.5k tokens, hence 126976.

Read it straight out of the cached `config.json`. No module, no venv, no
transformers -- deliberately: `module load` followed by `source activate` in a
shell that already has `venv_v21` active leaves `sys.prefix` pointing at the
module instead of the venv and the interpreter dies with
`LookupError: unknown encoding: UTF-8`. The same sequence is fine inside a
SLURM job, where the shell starts clean, which is why `llm_v21_common.sh`
keeps it.

```bash
WS=$(ws_find llm_pilot 2>/dev/null || echo $HOME/llm_pilot_ws)
CFG=$(ls "$WS/hf_cache/hub/models--Qwen--Qwen3.6-27B/snapshots/"*/config.json | head -1)
grep -E 'max_position_embeddings|rope_theta|rope_scaling|num_hidden_layers|num_key_value_heads|head_dim' "$CFG"
sed -n '/"layer_types"/,/\]/p' "$CFG" | grep -o '"[a-z_]*"' | sort | uniq -c
```

Measured 2026-08-02 on the cached snapshot
(`hf_cache/hub/models--Qwen--Qwen3.6-27B/snapshots/6a9e13bd6fc8/config.json`):
`max_position_embeddings` is 262144, so the position limit is not binding at
any budget considered here, and no rope scaling is involved.

Memory is not binding either, which was not obvious in advance. The model is
hybrid: `layer_types` is 16 `full_attention` and 48 `linear_attention` layers,
so only 16 layers hold a KV cache. With 4 KV heads and `head_dim` 256 in
bf16 that is 2 x 4 x 256 x 2 B = 4 KB per token per attending layer, i.e.
64 KB per token overall -- not the 256 KB a 64-layer full-attention model of
this size would need:

| budget | KV cache | + 27B bf16 weights | of 95.8 GB |
|---|---|---|---|
| 65536 (tier 2, ran) | ~4 GB | ~58 GB | 61 % |
| 126976 (tier 3) | ~8 GB | ~62 GB | 65 % |
| 262144 (hypothetical) | ~17 GB | ~71 GB | 74 % |

The binding constraint is therefore wall time, not the GPU. 126976 tokens at
the observed tier-2 rate is roughly 50-55 min per case, so ~6 cases per shard
fit inside the 8 h limit. A 256k tier would need two passes per shard; it
stays technically possible if a large share of the 88 still hits the cap.

If a future model does make the ceiling binding, lower the budget with
`MAX_NEW_TOKENS=<n> sbatch ...`. Case counts (must print 12 and 88):

```bash
cd "$WS/llm_v21"
python3 select_llm_escalation.py --prompts prompts.jsonl \
  --answers 'answers_qwen36-27b_nothink.shard*.jsonl' \
  --answers 'answers_qwen36-27b_nothink_16k.shard*.jsonl' --output lines | wc -l
python3 select_llm_escalation.py --prompts prompts.jsonl \
  --answers 'answers_qwen36-27b_think.shard*.jsonl' \
  --answers 'answers_qwen36-27b_think_64k.shard*.jsonl' --output lines | wc -l
```

### Submit

Chain a second pass per tier: tier-2 array tasks hit the 8 h limit repeatedly,
and a repeat pass picks up exactly the cases a TIMEOUT left without an attempt
while skipping everything already length-capped.

```bash
scp slurm/llm_v21_qwen36_27b_nothink_32k.sbatch \
    slurm/llm_v21_qwen36_27b_think_128k.sbatch uc3:MasterArbeit/slurm/

ssh uc3
cd ~/MasterArbeit
n1=$(sbatch --parsable slurm/llm_v21_qwen36_27b_nothink_32k.sbatch)
sbatch --dependency=afterany:$n1 slurm/llm_v21_qwen36_27b_nothink_32k.sbatch
t1=$(sbatch --parsable slurm/llm_v21_qwen36_27b_think_128k.sbatch)
sbatch --dependency=afterany:$t1 slurm/llm_v21_qwen36_27b_think_128k.sbatch
```

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
tail -f results/llm_v21/logs/gemini-3.1-flash-lite_think.log
```

Stop/pause (restarting the same command resumes):

```bash
pkill -f run_llm_v21_gemini.sh || true
pkill -f 'api-key-env GEMINI_API_KEY' || true
```

Gemini findings (smoke runs + quota page, 2026-07-15):

- model id: `gemini-3-flash` does not exist (ListModels); the project's
  free-tier quotas allow only 20 requests/day on the big flash models
  (3.5/3/2.5) — a `gemini-3.5-flash` think run stalled at ~14 prompts.
  `gemini-3.1-flash-lite` has 500 requests/day at 15 RPM and is now the
  default: one full 420-prompt mode fits into one day, run the two modes
  on separate days (they share the model's daily quota). The abandoned
  partial `answers_gemini-3.5-flash_think.jsonl` (14 records) stays
  untouched. Multi-account key rotation was considered and rejected
  (Gemini ToS: quota circumvention; account-suspension risk).
- thought summaries arrive inline in `content` as
  `<thought>...</thought>` (not in `reasoning_content`); the runner
  splits them into the `reasoning` field via `split_think`, which
  handles both tag styles. Truncated thought blocks stay in `answer`
  and remain retryable.
- Gemini counts thinking tokens against `max_tokens`: 16384 hit
  "length" at `reasoning_effort=high`, budget raised to 32768.
- Gemini 400s when `reasoning_effort` and `thinking_config` are both
  sent; with `--include-thoughts` the effort moves into
  `thinking_config.thinking_level` (`gemini_thinking_body`).
- `reasoning_effort=none` is only honored by gemini-2.5 models; for
  gemini-3.x models `minimal` is the lowest level, so the second arm is
  "minimal thinking", not "no thinking";
- transient HTTP 503 "high demand" happens and is absorbed by the
  normal retries; persistent 429 is absorbed by the quota backoff.
- do not attach Cloud Billing to the project mid-run: that switches the
  project to the paid tier and requests start costing money. (If the
  big flash models are ever wanted, the official route is the Google
  Cloud trial credit / paid Tier 1 — a separate decision.)

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
