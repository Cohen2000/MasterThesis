# Qwen 3.6 non-walk screen

This is an exploratory, bounded run, not the final thesis comparison.

- model: `Qwen/Qwen3.6-27B`
- modes: thinking and non-thinking
- access strategies: all eight non-walk strategies
- budget: 800 events
- cases: one target-blind, hash-selected instance from each of four data blocks
- sample seed: 0
- conditions: sample and metadata-only control
- size: 32 cases and 64 prompts per mode, 128 generations total
- execution: four shards per mode, one pass, no unchanged-budget retries
- token caps: 8,192 non-thinking and 32,768 thinking, both recorded in the
  `max_tokens` field of every answer record

An earlier version of this document reported a 16,384 thinking cap. That figure
belongs to the smoke run: `answers_nonwalk_qwen36_think_smoke.jsonl` is the only
record in the whole screen written at 16,384, and all 64 answers of the actual
thinking arm carry `max_tokens = 32768`.

The correction matters less than what checking it exposed. **31 of the 64
thinking answers, 48%, stopped at the token cap rather than at an end of
answer**, against 4 of 64 (6%) in the non-thinking arm at 8,192:

| mode | cap | answers | truncated |
|---|---:|---:|---:|
| non-thinking | 8,192 | 64 | 4 (6%) |
| thinking | 32,768 | 64 | 31 (48%) |

A truncated answer has no final JSON object, so it is a failure, not a slow
answer. Truncation is also not spread evenly: it removes the cases the model
reasoned longest about. Roughly half the thinking arm is therefore missing in a
way that correlates with difficulty, which is enough to invalidate a strategy
ranking read off that arm -- the thinking rows of this screen should be read as
a statement about the token budget, not about the access mechanisms. The
non-thinking arm does not have this problem.

This screen has no escalation ladder to repair it, unlike the noise probe and
the later non-walk expansion, which rerun incomplete prompts at a higher cap.

The four cases per strategy are enough for a potential screen, but not for a
final claim or a reliable strategy ranking.

## 1. Prepare locally

Run from the repository root:

```bash
bash scripts/prepare_nonwalk_qwen36_screen.sh
```

## 2. Upload from the local machine

```bash
WS=$(ssh uc3 'ws_find llm_pilot 2>/dev/null || echo $HOME/llm_pilot_ws')
ssh uc3 "mkdir -p '$WS/llm_v21'"
scp src/run_llm_v2.py uc3:"$WS/llm_v21/"
scp results/nonwalk_llm_qwen36_screen/prompts.jsonl \
    uc3:"$WS/llm_v21/nonwalk_qwen36_screen_prompts.jsonl"
scp slurm/llm_v21_common.sh slurm/nonwalk_qwen36_screen.sbatch \
    uc3:MasterArbeit/slurm/
```

**The explicit target name matters.** `nonwalk_qwen36_screen.sbatch` reads
`${PROMPTS_FILE:-nonwalk_qwen36_screen_prompts.jsonl}`, not `prompts.jsonl`.
Copying this file into `$WS/llm_v21/` under its own basename would both leave
the job without its prompts *and* overwrite the frozen 420-prompt V2.1 suite
that lives at `$WS/llm_v21/prompts.jsonl`, silently changing what any resumed
frozen-suite job would run. (Corrected 2026-08-20; the earlier command in this
file did exactly that.)

The selected truth rows remain local and are used only after inference.

## 3. Smoke on the cluster

```bash
ssh uc3
cd ~/MasterArbeit
SMOKE=1 sbatch slurm/nonwalk_qwen36_screen.sbatch
```

This starts one prompt per mode; the other array shards exit immediately.
Monitor with:

```bash
squeue --me
tail -f nonwalk_qwen36_screen_<JOBID>_0.out
tail -f nonwalk_qwen36_screen_<JOBID>_4.out
```

## 4. Full bounded screen

```bash
cd ~/MasterArbeit
sbatch slurm/nonwalk_qwen36_screen.sbatch
```

The one array job contains four non-thinking and four thinking tasks. It does
not automatically resubmit truncated prompts.

## 5. Download to the local machine

```bash
WS=$(ssh uc3 'ws_find llm_pilot 2>/dev/null || echo $HOME/llm_pilot_ws')
mkdir -p results/nonwalk_llm_qwen36_screen/answers
scp "uc3:$WS/llm_v21/answers_nonwalk_qwen36_*.jsonl" \
    results/nonwalk_llm_qwen36_screen/answers/
```

## 6. Evaluate locally

```bash
PYTHONPATH=src python3 src/evaluate_nonwalk_llm.py \
    --prompts results/nonwalk_llm_qwen36_screen/prompts.jsonl \
    --cases results/nonwalk_llm_qwen36_screen/selected_cases.csv \
    --answers 'results/nonwalk_llm_qwen36_screen/answers/answers_nonwalk_qwen36_*.jsonl' \
    --baseline-predictions results/nonwalk_screen/baselines/predictions.csv.gz \
    --out-dir results/nonwalk_llm_qwen36_screen/eval
```

The main report is `results/nonwalk_llm_qwen36_screen/eval/SUMMARY.md`.
It reports failure-penalized and complete-case profile MAE, answer validity,
missing jobs, the prompt-parity ExtraTrees controls, and the difference between
sample and metadata-only performance.
