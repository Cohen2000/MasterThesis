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
- token caps: 8,192 non-thinking and 16,384 thinking

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
