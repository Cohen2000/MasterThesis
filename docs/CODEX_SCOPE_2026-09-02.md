# Codex scope reduction — 2026-09-02

Amendment to `docs/FREEZE_2026-09.md`. It changes **what is run on Codex and in
what order**. It changes no arm, no budget, no sampler, no prompt text and no
prompt hash. The frozen prompt file is untouched; the two files written here are
selections from it.

## Why

Codex is quota-bound, not latency-bound: ~5M tokens per 5-hour window at a
measured mean of 55,871 tokens per call, i.e. ~85 usable calls per window. The
remaining 572 prompts would have cost ~32M tokens (~6.4 windows) and risked the
weekly allowance. Qwen carries the full 1,136-prompt set on the cluster at no
quota cost, so Codex's role is confirmatory across models, not primary.

## What changes

| | prompts | status |
|---|---|---|
| 2×2 factorial (`hidden`, `mechanism`, `direction_only`, `mechanism_direction`) | 640 | **run in full** |
| `metadata_only` | 32 | **on hold** |
| `mismatched` | 64 | **on hold** |

Of the 640 factorial prompts, 128 already carry a usable Codex answer from
Step 1. Step 1 used the same frozen prompts, the same CLI binary
(`codex-cli 0.146.0`, `gpt-5.6-sol`, effort `high`, `notools`, 0 tool events)
and the same protocol, so its generation-0 record **is** the Step 2
generation-0 record for those prompt_ids. They are not re-run.

Remaining Codex work: **399 calls**, ~22M tokens, ~4.7 windows.

## Consequences for the analysis

Codex generation-0 answers now live in **two** files and any evaluation must
read their union, deduplicated by `prompt_id`:

- `results/final_run_g2/answers/step1_codex_gen0.jsonl` (128 factorial prompts)
- `results/final_run_g2/answers/step2_codex_gen0.jsonl` (the rest)

`results/final_run_g2/answers/step1_codex_gen1.jsonl` is a *second* generation
of those same 128 prompts. It stays out of the single-generation Codex table and
is reported separately as the generation-noise check it was built for.

## What "on hold" means

Deferred, not dropped. `results/final_run_g2/prompts_codex_hold.jsonl` is a
runnable selection; if quota allows, it is run into the same append-only answer
file and the cells reappear in the analysis unchanged. The 21 answers those
cells already have (14 `mismatched`, 7 `metadata_only`) stay in the file and are
reported as partial, never as a cell.

The reduction was decided on quota and on cell membership. No answer from the
current run was inspected before deciding; `src/select_codex_core.py` reads
`prompt_id` only.

## Reproducing the split

```bash
python src/select_codex_core.py     # -> prompts_codex_core.jsonl, prompts_codex_hold.jsonl
```

## Command in force

```bash
python scripts/codex_screen/run_codex_screen.py --arm notools \
  --prompts results/final_run_g2/prompts_codex_core.jsonl \
  --condition "" --input-kind mask \
  --codex-bin /home/albert/.codex/packages/standalone/releases/0.146.0-x86_64-unknown-linux-musl/bin/codex \
  --out results/final_run_g2/answers/step2_codex_gen0.jsonl \
  --out-dir "$HOME/Dokumente/codex_screen" \
  --wait-for-reset --max-waits 64
```

The binary is pinned deliberately: the PATH default has moved to
`codex-cli 0.151.0`, and mixing two CLI versions inside one arm would change the
protocol mid-experiment without leaving a mark in the answer file.
