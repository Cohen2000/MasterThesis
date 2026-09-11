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

Two reductions, decided in that order on the same day. The second was taken
because the first still left the weekly allowance tight.

### First cut — the auxiliary cells

| | prompts | status |
|---|---|---|
| 2×2 factorial (`hidden`, `mechanism`, `direction_only`, `mechanism_direction`) | 640 | core |
| `metadata_only` | 32 | **on hold** |
| `mismatched` | 64 | **on hold** |

### Second cut — half the instances

The core is narrowed to **16 of the 32 instances**, giving 16 × 5 arms ×
4 conditions = 320 factorial prompts. The other 16 instances' factorial
prompts move to the hold file.

The subsample is stratified over the twelve graph groups, which are the
resampling unit of the cluster bootstrap; losing a group would cost more than
losing a variant inside one. Every group keeps at least one instance, leftover
slots go to the smallest groups first, and where a group has more instances
than slots the pick is drawn by `random.Random(20260901)` — the frozen master
seed. That lands on both instances of each of the four synthetic groups and
one of three from each of the eight real ones.

The choice is deliberately *not* by position: `prompt_id` order tracks the
graph family, so taking the first sixteen would have selected on exactly the
covariate the analysis is about. It is also deliberately not chosen to overlap
with what Codex had already answered — the answered prefix is alphabetical by
case_id and therefore family-structured, so maximizing reuse would have
imported the same bias. The 47 completed factorial answers that fall outside
the subsample stay in the append-only file and are reported as an extended
set, never as part of the primary cell.

Selected instances:

```
activity__n1500__f0__beta1              dar__dcsbm__n1500__f0__a0.1__c0.15
activity__n1500__f0__memoryless         dar__dcsbm__n1500__f0__a0.9__c0.15
activity__n500__f0__beta1               dar__dcsbm__n500__f0__a0.1__c0.15
activity__n500__f0__memoryless          dar__dcsbm__n500__f0__a0.9__c0.15
controlled__snap_bitcoin_otc__…rho0.55  controlled__sp_highschool2013__…rho0.55
controlled__snap_collegemsg__…rho0.15   controlled__sp_hypertext2009__…rho0.55
controlled__snap_email_eu__…rho0.55     controlled__sp_primaryschool__…rho0.55
controlled__snap_mathoverflow__…rho0.55 real__sp_hospital
```

### What it costs

The primary endpoint is the cluster-bootstrap slope of `Delta_i` on
`delta_i`, and its precision scales with the number of paired cases. Halving
the instances widens that interval; it does not bias it, and it leaves the
factorial balanced — all four conditions and all five arms keep equal case
counts, and the within-case pairing across conditions is untouched. Qwen
carries the full 1,136-prompt set, so the full-precision estimate exists; the
Codex row is the across-model confirmation at reduced precision, and the
write-up has to say so rather than present the two as equally powered.

Of the 640 factorial prompts, 128 already carry a usable Codex answer from
Step 1. Step 1 used the same frozen prompts, the same CLI binary
(`codex-cli 0.146.0`, `gpt-5.6-sol`, effort `high`, `notools`, 0 tool events)
and the same protocol, so its generation-0 record **is** the Step 2
generation-0 record for those prompt_ids. They are not re-run.

Of the 320 prompts in the halved core, 64 carry a usable Step 1 answer and 96
were already answered in Step 2.

Remaining Codex work: **160 calls**, ~8.9M tokens, ~1.9 windows — down from
572 before either cut.

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
python src/select_codex_core.py --instances 16
# -> prompts_codex_core.jsonl (256 to select from), prompts_codex_hold.jsonl (416)
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
