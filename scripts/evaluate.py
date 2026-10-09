#!/usr/bin/env python3
"""Rebuild the result tables in docs/results/final from the frozen predictions.

In plain words: PREDICTIONS.csv holds every estimate of the study, one row per sample, method
and answer. This script checks that table against the frozen inputs and then writes all
result tables from it. It calls no model and changes no prediction.

  1. The 384 samples and the raw answer files of the API models must be exactly the files
     listed in CHECKSUMS.json, and every prompt must be the one the repository builds.
  2. Every API answer is scored again from its raw text and must equal its row in PREDICTIONS.csv.
  3. Every stored error is recomputed from TRUTH.json and must equal the stored one.
  4. The tables and the two reports are written.

usage: python scripts/evaluate.py [--external ~/.local/share/masterthesis] [--out docs/results/final]

The samples (api_observations) and the raw answers (api_runs) are not in the repository.
"""
import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT/'production'))
from study.answer_format import parse_final, valid_profile  # noqa: E402
from study.common import sha, write_csv  # noqa: E402
from study.observation import messages  # noqa: E402
from study.tables import (LABEL, QWEN, APIM, errors, fixed_input_variability, history_markdown, history_summary,  # noqa: E402
                          main_markdown, paired_families, paired_methods, per_source, s_design_appendix, summary,
                          variability, variability_markdown, walk_markdown)
from api_runner import actual_usd, read_jsonl  # noqa: E402  (list prices of the providers)

FINAL = ROOT/'docs/results/final'
EXTERNAL = Path.home()/'.local/share/masterthesis'
# Run folder of each API model and the price list its spend is computed with.
RUNS = {'deepseek_flash': ('deepseek', 'deepseek'), 'gpt_6_sol': ('openai', 'openai'),
        'gpt_6_sol_tools': ('openai_tools', 'openai')}
ET_FITS = 11                # the reported fit plus ten fits on newly drawn training samples
SCORES = ('AE2', 'ProfileAE', 'signed_rho2')


# ---------------------------------------------------------------- 1. frozen inputs
def frozen_samples(external, checksums):
    """The 384 samples, checked file by file against CHECKSUMS.json."""
    folder, listed = external/'api_observations', checksums['observations']
    files = sorted(folder.glob('*.json'))
    if {p.name: sha(p) for p in files} != listed['files']:
        raise ValueError(f'{folder}: the samples differ from CHECKSUMS.json')
    samples = [json.loads(p.read_text()) for p in files]
    identity = [(s['id'], s['block_sha256'], s['prompt_sha256']) for s in sorted(samples, key=lambda s: s['id'])]
    if hashlib.sha256(json.dumps(identity, separators=(',', ':')).encode()).hexdigest() != listed['fingerprint_sha256']:
        raise ValueError('the samples differ from the fingerprint in CHECKSUMS.json')
    for s in samples:
        if s['messages'] != messages(s['block']):
            raise ValueError(f"{s['id']}: the stored prompt is not the prompt built from config/prompts")
    return {s['id']: s for s in samples}


def check_answer_files(external, checksums):
    for folder, listed in checksums['answers']['files'].items():
        found = {p.name: sha(p) for p in sorted((external/'api_runs'/folder).iterdir()) if p.is_file()}
        if found != listed:
            raise ValueError(f'api_runs/{folder}: the raw answer files differ from CHECKSUMS.json')


# ---------------------------------------------------------------- 2. API answers
def scored_answers(external, samples):
    """Every API answer scored from its raw text: {(sample, method, repeat): (prediction, reason)}.

    The same rule for every language model: the answer must be one JSON object with rho_2..rho_5,
    values in [0, 1], never increasing. An answer that stopped at the length limit is invalid.
    Nothing is repaired; a missing answer would show up as 'not_completed'.
    """
    scored = {}
    for method, (folder, _) in RUNS.items():
        answers = {a['id']: a for a in read_jsonl(external/'api_runs'/folder/'responses.jsonl')
                   if a.get('kind', 'main') == 'main'}
        for sample in samples:
            for repeat in (1, 2, 3):
                answer = answers.get(f'{sample}__{folder}__r{repeat}')
                values, reason = None, 'not_completed'
                if answer is not None and answer.get('limit_hit'):
                    reason = 'generation_limit'
                elif answer is not None:
                    values, reason = parse_final(answer.get('final_text') or '')
                scored[sample, method, repeat] = (values, reason)
    return scored


def check_api_rows(rows, scored):
    stored = {(r['observation_id'], r['method'], int(r['repeat_index'])): r for r in rows if r['method'] in RUNS}
    if set(stored) != set(scored):
        raise ValueError('PREDICTIONS.csv does not hold exactly three answers per sample and API model')
    for key, (values, reason) in scored.items():
        r = stored[key]
        if (r['prediction'], r['valid'], r['validation_reason']) != (values, values is not None, reason):
            raise ValueError(f'{key}: the raw answer does not give the row in PREDICTIONS.csv')


# ---------------------------------------------------------------- 3. the table of all predictions
def read_predictions(path):
    rows = []
    for raw in csv.DictReader(path.open()):
        r = dict(raw)
        r['prediction'] = json.loads(r['prediction']) if r['prediction'] else None
        r['valid'] = r['valid'] == 'True'
        r['replicate'] = int(r['replicate']) if r['replicate'] else None
        for key in ('truth_rho2', *SCORES):
            r[key] = float(r[key]) if r[key] else None
        rows.append(r)
    return rows


def check_predictions(rows, samples, truth):
    """29 rows per sample, and every stored error equals the error recomputed from the truth."""
    per_sample = Counter(r['observation_id'] for r in rows)
    if set(per_sample) != set(samples) or set(per_sample.values()) != {3 + ET_FITS + 3*len(QWEN+APIM)}:
        raise ValueError('PREDICTIONS.csv needs 29 rows for each of the 384 samples')
    for r in rows:
        s = samples[r['observation_id']]
        if (r['source'], r['group'], r['arm'], int(r['sample_index'])) != (s['graph_id'], s['stratum'], s['arm'], s['sample_index']):
            raise ValueError(f"{r['observation_id']}: row and sample disagree")
        if r['method'] == 'et' and (not r['valid'] or valid_profile(r['prediction']) != r['prediction']):
            raise ValueError(f"{r['observation_id']}: ExtraTrees output is not a valid profile")
        if r['valid'] != (r['prediction'] is not None) or r['truth_rho2'] != truth[r['source']][0]:
            raise ValueError(f"{r['observation_id']}/{r['method']}: validity or truth is inconsistent")
        if {k: r[k] for k in SCORES} != errors(r['prediction'], truth[r['source']]):
            raise ValueError(f"{r['observation_id']}/{r['method']}: stored error differs from the recomputed one")


# ---------------------------------------------------------------- 4. tables and reports
def write_tables(out, rows, sample_files):
    main = [r for r in rows if r['replicate'] in (None, 0)]        # ExtraTrees: the reported fit only
    table, methods, families, networks = summary(main), paired_methods(main), paired_families(main), per_source(main)
    write_csv(out/'SUMMARY.csv', table); write_csv(out/'PAIRED_METHODS.csv', methods)
    write_csv(out/'PAIRED_FAMILIES.csv', families); write_csv(out/'PER_SOURCE.csv', networks)
    design = s_design_appendix(sample_files, main)
    write_csv(out/'S_DESIGN_APPENDIX.csv', design)
    (out/'MAIN_RESULTS.md').write_text(main_markdown(table, methods, families, networks, design))
    training, sampling, response = variability(rows, ET_FITS - 1)
    write_csv(out/'VARIABILITY_TRAINING.csv', training)
    write_csv(out/'VARIABILITY_SAMPLING.csv', sampling)
    write_csv(out/'VARIABILITY_RESPONSE.csv', response)
    (out/'VARIABILITY.md').write_text(variability_markdown(training, sampling, response, ET_FITS - 1,
                                                            fixed_input_variability(rows)))
    losses, errs = history_summary(out/'HISTORY.csv')
    write_csv(out/'HISTORY_LOSSES.csv', losses); write_csv(out/'HISTORY_ERRORS.csv', errs)
    (out/'HISTORY.md').write_text(history_markdown(losses, errs))
    (out/'WALK.md').write_text(walk_markdown(list(csv.DictReader((out/'WALK.csv').open()))))


def recorded_spend(external):
    """USD per API model, from the token counts the providers reported (short test requests included)."""
    spend = {}
    for method, (folder, prices) in RUNS.items():
        run = external/'api_runs'/folder
        answers = (read_jsonl(run/'technical_responses.jsonl') if prices == 'openai' else []) + read_jsonl(run/'responses.jsonl')
        spend[method] = round(sum(actual_usd(a, prices) for a in answers), 4)
    return spend


def write_report(out, rows, spend):
    answers = Counter(r['method'] for r in rows)
    valid = Counter(r['method'] for r in rows if r['valid'])
    counts = '\n'.join(f'| {LABEL[m]} | {answers[m]:,} | {valid[m]:,} |' for m in QWEN+APIM)
    (out/'REPORT.md').write_text(f"""# Final results

This folder holds every result of the study. The [study design](../../DESIGN.md) explains the networks, the four sampling arms (R, S, H, B), the methods and the scoring; terms are explained in its [glossary](../../DESIGN.md#glossary). In short: 32 networks (12 real, a time-shuffled copy of each, 8 synthetic), each sampled three times in each arm, give 384 samples. Every method estimates `rho_2`, the share of interacting pairs that are active in at least two of five time windows, from exactly the same samples.

## Files

- [`MAIN_RESULTS.md`](MAIN_RESULTS.md): the main tables. For each group of networks, sampling arm and method: the average error (MAE_2 and ProfileMAE), its direction and the share of valid answers, plus method-against-method comparisons. The same numbers as CSV: `SUMMARY.csv` (per group), `PER_SOURCE.csv` (per network), `PAIRED_METHODS.csv` and `PAIRED_FAMILIES.csv` (comparisons).
- `PREDICTIONS.csv`: every single estimate, one row per sample and method (ExtraTrees: per fit; language models: per repeat). 29 rows per sample, {len(rows):,} rows. `TRUTH.json`: the true rho_2..rho_5 of every network.
- [`VARIABILITY.md`](VARIABILITY.md): how much results change between ExtraTrees fits, between the three samples of a network, and between the three answers of a language model.
- [`HISTORY.md`](HISTORY.md): why seeing only the last 60% of the time span (arm H) biases the estimates.
- [`WALK.md`](WALK.md): a check of the random walk (arm S) on all 32 networks, with 1000 simulated walks each.
- [`W_SENSITIVITY.md`](W_SENSITIVITY.md): how the true answer and the ordering of the networks change with 2 to 20 instead of 5 time windows (`scripts/window_sensitivity.py`; all values in `W_SENSITIVITY.csv` and `W_SENSITIVITY_NETWORKS.csv`).
- `S_DESIGN_APPENDIX.csv`: a simple re-weighted estimate for arm S, shown for comparison only.
- `CHECKSUMS.json`: SHA-256 fingerprints of the raw data files, of the 384 frozen samples given to the API models, and of every raw answer file, so anyone can verify they have the same data.

## When an answer counts

A language-model answer is valid when its final text is exactly one JSON object with the four values rho_2..rho_5, each between 0 and 1 and never increasing (one surrounding code block is allowed), and the model did not stop at its length limit. Invalid answers are counted and left out of the error; language-model answers are never corrected or asked again. ExtraTrees output is always turned into a valid profile: each value is limited to [0, 1] and each rho_k to at most rho_(k-1).

| Model | Answers | Valid |
|---|---:|---:|
{counts}

## Checks

- `PREDICTIONS.csv` has exactly 29 rows for each of the 384 samples: observed share, training median, statistical model, 11 ExtraTrees fits and 3 answers from each of the five language-model configurations.
- The evaluation recomputes every stored error from `TRUTH.json` and stops if one differs.
- Before any request, the 384 samples are checked against the fingerprint in `CHECKSUMS.json`, and every prompt is checked against its sample.
- Leakage check over all 528 ExtraTrees fits: no network being predicted (or its time-shuffled copy) appears in the training data or in the tuning of the fit that predicts it.
- ExtraTrees output: turning it into valid profiles changed 88 of 4,224 predictions (12 of 384 in the reported fit) slightly and no rho_2 value.
- Random-walk check: networks where the walk's bias cannot be corrected at the 10% budget are marked in `WALK.md`; no network is excluded.

## API spend

Spend computed from the token counts the providers reported, at list prices, including a few short test requests made before the main runs:

| Provider | Model | Spend (USD) |
|---|---|---:|
| DeepSeek | DeepSeek Flash | {spend['deepseek_flash']:.2f} |
| OpenAI | GPT-6 Sol | {spend['gpt_6_sol']:.2f} |
| OpenAI | GPT-6 Sol + Python | {spend['gpt_6_sol_tools']:.2f} |
| OpenAI | total | {spend['gpt_6_sol'] + spend['gpt_6_sol_tools']:.2f} |
""")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--external', type=Path, default=EXTERNAL, help='folder with api_observations and api_runs')
    ap.add_argument('--out', type=Path, default=FINAL, help='folder with PREDICTIONS.csv, TRUTH.json, HISTORY.csv, WALK.csv')
    a = ap.parse_args()
    checksums = json.loads((a.out/'CHECKSUMS.json').read_text())
    truth = json.loads((a.out/'TRUTH.json').read_text())
    samples = frozen_samples(a.external, checksums)
    check_answer_files(a.external, checksums)
    rows = read_predictions(a.out/'PREDICTIONS.csv')
    check_api_rows(rows, scored_answers(a.external, samples))
    check_predictions(rows, samples, truth)
    write_tables(a.out, rows, sorted((a.external/'api_observations').glob('*.json')))
    write_report(a.out, rows, recorded_spend(a.external))
    print(f'{len(rows):,} predictions checked; tables written to {a.out}')


if __name__ == '__main__':
    main()
