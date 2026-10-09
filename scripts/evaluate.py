#!/usr/bin/env python3
"""Rebuild the result tables in docs/results/final from the frozen table of all predictions.

PREDICTIONS.csv holds every estimate of the study, one row per sample, method and answer.
This script first checks it and then writes all tables from it. It calls no model.

  1. The samples and the raw answer files of the API models are the frozen ones (CHECKSUMS.json).
  2. Every answer of a language model, scored again from its raw text, gives its row in the table.
  3. Every stored error equals the error recomputed from TRUTH.json.

Needs results/samples (scripts/samples.py) and the raw answers in ~/.local/share/masterthesis.
"""
import csv
import json
import sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from study.answers import parse_final, valid_profile  # noqa: E402
from study.common import EXTERNAL, FINAL, OUT, REPEATS, read_json, sha, write_csv  # noqa: E402
from study.tables import (APIM, QWEN, errors, fixed_input_variability, history_markdown, history_summary, main_markdown,  # noqa: E402
                          paired_families, paired_methods, per_source, s_design_appendix, summary, variability,
                          variability_markdown, walk_markdown)

API_RUNS = {'deepseek_flash': 'deepseek', 'gpt_6_sol': 'openai', 'gpt_6_sol_tools': 'openai_tools'}   # method: answer folder
ET_FITS = 11                # the reported fit plus ten fits on newly drawn training samples
SCORES = ('AE2', 'ProfileAE', 'signed_rho2')


def frozen_inputs():
    """The 384 samples, after checking them and the API answer files against CHECKSUMS.json."""
    sums = read_json(FINAL/'CHECKSUMS.json')
    files = sorted((OUT/'samples').glob('*.json'))
    if {p.name: sha(p) for p in files} != sums['observations']['files']:
        raise SystemExit('results/samples are not the frozen samples; run scripts/samples.py')
    for folder, listed in sums['answers']['files'].items():
        if {p.name: sha(p) for p in (EXTERNAL/'api_runs'/folder).iterdir() if p.is_file()} != listed:
            raise SystemExit(f'api_runs/{folder}: the raw answers are not the frozen ones')
    return {s['id']: s for s in map(read_json, files)}


def scored_answers(samples):
    """Every answer of a language model, scored from its raw text: {(sample, method, repeat): (values, reason)}.
    An answer that stopped at the length limit or was cut off is invalid; nothing is repaired."""
    out = {}
    for method, folder in API_RUNS.items():
        lines = (EXTERNAL/'api_runs'/folder/'responses.jsonl').read_text().splitlines()
        answers = {a['id']: a for a in map(json.loads, lines) if a.get('kind', 'main') == 'main'}
        for sample in samples:
            for repeat in range(1, REPEATS+1):
                a = answers[f'{sample}__{folder}__r{repeat}']
                out[sample, method, repeat] = (None, 'generation_limit') if a.get('limit_hit') else parse_final(a.get('final_text') or '')
    files = {p.stem: p for p in (EXTERNAL/'qwen_runs').glob('**/answers/*_r*/*.json')}
    for sample, s in samples.items():
        version = 'panel888-access-v10-20260923' if s['arm'] == 'S' else 'panel888-access-v9-20260922'   # part of the answer's name
        for method in QWEN:
            for repeat in range(1, REPEATS+1):
                a = read_json(files[f'{sample}__{method}__r{repeat}__{version}'])
                failed = a.get('status') != 'completed' or a.get('technical_error')
                out[sample, method, repeat] = (None, a.get('end_state', 'technical_error')) if failed else parse_final(a.get('final_text', ''))
    return out


def read_predictions():
    rows = []
    for r in csv.DictReader((FINAL/'PREDICTIONS.csv').open()):
        r['prediction'] = json.loads(r['prediction']) if r['prediction'] else None
        r['valid'] = r['valid'] == 'True'
        r['replicate'] = int(r['replicate']) if r['replicate'] else None
        for key in ('truth_rho2', *SCORES): r[key] = float(r[key]) if r[key] else None
        rows.append(r)
    return rows


def check(rows, samples, truth):
    if set(Counter(r['observation_id'] for r in rows).values()) != {3+ET_FITS+REPEATS*len(QWEN+APIM)}:
        raise SystemExit('PREDICTIONS.csv needs 29 rows for each sample')
    scored = scored_answers(samples)
    for r in rows:
        s, name = samples[r['observation_id']], f"{r['observation_id']} {r['method']}"
        if (r['source'], r['group'], r['arm']) != (s['graph_id'], s['stratum'], s['arm']): raise SystemExit(f'{name}: wrong sample')
        if r['method'] in QWEN+APIM:
            values, reason = scored.pop((r['observation_id'], r['method'], int(r['repeat_index'])))
            if (r['prediction'], r['valid'], r['validation_reason']) != (values, values is not None, reason):
                raise SystemExit(f'{name}: the raw answer does not give this row')
        if r['method'] == 'et' and valid_profile(r['prediction']) != r['prediction']: raise SystemExit(f'{name}: not a valid profile')
        if r['valid'] != (r['prediction'] is not None) or {k: r[k] for k in SCORES} != errors(r['prediction'], truth[r['source']]):
            raise SystemExit(f'{name}: stored error differs from the recomputed one')
    if scored: raise SystemExit('answers without a row in PREDICTIONS.csv')


def main():
    samples, truth, rows = frozen_inputs(), read_json(FINAL/'TRUTH.json'), read_predictions()
    check(rows, samples, truth)
    main_rows = [r for r in rows if r['replicate'] in (None, 0)]        # ExtraTrees: the reported fit only
    table, methods, families, networks = summary(main_rows), paired_methods(main_rows), paired_families(main_rows), per_source(main_rows)
    write_csv(FINAL/'SUMMARY.csv', table); write_csv(FINAL/'PAIRED_METHODS.csv', methods)
    write_csv(FINAL/'PAIRED_FAMILIES.csv', families); write_csv(FINAL/'PER_SOURCE.csv', networks)
    design = s_design_appendix(samples.values(), main_rows)
    write_csv(FINAL/'S_DESIGN_APPENDIX.csv', design)
    (FINAL/'MAIN_RESULTS.md').write_text(main_markdown(table, methods, families, networks, design))
    training, sampling, response = variability(rows, ET_FITS-1)
    write_csv(FINAL/'VARIABILITY_TRAINING.csv', training); write_csv(FINAL/'VARIABILITY_SAMPLING.csv', sampling)
    write_csv(FINAL/'VARIABILITY_RESPONSE.csv', response)
    (FINAL/'VARIABILITY.md').write_text(variability_markdown(training, sampling, response, ET_FITS-1, fixed_input_variability(rows)))
    losses, errs = history_summary(FINAL/'HISTORY.csv')
    write_csv(FINAL/'HISTORY_LOSSES.csv', losses); write_csv(FINAL/'HISTORY_ERRORS.csv', errs)
    (FINAL/'HISTORY.md').write_text(history_markdown(losses, errs))
    (FINAL/'WALK.md').write_text(walk_markdown(list(csv.DictReader((FINAL/'WALK.csv').open()))))
    print(f'{len(rows):,} predictions checked, tables written')


if __name__ == '__main__':
    main()
