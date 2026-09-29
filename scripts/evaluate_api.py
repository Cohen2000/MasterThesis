#!/usr/bin/env python3
"""Evaluate the paid API answers (DeepSeek Flash, GPT-6 Sol, GPT-6 Sol + Python).

Joint evaluation over the 12 real sources, 12 surrogates and 8 synthetic graphs
(arms R/S/H/B), each block reported separately as equal-source means; originals
and surrogates are compared as pairs. API answers come from one run directory per
model (three repeats per observation). Offline methods
(plugin, median, MLE, ExtraTrees) and Qwen are read from the final PREDICTIONS.csv.
A method whose block is incomplete is reported as pending, never ranked. LLM accuracy is conditional
on valid final answers; LLM answers are never clipped, repaired or imputed. ExtraTrees output
is limited to a valid profile (values in [0, 1], never increasing).
"""
# How this script works, in plain words:
# 1. Read the true persistence values (TRUTH.json) of all 32 graphs.
# 2. Read the committed predictions of the methods that need no API (plugin, median,
#    MLE, ExtraTrees, both Qwen modes) from PREDICTIONS.csv and re-check their errors.
# 3. Read the raw API answers from the run directories (one folder per model) and turn
#    every answer into a prediction, or mark it invalid.
# 4. Score everything the same way and write CSV/Markdown tables into --out.
# The script never calls an API and never changes an answer.
import argparse
import csv
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

import numpy as np

# Make the repository's own modules importable when the script is run directly.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))
from study.common import write_csv, write_json  # noqa: E402
from study.answer_format import parse_final  # noqa: E402
from scripts.api_runner import (API_MAIN_ARMS, actual_usd, manifest, observations,  # noqa: E402
                                percentile, read_jsonl, usage_tokens)
from scripts.score_stage1 import draw_mcse, errors, mean_by_source  # noqa: E402

# Provider label used by the runner -> method name used in all result tables.
API_METHODS = {'deepseek': 'deepseek_flash', 'openai': 'gpt_6_sol', 'openai_tools': 'gpt_6_sol_tools'}
# Methods whose predictions are already in PREDICTIONS.csv (Qwen ran on the cluster, not via API).
OFFLINE = ('plugin', 'median', 'mle', 'et', 'qwen_thinking', 'qwen_nonthinking')
METHODS = OFFLINE + tuple(API_METHODS.values())
# Reference estimator per sampling arm: the plain count (plugin) under R, where every pair is
# equally likely to be observed; the maximum-likelihood estimator (MLE) elsewhere.
REFERENCE = {'R': 'plugin', 'S': 'mle', 'H': 'mle', 'B': 'mle'}
# The three graph blocks. They are always scored separately, never mixed.
GROUPS = ('real', 'surrogate', 'synthetic')


# True rho_2..rho_5 per graph, computed from the complete (unsampled) graph.
def truths(path):
    out = json.loads(Path(path).read_text())
    if len(out) != 32:
        raise ValueError('truth needs all 32 graphs')
    return out


# Load the non-API predictions and recompute each error from the truth. If a stored
# error differs from the recomputed one, something is inconsistent and we stop.
# Only ExtraTrees fit 0 (the production fit) is used here; the other 10 fits only
# serve the variability analysis.
def committed_rows(path, truth):
    rows = []
    for r in csv.DictReader(open(path)):
        if r['arm'] not in API_MAIN_ARMS or r['method'] not in OFFLINE or r.get('replicate') not in (None, '', '0'):
            continue
        stratum = r.get('stratum') or r['group']
        rid = r.get('id') or f"{r['observation_id']}__{r['method']}__r{r['repeat_index']}"
        prediction = json.loads(r['prediction']) if r['prediction'] else None
        valid = r['valid'] == 'True'
        # Invalid LLM answers get no score; ExtraTrees is always scored on its raw output.
        values = prediction if valid or r['method'] == 'et' else None
        check = errors(values, truth[r['source']])
        stored = float(r['AE2']) if r['AE2'] else None
        if (stored is None) != (check['AE2'] is None) or (stored is not None and abs(stored - check['AE2']) > 1e-9):
            raise ValueError(f"truth mismatch with committed predictions: {rid}")
        rows.append({'id': rid, 'observation_id': r['observation_id'], 'source': r['source'],
                     'stratum': stratum, 'arm': r['arm'], 'sample_index': int(r['sample_index']),
                     'repeat_index': int(r['repeat_index']) if r['repeat_index'] else None,
                     'method': r['method'], 'prediction': values, 'valid': valid,
                     'status': 'completed', 'validation_reason': r.get('validation_reason') or '', **check})
    return rows


# Turn the raw API answers of one run directory into scored rows. 'planned' is the
# full list of request IDs (observation x repeat), so a missing answer shows up as
# a row with status 'missing' instead of silently disappearing.
def api_rows(label, run_dir, planned, truth):
    records = {r['id']: r for r in read_jsonl(Path(run_dir) / 'responses.jsonl')
               if r.get('kind', 'main') == 'main'}
    rows = []
    for plan in planned:
        record = records.get(plan['id'])
        values, reason, status = None, 'not_completed', 'missing'
        if record is not None:
            status = 'completed'
            # The model hit its output-token limit: the answer counts as invalid.
            if record.get('limit_hit'):
                reason = 'generation_limit'
            else:
                # Same strict parser for every LLM: exactly the keys rho_2..rho_5, finite numbers in [0, 1],
                # non-increasing. Anything else is invalid; nothing is repaired or clipped.
                values, reason = parse_final(record.get('final_text') or '')
        rows.append({'id': plan['id'], 'observation_id': plan['observation_id'],
                     'source': plan['graph_id'], 'stratum': plan['stratum'], 'arm': plan['arm'],
                     'sample_index': plan['sample_index'], 'repeat_index': plan['repeat_index'],
                     'method': API_METHODS[label], 'prediction': values,
                     'valid': values is not None, 'status': status, 'validation_reason': reason,
                     **errors(values, truth[plan['graph_id']])})
    return rows


def in_group(row, group):
    return row['stratum'] == group


# Main table: one row per block x arm x method.
# MAE_2 = mean absolute error of rho_2, first averaged within each source (graph),
# then averaged over sources, so every source counts equally regardless of how
# many answers it has. A method missing a whole source is 'pending', not ranked.
def summary(rows):
    table = []
    for group in GROUPS:
        for arm in API_MAIN_ARMS:
            chosen = [r for r in rows if in_group(r, group) and r['arm'] == arm]
            plugin = mean_by_source([r for r in chosen if r['method'] == 'plugin'], 'AE2')
            plugin_mae = float(np.mean(list(plugin.values())))
            for method in METHODS:
                cell = [r for r in chosen if r['method'] == method]
                if not cell:
                    continue
                ae = mean_by_source(cell, 'AE2')
                block = {r['source'] for r in rows if in_group(r, group) and r['method'] == 'plugin'}
                complete = set(ae) == block
                mae = float(np.mean(list(ae.values()))) if complete else None
                table.append({
                    'group': group, 'arm': arm, 'method': method, 'sources': len(block),
                    'status': 'complete' if complete else 'pending',
                    'sources_with_valid': len(ae), 'planned_answers': len(cell),
                    'completed_answers': sum(r['status'] == 'completed' for r in cell),
                    'valid_answers': sum(r['valid'] for r in cell),
                    'validity_of_completed': (sum(r['valid'] for r in cell) /
                                              max(1, sum(r['status'] == 'completed' for r in cell))),
                    'MAE_2': mae,
                    'ProfileMAE': float(np.mean(list(mean_by_source(cell, 'ProfileAE').values()))) if complete else None,
                    'signed_rho_2': float(np.mean(list(mean_by_source(cell, 'signed_rho2').values()))) if complete else None,
                    'draw_clustered_MCSE_2': draw_mcse(cell, 'AE2') if complete else None,
                    # Share of the plugin's error that the method removes (1 = perfect, 0 = no better, <0 = worse).
                    'skill_vs_plugin': 1 - mae / plugin_mae if mae is not None and plugin_mae > 0 else None})
    return table


# Exact paired test: flip the sign of every per-source difference in all possible
# ways and count how often the mean is at least as extreme as the observed one.
# Splitting the vector in two halves keeps this fast for 12 sources (4096 patterns).
def signflip_exact(values):
    """Two-sided exact sign-flip p value of the mean, enumerated in two halves."""
    v = np.asarray(values, float)
    half = len(v) // 2
    def sums(part):
        signs = np.array(np.meshgrid(*[[-1., 1.]] * len(part), indexing='ij')).reshape(len(part), -1).T
        return signs @ part if len(part) else np.zeros(1)
    total = sums(v[:half])[:, None] + sums(v[half:])[None, :]
    return float(np.mean(np.abs(total) >= abs(v.sum()) - 1e-12))


# Head-to-head comparisons on the same sources, e.g. GPT-6 Sol vs MLE under arm S:
# per source MAE difference, number of sources where the first method wins, exact p.
def paired(rows):
    """Source-level MAE_2 difference (first minus second); each method uses all of
    its valid answers for a source."""
    out = []
    for group in ('real', 'surrogate'):
        for arm in API_MAIN_ARMS:
            chosen = [r for r in rows if in_group(r, group) and r['arm'] == arm]
            means = {m: mean_by_source([r for r in chosen if r['method'] == m], 'AE2') for m in METHODS}
            comparisons = [(api, other) for api in API_METHODS.values()
                           for other in dict.fromkeys(('plugin', REFERENCE[arm], 'et', 'qwen_thinking'))]
            comparisons += [('deepseek_flash', 'gpt_6_sol'), ('gpt_6_sol_tools', 'gpt_6_sol')]
            for first, second in comparisons:
                if not means[first] or not means[second]:
                    continue
                sources = sorted(set(means[first]) & set(means[second]))
                expected = len(means['plugin'])
                row = {'group': group, 'arm': arm, 'comparison': f'{first} vs {second}',
                       'sources_with_pairs': len(sources)}
                if len(sources) == expected:
                    d = np.array([means[first][s] - means[second][s] for s in sources])
                    row.update({'sources': expected, 'mean_difference': float(d.mean()), 'first_better_sources': int((d < 0).sum()),
                                'exact_signflip_p': signflip_exact(d)})
                out.append(row)
    return out


# MAE per single source (graph) for the API methods; used for the per-graph tables.
def per_source(rows):
    out = []
    grouped = defaultdict(list)
    for r in rows:
        grouped[r['source'], r['stratum'], r['arm'], r['method']].append(r)
    for (source, stratum, arm, method), cell in sorted(grouped.items()):
        valid = [r for r in cell if r['AE2'] is not None]
        out.append({'source': source, 'stratum': stratum, 'arm': arm, 'method': method,
                    'answers': len(cell), 'valid': len(valid),
                    'MAE_2': float(np.mean([r['AE2'] for r in valid])) if valid else None,
                    'ProfileMAE': float(np.mean([r['ProfileAE'] for r in valid])) if valid else None,
                    'signed_rho_2': float(np.mean([r['signed_rho2'] for r in valid])) if valid else None})
    return out


# Bookkeeping for one run directory: answer counts, returned model names, validity
# reasons, token usage and recorded spend (an upper bound on the provider bill).
def run_report(provider, run_dir):
    records = read_jsonl(Path(run_dir) / ('technical_responses.jsonl' if provider == 'openai' else 'responses.jsonl'))
    if provider == 'openai':
        records += read_jsonl(Path(run_dir) / 'responses.jsonl')
    report = {}
    for kind in ('smoke', 'pilot', 'main'):
        chosen = [r for r in records if r.get('kind', 'main') == kind]
        if not chosen:
            continue
        output = [usage_tokens(r, provider)[1] for r in chosen]
        reasoning = [r['reasoning_tokens'] for r in chosen if isinstance(r.get('reasoning_tokens'), int)]
        tools = [r.get('tool_calls') or 0 for r in chosen]
        report[kind] = {
            'responses': len(chosen), 'returned_models': dict(Counter(r.get('returned_model') for r in chosen)),
            'validity_reasons': dict(Counter((r.get('validity') or '').split(':')[0] for r in chosen)),
            'generation_limit_hits': sum(bool(r.get('limit_hit')) for r in chosen),
            'output_tokens': {'mean': float(np.mean(output)), 'median': float(np.median(output)),
                              'p95': percentile(output, .95), 'max': max(output)},
            'reasoning_tokens_mean': float(np.mean(reasoning)) if reasoning else None,
            'tool_calls_mean': float(np.mean(tools)), 'answers_using_tools': sum(t > 0 for t in tools),
            'recorded_spend_usd': round(sum(actual_usd(r, provider) for r in chosen), 4)}
    report['total_recorded_spend_usd'] = round(sum(actual_usd(r, provider) for r in records), 4)
    failures = read_jsonl(Path(run_dir) / 'batch_failures.jsonl')
    if failures:
        report['unbilled_batch_failures'] = len(failures)
    return report


def fmt(value, digits=4):
    return '' if value is None else f'{value:.{digits}f}' if isinstance(value, float) else str(value)


# Real graph vs its own surrogate (same source family): does a method behave
# differently on the original than on the synthetic stand-in built from it?
def families(rows):
    """Original minus surrogate source-level MAE_2 per method, one pair per source family."""
    out = []
    for arm in API_MAIN_ARMS:
        for method in METHODS:
            real = mean_by_source([r for r in rows if r['stratum'] == 'real' and r['arm'] == arm and r['method'] == method], 'AE2')
            sur = mean_by_source([r for r in rows if r['stratum'] == 'surrogate' and r['arm'] == arm and r['method'] == method], 'AE2')
            fams = sorted(f for f in real if f + '__pwt' in sur)
            if not fams:
                continue
            d = np.array([real[f] - sur[f + '__pwt'] for f in fams])
            out.append({'arm': arm, 'method': method, 'families': len(fams), 'mean_difference': float(d.mean()),
                        'original_better': int((d < 0).sum()), 'exact_signflip_p': signflip_exact(d)})
    return out


# Human-readable version of the tables above (API_RESULTS.md).
def markdown(table, pairs, fams, runs, out):
    lines = ['# API results', '',
             'DeepSeek Flash (reasoning high, off-peak), GPT-6 Sol (reasoning high, Batch) and GPT-6 Sol',
             'with the hosted Python tool (`gpt_6_sol_tools`, otherwise identical), three repeats each, on the',
             'frozen R/S/H/B observations. Offline methods (plugin, median, MLE, ExtraTrees; reference plugin for R',
             'and the MLE otherwise) and Qwen are the final predictions. MAE_2 is the equal-source mean over valid',
             'LLM answers and all ExtraTrees profiles; a block missing any source is **pending** and not ranked. LLM',
             'answers are never clipped, repaired or imputed. Recorded spend comes from the providers\' token counts.', '']
    for provider, report in runs.items():
        main = report.get('main', {})
        lines.append(f"- {provider}: {main.get('responses', 0)} main answers, recorded spend "
                     f"USD {report['total_recorded_spend_usd']:.4f} (incl. smoke/pilot), generation-limit hits "
                     f"{main.get('generation_limit_hits', 0)}.")
    for group in GROUPS:
        lines += ['', f'## {group}', '',
                  '| Arm | Method | MAE_2 | ProfileMAE | Signed rho_2 | Valid/planned | Sources | MCSE_2 |',
                  '|---|---|---:|---:|---:|---:|---:|---:|']
        for r in table:
            if r['group'] == group:
                lines.append(f"| {r['arm']} | {r['method']} | {fmt(r['MAE_2'])} | {fmt(r['ProfileMAE'])} | "
                             f"{fmt(r['signed_rho_2'])} | {r['valid_answers']}/{r['planned_answers']} | "
                             f"{r['sources_with_valid']}/{r['sources']} {'' if r['status'] == 'complete' else '(pending)'} | "
                             f"{fmt(r['draw_clustered_MCSE_2'])} |")
        if group == 'synthetic':
            lines += ['', 'The eight synthetic graphs form four generator pairs with shared random numbers;',
                      'these MAE results are descriptive (minimum two-sided sign-flip p with four blocks: 0.125).']
    lines += ['', '## Paired original minus surrogate', '',
              '| Arm | Method | Families | Mean difference | Original better | Exact sign-flip p |', '|---|---|---:|---:|---:|---:|']
    for r in fams:
        lines.append(f"| {r['arm']} | {r['method']} | {r['families']} | {fmt(r['mean_difference'])} | "
                     f"{r['original_better']}/{r['families']} | {fmt(r['exact_signflip_p'])} |")
    lines += ['', '## Source-level paired differences within a block', '',
              '| Group | Arm | Comparison | Mean difference | First better | Exact sign-flip p |', '|---|---|---|---:|---:|---:|']
    for r in pairs:
        if 'mean_difference' in r:
            lines.append(f"| {r['group']} | {r['arm']} | {r['comparison']} | {fmt(r['mean_difference'])} | "
                         f"{r['first_better_sources']}/{r['sources']} | {fmt(r['exact_signflip_p'])} |")
    (out / 'API_RESULTS.md').write_text('\n'.join(lines) + '\n')


# Glue: parse arguments, load everything, score, write all output files.
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--observations', type=Path, required=True, help='frozen API observations (384 files)')
    ap.add_argument('--deepseek', type=Path, help='DeepSeek run directory')
    ap.add_argument('--openai', type=Path, help='GPT-6 Sol run directory')
    ap.add_argument('--openai-tools', type=Path, help='GPT-6 Sol + Python run directory')
    ap.add_argument('--truth', type=Path, default=ROOT / 'docs/results/final/TRUTH.json')
    ap.add_argument('--predictions', type=Path, default=ROOT / 'docs/results/final/PREDICTIONS.csv')
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    truth = truths(a.truth)
    obs = observations(a.observations)
    rows = committed_rows(a.predictions, truth)
    runs = {}
    # Every LLM is planned with three repeats per observation.
    for label, run_dir in (('deepseek', a.deepseek), ('openai', a.openai), ('openai_tools', a.openai_tools)):
        if run_dir:
            rows += api_rows(label, run_dir, manifest(obs, label, 3), truth)
            runs[API_METHODS[label]] = run_report('deepseek' if label == 'deepseek' else 'openai', run_dir)
    a.out.mkdir(parents=True, exist_ok=True)
    table, pairs, fams = summary(rows), paired(rows), families(rows)
    write_csv(a.out / 'SUMMARY.csv', table)
    write_csv(a.out / 'PAIRED.csv', pairs)
    write_csv(a.out / 'PAIRED_FAMILIES.csv', fams)
    write_csv(a.out / 'PER_SOURCE.csv', per_source([r for r in rows if r['method'] in API_METHODS.values()]))
    write_csv(a.out / 'API_PREDICTIONS.csv', [r for r in rows if r['method'] in API_METHODS.values()])
    write_json(a.out / 'API_RUNS.json', runs)
    markdown(table, pairs, fams, runs, a.out)
    print((a.out / 'API_RESULTS.md').read_text())

if __name__ == '__main__':
    main()
