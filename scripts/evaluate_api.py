#!/usr/bin/env python3
"""Evaluate the paid API answers (DeepSeek Flash, GPT-6 Sol, GPT-6 Sol + Python).

Joint evaluation over the 12 real sources, 12 surrogates and 8 synthetic graphs
(arms R/S/H/B), each block reported separately as equal-source means; originals
and surrogates are compared as pairs. API answers come from the v11 run directories
and, after the API extension, from the ext run directories. Offline methods
(plugin, median, MLE, ExtraTrees) and Qwen are read from the final PREDICTIONS.csv.
A method whose block is incomplete is reported as pending, never ranked. Accuracy is conditional on
valid final answers; nothing is clipped, repaired or imputed.
"""
import argparse
import csv
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))
from main_experiment.common import write_csv, write_json  # noqa: E402
from main_experiment.evaluation import parse_final  # noqa: E402
from scripts.api_runner import (API_MAIN_ARMS, actual_usd, manifest, observations,  # noqa: E402
                                percentile, read_jsonl, usage_tokens)
from scripts.build_v10_results import draw_mcse, errors, mean_by_source  # noqa: E402

API_METHODS = {'deepseek': 'deepseek_flash', 'openai': 'gpt_6_sol', 'openai_tools': 'gpt_6_sol_tools'}
OFFLINE = ('plugin', 'median', 'mle', 'et', 'qwen_thinking', 'qwen_nonthinking')
METHODS = OFFLINE + tuple(API_METHODS.values())
REF_METHOD = {'R': 'plugin', 'S': 'mle', 'H': 'mle', 'B': 'mle'}
GROUPS = ('real', 'surrogate', 'synthetic')


def truths(path):
    out = json.loads(Path(path).read_text())
    if len(out) != 32:
        raise ValueError('truth needs all 32 graphs')
    return out


def committed_rows(path, truth):
    rows = []
    for r in csv.DictReader(open(path)):
        if r['arm'] not in API_MAIN_ARMS or r['method'] not in OFFLINE or r.get('replicate') not in (None, '', '0'):
            continue
        stratum = r.get('stratum') or r['group']
        rid = r.get('id') or f"{r['observation_id']}__{r['method']}__r{r['repeat_index']}"
        prediction = json.loads(r['prediction']) if r['prediction'] else None
        valid = r['valid'] == 'True'
        values = prediction if valid else None
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


def api_rows(label, run_dir, planned, truth):
    records = {r['id']: r for r in read_jsonl(Path(run_dir) / 'responses.jsonl')
               if r.get('kind', 'main') == 'main'}
    rows = []
    for plan in planned:
        record = records.get(plan['id'])
        values, reason, status = None, 'not_completed', 'missing'
        if record is not None:
            status = 'completed'
            if record.get('limit_hit'):
                reason = 'generation_limit'
            else:
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
                    'skill_vs_plugin': 1 - mae / plugin_mae if mae is not None and plugin_mae > 0 else None})
    return table


def signflip_exact(values):
    """Two-sided exact sign-flip p value of the mean, enumerated in two halves."""
    v = np.asarray(values, float)
    half = len(v) // 2
    def sums(part):
        signs = np.array(np.meshgrid(*[[-1., 1.]] * len(part), indexing='ij')).reshape(len(part), -1).T
        return signs @ part if len(part) else np.zeros(1)
    total = sums(v[:half])[:, None] + sums(v[half:])[None, :]
    return float(np.mean(np.abs(total) >= abs(v.sum()) - 1e-12))


def paired(rows):
    """Source-level MAE_2 difference (first minus second); each method uses all of
    its valid answers for a source."""
    out = []
    for group in GROUPS:
        for arm in API_MAIN_ARMS:
            chosen = [r for r in rows if in_group(r, group) and r['arm'] == arm]
            means = {m: mean_by_source([r for r in chosen if r['method'] == m], 'AE2') for m in METHODS}
            comparisons = [(api, other) for api in API_METHODS.values()
                           for other in dict.fromkeys(('plugin', REF_METHOD[arm], 'et', 'qwen_thinking'))]
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


def markdown(table, pairs, fams, runs, out):
    lines = ['# API results', '',
             'DeepSeek Flash (reasoning high, off-peak, one repeat), GPT-6 Sol (reasoning high, Batch, three',
             'repeats) and GPT-6 Sol with the hosted Python tool (`gpt_6_sol_tools`, otherwise identical) on the',
             'frozen R/S/H/B observations. Offline methods (plugin, median, MLE, ExtraTrees; reference plugin for R',
             'and the MLE otherwise) and Qwen are the final predictions. MAE_2 is the equal-source mean over valid',
             'answers within each block; a block missing any source is **pending** and not ranked. Nothing is',
             'clipped, repaired or imputed. Recorded spend is an upper bound on the provider bill.', '']
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


def spec(values):
    """SET=PATH pairs, e.g. v11=/runs/openai_v11 ext=/runs/openai_ext."""
    return [tuple(v.split('=', 1)) for v in values or []]


def main():
    from scripts import api_runner
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--observations', nargs='+', required=True, help='SET=DIR frozen API observations (v11, ext)')
    ap.add_argument('--deepseek', nargs='*', help='SET=RUN_DIR (one repeat)')
    ap.add_argument('--openai', nargs='*', help='SET=RUN_DIR (three repeats)')
    ap.add_argument('--openai-tools', nargs='*', help='SET=RUN_DIR (three repeats)')
    ap.add_argument('--truth', type=Path, default=ROOT / 'docs/results/final_20260928/TRUTH.json')
    ap.add_argument('--predictions', type=Path, default=ROOT / 'docs/results/final_20260928/PREDICTIONS.csv')
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    truth = truths(a.truth)
    obs = {}
    for name, directory in spec(a.observations):
        api_runner.use_observation_set(name)
        obs[name] = observations(Path(directory))
    rows = committed_rows(a.predictions, truth)
    runs = {}
    for label, values, repeats in (('deepseek', a.deepseek, 1), ('openai', a.openai, 3), ('openai_tools', a.openai_tools, 3)):
        for name, run_dir in spec(values):
            api_runner.use_observation_set(name)
            rows += api_rows(label, run_dir, manifest(obs[name], label, repeats), truth)
            runs[f'{API_METHODS[label]}:{name}'] = run_report('deepseek' if label == 'deepseek' else 'openai', run_dir)
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
