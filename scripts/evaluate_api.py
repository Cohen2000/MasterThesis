#!/usr/bin/env python3
"""Evaluate the paid API answers (DeepSeek Flash, GPT-6 Sol) against the v11 panel.

Reports each group of eight graphs (real, surrogate, synthetic) and all 24
graphs as equal-source means. Offline methods and Qwen are read unchanged from
the committed v11 PREDICTIONS.csv (arms R/S/H/B). Accuracy is conditional on
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
OFFLINE = ('plugin', 'median', 'design', 'mle', 'et', 'qwen_thinking', 'qwen_nonthinking')
METHODS = OFFLINE + tuple(API_METHODS.values())
REF_METHOD = {'R': 'plugin', 'S': 'design', 'H': 'mle', 'B': 'mle'}
GROUPS = ('real', 'surrogate', 'synthetic', 'all24')


def truths(directory):
    out = {}
    for path in sorted(Path(directory).glob('*.json')):
        row = json.loads(path.read_text())
        if out.setdefault(row['graph_id'], row['truth']) != row['truth']:
            raise ValueError(f"inconsistent truth for {row['graph_id']}")
    if len(out) != 24:
        raise ValueError('truth needs all 24 main graphs')
    return out


def committed_rows(path, truth):
    rows = []
    for r in csv.DictReader(open(path)):
        if r['arm'] not in API_MAIN_ARMS:
            continue
        prediction = json.loads(r['prediction']) if r['prediction'] else None
        valid = r['valid'] == 'True'
        values = prediction if valid else None
        check = errors(values, truth[r['source']])
        stored = float(r['AE2']) if r['AE2'] else None
        if (stored is None) != (check['AE2'] is None) or (stored is not None and abs(stored - check['AE2']) > 1e-9):
            raise ValueError(f"truth mismatch with committed predictions: {r['id']} {r['method']}")
        rows.append({'id': r['id'], 'observation_id': r['observation_id'], 'source': r['source'],
                     'stratum': r['stratum'], 'arm': r['arm'], 'sample_index': int(r['sample_index']),
                     'repeat_index': int(r['repeat_index']) if r['repeat_index'] else None,
                     'method': r['method'], 'prediction': values, 'valid': valid,
                     'status': 'completed', 'validation_reason': r['validation_reason'], **check})
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
    return group == 'all24' or row['stratum'] == group


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
                sources = sorted({r['source'] for r in cell})
                ae = mean_by_source(cell, 'AE2')
                complete = len(ae) == len(sources)
                mae = float(np.mean(list(ae.values()))) if complete else None
                table.append({
                    'group': group, 'arm': arm, 'method': method, 'sources': len(sources),
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
                expected = 24 if group == 'all24' else 8
                row = {'group': group, 'arm': arm, 'comparison': f'{first} vs {second}',
                       'sources_with_pairs': len(sources)}
                if len(sources) == expected:
                    d = np.array([means[first][s] - means[second][s] for s in sources])
                    row.update({'mean_difference': float(d.mean()), 'first_better_sources': int((d < 0).sum()),
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


def markdown(table, pairs, runs, out):
    lines = ['# API results (v11)', '',
             'DeepSeek Flash (reasoning high, off-peak, one repeat), GPT-6 Sol (reasoning high, Batch, three',
             'repeats) and GPT-6 Sol with the hosted Python tool (`gpt_6_sol_tools`, otherwise identical) on the',
             '288 frozen R/S/H/B observations. Offline methods and Qwen (three repeats) are the committed',
             'v11 predictions. MAE_2 is the equal-source mean over valid answers; `all24` weights all 24 graphs',
             'equally. Nothing is clipped, repaired or imputed.', '',
             'Recorded spend charges all DeepSeek input at the cache-miss price and all GPT input at the',
             'cache-write price, so it is an upper bound on the provider bill.', '']
    for provider, report in runs.items():
        main = report.get('main', {})
        tools = f", answers using the tool {main.get('answers_using_tools', 0)}" if provider.endswith('tools') else ''
        lines.append(f"- {provider}: {main.get('responses', 0)} main answers, recorded spend "
                     f"USD {report['total_recorded_spend_usd']:.4f} (incl. smoke/pilot), generation-limit hits "
                     f"{main.get('generation_limit_hits', 0)}, mean output tokens "
                     f"{fmt(main.get('output_tokens', {}).get('mean'), 0)}{tools}.")
    for group in GROUPS:
        lines += ['', f'## {group}', '',
                  '| Arm | Method | MAE_2 | ProfileMAE | Signed rho_2 | Valid/planned | Sources valid | MCSE_2 | Skill vs plugin |',
                  '|---|---|---:|---:|---:|---:|---:|---:|---:|']
        for r in table:
            if r['group'] != group:
                continue
            lines.append(f"| {r['arm']} | {r['method']} | {fmt(r['MAE_2'])} | {fmt(r['ProfileMAE'])} | "
                         f"{fmt(r['signed_rho_2'])} | {r['valid_answers']}/{r['planned_answers']} | "
                         f"{r['sources_with_valid']}/{r['sources']} | {fmt(r['draw_clustered_MCSE_2'])} | "
                         f"{fmt(r['skill_vs_plugin'])} |")
    lines += ['', '## Source-level paired differences', '',
              'First minus second source-level MAE_2; exact two-sided sign-flip over sources (8 or 24).', '',
              '| Group | Arm | Comparison | Mean difference | First better | Exact sign-flip p |',
              '|---|---|---|---:|---:|---:|']
    for r in pairs:
        n = 24 if r['group'] == 'all24' else 8
        lines.append(f"| {r['group']} | {r['arm']} | {r['comparison']} | {fmt(r.get('mean_difference'))} | "
                     f"{fmt(r.get('first_better_sources'))}/{n} | {fmt(r.get('exact_signflip_p'))} |")
    (out / 'API_RESULTS.md').write_text('\n'.join(lines) + '\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--observations', type=Path, required=True, help='288 frozen API observations')
    ap.add_argument('--truth-observations', type=Path, required=True, help='360 v10 observations with truth')
    ap.add_argument('--deepseek', type=Path, required=True)
    ap.add_argument('--deepseek-repeats', type=int, default=1)
    ap.add_argument('--openai', type=Path, required=True)
    ap.add_argument('--openai-repeats', type=int, default=3)
    ap.add_argument('--openai-tools', type=Path, help='GPT run with the hosted Python tool')
    ap.add_argument('--openai-tools-repeats', type=int, default=3)
    ap.add_argument('--predictions', type=Path,
                    default=ROOT / 'docs/results/panel888_v11_main_20260923/PREDICTIONS.csv')
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    truth = truths(a.truth_observations)
    obs = observations(a.observations)
    rows = committed_rows(a.predictions, truth)
    runs_in = [('deepseek', a.deepseek, a.deepseek_repeats), ('openai', a.openai, a.openai_repeats)]
    if a.openai_tools:
        runs_in.append(('openai_tools', a.openai_tools, a.openai_tools_repeats))
    for label, run_dir, repeats in runs_in:
        rows += api_rows(label, run_dir, manifest(obs, label, repeats), truth)
    a.out.mkdir(parents=True, exist_ok=True)
    table, pairs = summary(rows), paired(rows)
    runs = {API_METHODS[label]: run_report('deepseek' if label == 'deepseek' else 'openai', run_dir)
            for label, run_dir, _ in runs_in}
    write_csv(a.out / 'SUMMARY.csv', table)
    write_csv(a.out / 'PAIRED.csv', pairs)
    write_csv(a.out / 'PER_SOURCE.csv', per_source([r for r in rows if r['method'] in API_METHODS.values()]))
    write_csv(a.out / 'API_PREDICTIONS.csv', [r for r in rows if r['method'] in API_METHODS.values()])
    write_json(a.out / 'API_RUNS.json', runs)
    markdown(table, pairs, runs, a.out)
    print((a.out / 'API_RESULTS.md').read_text())


if __name__ == '__main__':
    main()
