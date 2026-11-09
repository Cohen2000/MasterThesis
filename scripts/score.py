#!/usr/bin/env python3
"""Score the raw answers of the language models: one row of data/predictions.csv per answer.

An answer counts only if its final text is exactly one JSON object with rho_2 ... rho_5 (src/study/answers.py);
an answer that hit the length limit or failed is invalid, and nothing is repaired. Reads the folders that
scripts/api.py, scripts/qwen.py and scripts/ablation_walk_log.py write:

  results/api/deepseek, results/api/openai, results/api/openai_tools   (api.py --out ...)
  results/qwen                                                       (qwen.py --out results/qwen)
  results/ablation_walk_log

  python scripts/score.py      writes results/llm_predictions.csv and compares it with data/predictions.csv

The raw answers themselves are not in the repository; only their scores are (data/predictions.csv).
"""
import csv
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from study.answers import parse_final  # noqa: E402
from study.common import DATA, OUT, REPEATS, read_json, write_csv  # noqa: E402

SAMPLER = {'R': 'node', 'S': 'walk', 'H': 'truncation', 'B': 'event'}
API = {'deepseek_v4_1_flash': ('deepseek', OUT/'api/deepseek'), 'gpt_6_sol': ('openai', OUT/'api/openai'),
       'gpt_6_sol_python': ('openai_tools', OUT/'api/openai_tools'),
       'gpt_6_sol_no_walk_log': ('openai_noweights', OUT/'ablation_walk_log')}
QWEN_VERSION = {'S': 'panel888-access-v10-20260923'}           # part of the Qwen answer names
QWEN_DEFAULT = 'panel888-access-v9-20260922'


def row(s, method, repeat, values):
    return {'sample_id': s['id'], 'network': s['graph_id'].removesuffix('__pwt'), 'shuffled': int(s['stratum'] == 'surrogate'),
            'sampler': SAMPLER[s['arm']], 'sample': s['sample_index'], 'method': method, 'answer': repeat,
            'valid': values is not None, **{f'rho_{k}': (repr(float(values[k-2])) if values else '') for k in range(2, 6)}}


def main():
    samples = [read_json(p) for p in sorted((DATA/'samples').glob('*.json'))]
    rows, missing = [], {}
    for method, (run, folder) in API.items():
        path = folder/'responses.jsonl'
        if not path.exists(): continue
        answers = {a['id']: a for a in map(json.loads, path.read_text().splitlines()) if a.get('kind', 'main') == 'main'}
        for s in samples:
            if method == 'gpt_6_sol_no_walk_log' and s['arm'] != 'S': continue
            for repeat in range(1, REPEATS+1):
                a = answers.get(f"{s['id']}__{run}__r{repeat}")
                if a is None:
                    missing[method] = missing.get(method, 0)+1
                    continue
                values = None if a.get('limit_hit') else parse_final(a.get('final_text') or '')[0]
                rows.append(row(s, method, repeat, values))
    files = {p.stem: p for p in (OUT/'qwen').glob('thinking_r*/*.json')}
    if files:
        for s in samples:
            for repeat in range(1, REPEATS+1):
                path = files.get(f"{s['id']}__qwen_thinking__r{repeat}__{QWEN_VERSION.get(s['arm'], QWEN_DEFAULT)}")
                if path is None:
                    missing['qwen3_6'] = missing.get('qwen3_6', 0)+1
                    continue
                a = read_json(path)
                failed = a.get('status') != 'completed' or a.get('technical_error')
                rows.append(row(s, 'qwen3_6', repeat, None if failed else parse_final(a.get('final_text', ''))[0]))
    if not rows: raise SystemExit('no answer folders found, see the usage above')
    write_csv(OUT/'llm_predictions.csv', rows)
    frozen = {(r['sample_id'], r['method'], r['answer']): r for r in csv.DictReader((DATA/'predictions.csv').open())}
    differ = [r for r in rows if {k: str(v) for k, v in r.items()} != frozen[r['sample_id'], r['method'], str(r['answer'])]]
    print(f'{len(rows)} answers scored, {len(differ)} differ from data/predictions.csv'
          + (f'; missing answers: {missing}' if missing else ''))
    if differ: raise SystemExit(1)


if __name__ == '__main__':
    main()
