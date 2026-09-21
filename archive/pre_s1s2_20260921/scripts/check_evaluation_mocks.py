#!/usr/bin/env python3
"""Offline check of the evaluation path with explicitly marked fake responses.

Five mock answers exercise invalid text, a technical error, an output-limit hit,
a never-started request and an in-progress request. The check also requires that
mock records are refused as production answers and that a duplicate response ID
is refused. No model is called.
"""
import csv
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from main_experiment.common import AUDIT, PREPARED, fresh_directory, read_json, read_jsonl, write_json
from evaluate_responses import evaluate


def mock_records(requests):
    records = []
    for i, r in enumerate(requests):
        record = {'id': r['id'], 'mock': True, 'started': True, 'terminal': True,
                  'final_text': '{"rho_2":0.4,"rho_3":0.3,"rho_4":0.2,"rho_5":0.1}',
                  'reasoning': 'MOCK reasoning includes {"rho_2":999}; it must never be parsed.',
                  'prompt_sha256': r['prompt_sha256'], 'payload_sha256': r['payload_sha256'],
                  'finish_reason': 'mock_stop', 'limit_hit': i == 2, 'technical_error': False}
        if i == 0: record['final_text'] = 'MOCK invalid answer'
        if i == 1: record['final_text'] = ''; record['technical_error'] = True
        if i == 3: record['started'] = False; record['terminal'] = False
        if i == 4: record['terminal'] = False
        records.append(record)
    return records


def must_fail(function, *args):
    try: function(*args)
    except ValueError: return True
    raise AssertionError(f'{function.__name__} accepted invalid input')


def main():
    out = fresh_directory(AUDIT/'mock_evaluation')
    requests = read_jsonl(PREPARED/'requests.jsonl')
    records = mock_records(requests)
    source = out/'mock_responses.jsonl'
    source.write_text(''.join(json.dumps(r)+'\n' for r in records))
    evaluate(source, out/'evaluation', True)
    rows = list(csv.DictReader(open(out/'evaluation/answer_errors.csv')))
    assert len(rows) == len(requests) == read_json(PREPARED/'report.json')['planned_logical_calls']
    assert rows[0]['prediction_json'] == '' and rows[1]['prediction_json'] == ''   # invalid / technical: no estimate
    assert rows[2]['valid'] == 'True' and rows[2]['limit_hit'] == 'True'
    assert rows[3]['status'] == 'not_started' and rows[3]['AE2'] == ''
    assert rows[4]['status'] == 'in_progress' and rows[4]['AE2'] == ''
    assert read_json(out/'evaluation/report.json')['mock'] and not read_json(out/'evaluation/report.json')['complete_main_result']
    must_fail(evaluate, source, out/'mock_as_real', False)                     # mock never counts as production
    duplicate = out/'duplicate_mock.jsonl'
    duplicate.write_text(json.dumps(records[0])+'\n'+json.dumps(records[0])+'\n')
    must_fail(evaluate, duplicate, out/'duplicate_mock', True)                 # one answer per request
    write_json(out/'check_report.json', {'mock_only': True, 'inference_calls': 0, 'rows_checked': len(rows),
                                         'missing_predictions_and_pending_states_checked': True,
                                         'mock_as_real_rejected': True, 'duplicate_id_rejected': True})
    print(json.dumps(read_json(out/'check_report.json'), indent=1))


if __name__ == '__main__':
    main()
