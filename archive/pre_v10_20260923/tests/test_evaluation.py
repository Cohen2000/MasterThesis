"""Strict answer parsing, conditional accuracy with hierarchical MCSE, paired controls."""
import csv
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
from main_experiment.common import DESIGN_VERSION, PREPARED, REFERENCES, ROOT, read_json, write_json
from main_experiment.evaluation import complete_summary, conditional_summary, errors, parse_final, resolve, strip_fence
sys.path.insert(0, str(ROOT/'scripts'))
from audit_qwen import strict                                  # noqa: E402  independent second parser
from evaluate_paired_controls import delta_metrics             # noqa: E402

GOOD = '{"rho_2": 0.5, "rho_3": 0.2, "rho_4": 0.1, "rho_5": 0}'
INVALID = ['{}', '[]', 'prose\n```json\n'+GOOD+'\n```', GOOD+' {}', '{"rho_2":.2}',
           GOOD.replace('0.5', 'true'), GOOD.replace('0.5', '"0.5"'), GOOD.replace('0.5', 'null'),
           GOOD.replace('0.5', 'NaN'), GOOD.replace('0.5', 'Infinity'), GOOD.replace('0.5', '1.1'),
           GOOD.replace('0.5', '-0.1'), GOOD.replace('0.5', '0.01'), GOOD.replace('"rho_5": 0', '"rho_5": 0, "rho_5": 0'),
           GOOD[:-3], 'reasoning '+GOOD]


class ParserTests(unittest.TestCase):
    def test_valid_bare_and_single_fence(self):
        self.assertEqual(parse_final(' '+GOOD+'\n'), ([.5, .2, .1, 0], 'valid'))
        for wrapper in ('```json\n%s\n```', '```\n%s\n```'):
            self.assertEqual(parse_final(wrapper % GOOD), ([.5, .2, .1, 0], 'valid_after_fence'))
        text, fenced = strip_fence('```json\n```json\n%s\n```\n```' % GOOD)
        self.assertTrue(fenced and '```json' in text)               # only one enclosing fence is removed

    def test_invalid_answers_are_never_repaired(self):
        for text in INVALID:
            self.assertIsNone(parse_final(text)[0], text)
            self.assertIsNone(parse_final('```json\n%s\n```' % text)[0], text)

    def test_independent_audit_parser_agrees(self):
        for text in [GOOD, '```json\n'+GOOD+'\n```', *INVALID]:
            self.assertEqual(strict(text), parse_final(text)[0], text)

    def test_failures_have_no_estimate(self):
        o = {'D_obs': 5}
        for record in ({'started': True, 'terminal': True, 'final_text': 'oops'},
                       {'started': True, 'terminal': True, 'final_text': GOOD, 'technical_error': True},
                       {'started': True, 'terminal': True, 'final_text': GOOD, 'refusal': True}):
            out = resolve(o, record)
            self.assertIsNone(out['prediction']); self.assertIsNone(out['replacement'])
            self.assertTrue(out['terminal']); self.assertFalse(out['valid'])
        self.assertEqual(resolve(o)['status'], 'not_started')
        self.assertEqual(resolve(o, {'started': True})['status'], 'in_progress')
        limited = resolve(o, {'started': True, 'terminal': True, 'limit_hit': True, 'final_text': GOOD})
        self.assertTrue(limited['valid'] and limited['limit_hit'])
        self.assertIsNone(resolve({'D_obs': 0})['prediction'])


class AggregationTests(unittest.TestCase):
    def test_complete_cells_mcse_is_the_draw_level_formula(self):
        a = np.random.default_rng(13).normal(size=(5, 3))
        result = complete_summary({'x': a, 'y': a+2}, {'x': 5, 'y': 5})
        self.assertAlmostEqual(result['mean'], a.mean()+1)
        self.assertAlmostEqual(result['mcse']**2, np.var(a.mean(1), ddof=1)/5/2)
        cells = {str(i): np.tile(np.arange(5)[:, None]/10, (1, 3)) for i in range(6)}
        x = complete_summary(cells, {k: 5 for k in cells})
        self.assertAlmostEqual(x['mcse'], math.sqrt(np.var(np.arange(5)/10, ddof=1)/5/6))

    def test_conditional_weights_and_missing_sources(self):
        a = np.array([[.2, .4, np.nan], [.9, np.nan, np.nan], [np.nan]*3, [.1, .3, .5], [.6]*3])
        estimate = conditional_summary({'x': a}, {'x': 5})
        self.assertAlmostEqual(estimate['mean'], np.nanmean(a))
        n = np.isfinite(a).sum(1); total = np.nansum(a, axis=1); m = np.nanmean(a)
        self.assertAlmostEqual(estimate['mcse']**2, 5/4*np.sum((total-m*n)**2)/n.sum()**2)
        missing = conditional_summary({'x': a, 'y': np.full((5, 3), np.nan)}, {'x': 5, 'y': 5})
        self.assertIsNone(missing['mean']); self.assertEqual(missing['sources_with_valid_answers'], 1)
        with self.assertRaises(ValueError): complete_summary({'x': a}, {'x': 5})

    def test_error_before_averaging(self):
        self.assertEqual(errors([0]*4, [.5]*4)['AE2'], .5)
        self.assertEqual(np.mean([errors([0]*4, [.5]*4)['AE2'], errors([1]*4, [.5]*4)['AE2']]), .5)
        self.assertIsNone(errors(None, [.5]*4)['AE2'])


class PairedControlTests(unittest.TestCase):
    def test_delta_error_is_not_a_difference_of_absolute_errors(self):
        d = delta_metrics([.5]*4, [.7]*4, [.4]*4, [.8]*4)
        self.assertAlmostEqual(d['AE_Delta_2'], .2); self.assertAlmostEqual(d['Delta_ProfileAE'], .2)
        self.assertAlmostEqual(d['signed_delta_error_2'], .2); self.assertTrue(d['sign_agreement_2'])

    def test_missing_member_has_no_pair_value(self):
        d = delta_metrics([.5]*4, [.7]*4, None, [.8]*4)
        self.assertIsNone(d['AE_Delta_2']); self.assertIsNone(d['delta_rhohat'])


class EvaluationIntegrationTests(unittest.TestCase):
    """evaluate_responses on a small copy of the prepared study (skipped before it exists)."""

    def setUp(self):
        if not (REFERENCES/'primary_baselines.json').exists(): self.skipTest('prepared study absent')

    def test_invalid_incomplete_and_mixed_answers(self):
        from evaluate_responses import evaluate
        from main_experiment.requests import planned
        with tempfile.TemporaryDirectory(suffix='_mock') as td:
            run = Path(td)/'run'
            references = read_json(REFERENCES/'primary_baselines.json')
            observations = [read_json(p) for p in sorted((PREPARED/'observations/sample').glob('dar_a0_r*__R-*.json'))]
            for o in observations: write_json(run/'observations/sample'/f'{o["id"]}.json', o)
            write_json(run/'report.json', {'design_version': DESIGN_VERSION})
            write_json(run/'baselines.json', {'design_version': DESIGN_VERSION,
                                              'observations': {o['id']: references['observations'][o['id']] for o in observations}})
            requests = planned(observations)
            (run/'requests.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in requests))
            records = [{'id': r['id'], 'prompt_sha256': r['prompt_sha256'], 'payload_sha256': r['payload_sha256'],
                        'started': True, 'terminal': True, 'mock': True, 'final_text': 'invalid'} for r in requests]
            responses = run/'mock.jsonl'

            def run_evaluation(name):
                responses.write_text(''.join(json.dumps(r)+'\n' for r in records))
                evaluate(responses, run/name, True, run, run/'baselines.json')
                with open(run/name/'answer_errors.csv') as f: answers = list(csv.DictReader(f))
                return answers, read_json(run/name/'report.json')
            answers, report = run_evaluation('all_invalid_mock')
            self.assertTrue(report['complete_main_result'])
            self.assertTrue(all(r['prediction_json'] == '' and r['replacement'] == '' for r in answers))
            records[0]['terminal'] = False
            self.assertFalse(run_evaluation('pending_mock')[1]['complete_main_result'])
            for i, r in enumerate(records):
                r['terminal'] = True
                if i % 3: r['final_text'] = '{"rho_2":0.4,"rho_3":0.3,"rho_4":0.2,"rho_5":0.1}'
            answers, _ = run_evaluation('mixed_mock')
            for r in answers: self.assertEqual(r['baseline_AE2_matched'] == '', r['valid'] == 'False')
            records[0].pop('payload_sha256')                # missing provenance is rejected
            with self.assertRaises(ValueError): run_evaluation('bad_mock')


if __name__ == '__main__':
    unittest.main()
