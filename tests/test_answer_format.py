"""Strict answer parsing and the valid-profile rule for ExtraTrees output."""
import unittest
from study.answer_format import parse_final, strip_fence, valid_profile

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


class ValidProfile(unittest.TestCase):
    def test_extratrees_output_is_made_valid_and_valid_profiles_stay_unchanged(self):
        self.assertEqual(valid_profile([1.02, 0.5, 0.6, -0.01]), [1.0, 0.5, 0.5, 0.0])
        self.assertEqual(valid_profile([0.8, 0.6, 0.4, 0.2]), [0.8, 0.6, 0.4, 0.2])
