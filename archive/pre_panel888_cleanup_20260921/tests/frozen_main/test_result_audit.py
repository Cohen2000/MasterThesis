import importlib.util
import unittest
from main_experiment.common import ROOT

spec=importlib.util.spec_from_file_location('independent_audit',ROOT/'scripts/audit_panel888.py')
audit=importlib.util.module_from_spec(spec); spec.loader.exec_module(audit)

class IndependentAuditTests(unittest.TestCase):
    def test_strict_valid_profile_and_whole_fence(self):
        text='{"rho_2":0.8,"rho_3":0.6,"rho_4":0.2,"rho_5":0}'
        self.assertEqual(audit.strict(text),[.8,.6,.2,0])
        self.assertEqual(audit.strict('```json\n'+text+'\n```'),[.8,.6,.2,0])

    def test_no_duplicate_boolean_nonfinite_or_nonmonotone(self):
        for text in ('{"rho_2":1,"rho_2":0.8,"rho_3":0.6,"rho_4":0.2,"rho_5":0}',
                     '{"rho_2":true,"rho_3":0.6,"rho_4":0.2,"rho_5":0}',
                     '{"rho_2":NaN,"rho_3":0.6,"rho_4":0.2,"rho_5":0}',
                     '{"rho_2":0.5,"rho_3":0.6,"rho_4":0.2,"rho_5":0}',
                     'reasoning {"rho_2":0.8,"rho_3":0.6,"rho_4":0.2,"rho_5":0}'):
            with self.subTest(text=text): self.assertIsNone(audit.strict(text))
