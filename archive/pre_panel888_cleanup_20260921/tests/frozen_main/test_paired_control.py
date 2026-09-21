import sys,unittest
from main_experiment.common import ROOT
sys.path.insert(0,str(ROOT/'scripts'))
from evaluate_panel888_pairs import delta_metrics
class PairedControlTests(unittest.TestCase):
    def test_difference_is_not_difference_of_absolute_errors(self):
        d=delta_metrics([.5]*4,[.7]*4,[.4]*4,[.8]*4)
        self.assertAlmostEqual(d['AE_Delta_2'],.2)
        self.assertAlmostEqual(d['Delta_ProfileAE'],.2)
        self.assertAlmostEqual(d['signed_delta_error_2'],.2)
        self.assertTrue(d['sign_agreement_2'])
    def test_missing_member_has_no_pair_prediction(self):
        d=delta_metrics([.5]*4,[.7]*4,None,[.8]*4)
        self.assertIsNone(d['AE_Delta_2']);self.assertIsNone(d['delta_rhohat'])
