import json
import unittest

from persistence_evaluation import (evaluate_response, mean_occupancy,
                                    parse_final_profile, profile_diagnostics,
                                    profile_mae)


class RawProfileEvaluation(unittest.TestCase):
    def setUp(self):
        self.truth = {"rho_2": 0.8, "rho_3": 0.6, "rho_4": 0.4, "rho_5": 0.2}

    def test_profile_mae_uses_all_four_components(self):
        prediction = {"rho_2": 0.7, "rho_3": 0.8, "rho_4": 0.1, "rho_5": 0.6}
        self.assertAlmostEqual(profile_mae(prediction, self.truth), 0.25)
        self.assertEqual(profile_mae(self.truth, self.truth), 0)

    def test_free_reasoning_before_final_json(self):
        response = 'Earlier draft: {"rho_2": 0}\nReasoning.\n' + json.dumps(self.truth) + "\n"
        self.assertEqual(parse_final_profile(response), self.truth)

    def test_no_clipping_sorting_or_raw_response_rewrite(self):
        prediction = {"rho_2": 1.2, "rho_3": 0.1, "rho_4": 0.9, "rho_5": -0.2}
        response = "Reasoning.\n" + json.dumps(prediction)
        result = evaluate_response(response, self.truth)
        self.assertEqual(result["raw_response"], response)
        self.assertEqual(result["prediction"], prediction)
        self.assertTrue(result["bounds_violation"])
        self.assertTrue(result["monotonicity_violation"])
        self.assertAlmostEqual(result["profile_mae"], 0.45)

    def test_non_monotone_in_range_values_are_not_repaired(self):
        prediction = dict(self.truth, rho_4=0.7)
        self.assertEqual(profile_diagnostics(prediction),
                         {"bounds_violation": False, "monotonicity_violation": True})
        self.assertAlmostEqual(profile_mae(prediction, self.truth), 0.075)

    def test_schema_numbers_duplicates_and_final_line_are_strict(self):
        invalid = ["", "{}", json.dumps(dict(self.truth, lo90=0.1)),
                   json.dumps(dict(self.truth, rho_2="0.8")),
                   json.dumps(dict(self.truth, rho_2=True)),
                   json.dumps(dict(self.truth, rho_2=float("nan"))),
                   json.dumps(dict(self.truth, rho_2=float("inf"))),
                   json.dumps(self.truth) + "\ntrailing text",
                   '{"rho_2":0.8,"rho_2":0.9,"rho_3":0.6,"rho_4":0.4,"rho_5":0.2}']
        for response in invalid:
            with self.subTest(response=response), self.assertRaises(ValueError):
                parse_final_profile(response)

    def test_unparseable_response_keeps_raw_text_and_has_no_invented_score(self):
        result = evaluate_response("incomplete", self.truth)
        self.assertEqual(result["raw_response"], "incomplete")
        self.assertIsNone(result["profile_mae"])
        self.assertIsNotNone(result["parse_error"])

    def test_occupancy_is_derived(self):
        self.assertAlmostEqual(mean_occupancy(self.truth), 0.6)

    def test_repeats_can_be_scored_individually_without_collapsing(self):
        raw = [json.dumps(dict(self.truth, rho_2=value)) for value in (0.7, 0.8, 0.9)]
        scores = [evaluate_response(response, self.truth) for response in raw]
        self.assertEqual(len(scores), 3)
        self.assertEqual([score["raw_response"] for score in scores], raw)
