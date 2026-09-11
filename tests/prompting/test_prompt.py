import unittest

from persistence_prompt import build_prompt


class GenericPrompt(unittest.TestCase):
    def test_sections_target_and_schema(self):
        prompt = build_prompt("Dyad a-b: windows {0,2}",
                              selection_access="A uniform node panel.",
                              history_access="Complete histories of incident dyads.")
        sections = ["TARGET / DEFINITIONS", "OBSERVATION MECHANISM",
                    "OBSERVED DATA", "TASK", "FINAL JSON RESULT"]
        positions = [prompt.index(section) for section in sections]
        self.assertEqual(positions, sorted(positions))
        self.assertIn("1 >= rho_2 >= rho_3 >= rho_4 >= rho_5 >= 0", prompt)
        self.assertIn("E_full", prompt)
        self.assertIn("You may reason freely", prompt)
        for obsolete in ("C_one_step", "lifetime", "lo90", "hi90", "rho_k2",
                         "underestimate", "overestimate", "persona", "solved example"):
            self.assertNotIn(obsolete, prompt)

    def test_access_semantics_cannot_be_omitted(self):
        with self.assertRaises(ValueError):
            build_prompt("data", selection_access="", history_access="full history")
