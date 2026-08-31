import re
import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import prompt_contract_g1 as C  # noqa: E402
from make_llm_prompts_v2 import MECHANISM  # noqa: E402


def toy_row(arm="time_agnostic_t"):
    return {"strategy": arm, "n_nodes_true": 500, "n_events_true": 12345,
            "input__nmask_exact_json": '{"1,01":7,"2,03":2}'}


class NeutralityTest(unittest.TestCase):
    """The neutral texts must not leak the direction they exist to withhold."""

    DIRECTIONAL = re.compile(
        r"\b(underestimat|overestimat|approximately equal to|too low|too high|"
        r"biased toward|easier to record|unbiased|inflat|deflat|"
        r"Consequence:)", re.I)

    def test_no_mechanism_text_names_a_direction(self):
        for arm, text in C.MECHANISM_NEUTRAL.items():
            hit = self.DIRECTIONAL.search(text)
            self.assertIsNone(hit, f"{arm} leaks {hit.group(0) if hit else ''}")

    def test_arm_a_mechanism_does_not_claim_unbiasedness(self):
        # A statistician must *derive* it from the stated inclusion rule; the
        # text stating it outright would make `mechanism` a direction condition.
        self.assertNotIn("unbiased", C.MECHANISM_NEUTRAL[C.NODE_PANEL])

    def test_consequence_lines_are_gone_from_the_neutral_texts(self):
        # The historical prompts carry them; that is the confound being split.
        for arm in ("time_respecting", "recent_history_k20"):
            self.assertIn("Consequence:", MECHANISM[arm])
            self.assertNotIn("Consequence:", C.MECHANISM_NEUTRAL[arm])

    def test_direction_only_names_a_direction_for_every_arm(self):
        for arm in C.ARMS:
            self.assertIsNotNone(self.DIRECTIONAL.search(C.DIRECTION_ONLY[arm]),
                                 f"{arm} states no direction")

    def test_arm_a_direction_is_written_not_skipped(self):
        text = C.DIRECTION_ONLY[C.NODE_PANEL]
        self.assertIn("approximately equal to", text)

    def test_walks_and_arm_b_point_opposite_ways(self):
        for arm in C.WALK_ARMS:
            self.assertIn("an underestimate of", C.DIRECTION_ONLY[arm])
        self.assertIn("an overestimate of", C.DIRECTION_ONLY[C.TWO_PHASE])


class NoSharedPhrasingTest(unittest.TestCase):
    """Processes that differ must not be described in shared sentences."""

    def _sentences(self, text):
        body = text.split("\n", 1)[1].replace("\n", " ")
        return {s.strip() for s in body.split(". ") if len(s.strip()) > 40}

    def test_no_sentence_crosses_the_walk_full_history_boundary(self):
        # Where the processes genuinely differ, nothing may be shared. A walk
        # censors what it sees; a whole-entity retrieval does not.
        walk = set().union(*(self._sentences(C.MECHANISM_NEUTRAL[a])
                             for a in C.WALK_ARMS))
        full = set().union(*(self._sentences(C.MECHANISM_NEUTRAL[a])
                             for a in (C.NODE_PANEL, C.TWO_PHASE)))
        self.assertEqual(walk & full, set())

    def test_the_full_history_arms_differ_on_selection(self):
        # They agree that records are complete -- that sentence is shared, and
        # correctly so. What must differ is how an entity is selected, which is
        # where their whole bias difference comes from.
        panel = C.MECHANISM_NEUTRAL[C.NODE_PANEL]
        two_phase = C.MECHANISM_NEUTRAL[C.TWO_PHASE]
        self.assertIn("uniformly at random and without replacement", panel)
        self.assertIn("at least one of its two endpoints was recruited", panel)
        self.assertIn("uniformly random order", two_phase)
        self.assertIn("one of its own event records was reached", two_phase)
        for marker in ("endpoints was recruited", "recruiting"):
            self.assertNotIn(marker, two_phase)

    def test_each_arm_is_mostly_its_own_prose(self):
        # Shared phrasing is allowed only where the process fact is genuinely
        # identical -- the three walkers all record the event they moved
        # across -- so a little overlap among the walks is correct, and
        # inventing differences there would be its own confound. What is not
        # allowed is an arm that is mostly another arm's text.
        for arm in C.ARMS:
            mine = self._sentences(C.MECHANISM_NEUTRAL[arm])
            others = set().union(*(self._sentences(C.MECHANISM_NEUTRAL[a])
                                   for a in C.ARMS if a != arm))
            unique = len(mine - others) / len(mine)
            self.assertGreaterEqual(
                unique, 0.6, f"{arm} is only {unique:.0%} its own prose")

    def test_walk_censoring_language_never_appears_on_the_full_history_arms(self):
        # Complete histories are returned, so nothing is bounded by visit count.
        for arm in (C.NODE_PANEL, C.TWO_PHASE):
            text = C.MECHANISM_NEUTRAL[arm]
            self.assertIn("COMPLETE", text)
            self.assertNotIn("traversed", text)
            self.assertNotIn("walker", text)

    def test_full_history_arms_state_that_n_and_mask_are_true_values(self):
        for arm in (C.NODE_PANEL, C.TWO_PHASE):
            self.assertIn("true total number of events",
                          C.MECHANISM_NEUTRAL[arm])


class FactorialTest(unittest.TestCase):
    def test_mechanism_direction_is_mechanism_plus_the_direction_sentence(self):
        for arm in C.ARMS:
            self.assertEqual(
                C.MECHANISM_DIRECTION[arm],
                f"{C.MECHANISM_NEUTRAL[arm]}\n\n{C.DIRECTION_ONLY[arm]}")

    def test_the_direction_factor_is_a_real_contrast_on_every_arm(self):
        # The historical text for time_agnostic_t names no direction, so reusing
        # it verbatim would collapse this contrast on that arm.
        for arm in C.ARMS:
            self.assertNotEqual(C.MECHANISM_NEUTRAL[arm],
                                C.MECHANISM_DIRECTION[arm])

    def test_historical_bridge_is_byte_identical(self):
        for arm in C.WALK_ARMS:
            self.assertEqual(C.DISCLOSED_HISTORICAL[arm], MECHANISM[arm])


class MismatchTest(unittest.TestCase):
    def test_mismatched_shows_the_other_arms_neutral_text(self):
        a, b = C.MISMATCH_PAIR
        self.assertEqual(C.context_block(a, "mismatched", b),
                         C.MECHANISM_NEUTRAL[b])
        self.assertEqual(C.context_block(b, "mismatched", a),
                         C.MECHANISM_NEUTRAL[a])

    def test_mismatched_refuses_arms_outside_the_pair(self):
        with self.assertRaises(ValueError):
            C.context_block("time_respecting", "mismatched", C.NODE_PANEL)

    def test_mismatched_requires_a_stated_arm(self):
        with self.assertRaises(ValueError):
            C.context_block(C.MISMATCH_PAIR[0], "mismatched")

    def test_mismatched_carries_no_direction(self):
        a, b = C.MISMATCH_PAIR
        self.assertNotIn("DIRECTION OF THE SAMPLING BIAS",
                         C.context_block(a, "mismatched", b))


class AssemblyTest(unittest.TestCase):
    def test_metadata_only_shows_no_sample(self):
        text = C.build_prompt(toy_row(), "metadata_only")
        self.assertNotIn("01", text.split("NO SAMPLE")[1])
        self.assertIn("n_nodes", text)

    def test_metadata_only_is_identical_across_arms(self):
        rendered = {C.build_prompt(toy_row(arm), "metadata_only")
                    for arm in C.ARMS}
        self.assertEqual(len(rendered), 1)

    def test_every_other_condition_shows_the_sample(self):
        for condition in ("hidden", "direction_only", "mechanism",
                          "mechanism_direction"):
            text = C.build_prompt(toy_row(), condition)
            self.assertIn('{"1,01":7,"2,03":2}', text)

    def test_section_order_is_identical_across_conditions(self):
        order = []
        for condition in ("hidden", "direction_only", "mechanism",
                          "mechanism_direction"):
            text = C.build_prompt(toy_row(), condition)
            order.append((text.index("DEFINITIONS"),
                          text.index("NOW THE ACTUAL OBSERVATION"),
                          text.index("TASK")))
        for a, b, c in order:
            self.assertLess(a, b)
            self.assertLess(b, c)

    def test_unknown_condition_is_an_error(self):
        with self.assertRaises(ValueError):
            C.build_prompt(toy_row(), "disclosed")


class LengthBandTest(unittest.TestCase):
    def test_mechanism_texts_share_a_length_band(self):
        words = [len(C.MECHANISM_NEUTRAL[a].split()) for a in C.ARMS]
        self.assertLessEqual(max(words) / min(words), 1.35)

    def test_placebo_sits_inside_the_mechanism_band(self):
        words = [len(C.MECHANISM_NEUTRAL[a].split()) for a in C.ARMS]
        placebo = len(C.IRRELEVANT_CONTEXT.split())
        self.assertGreaterEqual(placebo, min(words))
        self.assertLessEqual(placebo, max(words))

    def test_every_mechanism_text_has_the_same_heading(self):
        for arm in C.ARMS:
            self.assertTrue(
                C.MECHANISM_NEUTRAL[arm].startswith("SAMPLING MECHANISM\n"))


if __name__ == "__main__":
    unittest.main()
