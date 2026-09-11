import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from g0d_budget_ladder import (  # noqa: E402
    ladder_node_panel,
    ladder_two_phase,
)
from nonwalk_samplers import (  # noqa: E402
    event_sample_then_full_history,
    node_panel_full_history,
    prepare_dyad_histories,
    prepare_events,
)
from report_g0d_headroom import apply_seed_rule  # noqa: E402


def toy_events():
    """Four dyads with deliberately uneven history lengths."""
    rows = []
    for t in np.linspace(0.01, 0.99, 9):
        rows.append((0, 1, float(t)))
    for t in np.linspace(0.05, 0.95, 4):
        rows.append((1, 2, float(t)))
    rows.append((2, 3, 0.5))
    rows.append((3, 4, 0.7))
    return pd.DataFrame(rows, columns=["u", "v", "t"])


def panel_frame(records):
    """Minimal frame with the columns the seed rule reads."""
    return pd.DataFrame([
        {"instance_id": instance, "strategy": arm, "sample_seed": seed,
         "budget": budget}
        for instance, arm, seed, budget in records])


class SeedRuleTest(unittest.TestCase):
    def test_takes_first_non_empty_slots_in_sequence_order(self):
        frame = panel_frame([
            ("g", "arm", 0, 0), ("g", "arm", 1, 50), ("g", "arm", 2, 0),
            ("g", "arm", 3, 60), ("g", "arm", 4, 70),
        ])
        accepted, log = apply_seed_rule(frame, 2)
        self.assertEqual(list(accepted.sample_seed), [1, 3])
        self.assertEqual(list(accepted.seed_slot), [0, 1])
        self.assertEqual(int(log.seed_advances.iloc[0]), 2)
        self.assertEqual(int(log.highest_seed_index_used.iloc[0]), 3)

    def test_no_empties_means_no_advances(self):
        frame = panel_frame([("g", "arm", s, 10) for s in range(4)])
        accepted, log = apply_seed_rule(frame, 3)
        self.assertEqual(list(accepted.sample_seed), [0, 1, 2])
        self.assertEqual(int(log.seed_advances.iloc[0]), 0)

    def test_cases_are_never_silently_dropped(self):
        frame = panel_frame([("g", "arm", 0, 0), ("g", "arm", 1, 0)])
        with self.assertRaises(RuntimeError):
            apply_seed_rule(frame, 1)

    def test_each_case_and_arm_keeps_its_own_sequence(self):
        frame = panel_frame([
            ("g1", "a", 0, 0), ("g1", "a", 1, 5),
            ("g1", "b", 0, 7), ("g1", "b", 1, 9),
            ("g2", "a", 0, 3), ("g2", "a", 1, 4),
        ])
        accepted, log = apply_seed_rule(frame, 1)
        picked = dict(zip(zip(accepted.instance_id, accepted.strategy),
                          accepted.sample_seed))
        self.assertEqual(picked[("g1", "a")], 1)
        self.assertEqual(picked[("g1", "b")], 0)
        self.assertEqual(picked[("g2", "a")], 0)
        self.assertEqual(int(log.seed_advances.sum()), 1)


class BudgetLadderTest(unittest.TestCase):
    """The ladder is only usable if it reproduces the production samplers."""

    def setUp(self):
        self.prepared = prepare_events(toy_events())
        self.dyads = prepare_dyad_histories(self.prepared)

    def _observed(self, result):
        log = result.log
        if not len(log):
            return 0, 0
        dyads = set(map(tuple, log[["u", "v"]].astype(int).to_numpy()))
        return len(log), len(dyads)

    def test_node_panel_ladder_matches_sampler(self):
        budgets = [1, 2, 3, 5, 8, 13, 20, 40]
        for seed in range(6):
            rungs = ladder_node_panel(self.prepared, seed, budgets)
            for budget, events, dyads, _ in rungs:
                got = self._observed(
                    node_panel_full_history(self.prepared, budget, seed))
                self.assertEqual(got, (events, dyads),
                                 f"seed {seed}, budget {budget}")

    def test_two_phase_ladder_matches_sampler(self):
        budgets = [1, 2, 3, 5, 8, 13, 20, 40]
        for seed in range(6):
            rungs = ladder_two_phase(self.dyads, seed, budgets)
            for budget, events, dyads, _ in rungs:
                got = self._observed(
                    event_sample_then_full_history(self.dyads, budget, seed))
                self.assertEqual(got, (events, dyads),
                                 f"seed {seed}, budget {budget}")

    def test_ladder_reports_the_empty_sample_rather_than_hiding_it(self):
        # Budget 1 cannot fit any dyad here, so every rung must be empty.
        rungs = ladder_two_phase(self.dyads, 0, [1])
        self.assertEqual(rungs[0][1:], (0, 0, 0))


if __name__ == "__main__":
    unittest.main()
