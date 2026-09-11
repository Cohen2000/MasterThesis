import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from make_nonwalk_llm_prompts import select_cases  # noqa: E402


class BalancedNonwalkSelection(unittest.TestCase):
    def test_same_target_blind_instances_are_used_for_each_strategy(self):
        rows = []
        for block in ("a", "b"):
            for instance in (f"{block}1", f"{block}2"):
                for strategy in ("s1", "s2"):
                    for sample_seed in (0, 1):
                        rows.append({
                            "case_id": f"{instance}|{strategy}|{sample_seed}",
                            "instance_id": instance, "data_block": block,
                            "strategy": strategy, "target_budget": 800,
                            "sample_seed": sample_seed,
                            # Deliberately extreme truth values: selection must
                            # not inspect them.
                            "rho_W5_k2": 0.0 if instance.endswith("1") else 1.0,
                        })
        frame = pd.DataFrame(rows)
        selected = select_cases(
            frame, ["s1", "s2"], 800, sample_seed=0,
            instances_per_block=1, selection_seed=7)
        self.assertEqual(len(selected), 4)
        self.assertEqual(selected.groupby("strategy").size().to_dict(),
                         {"s1": 2, "s2": 2})
        instances = selected.groupby("strategy").instance_id.apply(set)
        self.assertEqual(instances["s1"], instances["s2"])
        self.assertTrue((selected.sample_seed == 0).all())


if __name__ == "__main__":
    unittest.main()
