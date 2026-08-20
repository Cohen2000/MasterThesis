import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evaluate_benchmark import _fit_models, evaluate_strategy  # noqa: E402


class MedianFloor(unittest.TestCase):
    def test_uses_only_training_groups_in_each_fold(self):
        rows = []
        for group, truth, feature in (
                ("g0", 0.0, -1.0), ("g1", 0.4, 0.0), ("g2", 1.0, 1.0)):
            for rep in range(2):
                rows.append({
                    "case_id": f"{group}-{rep}",
                    "group_id": group,
                    "strategy": "walk",
                    "rho_W5_k2": truth,
                    "crawl__signal": feature + rep / 100,
                })
        cases = pd.DataFrame(rows)
        records = evaluate_strategy(
            cases,
            "walk",
            ["rho_W5_k2"],
            ["crawl"],
            [],
            folds=3,
            jobs=1,
            cfg={},
        )
        pred = pd.concat(
            [r for r in records if r["model"].iloc[0] == "median_floor"],
            ignore_index=True,
        )
        for _, row in pred.iterrows():
            train_truth = cases.loc[
                cases["group_id"] != row["group_id"], "rho_W5_k2"]
            self.assertAlmostEqual(row["prediction"], train_truth.median())


class KNNRetrieval(unittest.TestCase):
    def test_returns_bounded_multioutput_predictions(self):
        train = pd.DataFrame({
            "case_id": [f"tr{i}" for i in range(8)],
            "group_id": [f"g{i // 2}" for i in range(8)],
            "strategy": ["walk"] * 8,
            "crawl__x": np.linspace(-1, 1, 8),
            "crawl__y": np.linspace(1, -1, 8),
            "rho_W5_k2": np.linspace(0.1, 0.8, 8),
            "rho_W5_k3": np.linspace(0.0, 0.7, 8),
        })
        test = pd.DataFrame({
            "case_id": ["te0", "te1"],
            "group_id": ["held", "held"],
            "strategy": ["walk", "walk"],
            "crawl__x": [-0.25, 0.25],
            "crawl__y": [0.25, -0.25],
            "rho_W5_k2": [0.3, 0.6],
            "rho_W5_k3": [0.2, 0.5],
        })
        records = _fit_models(
            train,
            test,
            ["rho_W5_k2", "rho_W5_k3"],
            ["crawl"],
            ["knn"],
            jobs=1,
        )
        self.assertEqual(len(records), 2)
        for record in records:
            self.assertEqual(record["model"].unique().tolist(), ["knn"])
            self.assertTrue(np.isfinite(record["prediction"]).all())
            self.assertTrue(record["prediction"].between(0.0, 1.0).all())


if __name__ == "__main__":
    unittest.main()
