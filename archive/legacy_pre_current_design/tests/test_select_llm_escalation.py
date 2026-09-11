import json
import os
import tempfile
import unittest

import select_llm_escalation as selector


COMPLETE = {
    "rho_k2": 0.4, "rho_k3": 0.3, "rho_k4": 0.2, "rho_k5": 0.1,
    "mean_occupancy": 0.5, "C_one_step": 0.6,
    "lifetime_mean_over_T": 0.2, "lo90": 0.3, "hi90": 0.5,
}


class TestSelectEscalation(unittest.TestCase):
    def write(self, rows):
        fd, path = tempfile.mkstemp(suffix=".jsonl")
        with os.fdopen(fd, "w") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")
        self.addCleanup(os.unlink, path)
        return path

    def test_selects_missing_and_latest_incomplete(self):
        prompts = self.write([
            {"prompt_id": "complete"},
            {"prompt_id": "recovered"},
            {"prompt_id": "length"},
            {"prompt_id": "missing"},
        ])
        answers = self.write([
            {"prompt_id": "complete", "answer": json.dumps(COMPLETE),
             "finish_reason": "stop"},
            {"prompt_id": "recovered", "answer": "",
             "finish_reason": "length"},
            {"prompt_id": "recovered", "answer": json.dumps(COMPLETE),
             "finish_reason": "stop"},
            {"prompt_id": "length", "answer": "",
             "finish_reason": "length"},
        ])
        self.assertEqual(
            selector.select_ids(prompts, [answers]),
            ["length", "missing"],
        )


if __name__ == "__main__":
    unittest.main()
