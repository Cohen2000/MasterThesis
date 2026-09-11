import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from make_qwen_prompts_g3 import SCOPE, build  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def toy_records():
    out = []
    for subset in ("factorial", "mismatched", "metadata_only",
                   "seed_replication", "irrelevant_context"):
        for i in range(4):
            out.append({"prompt_id": f"{subset}|{i}", "subset": subset,
                        "prompt": f"text {subset} {i}"})
    return out


class QwenPromptSeedTest(unittest.TestCase):
    """Without a varying gen_seed the response-noise measurement is a lie."""

    def test_generations_never_share_a_seed(self):
        records = toy_records()
        seen = set()
        for generation in range(3):
            rows = build(records, "qwen36-27b_think", generation, 20260901)
            seeds = {r["gen_seed"] for r in rows}
            self.assertEqual(len(seeds), len(rows), "seeds collide within a generation")
            self.assertFalse(seeds & seen, "seed reused across generations")
            seen |= seeds

    def test_every_record_carries_a_seed(self):
        rows = build(toy_records(), "qwen36-27b_think", 0, 20260901)
        self.assertTrue(rows)
        for row in rows:
            self.assertIsInstance(row["gen_seed"], int)

    def test_seeding_is_reproducible_from_the_master_seed(self):
        first = build(toy_records(), "qwen36-27b_think", 1, 20260901)
        second = build(toy_records(), "qwen36-27b_think", 1, 20260901)
        self.assertEqual([r["gen_seed"] for r in first],
                         [r["gen_seed"] for r in second])

    def test_a_different_master_seed_gives_different_seeds(self):
        a = build(toy_records(), "qwen36-27b_think", 0, 20260901)
        b = build(toy_records(), "qwen36-27b_think", 0, 20260902)
        self.assertNotEqual([r["gen_seed"] for r in a],
                            [r["gen_seed"] for r in b])

    def test_the_two_models_carry_their_rostered_scope(self):
        think = build(toy_records(), "qwen36-27b_think", 0, 1)
        nothink = build(toy_records(), "qwen36-27b_nothink", 0, 1)
        self.assertEqual({r["subset"] for r in think},
                         set(SCOPE["qwen36-27b_think"]))
        self.assertEqual({r["subset"] for r in nothink},
                         set(SCOPE["qwen36-27b_nothink"]))
        # seed replication is a thinking-model subset only
        self.assertIn("seed_replication", {r["subset"] for r in think})
        self.assertNotIn("seed_replication", {r["subset"] for r in nothink})


class GenerationNoiseCheckTest(unittest.TestCase):
    """The guard has to actually fail on the failure it exists to catch."""

    SCRIPT = REPO / "src" / "check_generation_noise.py"

    def _run(self, gen0, gen1):
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for name, records in (("g0", gen0), ("g1", gen1)):
                path = Path(tmp) / f"{name}.jsonl"
                path.write_text("".join(
                    json.dumps(r) + "\n" for r in records))
                paths.append(str(path))
            return subprocess.run(
                [sys.executable, str(self.SCRIPT), "--answers", *paths],
                capture_output=True, text=True,
                env={"PYTHONPATH": str(REPO / "src"), "PATH": "/usr/bin:/bin"})

    def test_passes_when_generations_differ(self):
        gen0 = [{"prompt_id": f"p{i}", "answer": f"a{i}", "seed": i}
                for i in range(10)]
        gen1 = [{"prompt_id": f"p{i}", "answer": f"b{i}", "seed": 100 + i}
                for i in range(10)]
        self.assertEqual(self._run(gen0, gen1).returncode, 0)

    def test_fails_when_answers_are_identical_despite_varied_seeds(self):
        # The nastier case: gen_seed *was* varied, but decoding is effectively
        # deterministic, so the noise measurement is still an artifact.
        gen0 = [{"prompt_id": f"p{i}", "answer": f"a{i}", "seed": i}
                for i in range(10)]
        gen1 = [{"prompt_id": f"p{i}", "answer": f"a{i}", "seed": 100 + i}
                for i in range(10)]
        result = self._run(gen0, gen1)
        self.assertEqual(result.returncode, 1)
        self.assertIn("byte-identical", result.stderr)

    def test_fails_when_the_seed_was_not_varied(self):
        gen0 = [{"prompt_id": f"p{i}", "answer": f"a{i}", "seed": 7}
                for i in range(10)]
        gen1 = [{"prompt_id": f"p{i}", "answer": f"b{i}", "seed": 7}
                for i in range(10)]
        result = self._run(gen0, gen1)
        self.assertEqual(result.returncode, 1)
        self.assertIn("reused a seed", result.stderr)

    def test_fails_when_no_prompt_is_shared(self):
        gen0 = [{"prompt_id": "p1", "answer": "a", "seed": 1}]
        gen1 = [{"prompt_id": "p2", "answer": "b", "seed": 2}]
        self.assertEqual(self._run(gen0, gen1).returncode, 1)


if __name__ == "__main__":
    unittest.main()
