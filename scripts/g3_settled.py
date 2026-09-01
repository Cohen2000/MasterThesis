#!/usr/bin/env python3
"""How many prompts of one generation are settled, i.e. will not be retried.

Settled means complete, or having burned its attempt ceiling. Some generations
never terminate at any budget -- the same prompts truncate at 16k and again at
24k -- so counting only complete ones makes a resume loop reload the model for
them indefinitely. A prompt that genuinely cannot be answered is a result under
the failure-penalized rule, not a gap to keep filling.

Counts distinct prompt_ids, not lines: a prompt can appear in more than one
shard file when the shard count changed between runs.
"""

import glob
import json
import sys

MAX_ATTEMPTS = 3


def main():
    tag, generation = sys.argv[1], sys.argv[2]
    pattern = f"llm_g3/answers_vllm_qwen36-27b_{tag}_g{generation}.shard*.jsonl"
    done, tries = set(), {}
    for path in glob.glob(pattern):
        with open(path) as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                pid = record.get("prompt_id")
                tries[pid] = tries.get(pid, 0) + 1
                if record.get("finish_reason") == "stop":
                    done.add(pid)
    burned = {p for p, n in tries.items() if n >= MAX_ATTEMPTS}
    print(len(done | burned))


if __name__ == "__main__":
    main()
