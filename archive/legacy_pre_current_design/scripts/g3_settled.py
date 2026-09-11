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

sys.path.insert(0, "llm_g3")
from run_llm_v2 import is_complete_record  # noqa: E402

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
                # `finish_reason == "stop"` is not completeness: a stopped
                # generation whose final JSON is unparseable or missing keys
                # is still retryable, and counting it as done would stop the
                # keeper while real work remains.
                if is_complete_record(record):
                    done.add(pid)
    burned = {p for p, n in tries.items() if n >= MAX_ATTEMPTS}
    print(len(done | burned))


if __name__ == "__main__":
    main()
