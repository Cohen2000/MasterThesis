#!/usr/bin/env python3
"""Variant of Sections 4.4 and 5.3 of the thesis: GPT-6 Sol under the random walk without the walk's log.

The 72 random-walk samples (12 networks and their copies, three samples each) are shown with the
same prompt, except that the three log columns (inv_events, traversals, traversals_per_event) and
their description are removed; only the walk length L is added. Three answers per sample.

  python scripts/ablation_walk_log.py              dry run: how many answers are missing
  python scripts/ablation_walk_log.py --execute    send the requests (needs OPENAI_API_KEY)
  python scripts/ablation_walk_log.py --retry      also resend requests that were sent but brought no answer

The answers go to results/ablation_walk_log/responses.jsonl; scripts/score.py turns them into rows
of data/predictions.csv (method gpt_6_sol_no_walk_log).
"""
import argparse
import json
import sys
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import api  # noqa: E402
from study.common import DATA, OUT, digest, read_json  # noqa: E402
from study.sample import HEAD, PROMPTS, WALK_HEAD  # noqa: E402

RUN = 'openai_noweights'        # part of every request ID


def samples():
    """The random-walk samples without the walk's log; the sampling rule and the walk length L stay."""
    read = lambda name: (PROMPTS/name).read_text().rstrip('\n')
    rule = read('S.txt').split('The table also gives', 1)[0].rstrip()+' The walk length L is released.'
    selected = []
    for path in sorted((DATA/'samples').glob('*.json')):
        s = read_json(path)
        if s['arm'] != 'S': continue
        lines = s['block'].splitlines()
        head = lines.index(WALK_HEAD)
        rows = [line.split(',') for line in lines[head+1:]]
        block = '\n'.join(lines[:head]+[f'L={sum(int(r[4]) for r in rows)}', HEAD]+[','.join(r[:3]) for r in rows])
        user = [read('task.txt'), 'Sampling rule: '+rule, block, 'Return the four full-archive estimates in the specified JSON format.']
        messages = [{'role': 'system', 'content': read('system.txt')}, {'role': 'user', 'content': '\n'.join(user)}]
        assert all(word not in m['content'] for m in messages for word in ('inv_events', 'traversals', 'c_e'))
        selected.append({**s, 'original_block': s['block'], 'block': block, 'block_sha256': digest(block),
                         'messages': messages, 'prompt_sha256': digest(messages)})
    assert len(selected) == 72
    return selected


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--execute', action='store_true', help='send the requests')
    ap.add_argument('--retry', action='store_true', help='resend requests sent without an answer')
    a = ap.parse_args()
    out = OUT/'ablation_walk_log'
    lines = lambda name: [json.loads(x) for x in (out/name).read_text().splitlines()] if (out/name).exists() else []
    todo = api.planned(samples(), RUN)
    done = {r['id'] for r in lines('responses.jsonl')}
    sent = {r['id'] for r in lines('attempts.jsonl')}-done
    todo = [r for r in todo if r['id'] not in done and (a.retry or r['id'] not in sent)]
    print(f'{len(done)} answers stored, {len(sent)} sent without an answer, {len(todo)} to ask')
    if a.execute and todo:
        if any(digest(r['messages']) != r['prompt_sha256'] for r in todo): raise SystemExit('a prompt does not match its sample')
        out.mkdir(parents=True, exist_ok=True)
        api.run_openai(todo, out, tools=False)


if __name__ == '__main__':
    main()
