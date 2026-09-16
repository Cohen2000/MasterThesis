#!/usr/bin/env python3
"""Validity of the stored answers under each of the three parser rules.

The rules are versioned because they were fixed at different times, and only the
second one governs the main result:

  v1 bare        the original contract: the answer is exactly one JSON object.
  v2 fence       v1 plus one whole-answer markdown fence. Fixed BEFORE the main
                 run, after a four-answer smoke test showed fenced replies.
                 This is the main rule.
  v3 trailing    v2 plus a JSON object at the very end of the answer. Fixed AFTER
                 seeing that v2 yields zero valid non-thinking answers. Reported
                 as a sensitivity analysis only, and applied identically to every
                 configuration -- not only to the one it was prompted by.

The strict/loose split follows IFEval (arXiv:2311.07911), which reports both a
strict and a relaxed instruction-following score rather than choosing one.

Examples are selected mechanically: the first request ids in sort order per
configuration. No example is chosen for having a good or bad estimate.
"""
import argparse, json, re, sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from main_experiment.evaluation import parse_final, strip_fence

FENCE_ONLY = re.compile(r'\A```[A-Za-z0-9_+-]*\s*\n(.*?)\n?```\s*\Z', re.S)
TRAILING = re.compile(r'(?:```[A-Za-z0-9_+-]*\s*\n)?(\{[^{}]*"rho_2"[^{}]*\})\s*(?:\n```)?\s*\Z', re.S)


def v1(text):
    """Original rule: the whole answer must be the JSON object."""
    if FENCE_ONLY.match(text.strip()):
        return None, 'fenced'          # a fence is not bare JSON under v1
    return parse_final(text)


def v2(text):
    """Main rule: v1 plus one whole-answer fence."""
    return parse_final(text)


def v3(text):
    """Sensitivity rule: v2 plus a trailing JSON object."""
    v, r = parse_final(text)
    if v is not None:
        return v, r
    m = TRAILING.search(text.rstrip())
    if not m:
        return None, 'no_trailing_json'
    v, r = parse_final(m.group(1))
    return v, ('valid_after_trailing' if v is not None else r)


RULES = [('v1_bare', v1), ('v2_fence', v2), ('v3_trailing', v3)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--answers', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--examples', type=int, default=2)
    ap.add_argument('--prompts', help='rendered_prompts.jsonl from the archive')
    a = ap.parse_args()

    recs = {}
    for f in sorted(Path(a.answers).rglob('*.json')):
        d = json.loads(f.read_text())
        recs[d['id']] = d
    prompts = {}
    if a.prompts:
        for line in Path(a.prompts).read_text().splitlines():
            r = json.loads(line)
            prompts[(r['observation_id'], r['mode'])] = r

    counts = {}
    for mode in sorted({d['mode'] for d in recs.values()}):
        sub = [d for d in recs.values() if d['mode'] == mode]
        counts[mode] = {'n': len(sub)}
        for name, fn in RULES:
            c = Counter()
            for d in sub:
                v, r = fn(d['final_text'])
                c['valid' if v is not None else 'invalid'] += 1
                c['reason:' + r] += 1
            counts[mode][name] = dict(c)

    examples = []
    for mode in sorted({d['mode'] for d in recs.values()}):
        chosen = sorted(d['id'] for d in recs.values() if d['mode'] == mode)[:a.examples]
        for rid in chosen:
            d = recs[rid]
            p = prompts.get((d['observation_id'], mode))
            examples.append({
                'request_id': rid, 'mode': mode,
                'selection': 'first request ids in sort order; not chosen by outcome',
                'prompt_tail': (p['rendered'][-160:] if p else None),
                'output_head': d['raw_text'][:200],
                'output_tail': d['raw_text'][-200:],
                'output_tokens': d['output_tokens'],
                'end_state': d['end_state'],
                'verdicts': {name: fn(d['final_text'])[1] for name, fn in RULES}})

    report = {'rules': {'v1_bare': 'original contract, bare JSON only',
                        'v2_fence': 'MAIN RULE, fixed before the run: one whole-answer fence allowed',
                        'v3_trailing': 'SENSITIVITY, fixed after the run: trailing JSON object accepted'},
              'applied_to_all_configurations': True,
              'counts': counts, 'examples': examples}
    Path(a.out).write_text(json.dumps(report, indent=1, sort_keys=True) + '\n')
    for mode, c in counts.items():
        line = ' | '.join(f"{n}: {c[n].get('valid', 0)}/{c['n']}" for n, _ in RULES)
        print(f'{mode:12} {line}')
    print(f'written to {a.out}')


if __name__ == '__main__':
    main()
