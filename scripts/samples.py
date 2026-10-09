#!/usr/bin/env python3
"""Draw the 384 samples: 32 networks x 4 sampling arms x 3 draws, with the prompt of each.

Writes results/samples/ and checks every file against the fingerprints in CHECKSUMS.json, so the
samples are exactly those the language models answered. Needs the raw networks in data/raw.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from study.common import ARMS, DRAWS, FINAL, OUT, REAL, digest, group, read_json, sample_id, sha, write_json  # noqa: E402
from study.data import network, twin  # noqa: E402
from study.sample import prompt, text  # noqa: E402
from study.sampling import draw, sizes  # noqa: E402
from study.synthetic import test_networks  # noqa: E402
from study.walk import Walk  # noqa: E402


def networks():
    """The 32 test networks: 12 real, their 12 time-shuffled twins and 8 synthetic."""
    for key in REAL:
        g = network(key)
        yield g
        yield twin(g)
    yield from test_networks()


def main():
    truth = {}
    for g in networks():
        walk = Walk(g)
        size = sizes(g, walk)
        truth[g.key] = g.truth
        for arm in ARMS:
            for index in range(1, DRAWS+1):
                block = text(g, arm, size, *draw(g, arm, index, 'sample', size, walk))
                messages = prompt(block)
                name = sample_id(g.key, arm, index)
                write_json(OUT/'samples'/f'{name}.json', {
                    'id': name, 'graph_id': g.key, 'stratum': group(g.key), 'arm': arm, 'sample_index': index,
                    'block': block, 'block_sha256': digest(block), 'messages': messages, 'prompt_sha256': digest(messages)})
    frozen = read_json(FINAL/'CHECKSUMS.json')['observations']['files']
    if {p.name: sha(p) for p in (OUT/'samples').glob('*.json')} != frozen or truth != read_json(FINAL/'TRUTH.json'):
        raise SystemExit('the samples or the true values differ from the frozen ones')
    print(f'{len(frozen)} samples written; identical to the frozen samples')


if __name__ == '__main__':
    main()
