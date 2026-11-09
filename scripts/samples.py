#!/usr/bin/env python3
"""Draw the 288 samples: 24 networks (12 real, 12 time-shuffled copies) x 4 samplers x 3 draws.

Every sample is written with the exact prompt the language models received. The script then checks
every file against data/checksums.json and the true persistence against data/truth.json, so the
samples are exactly those of the thesis. Needs the raw networks in data/raw (see config/networks.yaml).

  python scripts/samples.py            writes results/samples/
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from study.common import ARMS, DATA, DRAWS, OUT, REAL, digest, group, read_json, sample_id, sha, write_json  # noqa: E402
from study.data import network, twin  # noqa: E402
from study.sample import prompt, text  # noqa: E402
from study.sampling import draw, sizes  # noqa: E402
from study.walk import Walk  # noqa: E402


def networks():
    """The 12 real networks and their time-shuffled copies."""
    for key in REAL:
        g = network(key)
        yield g
        yield twin(g)


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
        print(g.key, 'done', flush=True)
    frozen = read_json(DATA/'checksums.json')
    if {p.name: sha(p) for p in (OUT/'samples').glob('*.json')} != frozen or truth != read_json(DATA/'truth.json'):
        raise SystemExit('the samples or the true values differ from the frozen ones')
    print(f'{len(frozen)} samples written to results/samples; identical to data/samples')


if __name__ == '__main__':
    main()
