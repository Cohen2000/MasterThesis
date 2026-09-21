#!/usr/bin/env python3
"""Copy verified unchanged graphs, calibration, and models into the final run."""
import sys, shutil
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from main_experiment.common import ROOT, CURRENT_RUN, PREVIOUS_RUN, sha, read_json, write_json

def main():
    run = ROOT / CURRENT_RUN
    prev = PREVIOUS_RUN
    if run.exists():
        raise ValueError(f'Destination {run} already exists')
    for name, h in read_json(prev / 'checksums.json').items():
        assert sha(prev / name) == h, name
    copied = {}
    for folder in ('graphs', 'calibration', 'models'):
        for source in (prev / folder).rglob('*'):
            if not source.is_file():
                continue
            rel = source.relative_to(prev)
            dest = run / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dest)
            assert sha(source) == sha(dest)
            copied[str(rel)] = {'source': str(source.relative_to(ROOT)), 'sha256': sha(dest)}
    write_json(run / 'reused_artifacts.json', copied)
    print(f'{len(copied)} artifact files (graphs, calibration, models) verified and reused from {prev.name}')

if __name__ == '__main__':
    main()
