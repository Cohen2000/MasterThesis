#!/usr/bin/env python3
"""Seed a NEW run with verified unchanged graphs and walk-only checkpoints."""
import sys, shutil
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from main_experiment.common import ROOT,CURRENT_RUN,CURRENT_REVISION,PREVIOUS_RUN,PREVIOUS_REVISION,sha,read_json,write_json
from main_experiment.pool import pool_definition


def main():
    run=ROOT/CURRENT_RUN; rev=ROOT/CURRENT_REVISION
    if run.exists() or rev.exists(): raise ValueError('new destinations required')
    for name,h in read_json(PREVIOUS_RUN/'checksums.json').items():
        if sha(PREVIOUS_RUN/name)!=h: raise ValueError(f'old offline checksum {name}')
    assert read_json(PREVIOUS_REVISION/'pool_definition.json')==pool_definition()
    copied={}
    def copy(source,dest):
        dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(source,dest)
        assert sha(source)==sha(dest)
        copied[str(dest.relative_to(ROOT))]={'source':str(source.relative_to(ROOT)),'sha256':sha(dest)}
    for p in (PREVIOUS_RUN/'graphs').rglob('*'):
        if p.is_file(): copy(p,run/p.relative_to(PREVIOUS_RUN))
    for old,new in ((PREVIOUS_RUN/'calibration',run/'calibration'),
                    (PREVIOUS_REVISION/'pool/calibration',rev/'pool/calibration')):
        for p in old.rglob('*'):
            if p.is_file() and (p.name in ('calibrated.json','timing.json') or p.name.startswith(('prefix_','validation_'))):
                copy(p,new/p.relative_to(old))
    write_json(run/'reused_offline_inputs.json',copied)
    print(f'verified/copied {len(copied)} immutable inputs; no models or H observations reused')


if __name__=='__main__': main()
