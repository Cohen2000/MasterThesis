#!/usr/bin/env python3
"""Copy verified unchanged graphs only; never reuse event-weighted calibration."""
import sys,shutil
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from main_experiment.common import ROOT,CURRENT_RUN,CURRENT_REVISION,PREVIOUS_RUN,PREVIOUS_REVISION,sha,read_json,write_json
from main_experiment.pool import pool_definition

def main():
    run=ROOT/CURRENT_RUN; rev=ROOT/CURRENT_REVISION
    if run.exists() or rev.exists(): raise ValueError('new empty destinations required')
    for name,h in read_json(PREVIOUS_RUN/'checksums.json').items(): assert sha(PREVIOUS_RUN/name)==h,name
    assert read_json(PREVIOUS_REVISION/'pool_definition.json')['graphs']==pool_definition()['graphs']
    copied={}
    for source in (PREVIOUS_RUN/'graphs').rglob('*'):
        if not source.is_file(): continue
        rel=source.relative_to(PREVIOUS_RUN); dest=run/rel
        dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(source,dest)
        assert sha(source)==sha(dest)
        copied[str(rel)]={'source':str(source.relative_to(ROOT)),'sha256':sha(dest)}
    write_json(run/'reused_graphs.json',copied)
    print(f'{len(copied)} graph files verified; zero walk checkpoints reused')

if __name__=='__main__': main()
