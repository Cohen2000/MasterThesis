#!/usr/bin/env python3
"""Publish completed Qwen evidence separately from the pre-inference offline seal."""
import csv,json,sys,shutil
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from main_experiment.common import *
run=ROOT/CURRENT_RUN;rev=ROOT/CURRENT_REVISION;qwen=run.with_name(run.name+'_qwen')
audit=read_json(rev/'independent_audit.json');assert audit['qwen']['complete']
out=ROOT/'docs/results/panel888_qwen_20260921';out.mkdir(parents=True,exist_ok=True)
for name,source in {'summary.csv':qwen/'evaluation/summary.csv','source_results.csv':qwen/'evaluation/source_results.csv',
    'paired_summary.csv':qwen/'paired_control/paired_summary.csv','paired_sources.csv':qwen/'paired_control/paired_sources.csv',
    'synthetic_within_replicate_contrasts.csv':qwen/'paired_control/synthetic_within_replicate_contrasts.csv',
    'independent_audit.json':rev/'independent_audit.json'}.items():
    dest=out/name
    if dest.exists() and sha(dest)!=sha(source):raise ValueError('refusing to overwrite completed result evidence')
    shutil.copy2(source,dest)
rows=list(csv.DictReader((out/'summary.csv').open()))
qrows=[r for r in rows if r['config_id'].startswith('qwen')];assert qrows and all(r['complete']=='True' for r in qrows)
api=read_json(run.with_name(run.name+'_api')/'ledger.json');assert not api['requests'] and not api['batches']
raw=[json.loads(x) for x in (qwen/'responses.jsonl').read_text().splitlines()]
report={'design_version':DESIGN_VERSION,'qwen':audit['qwen'],'sol_deepseek_started':False,
    'technical_errors':sum(bool(r.get('technical_error')) for r in raw),'output_limit_hits':sum(bool(r.get('limit_hit')) for r in raw),
    'complete_qwen_cells':len(qrows),'offline_freeze_sha256':sha(ROOT/'docs/results/panel888_20260921/FREEZE.json'),
    'checksums':{str(p.relative_to(ROOT)):sha(p) for p in sorted(qwen.rglob('*')) if p.is_file()}}
write_json(out/'RESULT_FREEZE.json',report)
text=f'''# Final Qwen results: {DESIGN_VERSION}

There is one final panel: 8 real, 8 matched temporal surrogates and 8 synthetic
instances. Earlier results are development provenance. Qwen responses were all
newly generated; no earlier answer was reused.

Completed {len(raw)} Qwen requests; {audit['qwen']['valid']} valid,
{audit['qwen']['invalid']} invalid. Technical errors: {report['technical_errors']};
output limits: {report['output_limit_hits']}. Accuracy is conditional on valid
answers, with equal-source weighting within each evidence block/condition.
Sol and DeepSeek remain unstarted, with identical prepared observations/repeats.

Ordinary accuracy and validity: [summary](results/panel888_qwen_20260921/summary.csv).
Paired temporal control: [paired summary](results/panel888_qwen_20260921/paired_summary.csv).
Synthetic contrasts retain within-r pairing:
[contrasts](results/panel888_qwen_20260921/synthetic_within_replicate_contrasts.csv).
Offline sensitivities and methodological qualifications are retained in the
[offline freeze](results/panel888_20260921/FREEZE.json) and protocol.
'''
Path('docs/RESULTS_PANEL888.md').write_text(text)
print(json.dumps({k:v for k,v in report.items() if k!='checksums'},indent=2))
