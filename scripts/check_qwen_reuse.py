#!/usr/bin/env python3
"""Which stored Qwen answers may be reused under the current design, and why.

An answer is reusable only if everything that determined it is unchanged:
the observation block, the rendered prompt messages (by hash), the request id,
the request seed and the complete Qwen payload in the manifest, the stored
answer's own prompt hash and seed, and the per-request sampling configuration of
the runner. Arm H answers of the previous design are never reusable: the H prompt
changed by design, and the check establishes that instead of assuming it.
"""
import argparse, json, sys, hashlib, importlib.util
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from main_experiment.common import read_json, write_json, QWEN_CONFIGS


# Payload fields that determine a Qwen generation. The remaining manifest fields
# (response_format, stream, stream_options in the old manifest; executed_* and
# planned_but_not_used in the new one) only describe the transport. Commit 9c5b04b
# corrected them to what was executed; run_qwen_batch.py never read the payload,
# it takes messages from the observation file and sampling from its MODES table.
GENERATION_KEYS = ('model', 'messages', 'max_tokens', 'temperature', 'top_p', 'top_k', 'min_p',
                   'presence_penalty', 'repetition_penalty', 'chat_template_kwargs', 'seed')


def load(path):
    return {r['id']: r for r in map(json.loads, Path(path).read_text().splitlines())}


def module(path):
    spec = importlib.util.spec_from_file_location(Path(path).stem, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--old-run', required=True)
    ap.add_argument('--new-run', required=True)
    ap.add_argument('--old-answers', required=True, help='directory with <mode>_r<n>/<id>.json')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    old, new = load(Path(a.old_run) / 'requests.jsonl'), load(Path(a.new_run) / 'requests.jsonl')
    here = Path(__file__).resolve().parent
    batch, engine = module(here / 'run_qwen_batch.py'), module(here / 'run_qwen_engine.py')
    same_sampling = (batch.MODES == engine.MODES and batch.TOP_K == engine.TOP_K
                     and batch.PRESENCE_PENALTY == engine.PRESENCE_PENALTY)
    report = {'same_sampling_configuration': same_sampling, 'reusable': 0, 'not_reusable': {},
              'new_requests_needing_generation': 0, 'by_arm': {}}
    reuse = []
    answers = {}
    for f in Path(a.old_answers).rglob('*.json'):
        d = json.loads(f.read_text()); answers[d['id']] = d
    old_obs, new_obs, doc = {}, {}, {}
    uses_payload = 'payload' in (here / 'run_qwen_batch.py').read_text().split('def load_requests', 1)[1].split('def main', 1)[0]
    report['old_runner_reads_payload'] = uses_payload
    if uses_payload: raise SystemExit('run_qwen_batch.py reads the payload; documentation fields would matter')
    for rid, r in sorted(new.items()):
        if r['config_id'] not in QWEN_CONFIGS: continue
        reasons = []
        o = old.get(rid)
        if o is None: reasons.append('no_request_with_this_id_in_previous_design')
        else:
            if o['arm'] != r['arm']: reasons.append('arm')
            if o['prompt_sha256'] != r['prompt_sha256']: reasons.append('prompt')
            if o['seed'] != r['seed']: reasons.append('seed')
            if any(o['payload'].get(k) != r['payload'].get(k) for k in GENERATION_KEYS):
                reasons.append('payload')
            for k in set(o['payload']) | set(r['payload']):
                if k not in GENERATION_KEYS and o['payload'].get(k) != r['payload'].get(k):
                    doc[k] = doc.get(k, 0) + 1
            if o['observation_id'] not in old_obs:
                old_obs[o['observation_id']] = read_json(Path(a.old_run) / 'observations/sample' / f"{o['observation_id']}.json")
            if r['observation_id'] not in new_obs:
                new_obs[r['observation_id']] = read_json(Path(a.new_run) / 'observations/sample' / f"{r['observation_id']}.json")
            no, nn = old_obs[o['observation_id']], new_obs[r['observation_id']]
            if no['block'] != nn['block'] or no['messages'] != nn['messages']: reasons.append('observation')
            ans = answers.get(rid)
            if ans is None: reasons.append('no_stored_answer')
            elif ans.get('prompt_sha256') != r['prompt_sha256'] or ans.get('seed') != r['seed']:
                reasons.append('stored_answer_metadata')
        if not same_sampling: reasons.append('generation_configuration')
        arm = report['by_arm'].setdefault(r['arm'], {'reusable': 0, 'generate': 0})
        if reasons:
            report['new_requests_needing_generation'] += 1; arm['generate'] += 1
            key = ','.join(reasons); report['not_reusable'][key] = report['not_reusable'].get(key, 0) + 1
        else:
            report['reusable'] += 1; arm['reusable'] += 1; reuse.append(rid)
    old_h = sorted(k for k, v in old.items() if v['arm'] == 'H' and v['config_id'] in QWEN_CONFIGS)
    report['documentation_only_payload_differences'] = doc
    report['generation_keys_compared'] = list(GENERATION_KEYS)
    report['previous_H_requests_excluded'] = len(old_h)
    report['previous_H_ids_colliding_with_new_ids'] = len(set(old_h) & set(new))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    write_json(a.out, {**report, 'reusable_ids_sha256': hashlib.sha256('\n'.join(reuse).encode()).hexdigest()})
    Path(str(a.out) + '.reusable_ids.txt').write_text('\n'.join(reuse) + '\n')
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
