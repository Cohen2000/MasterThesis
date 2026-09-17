#!/usr/bin/env python3
"""Build the durable archive of a Qwen run (main run, or the H revision with --hrecent5).

The scratch workspace expires, so everything needed to re-read, re-check or
re-evaluate this run is collected once, hashed, and verified by reading it back.
Included: the raw answers, the request manifest, the observation blocks, the
prompts as the tokenizer actually rendered them for each mode, the model,
tokenizer and chat-template hashes, the pinned environment and the job logs.
"""
import hashlib, json, os, shutil, sys, tarfile, time
from pathlib import Path

WS = Path('/pfs/work9/workspace/scratch/tu_zxokn55-llm_pilot')
MODEL = WS / 'models/Qwen3.6-35B-A3B'
# --hrecent5: archive the H-revision generation. Only its own (new H) prompts are
# rendered; the reused R/S/B answers are already in the main-run archive.
HREC = '--hrecent5' in sys.argv
ARGS = [a for a in sys.argv[1:] if a != '--hrecent5']
EXP = WS / ('hrecent5/mainexp' if HREC else 'mainexp')
RUN = EXP / 'run'
SRC = WS / ('hrecent5/src' if HREC else 'src')
OUT = Path(ARGS[0] if ARGS else EXP / 'archive')
MODES = {'thinking': True, 'nonthinking': False}


def sha(p, chunk=1 << 20):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(chunk), b''):
            h.update(b)
    return h.hexdigest()


def main():
    sys.path.insert(0, str(SRC))
    from transformers import AutoTokenizer
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.time()

    # 1. the prompts as actually rendered, per mode
    tok = AutoTokenizer.from_pretrained(str(MODEL))
    rendered = OUT / 'rendered_prompts.jsonl'
    n = 0
    with open(rendered, 'w') as out:
        for f in sorted((RUN / 'observations/sample').glob('*.json')):
            if HREC and '__H-recent5__' not in f.name:
                continue
            d = json.loads(f.read_text())
            for mode, think in MODES.items():
                text = tok.apply_chat_template(d['messages'], tokenize=False,
                                               add_generation_prompt=True,
                                               enable_thinking=think)
                ids = tok(text, add_special_tokens=False)['input_ids']
                out.write(json.dumps({
                    'observation_id': d['id'], 'mode': mode,
                    'prompt_sha256': d['prompt_sha256'],
                    'rendered_sha256': hashlib.sha256(text.encode()).hexdigest(),
                    'input_tokens': len(ids), 'rendered': text}, sort_keys=True) + '\n')
                n += 1
    print(f'rendered {n} prompts', flush=True)

    # 2. model, tokenizer and template identity
    ident = {'model_dir': str(MODEL), 'revision_pinned':
             '995ad96eacd98c81ed38be0c5b274b04031597b0'}
    for name in ('config.json', 'generation_config.json', 'tokenizer.json',
                 'tokenizer_config.json', 'chat_template.jinja',
                 'model.safetensors.index.json'):
        p = MODEL / name
        if p.exists():
            ident[name] = sha(p)
    ident['safetensors'] = {p.name: sha(p) for p in sorted(MODEL.glob('*.safetensors'))}
    (OUT / 'model_identity.json').write_text(json.dumps(ident, indent=1, sort_keys=True) + '\n')
    print(f'hashed {len(ident["safetensors"])} weight shards', flush=True)

    # 3. everything else, copied verbatim
    shutil.copytree(EXP / 'answers', OUT / 'answers', dirs_exist_ok=True)
    shutil.copy2(RUN / 'requests.jsonl', OUT / 'requests.jsonl')
    shutil.copytree(RUN / 'observations', OUT / 'observations', dirs_exist_ok=True)
    shutil.copy2(WS / 'mainexp/requirements.pinned.txt', OUT / 'requirements.pinned.txt')
    logs = OUT / 'logs'; logs.mkdir(exist_ok=True)
    for f in (EXP / 'logs').glob('*.out'):
        shutil.copy2(f, logs / f.name)
    for extra in ('probe_result_tp1.json', 'qwen_files.txt', 'run_qwen_engine.py', 'qwen_hrecent5.sbatch'):
        p = EXP / extra
        if p.exists():
            shutil.copy2(p, OUT / extra)

    # 4. checksum every file, then read it all back and confirm
    files = sorted(p for p in OUT.rglob('*') if p.is_file() and p.name != 'CHECKSUMS.json')
    sums = {str(p.relative_to(OUT)): sha(p) for p in files}
    (OUT / 'CHECKSUMS.json').write_text(json.dumps(sums, indent=1, sort_keys=True) + '\n')
    bad = [k for k, v in sums.items() if sha(OUT / k) != v]
    total = sum((OUT / k).stat().st_size for k in sums)
    report = {'files': len(sums), 'bytes': total,
              'answers': len(list((OUT / 'answers').rglob('*.json'))),
              'rendered_prompts': n, 'readback_mismatches': bad,
              'seconds': time.time() - started}
    (OUT / 'ARCHIVE_REPORT.json').write_text(json.dumps(report, indent=1, sort_keys=True) + '\n')
    print('ARCHIVE ' + json.dumps(report))
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
