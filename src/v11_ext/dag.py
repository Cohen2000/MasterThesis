"""Task graph of the v11 extension and its SLURM orchestration.

Tasks whose outputs exist are never submitted. Each (stage, replicate) becomes one
SLURM array; an array waits (afterok) for every array holding one of its inputs.
Qwen runs on H100s as soon as its bundle exists, concurrently with the ET work;
a second Qwen round (afterany) resumes only never-started requests.
"""
from collections import defaultdict
import json
import subprocess
import time
from main_experiment.common import TRAIN, read_json, write_json
from .core import ARMS, CFG, EXT, NEW_SOURCES, ROOT, WORK, Task, frozen_inputs
from .et import FOLDS, V11_FOLDS

POOL_GRAPHS = 400
QWEN_REQUESTS_PER_OBS = 6
# Expected minutes per task (dry-run estimate), from the v10 runs of the same code:
# prepare 3 min for 40 graphs, pool 3 min for 500 graphs, selection 1.5-5.5 min on 8 CPUs, fits < 30 s.
QUEUE_MINUTES = 3          # scheduling latency per dependency level
QWEN_MINUTES_PER_SHARD = 60


def build(replicates):
    frozen = frozen_inputs()
    tasks = {}

    def add(stage, name, params, deps=(), **resources):
        task = Task(stage, name, params, list(deps), **resources).finalize(frozen)
        tasks[name] = task
        return task

    sources = [add('source', f'source:{s}', {'source': s}, cpus=4, mem_gb=96, minutes=360,
                   estimate=25 if s == 'lkml_reply' else 10) for s in NEW_SOURCES]
    testset = add('testset', 'testset', {}, sources, cpus=2, mem_gb=32, minutes=60, estimate=4)
    add('qwen_bundle', 'qwen_bundle', {}, sources, cpus=1, mem_gb=8, minutes=30, estimate=1)
    chunk = CFG['pool_chunk']
    for k in range(replicates+1):
        real, pool = {}, []
        if k:
            real = {arm: [add('draw_real', f'draw_real:{k}:{s}:{arm}', {'k': k, 'source': s, 'arm': arm},
                              cpus=1, mem_gb=24, minutes=60, estimate=1.5) for s in TRAIN] for arm in ARMS}
            pool = [add('draw_pool', f'draw_pool:{k}:{i:03d}', {'k': k, 'first': i*chunk, 'last': (i+1)*chunk},
                        cpus=1, mem_gb=16, minutes=120, estimate=4) for i in range(POOL_GRAPHS//chunk)]
        for arm in ARMS:
            draws = real.get(arm, [])+pool
            for fold in FOLDS:
                select = None
                if not (k == 0 and fold in V11_FOLDS):
                    select = add('select', f'select:{k}:{arm}:{fold}', {'k': k, 'arm': arm, 'fold': fold},
                                 draws, cpus=8, mem_gb=16, minutes=120, estimate=5)
                add('train', f'train:{k}:{arm}:{fold}', {'k': k, 'arm': arm, 'fold': fold},
                    [testset, *([select] if select else []), *draws], cpus=4, mem_gb=16, minutes=30, estimate=1)
    trains = [t for t in tasks.values() if t.stage == 'train']
    add('report', 'report', {'replicates': replicates}, [*sources, testset, *trains, tasks['qwen_bundle']],
        cpus=2, mem_gb=32, minutes=60, estimate=5)
    return tasks


def arrays(tasks):
    """Pending tasks grouped into arrays by (stage, replicate), in dependency order."""
    groups = defaultdict(list)
    for t in tasks.values():
        if not t.done: groups[(t.stage, t.params.get('k', -1))].append(t)
    order = ['source', 'testset', 'qwen_bundle', 'draw_real', 'draw_pool', 'select', 'train', 'report']
    return sorted(groups.items(), key=lambda kv: (order.index(kv[0][0]), kv[0][1]))


def qwen_state():
    from .qwen import QWEN_DIR
    requests = QWEN_DIR/'mainexp/run/requests.jsonl'
    if not requests.exists():      # about 15 observations per source before drawing
        return {'installed': False, 'requests': 15*len(NEW_SOURCES)*QWEN_REQUESTS_PER_OBS, 'answers': 0}
    n = len(requests.read_text().splitlines())
    answered = len(list((QWEN_DIR/'mainexp/answers').glob('*_r*/*.json')))
    return {'installed': True, 'requests': n, 'answers': answered}


def estimate(tasks):
    pending = [t for t in tasks.values() if not t.done]
    cpu_hours = sum(t.estimate*t.cpus for t in pending)/60
    finish = {}

    def done_at(t):
        if t.name not in finish:
            start = max((done_at(d) for d in t.deps), default=0.)
            finish[t.name] = start + (0. if t.done else t.estimate+QUEUE_MINUTES)
        return finish[t.name]
    cpu_wall = max(map(done_at, tasks.values()))/60
    q = qwen_state()
    shards = CFG['cluster']['qwen_shards']
    qwen_left = q['requests'] == 0 or q['answers'] < q['requests']
    gpu_hours = shards*QWEN_MINUTES_PER_SHARD/60 if qwen_left else 0.
    qwen_wall = (done_at(tasks['qwen_bundle'])+QWEN_MINUTES_PER_SHARD+QUEUE_MINUTES)/60 if qwen_left else 0.
    return {'pending_tasks': len(pending), 'cpu_hours': round(cpu_hours, 1), 'gpu_hours': round(gpu_hours, 1),
            'cpu_critical_path_hours': round(cpu_wall, 2), 'qwen_path_hours': round(qwen_wall, 2),
            'wall_hours': round(max(cpu_wall, qwen_wall)+5/60, 2)}


def plan(replicates=None):
    """Build the DAG; fall back to the smaller replicate count if the wall estimate exceeds the limit."""
    replicates = replicates or CFG['et_replicates']
    tasks = build(replicates)
    est = estimate(tasks)
    if est['wall_hours'] > CFG['wall_limit_hours'] and replicates > CFG['et_replicates_fallback']:
        replicates = CFG['et_replicates_fallback']
        tasks = build(replicates); est = estimate(tasks)
    return replicates, tasks, est


def describe(tasks, est, replicates):
    lines = [f'v11 extension plan: {replicates} ET replicates (+ production replicate 0)']
    for (stage, k), group in arrays(tasks):
        t = group[0]
        lines.append(f'  array {stage:<12} k={k:<3} tasks={len(group):<4} cpus={max(x.cpus for x in group):<2} '
                     f'mem={max(x.mem_gb for x in group)}G time={max(x.minutes for x in group)}min')
    total = defaultdict(lambda: [0, 0])
    for t in tasks.values():
        total[t.stage][0] += 1; total[t.stage][1] += t.done
    lines.append('  done/total: '+', '.join(f'{s} {d}/{n}' for s, (n, d) in total.items()))
    q = qwen_state()
    lines.append(f'  qwen: bundle installed={q["installed"]} answers {q["answers"]}/{q["requests"]}')
    lines.append('  estimate: '+json.dumps(est))
    return '\n'.join(lines)


def sbatch(args, cwd=None):
    out = subprocess.run(['sbatch', '--parsable', *args], cwd=cwd, check=True, capture_output=True, text=True)
    return out.stdout.strip().split(';')[0]


def queued():
    out = subprocess.run(['squeue', '--me', '-h', '-o', '%i %j %T %M %R'], capture_output=True, text=True, check=True).stdout
    return [line for line in out.splitlines() if ' v11x_' in line]


def submit(replicates, tasks):
    if queued(): raise RuntimeError('v11x jobs are still queued or running; one chain at a time')
    stamp = time.strftime('%Y%m%dT%H%M%S')
    folder = WORK/'plans'/stamp
    logs = EXT/'logs'
    logs.mkdir(parents=True, exist_ok=True)
    job_of = {}          # task name -> job id of the array that produces it
    jobs = []
    venv = CFG['cluster']['venv']
    for (stage, k), group in arrays(tasks):
        plan_file = folder/f'{stage}_{k}.json'
        write_json(plan_file, {'replicates': replicates, 'tasks': [[t.name, t.key] for t in group]})
        upstream = sorted({job_of[d.name] for t in group for d in t.deps if d.name in job_of})
        args = [f'--job-name=v11x_{stage}_{k}', f'--array=0-{len(group)-1}', '--partition=cpu',
                f'--cpus-per-task={max(t.cpus for t in group)}', f'--mem={max(t.mem_gb for t in group)}G',
                f'--time={max(t.minutes for t in group)}', f'--output={logs}/%x_%A_%a.out', f'--chdir={ROOT}']
        if stage == 'report' and 'qwen' in job_of:
            upstream_any = [job_of['qwen']]
            args.append('--dependency=' + ','.join([*(f'afterok:{j}' for j in upstream),
                                                    *(f'afterany:{j}' for j in upstream_any)]))
        elif upstream:
            args.append('--dependency=afterok:'+':'.join(upstream))
        job = sbatch([*args, str(ROOT/'cluster/v11_ext_task.sbatch'), venv, str(plan_file)])
        for t in group: job_of[t.name] = job
        jobs.append({'array': f'{stage}_{k}', 'job': job, 'tasks': len(group), 'after': upstream})
        print('SUBMITTED', stage, k, job, len(group), flush=True)
        if stage == 'qwen_bundle':
            job_of['qwen'] = submit_qwen(job, logs, jobs)
    if 'qwen' not in job_of:
        q = qwen_state()
        if q['installed'] and q['answers'] < q['requests']:
            job_of['qwen'] = submit_qwen(None, logs, jobs)
    write_json(folder/'jobs.json', jobs)
    return jobs


def submit_qwen(bundle_job, logs, jobs):
    from .qwen import QWEN_DIR
    main = QWEN_DIR/'mainexp'
    (main/'logs').mkdir(parents=True, exist_ok=True)
    shards = CFG['cluster']['qwen_shards']
    exp = str(QWEN_DIR.relative_to(EXT.parent))
    common = [f'--array=0-{shards-1}', f'--chdir={main}', f'--output={logs}/%x_%A_%a.out']
    first = sbatch([*common, '--job-name=v11x_qwen_r1',
                    *([f'--dependency=afterok:{bundle_job}'] if bundle_job else []),
                    str(ROOT/'cluster/qwen_engine.sbatch'), exp, str(shards), str(CFG['cluster']['qwen_max_num_seqs']), 'all'])
    second = sbatch([*common, '--job-name=v11x_qwen_r2', f'--dependency=afterany:{first}',
                     str(ROOT/'cluster/qwen_engine.sbatch'), exp, str(shards), str(CFG['cluster']['qwen_max_num_seqs']), 'all'])
    jobs += [{'array': 'qwen_r1', 'job': first, 'tasks': shards}, {'array': 'qwen_r2', 'job': second, 'tasks': shards}]
    print('SUBMITTED qwen', first, second, flush=True)
    return second


def status():
    mine = queued()
    return '\n'.join(['queue:', *mine[:60], f'({len(mine)} array rows in queue)'])


def run_planned(plan_file, index):
    spec = read_json(plan_file)
    name, key = spec['tasks'][index]
    task = build(spec['replicates'])[name]
    if task.key != key: raise RuntimeError(f'{name}: code, config or inputs changed since submission')
    execute(task)


def run_named(name, replicates=None):
    execute(build(replicates or CFG['et_replicates'])[name])


def execute(task):
    from . import core, et, qwen, replicates, report, sources
    fn = {'source': sources.source_stage, 'testset': replicates.testset, 'draw_real': replicates.draw_real,
          'draw_pool': replicates.draw_pool, 'select': et.select, 'train': et.train,
          'qwen_bundle': qwen.bundle, 'report': report.report}[task.stage]
    core.run(task, fn)
