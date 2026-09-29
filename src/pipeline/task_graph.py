"""Task graph of stage 2 and its SLURM orchestration.

Tasks whose outputs exist are never submitted. Each (stage, replicate) becomes one
SLURM array; an array waits (afterok) for every array holding one of its inputs.
Qwen runs on H100s as soon as its bundle exists, concurrently with the ET work;
a second Qwen round (afterany) resumes only never-started requests.
"""
# Plain-words overview:
# - build() lists every task of the cluster pipeline (download/prepare a source graph,
#   draw observations, fit ExtraTrees, run Qwen, write the report, ...) together with
#   the tasks it depends on. Each task gets a key = hash of its code, config, inputs
#   and dependencies, so a task counts as done only if its output was produced by
#   exactly this version of everything.
# - arrays() groups the unfinished tasks into SLURM job arrays, one per (stage, replicate).
# - submit() sends those arrays to SLURM, chained so that an array only starts after
#   the arrays producing its inputs have finished successfully.
# - execute() is what runs inside a job: it calls the Python function of the stage.
from collections import defaultdict
import json
import subprocess
import time
from study.common import TRAINING_SOURCES, read_json, write_json
from .core import ARMS, CFG, PIPELINE_DIR, MLE_ANCHOR_ARMS, STAGE2_SOURCES, RADOSLAW, ROOT, WORK, Task, frozen_inputs
from .extratrees_fits import FOLDS, STAGE1_FOLDS
from . import surrogates_and_checks as panel
from .surrogates_and_checks import REPORTED_ARMS, SURROGATES

# Number of synthetic training graphs in the ExtraTrees training pool.
POOL_GRAPHS = 400
# Qwen requests per observation: 2 modes (thinking / non-thinking) x 3 repeats.
QWEN_REQUESTS_PER_OBS = 6
# Expected minutes per task (dry-run estimate), from stage-1 runs of the same code:
# prepare 3 min for 40 graphs, pool 3 min for 500 graphs, selection 1.5-5.5 min on 8 CPUs, fits < 30 s.
QUEUE_MINUTES = 3          # scheduling latency per dependency level
QWEN_MINUTES_PER_SHARD = 60


# Build the whole task graph. 'replicates' = number of extra ExtraTrees fits besides the
# production fit (k = 0); the extra fits only measure how much ET results vary.
def build(replicates):
    frozen = frozen_inputs()
    tasks = {}

    # Create one task, compute its key and register it under its unique name.
    def add(stage, name, params, deps=(), **resources):
        task = Task(stage, name, params, list(deps), **resources).finalize(frozen)
        tasks[name] = task
        return task

    # Step 1: turn each raw source dataset into a prepared temporal graph.
    sources = [add('source', f'source:{s}', {'source': s}, cpus=4, mem_gb=96, minutes=360,
                   estimate=25 if s == 'lkml_reply' else 10) for s in STAGE2_SOURCES]
    # Step 2: draw the test observations (arms x 3 draws per graph) the methods are scored on.
    testset = add('testset', 'testset', {}, sources, cpus=2, mem_gb=32, minutes=60, estimate=4)
    add('qwen_bundle', 'qwen_bundle', {}, sources, cpus=1, mem_gb=8, minutes=30, estimate=1)
    anchor = {arm: add('anchor0', f'anchor0:{arm}', {'arm': arm}, cpus=1, mem_gb=16, minutes=120, estimate=5)
              for arm in MLE_ANCHOR_ARMS}
    # Step 3: ExtraTrees. For every replicate k: draw training observations, choose the
    # hyper-parameters (select) and fit the forest (train), per arm and per cross-validation fold.
    # k = 0 reuses the frozen production draws where they exist.
    chunk = CFG['pool_chunk']
    for k in range(replicates+1):
        real, pool = {}, []
        if k:
            real = {arm: [add('draw_real', f'draw_real:{k}:{s}:{arm}', {'k': k, 'source': s, 'arm': arm},
                              cpus=1, mem_gb=24, minutes=60, estimate=1.5) for s in TRAINING_SOURCES] for arm in ARMS}
            pool = [add('draw_pool', f'draw_pool:{k}:{i:03d}', {'k': k, 'first': i*chunk, 'last': (i+1)*chunk},
                        cpus=1, mem_gb=16, minutes=120, estimate=4) for i in range(POOL_GRAPHS//chunk)]
        for arm in ARMS:
            draws = real.get(arm, [])+pool+([anchor[arm]] if k == 0 and arm in anchor else [])
            for fold in FOLDS:
                select = None
                if not (k == 0 and fold in STAGE1_FOLDS and arm not in MLE_ANCHOR_ARMS):
                    select = add('select', f'select:{k}:{arm}:{fold}', {'k': k, 'arm': arm, 'fold': fold},
                                 draws, cpus=8, mem_gb=16, minutes=120, estimate=5)
                add('train', f'train:{k}:{arm}:{fold}', {'k': k, 'arm': arm, 'fold': fold},
                    [testset, *([select] if select else []), *draws], cpus=4, mem_gb=16, minutes=30, estimate=1)
    # 12 real sources, 12 surrogates, 8 synthetic graphs; arms R/S/H/B (S_obs is drawn for reproducibility only).
    # Step 4: one surrogate per real source: the same events with their timestamps shuffled
    # (every pair keeps its number of events, only the timing is randomised), plus its
    # test set and the ExtraTrees fits that score the surrogates.
    sur = [add('surrogate', f'surrogate:{s}', {'source': s}, [tasks[f'source:{s}']], cpus=4, mem_gb=96,
               minutes=360, estimate=25 if s == 'lkml_reply' else 10) for s in STAGE2_SOURCES]
    testset_sur = add('testset_sur', 'testset_sur', {}, sur, cpus=1, mem_gb=8, minutes=30, estimate=1)
    for k in range(replicates+1):
        for arm in REPORTED_ARMS:
            for fold in ('synthetic', RADOSLAW):
                base = tasks[f'train:{k}:{arm}:{fold}']
                add('train_sur', f'train_sur:{k}:{arm}:{fold}', {'k': k, 'arm': arm, 'fold': fold},
                    [*base.deps, base, testset_sur], cpus=4, mem_gb=16, minutes=30, estimate=1)
    # Step 5: Qwen request bundle for the surrogates, frozen API observations, history and walk diagnostics.
    add('qwen_sur', 'qwen_sur', {}, sur, cpus=1, mem_gb=8, minutes=30, estimate=1)
    add('api_freeze', 'api_freeze', {}, [*sources, *sur], cpus=1, mem_gb=8, minutes=30, estimate=1)
    add('history', 'history', {}, [*sources, *sur], cpus=2, mem_gb=64, minutes=120, estimate=5)
    for key in (*STAGE2_SOURCES, *SURROGATES):
        parent = panel.family(key)
        add('walkdiag', f'walkdiag:{key}', {'graph': key},
            [tasks[f'source:{parent}'], *([tasks[f'surrogate:{parent}']] if key in SURROGATES else [])],
            cpus=2, mem_gb=64, minutes=360, estimate=30 if parent == 'lkml_reply' else 5)
    # Step 6: the final report waits for everything above.
    report_deps = [t for t in tasks.values() if t.stage in ('source', 'surrogate', 'testset', 'testset_sur', 'train_sur',
                                                            'qwen_bundle', 'qwen_sur', 'api_freeze', 'history', 'walkdiag')
                   or (t.stage == 'train' and t.params['arm'] in REPORTED_ARMS)]
    add('report', 'report', {'replicates': replicates}, report_deps, cpus=2, mem_gb=32, minutes=60, estimate=5)
    return tasks


# Unfinished tasks, grouped per (stage, replicate) and sorted so that producers come first.
def arrays(tasks):
    """Pending tasks grouped into arrays by (stage, replicate), in dependency order."""
    groups = defaultdict(list)
    for t in tasks.values():
        if not t.done: groups[(t.stage, t.params.get('k', -1))].append(t)
    order = ['source', 'surrogate', 'testset', 'testset_sur', 'qwen_bundle', 'qwen_sur', 'api_freeze', 'history',
             'walkdiag', 'anchor0', 'draw_real', 'draw_pool', 'select', 'train', 'train_sur', 'report']
    return sorted(groups.items(), key=lambda kv: (order.index(kv[0][0]), kv[0][1]))


# Qwen bundle folders: the main test set and the surrogates.
def bundles():
    from .qwen import QWEN_DIR
    return {'qwen_bundle': QWEN_DIR, 'qwen_sur': panel.QWEN_SUR_DIR}


# How many Qwen requests exist in a bundle and how many already have an answer.
def qwen_state(folder=None):
    from .qwen import QWEN_DIR
    folder = folder or QWEN_DIR
    requests = folder/'mainexp/run/requests.jsonl'
    if not requests.exists():      # about 12-15 observations per graph before drawing
        return {'installed': False, 'requests': 12*len(STAGE2_SOURCES)*QWEN_REQUESTS_PER_OBS, 'answers': 0}
    n = len(requests.read_text().splitlines())
    answered = len(list((folder/'mainexp/answers').glob('*_r*/*.json')))
    return {'installed': True, 'requests': n, 'answers': answered}


# Rough CPU/GPU hours and wall time of the remaining work (only for the dry-run printout).
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
    shards = CFG['cluster']['qwen_shards']
    left = [name for name, folder in bundles().items() if qwen_state(folder)['answers'] < qwen_state(folder)['requests']]
    gpu_hours = len(left)*shards*QWEN_MINUTES_PER_SHARD/60
    qwen_wall = max([(done_at(tasks[b])+QWEN_MINUTES_PER_SHARD+QUEUE_MINUTES)/60 for b in left], default=0.)
    return {'pending_tasks': len(pending), 'cpu_hours': round(cpu_hours, 1), 'gpu_hours': round(gpu_hours, 1),
            'cpu_critical_path_hours': round(cpu_wall, 2), 'qwen_path_hours': round(qwen_wall, 2),
            'wall_hours': round(max(cpu_wall, qwen_wall)+5/60, 2)}


# Build the graph with the configured number of ET replicates; use fewer if it would take too long.
def plan(replicates=None):
    """Build the DAG; fall back to the smaller replicate count if the wall estimate exceeds the limit."""
    replicates = replicates or CFG['et_replicates']
    tasks = build(replicates)
    est = estimate(tasks)
    if est['wall_hours'] > CFG['wall_limit_hours'] and replicates > CFG['et_replicates_fallback']:
        replicates = CFG['et_replicates_fallback']
        tasks = build(replicates); est = estimate(tasks)
    return replicates, tasks, est


# Text summary of the plan: one line per job array, progress per stage, Qwen progress.
def describe(tasks, est, replicates):
    lines = [f'stage-2 plan: {replicates} ET replicates (+ production replicate 0)']
    for (stage, k), group in arrays(tasks):
        t = group[0]
        lines.append(f'  array {stage:<12} k={k:<3} tasks={len(group):<4} cpus={max(x.cpus for x in group):<2} '
                     f'mem={max(x.mem_gb for x in group)}G time={max(x.minutes for x in group)}min')
    total = defaultdict(lambda: [0, 0])
    for t in tasks.values():
        total[t.stage][0] += 1; total[t.stage][1] += t.done
    lines.append('  done/total: '+', '.join(f'{s} {d}/{n}' for s, (n, d) in total.items()))
    for name, folder in bundles().items():
        q = qwen_state(folder)
        lines.append(f'  {name}: installed={q["installed"]} answers {q["answers"]}/{q["requests"]}')
    lines.append('  estimate: '+json.dumps(est))
    return '\n'.join(lines)


# Submit one SLURM job and return its job id.
def sbatch(args, cwd=None):
    out = subprocess.run(['sbatch', '--parsable', *args], cwd=cwd, check=True, capture_output=True, text=True)
    return out.stdout.strip().split(';')[0]


# Our pipeline jobs (names start with 'pipe_') that are still queued or running.
def queued():
    out = subprocess.run(['squeue', '--me', '-h', '-o', '%i %j %T %M %R'], capture_output=True, text=True, check=True).stdout
    return [line for line in out.splitlines() if ' pipe_' in line]


# Submit all unfinished tasks as chained SLURM arrays. Each array gets a plan file listing
# its tasks and their keys; a job refuses to run if the code changed after submission.
def submit(replicates, tasks, allow_queued=False):
    """Submit every pending task. With allow_queued, the caller asserts that no queued
    v11x array produces any pending task (e.g. after cancelling a failed branch)."""
    if queued() and not allow_queued: raise RuntimeError('v11x jobs are still queued or running; one chain at a time')
    # Arrays still queued keep producing their tasks; new arrays depend on them instead of duplicating them.
    live = {line.split()[1]: line.split()[0].split('_')[0] for line in queued()}
    stamp = time.strftime('%Y%m%dT%H%M%S')
    folder = WORK/'plans'/stamp
    logs = PIPELINE_DIR/'logs'
    logs.mkdir(parents=True, exist_ok=True)
    job_of = {}          # task name -> job id of the array that produces it
    jobs = []
    venv = CFG['cluster']['venv']
    # Qwen whose bundle already exists starts now, before any array that must wait for it.
    for name, qdir in bundles().items():
        q = qwen_state(qdir)
        if f'pipe_{name}_r2' in live:
            job_of[f'qwen:{name}'] = live[f'pipe_{name}_r2']
        elif tasks[name].done and q['answers'] < q['requests']:
            job_of[f'qwen:{name}'] = submit_qwen(None, logs, jobs, name, qdir)
    for (stage, k), group in arrays(tasks):
        if f'pipe_{stage}_{k}' in live:
            for t in group: job_of[t.name] = live[f'pipe_{stage}_{k}']
            print('ATTACHED', stage, k, live[f'pipe_{stage}_{k}'], flush=True)
            continue
        plan_file = folder/f'{stage}_{k}.json'
        write_json(plan_file, {'replicates': replicates, 'tasks': [[t.name, t.key] for t in group]})
        upstream = sorted({job_of[d.name] for t in group for d in t.deps if d.name in job_of})
        args = [f'--job-name=pipe_{stage}_{k}', f'--array=0-{len(group)-1}', '--partition=cpu',
                f'--cpus-per-task={max(t.cpus for t in group)}', f'--mem={max(t.mem_gb for t in group)}G',
                f'--time={max(t.minutes for t in group)}', f'--output={logs}/%x_%A_%a.out', f'--chdir={ROOT}']
        upstream_any = [j for n, j in job_of.items() if n.startswith('qwen:')]
        if stage == 'report' and upstream_any:
            args.append('--dependency=' + ','.join([*(f'afterok:{j}' for j in upstream),
                                                    *(f'afterany:{j}' for j in upstream_any)]))
        elif upstream:
            args.append('--dependency=afterok:'+':'.join(upstream))
        job = sbatch([*args, str(ROOT/'cluster/pipeline_task.sbatch'), venv, str(plan_file)])
        for t in group: job_of[t.name] = job
        jobs.append({'array': f'{stage}_{k}', 'job': job, 'tasks': len(group), 'after': upstream})
        print('SUBMITTED', stage, k, job, len(group), flush=True)
        if stage in bundles() and f'qwen:{stage}' not in job_of:
            q = qwen_state(bundles()[stage])
            if not (q['installed'] and q['answers'] >= q['requests']):
                job_of[f'qwen:{stage}'] = submit_qwen(job, logs, jobs, stage, bundles()[stage])
    write_json(folder/'jobs.json', jobs)
    return jobs


# Qwen on the GPU partition, submitted twice: the second round (afterany) only picks up
# requests that never started in the first round, e.g. because a shard ran out of time.
def submit_qwen(bundle_job, logs, jobs, name, qdir):
    main = qdir/'mainexp'
    (main/'logs').mkdir(parents=True, exist_ok=True)
    shards = CFG['cluster']['qwen_shards']
    exp = str(qdir.relative_to(PIPELINE_DIR.parent))
    common = [f'--array=0-{shards-1}', f'--chdir={main}', f'--output={logs}/%x_%A_%a.out']
    first = sbatch([*common, f'--job-name=pipe_{name}_r1',
                    *([f'--dependency=afterok:{bundle_job}'] if bundle_job else []),
                    str(ROOT/'cluster/qwen_engine.sbatch'), exp, str(shards), str(CFG['cluster']['qwen_max_num_seqs']), 'all'])
    second = sbatch([*common, f'--job-name=pipe_{name}_r2', f'--dependency=afterany:{first}',
                     str(ROOT/'cluster/qwen_engine.sbatch'), exp, str(shards), str(CFG['cluster']['qwen_max_num_seqs']), 'all'])
    jobs += [{'array': f'{name}_r1', 'job': first, 'tasks': shards}, {'array': f'{name}_r2', 'job': second, 'tasks': shards}]
    print('SUBMITTED', name, 'GPU', first, second, flush=True)
    return second


# First 60 lines of our SLURM queue.
def status():
    mine = queued()
    return '\n'.join(['queue:', *mine[:60], f'({len(mine)} array rows in queue)'])


# Inside an array job: run task number 'index' of the plan, after checking its key still matches.
def run_planned(plan_file, index):
    spec = read_json(plan_file)
    name, key = spec['tasks'][index]
    task = build(spec['replicates'])[name]
    if task.key != key: raise RuntimeError(f'{name}: code, config or inputs changed since submission')
    execute(task)


# Run one task by name in the current process (used for smoke tests).
def run_named(name, replicates=None):
    execute(build(replicates or CFG['et_replicates'])[name])


# Map each stage name to the function that does the work; core.run writes the outputs
# into a fresh folder named after the task key.
def execute(task):
    from . import core, qwen, report, real_networks, training_draws
    from . import extratrees_fits as et
    fn = {'source': real_networks.source_stage, 'testset': training_draws.testset, 'draw_real': training_draws.draw_real,
          'draw_pool': training_draws.draw_pool, 'select': et.select, 'train': et.train,
          'qwen_bundle': qwen.bundle, 'report': report.report, 'anchor0': training_draws.anchor0,
          'surrogate': panel.surrogate, 'testset_sur': panel.testset_sur, 'train_sur': panel.train_sur,
          'qwen_sur': panel.qwen_sur, 'api_freeze': panel.api_freeze, 'history': panel.history,
          'walkdiag': panel.walkdiag}[task.stage]
    core.run(task, fn)
