# V2 Linux laptop run

Assumed project directory:

```text
~/Dokumente/MasterArbeit_benchmark_v2
```

## 1. Python environment

```bash
cd "$HOME/Dokumente/MasterArbeit_benchmark_v2"
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 2. Real data

Copy the already downloaded v1 files:

```bash
mkdir -p data/raw
cp -av "$HOME/Dokumente/MasterArbeit_benchmark_v1/data/raw/." data/raw/
```

Recommended optional SNAP additions:

```bash
bash scripts/download_v2_snap_data.sh
python src/check_real_data.py --preset v2
```

Six recognized files are sufficient. Missing optional sources are listed but do
not stop the run once the configured minimum is met.

## 3. Smoke

```bash
bash run_benchmark_v2_smoke.sh 2>&1 | tee v2_smoke.log
```

Expected final line:

```text
V2 SMOKE OK: .../results/benchmark_v2_smoke/SCREEN_SUMMARY.md
```

## 4. Full resumable run

```bash
nohup systemd-inhibit \
  --what=sleep:idle \
  --why="Masterarbeit benchmark v2" \
  env SHARDS=12 JOBS=1 RESET=0 \
  bash run_benchmark_v2_local.sh \
  > v2_nohup.log 2>&1 &

echo $! | tee v2_run.pid
```

The runner executes shards serially to stay within laptop RAM. A restart with
`RESET=0` reuses the manifest and skips valid finished shard files.

## 5. Monitor

```bash
tail -f local_logs_v2/master.log
```

```bash
find results/benchmark_v2 -maxdepth 1 \
  -name 'cases_shard_*.csv.gz' -size +0c | wc -l
```

```bash
free -h
ps -eo pid,comm,%cpu,%mem,rss --sort=-rss | head -12
```

## 6. Result

```bash
tail -n 60 local_logs_v2/master.log
ls -lh benchmark_v2_results_to_share.zip
unzip -t benchmark_v2_results_to_share.zip
```

Upload `benchmark_v2_results_to_share.zip` for the next analysis step.
