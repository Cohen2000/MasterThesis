# V2 validation record

The packaged source was checked with:

```bash
python -m py_compile src/*.py tests/*.py
PYTHONPATH=src python -m unittest discover -s tests -v
bash -n run_benchmark_v2_local.sh run_benchmark_v2_smoke.sh scripts/download_v2_snap_data.sh
```

Results:

- 9/9 unit and invariant tests passed.
- Full `v2_smoke` completed end to end:
  - 44 instances;
  - 7 independent groups;
  - 544 cases;
  - 7 targets;
  - all configured smoke access mechanisms;
  - case/manifest validation, model evaluation, plots and result bundle.
- The large v2 synthetic generator completed with all optional real data absent:
  - 474 synthetic instances;
  - 30 independent synthetic groups;
  - all DCSBM/LFR, homogeneous/heterogeneous/community-DAR, activity,
    renewal and controlled-twin blocks generated and validated.
- Legacy v1 smoke regression:
  - 206 case ids identical;
  - 509 legacy numeric target/coverage/estimator/feature fields bit-identical.

Raw real datasets are not distributed in this repository and must be copied or
downloaded by the user.
