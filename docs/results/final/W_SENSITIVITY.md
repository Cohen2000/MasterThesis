# Number of time windows

How to read this: the study uses W = 5 windows. Here the true persistence of the 12 real networks
and their time-shuffled copies is recomputed for other W (no sampling, no methods). "Fixed k" keeps
k = 2 (active in at least 2 windows); "fixed share" uses k = ceil(0.4 W), i.e. active in at least 40%
of the windows, which equals the study target at W = 5. Spearman is the rank correlation of the
networks with their W = 5 values: 1 means the same ordering. Computed by `scripts/window_sensitivity.py`,
which first checks that W = 5 reproduces `TRUTH.json` exactly.

## Real networks

| W | Reading | k | Mean rho | Mean change vs W = 5 | Max change | Spearman vs W = 5 | Networks with empty windows |
|---:|---|---:|---:|---:|---:|---:|---:|
| 2 | fixed k | 2 | 0.245 | 0.104 | 0.301 | 0.881 | 0/12 |
| 3 | fixed k | 2 | 0.301 | 0.048 | 0.094 | 0.986 | 0/12 |
| 3 | fixed share | 2 | 0.301 | 0.048 | 0.094 | 0.986 | 0/12 |
| 4 | fixed k | 2 | 0.331 | 0.018 | 0.036 | 0.986 | 0/12 |
| 4 | fixed share | 2 | 0.331 | 0.018 | 0.036 | 0.986 | 0/12 |
| 5 | fixed k | 2 | 0.348 | 0.000 | 0.000 | 1.000 | 1/12 |
| 5 | fixed share | 2 | 0.348 | 0.000 | 0.000 | 1.000 | 1/12 |
| 6 | fixed k | 2 | 0.358 | 0.011 | 0.047 | 1.000 | 1/12 |
| 6 | fixed share | 3 | 0.200 | 0.149 | 0.279 | 0.993 | 1/12 |
| 8 | fixed k | 2 | 0.374 | 0.025 | 0.061 | 0.986 | 1/12 |
| 8 | fixed share | 4 | 0.142 | 0.207 | 0.342 | 0.972 | 1/12 |
| 10 | fixed k | 2 | 0.384 | 0.036 | 0.076 | 0.986 | 3/12 |
| 10 | fixed share | 4 | 0.152 | 0.197 | 0.351 | 0.979 | 3/12 |
| 12 | fixed k | 2 | 0.388 | 0.040 | 0.087 | 0.993 | 3/12 |
| 12 | fixed share | 5 | 0.118 | 0.230 | 0.397 | 0.965 | 3/12 |
| 15 | fixed k | 2 | 0.398 | 0.050 | 0.094 | 0.972 | 3/12 |
| 15 | fixed share | 6 | 0.102 | 0.246 | 0.406 | 0.965 | 3/12 |
| 20 | fixed k | 2 | 0.409 | 0.060 | 0.106 | 0.972 | 4/12 |
| 20 | fixed share | 8 | 0.076 | 0.273 | 0.451 | 0.937 | 4/12 |

## Time-shuffled copies

| W | Reading | k | Mean rho | Mean change vs W = 5 | Max change | Spearman vs W = 5 | Networks with empty windows |
|---:|---|---:|---:|---:|---:|---:|---:|
| 2 | fixed k | 2 | 0.533 | 0.082 | 0.253 | 0.958 | 0/12 |
| 3 | fixed k | 2 | 0.575 | 0.040 | 0.174 | 0.979 | 0/12 |
| 3 | fixed share | 2 | 0.575 | 0.040 | 0.174 | 0.979 | 0/12 |
| 4 | fixed k | 2 | 0.600 | 0.015 | 0.083 | 0.986 | 0/12 |
| 4 | fixed share | 2 | 0.600 | 0.015 | 0.083 | 0.986 | 0/12 |
| 5 | fixed k | 2 | 0.615 | 0.000 | 0.000 | 1.000 | 1/12 |
| 5 | fixed share | 2 | 0.615 | 0.000 | 0.000 | 1.000 | 1/12 |
| 6 | fixed k | 2 | 0.621 | 0.006 | 0.025 | 1.000 | 1/12 |
| 6 | fixed share | 3 | 0.459 | 0.156 | 0.268 | 0.965 | 1/12 |
| 8 | fixed k | 2 | 0.628 | 0.014 | 0.038 | 0.993 | 1/12 |
| 8 | fixed share | 4 | 0.386 | 0.229 | 0.365 | 0.972 | 1/12 |
| 10 | fixed k | 2 | 0.632 | 0.017 | 0.051 | 0.993 | 3/12 |
| 10 | fixed share | 4 | 0.395 | 0.220 | 0.338 | 0.965 | 3/12 |
| 12 | fixed k | 2 | 0.636 | 0.021 | 0.061 | 0.993 | 3/12 |
| 12 | fixed share | 5 | 0.344 | 0.271 | 0.398 | 0.965 | 3/12 |
| 15 | fixed k | 2 | 0.638 | 0.024 | 0.072 | 0.986 | 3/12 |
| 15 | fixed share | 6 | 0.310 | 0.304 | 0.427 | 0.958 | 3/12 |
| 20 | fixed k | 2 | 0.642 | 0.027 | 0.083 | 0.986 | 4/12 |
| 20 | fixed share | 8 | 0.260 | 0.355 | 0.464 | 0.965 | 4/12 |

All k and every network: `W_SENSITIVITY.csv`, `W_SENSITIVITY_NETWORKS.csv`.
