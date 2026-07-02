# Bias identifiability and representation: results

All numbers below are reproducible with the scripts in `src/`
(`bias_identifiability.py`, `per_window_bias_test.py`, `analyze_coverage_band.py`).
Estimand is window persistence `rho_headline` (results are identical for
`mean_span_frac`; the two are collinear, see below).

## 1. Coverage band per walk (`analyze_coverage_band.py` on the cluster grid)

MLE-fraction = (floor - MLE) / (floor - oracle). Negative = MLE worse than
guessing the population mean.

- **time_agnostic** (negative control): oracle >= floor, no recoverable rho
  signal. Confirms rho is a genuinely temporal property.
- **time_agnostic_t** (clean): MLE-fraction rises 0.43 -> 0.83 -> 1.00 across
  low/mid/high coverage; at high coverage the uniform MLE is optimal.
- **recency_biased**: MLE-fraction -0.40 -> 0.28 -> 0.90 (intermediate bias).
- **time_respecting** (forward): MLE-fraction -1.15 -> -0.58 -> -0.00. The
  uniform occupancy MLE is BELOW the floor at essentially all coverage; the
  oracle recovers rho (MAE ~0.05-0.09 vs floor ~0.14) at all coverage.

Ranking-vs-calibration dissociation (forward walk): Spearman(MLE, true) rises
0.56 -> 0.70 -> 0.90 while MLE MAE stays 0.19 -> 0.18 -> 0.15 (worse than floor).
The MLE knows the ordering but cannot put a calibrated number on it under bias.

## 2. Is the bias correctable label-free? (`bias_identifiability.py`)

Bias-aware MLE = occupancy MLE with one bias parameter beta (later active windows
favoured, weights exp(beta*rank)); beta=0 reproduces the uniform MLE.

| beta source | forward-walk result | verdict |
|---|---|---|
| uniform (beta=0) | MLE > floor at all coverage | fails (inconsistent, not just inefficient) |
| **beta-fit** (max profile log-likelihood, LABEL-FREE) | picks beta~0, MLE ~ uniform | **fails** |
| beta-calib (one beta per walk, uses labels) | reaches oracle at mid/high cov; ~55% of the gap at low cov | works given external beta |

Smoking gun (one forward instance, rho_true=0.40): the profile log-likelihood is
MAXIMISED at beta=0 (rho_hat=0.187, wrong); the correct beta~2 (rho_hat~0.42) has
LOWER likelihood. So maximum-likelihood chooses the no-correction value. The bias
strength is not identifiable from the biased sample; it must come from outside
(calibration on labels, or disclosure).

## 3. Does a richer input (per-window counts) fix it? (`per_window_bias_test.py`)

Store per edge the observation count in each window, e.g. (0,0,0,0,4).

- The generator's true activity is positionally uniform: P(window j active | edge
  active) = [0.30, 0.31, 0.31, 0.31, 0.31]. So any observation skew is pure bias.
- Per-window DOES detect the bias: the per-window MLE's likelihood peaks at
  beta>0 for forward (beta=0 for clean), unlike the (n,w) MLE. But the free fit
  OVERCORRECTS rho to ~1.0 (bimodal: 0.19 at beta=0, ~1.0 for beta>=0.5).
- Mechanism: P(window j observed | truly active), forward walk =
  [0.004, 0.019, 0.037, 0.089, 0.298] (clean: flat ~0.14). A forward crawl
  observes an early window ~0.4% of the time. The per-edge early activity is
  structurally never sampled, so no representation of the observed data recovers
  it. **The obstruction is the sampling, not the representation.**

## 4. Representation / second-estimand redundancy (`analyze_coverage_band.py`)

On the full 768-instance grid: rho vs mean_occupancy Pearson 1.000; rho vs
C_one_step Pearson 0.988 (Spearman 0.997); occupancy vs C Pearson 0.989. The
(n,w) count histogram is a sufficient PREDICTOR for all of them (an earlier test
showed oracle(histogram) ~ oracle(full summary) ~ oracle(window-bitmask) for rho
and C). C is redundant with occupancy on synthetic data, so it does not rescue a
representation axis here; a real-data check of the occupancy-C correlation is the
only place C could become non-redundant.

## Take-away for the thesis

The uniform occupancy MLE is inconsistent under forward-time sampling. Correction
is impossible from the biased sample alone (beta non-identifiable; early windows
unobserved), and possible only with externally supplied bias strength. This is
exactly what the disclosure experiment (Phase 3) tests: whether an LLM told the
sampling is forward-biased uses that information like the beta-calibrated MLE
(which provably reaches the oracle) or reverts to the beta=0 default.
