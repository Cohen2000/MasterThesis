# Design revision `budget10-hrecent5-20260917` — report

Branch `experiment/offline-freeze-20260916`. Specification: last section of
[`docs/MAIN_EXPERIMENT_IMPLEMENTATION.md`](../../docs/MAIN_EXPERIMENT_IMPLEMENTATION.md),
fixed in commit `0468cf4` before any H answer of this design was generated.

**Status of this revision.** It was specified after the results of
`budget10-20261001`, including the Qwen answers, had been seen. It is a documented
revision, not a retroactive preregistration. Only arm H changed. Superseded designs
stay on disk and in Git: `budget10_20261001` and `baseline_revision_20261001`
hold the suffix-panel H, and that arm remains reproducible as the development
variant `H_suffix_v1` (all 70 stored H blocks and prompts are reproduced byte for
byte).

## 1. What changed

Arm H is now a **uniform sample of d active dyads, each with its min(5, m_e) most
recent events**. d is fixed before any draw as the integer whose expected volume
d·C/D, with C = Σ min(5, m_e), is closest to the ten-percent budget. The block
lists all 31 five-window patterns plus `at_cap_dyads`, the number of listed dyads
with exactly five retrieved events. Such a dyad is possibly, not provably,
truncated. The fixed H reference is the midpoint of per-dyad bounds
J ≤ K ≤ J + b − 1 (b = earliest observed window, only for at-cap dyads), which
bound the profile **of the sampled dyads**. The three-to-five-window correction and
its features are gone from arm H.

Literature: Gibbons (2001) samples distinct values uniformly with a per-value row
limit, keeps a reservoir (Vitter 1985) above the limit, and stores exact counts.
Zhou et al. (2022, TGL) name uniform and most-recent temporal neighbour sampling
as the two standard strategies. **Our own work** is the combination: dyads as the
distinct values, most-recent instead of reservoir retention, cap 5, no exact
count, the budget rule, and the bound derivation. The literature supports the
building blocks, not this configuration or any ranking.

## 2. What was regenerated and what was reused

| Item | Status |
| --- | --- |
| Graphs, labels, pool definition, train/dev split | unchanged; the pool rebuild reproduced all 7 500 R/S/B pool blocks and all 500 labels byte for byte |
| R, S, B observations and prompts (main and training) | unchanged, byte-identical |
| H observations: 16 real training sources, 400 + 100 pool graphs, 14 main graphs | **new** |
| ExtraTrees, 7 pooled + 7 real-only folds | **refitted** (195 features), all arms re-predicted |
| Development check (100 graphs), main-panel baselines, error decompositions | **recomputed** |
| Qwen R/S/B answers (1 260) | **reused** after `check_qwen_reuse.py`: same request id, block, prompt hash, seed, stored-answer metadata, generation-relevant payload and runner sampling. Differing payload fields are the transport documentation corrected in 9c5b04b and never read by the runner. |
| Qwen H answers (396) | **new**, see §6. No previous H answer is used. |
| Sol, DeepSeek | not run |

The offline run was regenerated from the spec commit and compared file by file with
the pre-commit run: all 1 498 files are identical apart from timing fields and the
source inventory. `verify_main_offline.py` re-derives all 592 blocks from their seeds
and checks every H block against the backward-fill rule; 112 unit tests pass.

**Sizes.** 276 main observations (13 graphs × 20 + 16 for `sp_highschool2013`),
316 training observations, 3 312 planned calls, 1 656 of them Qwen.

## 3. Budget, saturation and coverage

Every graph is budget-matched on every arm: maximum absolute relative error R
0.63 %, H 0.98 %, walk validation 1.87 %; B keeps p = 0.10 exactly.

| Condition | Graphs |
| --- | --- |
| H target unreachable (C < B) | `sp_highschool2013` (C = 18 667, B = 18 851) |
| H saturated (d = D), therefore **deterministic H**, one observation | `sp_highschool2013` only; no pool graph |
| H outside the 5 % tolerance | none (`sp_highschool2013`: −0.98 %) |

Median coverage on the main panel (production draws):

| stratum | arm | dyads | active dyad-windows | events | D_obs median (min) |
|---|---|---|---|---|---|
| real | R | 9.7 % | 9.8 % | 9.6 % | 1472 (96) |
| real | S | 1.6 % | 2.4 % | 10.2 % | 276 (19) |
| real | **H** | **63.4 %** | **45.8 %** | 10.0 % | 10181 (820) |
| real | B | 43.4 % | 37.1 % | 10.0 % | 5371 (607) |
| synthetic | R | 9.8 % | 10.0 % | 10.1 % | 256 (150) |
| synthetic | S | 7.5 % | 9.3 % | 10.0 % | 196 (122) |
| synthetic | **H** | **12.2 %** | **10.6 %** | 10.0 % | 319 (253) |
| synthetic | B | 31.6 % | 17.9 % | 10.0 % | 804 (719) |

With the cap, H needs few events per dyad, so on the event-dense real sources it
covers most dyads (72 % on Hospital, 89 % on Copenhagen). Equal event volume is not
equal target information.

## 4. Offline check of arm H (20 draws per main graph)

`scripts/check_h_recent.py`, separate streams, one draw for the saturated graph.
Outputs in [`results/h_recent5_check_20260917/`](../h_recent5_check_20260917/).

**Selection behaves as a simple random sample.** Across all graphs the mean
selection error on ρ₂ lies within about one Monte-Carlo standard error of zero (max |z| =
1.01), and the empirical spread over 20 draws is 0.82–1.28 times the analytic SRS
value (1 − d/D)·S²/d, consistent with the ±16 % sampling error of a 20-draw SD.
No draw violated its sample bounds; realised volume averaged 0.998–1.002 B per stratum.

**Lost recurrence information, computed over all dyads.** Under a fixed-size
uniform dyad sample E[plug-in] = ρ^J exactly, so the history loss below is the
population value, not an estimate:

| graph | d/D | rel. budget err. | saturated | m_e>5 | J<K (recent) | J<K (uniform subset) | rho2 | hist rho2 | hist rho2 uniform | hist profile | mid bias rho2 | width rho2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sp_hospital | 0.720 | +0.02 % | no | 0.586 | 0.292 | 0.158 | 0.445 | -0.253 | -0.065 | -0.113 | -0.017 | 0.471 |
| sp_highschool2013 | 1.000 | -0.98 % | yes | 0.406 | 0.275 | 0.223 | 0.486 | -0.183 | -0.064 | -0.140 | -0.061 | 0.245 |
| copenhagen_bluetooth | 0.889 | +0.00 % | no | 0.469 | 0.280 | 0.196 | 0.447 | -0.232 | -0.052 | -0.142 | -0.051 | 0.362 |
| snap_email_eu | 0.634 | +0.00 % | no | 0.396 | 0.292 | 0.216 | 0.506 | -0.132 | -0.028 | -0.139 | -0.050 | 0.165 |
| snap_collegemsg | 0.167 | -0.00 % | no | 0.190 | 0.030 | 0.020 | 0.103 | -0.023 | -0.013 | -0.009 | +0.020 | 0.087 |
| snap_mathoverflow | 0.117 | -0.00 % | no | 0.051 | 0.016 | 0.011 | 0.084 | -0.008 | -0.004 | -0.006 | +0.008 | 0.031 |
| dar_a0_r1 | 0.106 | -0.10 % | no | 0.096 | 0.033 | 0.015 | 0.391 | -0.006 | -0.001 | -0.009 | +0.001 | 0.015 |
| dar_a08_r1 | 0.155 | -0.02 % | no | 0.540 | 0.464 | 0.370 | 0.774 | -0.015 | -0.000 | -0.219 | -0.006 | 0.018 |
| dar_a0_r2 | 0.107 | +0.11 % | no | 0.097 | 0.040 | 0.016 | 0.390 | -0.010 | -0.002 | -0.011 | -0.002 | 0.014 |
| dar_a08_r2 | 0.152 | -0.06 % | no | 0.533 | 0.448 | 0.356 | 0.767 | -0.012 | -0.000 | -0.212 | -0.005 | 0.013 |
| ad_memoryless_r1 | 0.100 | -0.03 % | no | 0.000 | 0.000 | 0.000 | 0.049 | +0.000 | +0.000 | +0.000 | +0.000 | 0.000 |
| ad_memory_r1 | 0.141 | -0.12 % | no | 0.365 | 0.260 | 0.218 | 0.789 | -0.012 | -0.002 | -0.104 | -0.006 | 0.013 |
| ad_memoryless_r2 | 0.100 | -0.05 % | no | 0.000 | 0.000 | 0.000 | 0.043 | +0.000 | +0.000 | +0.000 | +0.000 | 0.000 |
| ad_memory_r2 | 0.136 | -0.12 % | no | 0.348 | 0.245 | 0.199 | 0.775 | -0.008 | -0.001 | -0.090 | -0.004 | 0.009 |

"hist" is ρ^J − ρ (plug-in bias), "uniform" the same for a uniform subset of the same size, "hist profile" the mean over k = 2..5, "mid bias" the bias of the bound midpoint, "width" the expected U₂ − L₂.

**Is H history-shaped?** On the four event-dense real sources, yes: the plug-in
loses 13–25 percentage points of ρ₂, and on the main draws the history component is
98 % of the summed absolute components (|history| 0.132 against |selection|
0.002; table in §5). The boundary cases are reported as such:

* `snap_mathoverflow` (−0.008 on ρ₂, −0.006 on the profile) and `snap_collegemsg`
  (−0.023, −0.009) lose little: most of their dyads have fewer than six events.
* DAR α = 0 (−0.007 / −0.010) and activity-driven without memory (exactly 0: no dyad
  has more than five events) lose essentially nothing.
* DAR α = 0.8 and activity-driven with memory lose little on **ρ₂** (−0.008 to
  −0.015), because long-lived dyads keep at least two recent windows, but a lot on
  **ρ₃–ρ₅** (profile mean −0.21 to −0.22 and −0.09 to −0.10).

So H is history-shaped on the real panel as a whole and on the profile of two of
the four synthetic conditions; on ρ₂ of the synthetic panel its error is mostly
selection noise.

**Per-draw errors and baselines (MAE₂, mean over 20 draws):**

| graph | draws | selection (MCSE) | SD emp/analytic | history | midpoint err | covers truth | volume/B | at-cap | AE2 plug-in | AE2 midpoint | AE2 ET pooled | AE2 median | AE2 uniform-subset plug-in |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sp_hospital | 20 | -0.0001 (0.0019) | 0.91 | -0.253 | -0.017 | 1.00 | 1.001 | 0.634 | 0.253 | 0.017 | 0.021 | 0.155 | 0.066 |
| sp_highschool2013 | 1 | +0.0000 (0.0000) | — | -0.183 | -0.061 | 1.00 | 0.990 | 0.444 | 0.183 | 0.061 | 0.032 | 0.195 | 0.064 |
| copenhagen_bluetooth | 20 | +0.0001 (0.0001) | 0.82 | -0.232 | -0.051 | 1.00 | 1.000 | 0.507 | 0.232 | 0.051 | 0.004 | 0.157 | 0.052 |
| snap_email_eu | 20 | +0.0005 (0.0007) | 1.01 | -0.133 | -0.050 | 1.00 | 1.002 | 0.442 | 0.133 | 0.050 | 0.042 | 0.216 | 0.028 |
| snap_collegemsg | 20 | -0.0006 (0.0014) | 1.12 | -0.023 | +0.020 | 1.00 | 1.002 | 0.240 | 0.023 | 0.020 | 0.064 | 0.316 | 0.014 |
| snap_mathoverflow | 20 | +0.0003 (0.0004) | 1.00 | -0.008 | +0.008 | 1.00 | 1.001 | 0.075 | 0.008 | 0.008 | 0.052 | 0.334 | 0.003 |
| dar_a0_r1 | 20 | -0.0022 (0.0067) | 1.24 | -0.007 | -0.002 | 0.20 | 0.998 | 0.174 | 0.026 | 0.024 | 0.021 | 0.036 | 0.025 |
| dar_a08_r1 | 20 | +0.0026 (0.0063) | 1.16 | -0.016 | -0.004 | 0.30 | 1.005 | 0.623 | 0.023 | 0.022 | 0.021 | 0.419 | 0.023 |
| dar_a0_r2 | 20 | -0.0007 (0.0053) | 0.98 | -0.010 | -0.004 | 0.25 | 0.998 | 0.173 | 0.022 | 0.019 | 0.017 | 0.035 | 0.021 |
| dar_a08_r2 | 20 | -0.0036 (0.0070) | 1.28 | -0.011 | -0.008 | 0.10 | 0.999 | 0.622 | 0.029 | 0.026 | 0.025 | 0.412 | 0.025 |
| ad_memoryless_r1 | 20 | +0.0017 (0.0017) | 1.14 | +0.000 | +0.002 | 0.00 | 1.002 | 0.000 | 0.005 | 0.005 | 0.006 | 0.305 | 0.005 |
| ad_memory_r1 | 20 | +0.0024 (0.0043) | 0.85 | -0.012 | -0.003 | 0.05 | 1.006 | 0.472 | 0.018 | 0.017 | 0.009 | 0.434 | 0.017 |
| ad_memoryless_r2 | 20 | +0.0012 (0.0012) | 0.86 | +0.000 | +0.001 | 0.00 | 0.999 | 0.000 | 0.004 | 0.004 | 0.003 | 0.311 | 0.004 |
| ad_memory_r2 | 20 | -0.0058 (0.0057) | 1.09 | -0.007 | -0.009 | 0.05 | 0.994 | 0.452 | 0.023 | 0.021 | 0.007 | 0.421 | 0.020 |

Stratum means (sources equally weighted):

| stratum | plug-in | midpoint | ET pooled | ET real-only | median | uniform-subset plug-in | exp. hist rho2 recent | exp. hist rho2 uniform | J<K |
|---|---|---|---|---|---|---|---|---|---|
| real | 0.1387 | 0.0344 | 0.0360 | 0.0355 | 0.2288 | 0.0380 | -0.1386 | -0.0375 | 0.198 |
| dar_a0 | 0.0240 | 0.0216 | 0.0186 | 0.0156 | 0.0357 | 0.0226 | -0.0080 | -0.0013 | 0.036 |
| dar_a08 | 0.0260 | 0.0239 | 0.0231 | 0.2725 | 0.4158 | 0.0235 | -0.0136 | -0.0005 | 0.456 |
| ad_memoryless | 0.0049 | 0.0049 | 0.0043 | 0.0274 | 0.3082 | 0.0049 | +0.0000 | +0.0000 | 0.000 |
| ad_memory | 0.0205 | 0.0191 | 0.0080 | 0.2930 | 0.4276 | 0.0183 | -0.0103 | -0.0015 | 0.253 |

The bound intervals cover the full-archive ρ₂ on every real draw (they are wide:
mean width 0.23) and on only 0–30 % of synthetic draws, where they are narrow
(≤ 0.02) and selection noise dominates. That is expected: they bound the sampled
dyads, not the archive.

**Mechanism check: most recent versus uniform subset of the same size.** On the
same sampled dyads, keeping min(5, m_e) events drawn uniformly without replacement
(identical volume by construction, no model fit, no LLM call, no bounds applied):

* dyads with J < K: 0.198 (recent) against 0.138 (uniform) on the real sources;
* expected ρ₂ loss: −0.139 against −0.038 on the real sources, −0.008 to −0.014
  against −0.000 to −0.002 on the synthetic conditions;
* plug-in MAE₂: 0.139 against 0.038 on the real sources (paired difference
  −0.101, between-source SE 0.032).

Recency concentrates the loss on the early windows, and that is what removes
recurrence; a uniform subset of the same size spreads the retained events over the
dyad's lifetime and keeps more of its active windows. The comparison describes the
mechanism; it is not an alternative design.

## 5. Baselines

### Development graphs (100 held-out synthetic, 2 000 observations)

| Arm | plug-in | fixed reference | B mixture | ET pooled | ET real-only | median |
| --- | --- | --- | --- | --- | --- | --- |
| R | 0.0176 | = plug-in | — | 0.0178 | 0.1677 | 0.3055 |
| S | 0.1369 | 0.0318 | — | 0.0210 | 0.1749 | 0.3055 |
| **H** | **0.0516** | **0.0233** (midpoint) | — | **0.0152** | 0.1733 | 0.3055 |
| B | 0.3839 | 0.1164 | 0.0723 | 0.0299 | 0.1890 | 0.3055 |

R, S and B plug-in, correctors and mixture are unchanged from the previous revision,
as they must be; ExtraTrees moves slightly (e.g. B 0.0305 → 0.0299) because it was
refitted with the new H rows and features. Previous-design H for comparison:
plug-in 0.1010, homogeneous corrector 0.1199, mixture candidate 0.0380, ET 0.0246.

### Main panel, five production draws (MAE₂ ± between-source SE)

real
| arm | plug-in | fixed reference | B mixture | ET pooled | ET real-only | median |
|---|---|---|---|---|---|---|
| R | 0.0197 ± 0.0083 | 0.0197 ± 0.0083 | — | 0.0263 ± 0.0078 | 0.0433 ± 0.0082 | 0.2288 ± 0.0320 |
| S | 0.3222 ± 0.0548 | 0.0972 ± 0.0440 | — | 0.0592 ± 0.0172 | 0.0693 ± 0.0221 | 0.2288 ± 0.0320 |
| H | 0.1392 ± 0.0424 | 0.0350 ± 0.0088 | — | 0.0367 ± 0.0087 | 0.0352 ± 0.0089 | 0.2288 ± 0.0320 |
| B | 0.0823 ± 0.0146 | 0.0555 ± 0.0184 | 0.0592 ± 0.0190 | 0.0497 ± 0.0156 | 0.0785 ± 0.0228 | 0.2288 ± 0.0320 |

**synthetic**

| arm | plug-in | fixed reference | B mixture | ET pooled | ET real-only | median |
|---|---|---|---|---|---|---|
| R | 0.0203 ± 0.0035 | 0.0203 ± 0.0035 | — | 0.0183 ± 0.0045 | 0.1467 ± 0.0455 | 0.2968 ± 0.0597 |
| S | 0.1255 ± 0.0207 | 0.0323 ± 0.0069 | — | 0.0161 ± 0.0037 | 0.1505 ± 0.0444 | 0.2968 ± 0.0597 |
| H | 0.0132 ± 0.0029 | 0.0119 ± 0.0022 | — | 0.0103 ± 0.0023 | 0.1525 ± 0.0498 | 0.2968 ± 0.0597 |
| B | 0.3629 ± 0.0791 | 0.0962 ± 0.0292 | 0.0661 ± 0.0175 | 0.0240 ± 0.0058 | 0.1752 ± 0.0481 | 0.2968 ± 0.0597 |

"fixed reference" is the plug-in for R, the walk ratio for S, the bound midpoint for H and the homogeneous corrector for B; the primary reference for B is the mixture.

Arm H per source (MAE₂ over the production draws; `sp_highschool2013` has one
deterministic draw):

| source | draws | plug-in | midpoint | ET pooled | ET real-only | median |
|---|---|---|---|---|---|---|
| sp_hospital | 5 | 0.2546 | 0.0193 | 0.0245 | 0.0113 | 0.1550 |
| sp_highschool2013 | 1 | 0.1832 | 0.0607 | 0.0317 | 0.0338 | 0.1954 |
| copenhagen_bluetooth | 5 | 0.2322 | 0.0516 | 0.0046 | 0.0221 | 0.1568 |
| snap_email_eu | 5 | 0.1310 | 0.0493 | 0.0420 | 0.0660 | 0.2156 |
| snap_collegemsg | 5 | 0.0262 | 0.0212 | 0.0652 | 0.0212 | 0.3158 |
| snap_mathoverflow | 5 | 0.0080 | 0.0077 | 0.0521 | 0.0569 | 0.3345 |
| dar_a0_r1 | 5 | 0.0065 | 0.0116 | 0.0087 | 0.0044 | 0.0361 |
| dar_a0_r2 | 5 | 0.0232 | 0.0165 | 0.0182 | 0.0311 | 0.0352 |
| dar_a08_r1 | 5 | 0.0192 | 0.0148 | 0.0128 | 0.2779 | 0.4192 |
| dar_a08_r2 | 5 | 0.0193 | 0.0154 | 0.0197 | 0.2706 | 0.4124 |
| ad_memoryless_r1 | 5 | 0.0049 | 0.0049 | 0.0025 | 0.0248 | 0.3053 |
| ad_memoryless_r2 | 5 | 0.0022 | 0.0022 | 0.0024 | 0.0238 | 0.3112 |
| ad_memory_r1 | 5 | 0.0095 | 0.0093 | 0.0062 | 0.2994 | 0.4343 |
| ad_memory_r2 | 5 | 0.0208 | 0.0208 | 0.0121 | 0.2881 | 0.4210 |

Paired against the plug-in on the real sources: H midpoint −0.104 ± 0.039,
H ET pooled −0.103 ± 0.050. On synthetic H the fixed references are close to the
plug-in (−0.001 ± 0.001 and −0.003 ± 0.001).

### Error decomposition on the main draws (ρ₂)

| stratum | arm | selection | history | net | abs. selection | abs. history | history share | cancellation |
|---|---|---|---|---|---|---|---|---|
| real | R | -0.0019 | +0.0000 | -0.0019 | 0.0197 | 0.0000 | 0 % | 1.00 |
| real | S | +0.3222 | +0.0000 | +0.3222 | 0.3222 | 0.0000 | 0 % | 1.00 |
| real | H | -0.0002 | -0.1390 | -0.1392 | 0.0022 | 0.1322 | 98 % | 1.01 |
| real | B | +0.1853 | -0.2676 | -0.0823 | 0.1853 | 0.2676 | 59 % | 5.50 |
| synthetic | R | +0.0015 | +0.0000 | +0.0015 | 0.0203 | 0.0000 | 0 % | 1.00 |
| synthetic | S | +0.1255 | +0.0000 | +0.1255 | 0.1255 | 0.0000 | 0 % | 1.00 |
| synthetic | H | -0.0003 | -0.0075 | -0.0078 | 0.0118 | 0.0075 | 39 % | 1.46 |
| synthetic | B | +0.1197 | -0.4826 | -0.3629 | 0.1197 | 0.4826 | 80 % | 1.66 |

Previous-design H for comparison: selection +0.127 / history −0.192 / net −0.065
(real), with cancellation factor 4.4. The new H has essentially no selection
component, so the net error is the history loss and nothing cancels it.

## 6. Qwen3.6-35B-A3B on the revised panel

### Execution

Specification commit `0468cf4`; runner commits `e027e9d` and `87d0af5` (execution
only, no design change). Model, revision `995ad96…`, all 26 weight-shard hashes,
tokenizer, chat template, pinned environment (vLLM 0.29.0, transformers 5.17.0,
torch 2.13.0) and per-request sampling are identical to the archived previous
run; the runner's own prompt token count equals the engine's for all 396 answers.

* **New:** 396 H answers (66 observations × 2 modes × 3 repeats), four two-hour
  jobs on `gpu_h100`, each shard finished in its first attempt (18–41 min, about
  1.6 GPU-hours in total, 3.72 million output tokens). No request was admitted and
  lost, none was generated twice. A separate smoke test (8 requests, job 7006435)
  wrote into `answers_smoke/` and is not evaluated; its first attempt (7006420)
  failed at start-up on a runner bug, which was fixed before any production job.
* **Reused:** 1 260 R/S/B answers from `qwen_archive_20261001.tgz`, verified per file
  against the archive checksums and per request by `check_qwen_reuse.py`.
* **End states:** 395 regular ends, **one output-limit hit**: the non-thinking answer
  `ad_memory_r1__H-recent5__s1__qwen_nonthinking__r1` fell into a repetition loop
  ("1,1,1,…") and ran to the 258 048-token allowance in 38 minutes. It is invalid
  under every parser rule and receives the plug-in replacement; there is no retry.
  The same request (same seed and prompt) ended normally after 9 269 tokens in the
  unevaluated smoke test, a concrete instance of GPU runs not being bitwise
  reproducible. No unclosed reasoning block, no empty answer, no technical failure.
  Output length otherwise: thinking median 8 121 (max 15 999), non-thinking median
  9 306 (max 17 246).
* **Archive:** `results/main_experiment/hrecent5_qwen_archive_20260917.tgz`
  (SHA-256 in the neighbouring file; copy in `$HOME/hrecent5_archive_20260917` on the
  cluster): 685 files — new answers, requests, observations, the 132 rendered H
  prompts, model identity, pinned and job-time environment, runner, job script and
  logs — with zero read-back mismatches on the cluster and after transfer.

**Parser rules on the new H answers.** Thinking: v1 145/198, **v2 198/198**, v3
198/198. Non-thinking: v1 0/198, **v2 0/198**, v3 166/198 (31 without a trailing
JSON object, 1 non-monotone). Over all 1 656 current answers: thinking v2 828/828,
non-thinking v2 0/828 and v3 747/828. As before, the non-thinking mode derives in
the answer field, so under the main rule v2 it has **no valid answer in any cell**;
its pipeline value would be the plug-in and is not reported as a model estimate.
The v3 column below is the post-hoc sensitivity reading.

### Main result on the six real sources (MAE₂)

| Arm | Qwen thinking (v2) | Qwen non-thinking (v2) | Qwen non-thinking (v3, sensitivity) | plug-in | primary reference | ET pooled | training median |
|---|---|---|---|---|---|---|---|
| R | 0.0199 | no valid answer | 0.0198 | 0.0197 | 0.0197 | 0.0263 | 0.2288 |
| S | 0.1384 | no valid answer | 0.1626 | 0.3222 | 0.0972 | 0.0592 | 0.2288 |
| H | 0.1402 | no valid answer | 0.1358 | 0.1392 | 0.0350 | 0.0367 | 0.2288 |
| B | 0.1214 | no valid answer | 0.1145 | 0.0823 | 0.0592 | 0.0497 | 0.2288 |

Qwen thinking, paired differences (mean ± between-source SE) and the Monte-Carlo
error of its MAE₂ split into model-repeat and sampler parts:

| Arm | vs primary | vs plug-in | vs ET pooled | vs median | MCSE of MAE2 (model / sampler) | signed ρ₂ Qwen | signed ρ₂ plug-in |
|---|---|---|---|---|---|---|---|
| R | +0.0001 ± 0.0004 | +0.0001 ± 0.0004 | -0.0065 ± 0.0083 | -0.2090 ± 0.0373 | 0.0038 (0.0004 / 0.0038) | -0.0015 | -0.0019 |
| S | +0.0412 ± 0.0112 | -0.1838 ± 0.0534 | +0.0792 ± 0.0381 | -0.0904 ± 0.0711 | 0.0181 (0.0107 / 0.0148) | +0.0758 | +0.3222 |
| H | +0.1053 ± 0.0345 | +0.0010 ± 0.0060 | +0.1036 ± 0.0456 | -0.0886 ± 0.0696 | 0.0049 (0.0041 / 0.0038) | -0.0987 | -0.1392 |
| B | +0.0623 ± 0.0229 | +0.0391 ± 0.0152 | +0.0718 ± 0.0233 | -0.1074 ± 0.0391 | 0.0132 (0.0144 / 0.0023) | -0.0166 | -0.0823 |

R, S and B are unchanged from the previous report because the answers are the same;
only the refitted ExtraTrees reference moved (e.g. S: +0.077 → +0.079).

### Arm H

| stratum | thinking | non-thinking (v3) | valid non-thinking (v3) | plug-in | midpoint | ET pooled | median |
|---|---|---|---|---|---|---|---|
| real | 0.1402 | 0.1358 | 0.86 | 0.1392 | 0.0350 | 0.0367 | 0.2288 |
| dar_a0 | 0.0452 | 0.0275 | 0.80 | 0.0148 | 0.0140 | 0.0135 | 0.0357 |
| dar_a08 | 0.0185 | 0.0488 | 0.87 | 0.0193 | 0.0151 | 0.0162 | 0.4158 |
| ad_memoryless | 0.0036 | 0.0036 | 0.90 | 0.0036 | 0.0036 | 0.0024 | 0.3082 |
| ad_memory | 0.0151 | 0.0427 | 0.73 | 0.0151 | 0.0150 | 0.0092 | 0.4276 |

Per real source, Qwen thinking (v2):

| source | Qwen thinking | MCSE (model / sampler) | signed | plug-in | midpoint | ET pooled |
|---|---|---|---|---|---|---|
| sp_hospital | 0.2542 | 0.0030 (0.0018 / 0.0024) | -0.1887 | 0.2546 | 0.0193 | 0.0245 |
| sp_highschool2013 | 0.1832 | 0.0000 (0.0000 / 0.0000) | -0.1832 | 0.1832 | 0.0607 | 0.0317 |
| copenhagen_bluetooth | 0.2082 | 0.0191 (0.0134 / 0.0136) | -0.1650 | 0.2322 | 0.0516 | 0.0046 |
| snap_email_eu | 0.1313 | 0.0087 (0.0175 / 0.0000) | -0.0631 | 0.1310 | 0.0493 | 0.0420 |
| snap_collegemsg | 0.0466 | 0.0205 (0.0099 / 0.0179) | +0.0008 | 0.0262 | 0.0212 | 0.0652 |
| snap_mathoverflow | 0.0178 | 0.0030 (0.0041 / 0.0000) | +0.0072 | 0.0080 | 0.0077 | 0.0521 |

`sp_highschool2013` has one deterministic observation; its three answers are
identical (all equal to the plug-in), so both MCSE components are zero there.

**Where the answers lie relative to the sample bounds** (valid answers, ρ₂, equality
within 5·10⁻⁴ because answers are rounded; descriptive only, nothing is clipped):

| stratum | config | rule | below L | = L (plug-in) | inside | = U | above U |
|---|---|---|---|---|---|---|---|
| real | qwen_thinking | v2 | 8 % | 71 % | 4 % | 0 % | 18 % |
| dar_a0 | qwen_thinking | v2 | 10 % | 70 % | 3 % | 10 % | 7 % |
| dar_a08 | qwen_thinking | v2 | 7 % | 73 % | 3 % | 13 % | 3 % |
| ad_memoryless | qwen_thinking | v2 | 10 % | 90 % | 0 % | 0 % | 0 % |
| ad_memory | qwen_thinking | v2 | 0 % | 87 % | 3 % | 7 % | 3 % |
| real | qwen_nonthinking | v3 | 7 % | 52 % | 19 % | 3 % | 18 % |
| dar_a0 | qwen_nonthinking | v3 | 12 % | 50 % | 12 % | 8 % | 17 % |
| dar_a08 | qwen_nonthinking | v3 | 27 % | 42 % | 12 % | 4 % | 15 % |
| ad_memoryless | qwen_nonthinking | v3 | 4 % | 96 % | 0 % | 0 % | 0 % |
| ad_memory | qwen_nonthinking | v3 | 32 % | 32 % | 23 % | 9 % | 5 % |

**Reading.** On the new H, Qwen thinking mostly reports the observed profile: 71 %
of its valid real-source answers equal the plug-in (the lower bound), 70–90 % on the
synthetic conditions. Its MAE₂ therefore matches the plug-in (0.140 against 0.139,
paired +0.001 ± 0.006) and stays far behind the bound midpoint (+0.105 ± 0.035) and
ExtraTrees (+0.104 ± 0.046). When it does correct, it often goes past the largest
value the sampled dyads allow (18 % of real answers above U₂), and it rarely lands
strictly between the bounds (4 %). The mean signed error moves from −0.139 to
−0.099: a partial correction on average, driven by a minority of answers. On the
event-dense sources where history is lost (Hospital, HighSchool2013, Email-Eu) it is
indistinguishable from the plug-in; on Copenhagen it corrects slightly (0.208
against 0.232); on CollegeMsg and MathOverflow, where little history is lost, it is
worse than the plug-in (0.047 against 0.026, 0.018 against 0.008). On the synthetic
panel it equals the plug-in except for DAR α = 0 (0.045 against 0.015).

This differs from the previous design, where Qwen over-corrected H (signed +0.128,
MAE₂ 0.208 against a plug-in of 0.072). The two arms carry different information,
so the numbers are not a like-for-like comparison.

All arms, Qwen thinking against the plug-in, by stratum (MAE₂ Qwen / plug-in):

| stratum | R | S | H | B |
|---|---|---|---|---|
| real | 0.020 / 0.020 | 0.138 / 0.322 | 0.140 / 0.139 | 0.121 / 0.082 |
| dar_a0 | 0.021 / 0.019 | 0.078 / 0.182 | 0.045 / 0.015 | 0.263 / 0.314 |
| dar_a08 | 0.032 / 0.033 | 0.076 / 0.132 | 0.019 / 0.019 | 0.370 / 0.523 |
| ad_memoryless | 0.009 / 0.009 | 0.016 / 0.037 | 0.004 / 0.004 | 0.050 / 0.042 |
| ad_memory | 0.021 / 0.021 | 0.072 / 0.151 | 0.015 / 0.015 | 0.462 / 0.573 |

### Secondary, post hoc (real sources)

| configuration | arm | MAE₂ (mean of answers) | median of three | 50/50 to plug-in | 50/50 to median | answer SD (ρ₂) | observations with 3 valid |
|---|---|---|---|---|---|---|---|
| qwen_thinking (v2) | R | 0.0199 | 0.0197 | 0.0197 | 0.1140 | 0.001 | 30 |
| qwen_thinking (v2) | S | 0.1384 | 0.1090 | 0.2023 | 0.1259 | 0.098 | 30 |
| qwen_thinking (v2) | H | 0.1402 | 0.1347 | 0.1262 | 0.1681 | 0.058 | 26 |
| qwen_thinking (v2) | B | 0.1214 | 0.0796 | 0.0889 | 0.1466 | 0.117 | 30 |
| qwen_nonthinking (v3) | R | 0.0198 | 0.0197 | 0.0196 | 0.1138 | 0.000 | 26 |
| qwen_nonthinking (v3) | S | 0.1626 | 0.1159 | 0.2128 | 0.1285 | 0.127 | 25 |
| qwen_nonthinking (v3) | H | 0.1358 | 0.1371 | 0.1205 | 0.1666 | 0.080 | 15 |
| qwen_nonthinking (v3) | B | 0.1145 | 0.0790 | 0.0874 | 0.1553 | 0.122 | 21 |

"Median of three" is the componentwise median of the three pipeline answers of one
observation; the 50/50 rows average each answer with the plug-in or with the fold's
training median (the agreed diagnostic named no target, so both are shown). None
of these is a main result and none was used to choose anything. The median of three
lowers MAE₂ where the answers scatter (B 0.121 → 0.080, S 0.138 → 0.109) and hardly
changes H (0.140 → 0.135), whose answers mostly coincide.

## 7. Descriptive budget and coverage sweep (post hoc)

Authorised by the user on 2026-09-17; it changes nothing in the design. Files in
[`results/budget_sweep_20260917/`](../budget_sweep_20260917/).

* **Arm H, budget at cap 5.** The expected plug-in bias does not depend on the
  budget (real −0.139, synthetic −0.008 at every share from 1 % to 50 %); the budget
  moves only coverage and SRS variance. Real sources saturate early (mean dyad
  coverage 0.59 at 10 %, 0.90 at 50 %). Midpoint MAE₂ on the real sources stays at
  0.034–0.039 for every budget.
* **Arm H, cap at 10 % budget.** The cap sets the history loss: real-source plug-in
  bias −0.345 / −0.241 / −0.191 / −0.139 / −0.088 / −0.049 / 0 for caps 1, 2, 3, 5,
  10, 20 and full history, while dyad coverage falls from 0.77 to 0.10. At cap 5
  the real sources are the only ones with a large ρ₂ loss; at caps ≤ 3 the synthetic
  graphs lose heavily as well (−0.104 at cap 3).
* **All arms, budget 2–30 %.** R plug-in MAE₂ falls with budget (real 0.040 →
  0.011); the S walk-ratio corrector improves strongly (real 0.170 → 0.041) while
  the S plug-in stays biased (≈ 0.31–0.34); B plug-in improves (real 0.150 → 0.042,
  synthetic 0.466 → 0.188) and the B mixture follows (real 0.092 → 0.035); H plug-in
  and midpoint are flat on the real sources (0.136–0.139 and 0.034). The walk
  length was recalibrated per budget (256 paths, no validation walks) and reached
  the target on every graph; the B mixture fell back to the homogeneous corrector
  on 4–11 % of draws per budget.

None of these values was used to choose the cap, the budget or any reference.

## 8. Limits

* Six real sources; most real-source differences are within a few between-source
  standard errors.
* The bound midpoint is a fixed, untuned reference; its intervals bound the sampled
  dyads, not the archive, and are wide on event-dense sources.
* The H history loss is concentrated in four real sources. On ρ₂ of the synthetic
  panel, and on CollegeMsg and MathOverflow, arm H is close to a pure selection arm.
* Development and pool graphs come from the two generator families of the main
  synthetic instances.
* One model, two modes. Sol and DeepSeek are not run; the four-configuration
  comparison is incomplete by design at this stage.
* Qwen's non-thinking mode has no valid answer under the main parser rule; its
  numbers exist only as a post-hoc sensitivity reading.

## 9. Answers to the questions of the revision

* **Newly produced:** H observations for all real training sources, the 500 pool
  graphs and the 14 main graphs; all ExtraTrees folds; all baseline predictions for
  all arms; the development check, main baselines and decompositions; the 20-draw
  check; the sweep; 396 Qwen H answers and the complete Qwen evaluation.
* **Reused:** graphs, labels, pool definition and split; every R, S and B
  observation and prompt (byte-identical); 1 260 Qwen R/S/B answers.
* **Deterministic H:** `sp_highschool2013` only (target unreachable, d = D, −0.98 %).
* **Is H history-shaped?** On the real panel, yes: the history component is 98 % of
  the absolute plug-in error, with ρ₂ losses of 13–25 points on four sources. On
  CollegeMsg, MathOverflow, DAR α = 0 and activity-driven without memory it is not;
  on DAR α = 0.8 and activity-driven with memory the loss is small on ρ₂ and large on
  ρ₃–ρ₅. These limit cases are reported, not tuned away.
