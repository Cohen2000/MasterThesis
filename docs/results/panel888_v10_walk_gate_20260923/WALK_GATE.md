# Amended v10 interaction-walk gate

Cluster job 7143961; 1,000 independent walks per graph at the calibrated 10% length.
Post-hoc amendment: the original absolute-bias threshold ignored Monte Carlo error and the
first-order bias of a finite-sample Hájek ratio. The walk and plain S estimator were unchanged.

For each real or surrogate source with absolute stationary interaction-walk shift above 0.05,
the amended gate requires absolute S bias at most 10% of absolute plugin bias and S RMSE
at most half the plugin RMSE. Failures remain in the study and are flagged as
**not correctable at this budget**.

| Source | Shift | Ratio ESS | Plugin bias | S bias ± MCSE | Predicted first-order bias | S / plugin bias | S / plugin RMSE | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| sp_hospital | +0.3121 | 18.7 | +0.2920 | +0.0174 ± 0.0040 | +0.0154 | 0.060 | 0.433 | pass |
| sp_highschool2013 | +0.4530 | 48.7 | +0.4014 | +0.0082 ± 0.0023 | +0.0067 | 0.020 | 0.181 | pass |
| copenhagen_bluetooth | +0.4574 | 500.1 | +0.4233 | +0.0009 ± 0.0006 | +0.0006 | 0.002 | 0.048 | pass |
| sp_workplace | +0.4341 | 47.7 | +0.3862 | +0.0052 ± 0.0016 | +0.0045 | 0.014 | 0.131 | pass |
| snap_email_eu | +0.4391 | 159.2 | +0.4141 | +0.0026 ± 0.0013 | +0.0021 | 0.006 | 0.098 | pass |
| snap_collegemsg | +0.1962 | 628.1 | +0.1488 | +0.0000 ± 0.0003 | +0.0001 | 0.000 | 0.071 | pass |
| snap_mathoverflow | +0.1839 | 11695.8 | +0.1525 | -0.0004 ± 0.0002 | +0.0000 | -0.003 | 0.041 | pass |
| sp_hospital__pwt | +0.1582 | 20.1 | +0.1559 | +0.0300 ± 0.0048 | +0.0168 | 0.193 | 0.983 | not correctable at this budget |
| sp_highschool2013__pwt | +0.3252 | 54.3 | +0.3130 | +0.0106 ± 0.0026 | +0.0071 | 0.034 | 0.261 | pass |
| copenhagen_bluetooth__pwt | +0.2749 | 647.4 | +0.2699 | +0.0004 ± 0.0008 | +0.0006 | 0.001 | 0.092 | pass |
| sp_workplace__pwt | +0.3454 | 49.3 | +0.3340 | +0.0093 ± 0.0026 | +0.0074 | 0.028 | 0.250 | pass |
| snap_email_eu__pwt | +0.2948 | 179.8 | +0.2852 | +0.0025 ± 0.0015 | +0.0021 | 0.009 | 0.165 | pass |
| snap_collegemsg__pwt | +0.3383 | 470.1 | +0.3150 | +0.0012 ± 0.0008 | +0.0006 | 0.004 | 0.085 | pass |
| snap_mathoverflow__pwt | +0.3126 | 9297.9 | +0.2862 | -0.0007 ± 0.0006 | +0.0000 | -0.002 | 0.069 | pass |

1 of 14 applicable sources failed the amended gate.

## Confirmation runs

Cluster array job 7143568; each condition used 1,000 walks at the original calibrated L.
The strength-start condition samples the initial vertex proportional to event strength;
the 4L condition retains the uniform start.

| Source | Uniform L bias ± MCSE | Strength-start L bias ± MCSE | Uniform 4L bias ± MCSE |
|---|---:|---:|---:|
| sp_hospital | +0.0174 ± 0.0040 | +0.0251 ± 0.0042 | +0.0045 ± 0.0021 |
| sp_hospital__pwt | +0.0300 ± 0.0048 | +0.0345 ± 0.0047 | +0.0086 ± 0.0026 |
| sp_highschool2013__pwt | +0.0106 ± 0.0026 | +0.0054 ± 0.0026 | +0.0025 ± 0.0013 |

The CSV and JSON give all 24 graph rows, stationary rho_2..rho_5 targets,
bias, SD and RMSE for plugin, S and S_obs, weight ESS, revisit and inclusion diagnostics.
