# v10 interaction-walk gate

Cluster job 7143357; 1,000 independent walks per graph at its recalibrated 10% length.
The 24-graph audit failed the prespecified gate. No v10 offline study or Qwen production was submitted.

Gate applies to each real or surrogate source with absolute stationary interaction-walk shift above 0.05:
absolute mean S design bias must be at most 0.01 and S design RMSE at most half the plugin RMSE.

| Source | Stationary shift | Plugin bias | Plugin RMSE | S bias | S RMSE | S RMSE / plugin | Gate |
|---|---:|---:|---:|---:|---:|---:|---|
| sp_hospital | +0.3121 | +0.2920 | 0.2960 | +0.0174 | 0.1282 | 0.433 | FAIL |
| sp_highschool2013 | +0.4530 | +0.4014 | 0.4018 | +0.0082 | 0.0727 | 0.181 | pass |
| copenhagen_bluetooth | +0.4574 | +0.4233 | 0.4234 | +0.0009 | 0.0204 | 0.048 | pass |
| sp_workplace | +0.4341 | +0.3862 | 0.3872 | +0.0052 | 0.0508 | 0.131 | pass |
| snap_email_eu | +0.4391 | +0.4141 | 0.4142 | +0.0026 | 0.0408 | 0.098 | pass |
| snap_collegemsg | +0.1962 | +0.1488 | 0.1499 | +0.0000 | 0.0106 | 0.071 | pass |
| snap_mathoverflow | +0.1839 | +0.1525 | 0.1535 | -0.0004 | 0.0063 | 0.041 | pass |
| sp_hospital__pwt | +0.1582 | +0.1559 | 0.1562 | +0.0300 | 0.1535 | 0.983 | FAIL |
| sp_highschool2013__pwt | +0.3252 | +0.3130 | 0.3131 | +0.0106 | 0.0816 | 0.261 | FAIL |
| copenhagen_bluetooth__pwt | +0.2749 | +0.2699 | 0.2699 | +0.0004 | 0.0248 | 0.092 | pass |
| sp_workplace__pwt | +0.3454 | +0.3340 | 0.3342 | +0.0093 | 0.0835 | 0.250 | pass |
| snap_email_eu__pwt | +0.2948 | +0.2852 | 0.2852 | +0.0025 | 0.0469 | 0.165 | pass |
| snap_collegemsg__pwt | +0.3383 | +0.3150 | 0.3152 | +0.0012 | 0.0268 | 0.085 | pass |
| snap_mathoverflow__pwt | +0.3126 | +0.2862 | 0.2883 | -0.0007 | 0.0198 | 0.069 | pass |

3 of 14 applicable sources failed.
The CSV and JSON contain all 24 graph rows, rho_2..rho_5 stationary targets for uniform, degree, and interaction weights,
bias, SD, RMSE for plugin, S, and S_obs, and revisit, effective sample size, and inclusion diagnostics.
