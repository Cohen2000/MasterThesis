# Seeing only part of the time axis (arm H)

How to read this: arm H sees only the last 60% of the time axis (3 of 5 windows). A pair active in
two early windows looks active in fewer windows, and pairs active only early disappear. These tables
measure both effects on the complete networks, without any node sampling. Terms are explained in the [glossary](../../DESIGN.md#glossary).

- K = active windows of a pair among all five; J = among the three visible ones.
- **Numerator loss**: share of pairs with K >= k that no longer show J >= k.
- **Denominator loss**: share of all interacting pairs that become invisible (J = 0).
- The observed share is correct only when both losses are equal. With three visible windows, J >= 4
  is impossible, so the observed share of rho_4 and rho_5 is always 0.
- "Last 60%" is what arm H sees; "first 60%" is shown for comparison. "With node sampling" uses the
  actual H observations (time cut plus random nodes).
- Averages over the 12 real networks and over their 12 time-shuffled copies.

## Losses

| Networks | Visible part | k | Numerator loss | Denominator loss | Error of the observed share | Networks with |error| <= 0.01 |
|---|---|---:|---:|---:|---:|---:|
| real | last 60% | 2 | 0.5123 | 0.4010 | -0.0524 | 1/12 |
| real | last 60% | 3 | 0.6606 | 0.4010 | -0.0606 | 4/12 |
| real | last 60% | 4 | 1.0000 | 0.4010 | -0.1026 | 3/12 |
| real | last 60% | 5 | 1.0000 | 0.4010 | -0.0548 | 5/12 |
| real | first 60% | 2 | 0.3555 | 0.2137 | -0.0585 | 1/12 |
| real | first 60% | 3 | 0.5441 | 0.2137 | -0.0630 | 2/12 |
| real | first 60% | 4 | 1.0000 | 0.2137 | -0.1026 | 3/12 |
| real | first 60% | 5 | 1.0000 | 0.2137 | -0.0548 | 5/12 |
| time-shuffled copy | last 60% | 2 | 0.3514 | 0.2177 | -0.0980 | 1/12 |
| time-shuffled copy | last 60% | 3 | 0.5308 | 0.2177 | -0.1563 | 1/12 |
| time-shuffled copy | last 60% | 4 | 1.0000 | 0.2177 | -0.3391 | 1/12 |
| time-shuffled copy | last 60% | 5 | 1.0000 | 0.2177 | -0.2198 | 2/12 |
| time-shuffled copy | first 60% | 2 | 0.2524 | 0.1457 | -0.0595 | 1/12 |
| time-shuffled copy | first 60% | 3 | 0.4246 | 0.1457 | -0.1217 | 1/12 |
| time-shuffled copy | first 60% | 4 | 1.0000 | 0.1457 | -0.3391 | 1/12 |
| time-shuffled copy | first 60% | 5 | 1.0000 | 0.1457 | -0.2198 | 2/12 |

## Errors

| Networks | Estimator | Visible part | With node sampling | Signed error | MAE_2 | ProfileMAE |
|---|---|---:|---:|---:|---:|---:|
| real | Observed share (plug-in) | last 60% | no | -0.0524 | 0.0649 | 0.0727 |
| real | Statistical model (MLE) | last 60% | no | 0.0657 | 0.0689 | 0.0435 |
| real | Observed share (plug-in) | first 60% | no | -0.0585 | 0.0585 | 0.0697 |
| real | Statistical model (MLE) | first 60% | no | 0.0585 | 0.0601 | 0.0385 |
| real | Observed share (plug-in) | last 60% | yes | -0.0533 | 0.0639 | 0.0720 |
| real | Statistical model (MLE) | last 60% | yes | 0.0676 | 0.0706 | 0.0444 |
| time-shuffled copy | Observed share (plug-in) | last 60% | no | -0.0980 | 0.0980 | 0.2033 |
| time-shuffled copy | Statistical model (MLE) | last 60% | no | 0.0105 | 0.0456 | 0.0388 |
| time-shuffled copy | Observed share (plug-in) | first 60% | no | -0.0595 | 0.0595 | 0.1850 |
| time-shuffled copy | Statistical model (MLE) | first 60% | no | 0.0382 | 0.0467 | 0.0454 |
| time-shuffled copy | Observed share (plug-in) | last 60% | yes | -0.0934 | 0.0945 | 0.2008 |
| time-shuffled copy | Statistical model (MLE) | last 60% | yes | 0.0150 | 0.0491 | 0.0471 |

Per network and k: `HISTORY.csv`.
