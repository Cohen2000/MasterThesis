# History truncation (arm H)

Population level: every dyad of the graph, no node sampling. K counts active windows among all five,
J among the three retained ones (last 60% = windows 3-5, first 60% = windows 1-3). For k = 2..5 the
numerator loss is 1 - #(J>=k)/#(K>=k) and the denominator loss 1 - #(J>=1)/#(K>=1); the truncated
plugin equals rho_k (1 - numerator loss)/(1 - denominator loss), so the losses cancel only when
they are equal. With three windows J >= 4 is impossible, so the truncated plugin is 0 for k = 4, 5.
"H sampled" uses the actual H draws (last 60% plus the node panel): truncation plus node sampling.
Equal-graph means over the 12 real sources and the 12 surrogates.

## Losses

| Group | Window | k | Numerator loss | Denominator loss | Plugin signed error | |error| <= 0.01 |
|---|---|---:|---:|---:|---:|---:|
| real | last60 | 2 | 0.5123 | 0.4010 | -0.0524 | 1/12 |
| real | last60 | 3 | 0.6606 | 0.4010 | -0.0606 | 4/12 |
| real | last60 | 4 | 1.0000 | 0.4010 | -0.1026 | 3/12 |
| real | last60 | 5 | 1.0000 | 0.4010 | -0.0548 | 5/12 |
| real | first60 | 2 | 0.3555 | 0.2137 | -0.0585 | 1/12 |
| real | first60 | 3 | 0.5441 | 0.2137 | -0.0630 | 2/12 |
| real | first60 | 4 | 1.0000 | 0.2137 | -0.1026 | 3/12 |
| real | first60 | 5 | 1.0000 | 0.2137 | -0.0548 | 5/12 |
| surrogate | last60 | 2 | 0.3514 | 0.2177 | -0.0980 | 1/12 |
| surrogate | last60 | 3 | 0.5308 | 0.2177 | -0.1563 | 1/12 |
| surrogate | last60 | 4 | 1.0000 | 0.2177 | -0.3391 | 1/12 |
| surrogate | last60 | 5 | 1.0000 | 0.2177 | -0.2198 | 2/12 |
| surrogate | first60 | 2 | 0.2524 | 0.1457 | -0.0595 | 1/12 |
| surrogate | first60 | 3 | 0.4246 | 0.1457 | -0.1217 | 1/12 |
| surrogate | first60 | 4 | 1.0000 | 0.1457 | -0.3391 | 1/12 |
| surrogate | first60 | 5 | 1.0000 | 0.1457 | -0.2198 | 2/12 |

## Errors

| Group | Estimator | Truncation | Node sampling | Signed rho_2 | MAE_2 | ProfileMAE |
|---|---|---:|---:|---:|---:|---:|
| real | plugin | last60 | no | -0.0524 | 0.0649 | 0.0727 |
| real | mle | last60 | no | 0.0657 | 0.0689 | 0.0435 |
| real | plugin | first60 | no | -0.0585 | 0.0585 | 0.0697 |
| real | mle | first60 | no | 0.0585 | 0.0601 | 0.0385 |
| real | plugin | last60 | yes | -0.0533 | 0.0637 | 0.0717 |
| real | mle | last60 | yes | 0.0676 | 0.0685 | 0.0431 |
| surrogate | plugin | last60 | no | -0.0980 | 0.0980 | 0.2033 |
| surrogate | mle | last60 | no | 0.0105 | 0.0456 | 0.0388 |
| surrogate | plugin | first60 | no | -0.0595 | 0.0595 | 0.1850 |
| surrogate | mle | first60 | no | 0.0382 | 0.0467 | 0.0454 |
| surrogate | plugin | last60 | yes | -0.0934 | 0.0934 | 0.2005 |
| surrogate | mle | last60 | yes | 0.0150 | 0.0487 | 0.0453 |

Per graph and k: `HISTORY.csv`.
