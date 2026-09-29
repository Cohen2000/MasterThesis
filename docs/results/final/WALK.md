# Random-walk check (arm S)

How to read this: for every network, 1000 random walks of the arm-S length L were simulated. The
table shows how far the observed share lands from the truth on average (bias) and how much it
varies between walks (SD), for the plain observed share and for the re-weighted design ratio.
Busy pairs are visited more often, which pushes the observed share up. Terms are explained in the [glossary](../../DESIGN.md#glossary).

- **Shift**: how much the walk's preference for busy pairs moves rho_2 on its own.
- **Check needed**: yes when the shift is larger than 0.05. **Correctable**: yes when the design ratio
  removes at least 90% of that bias and halves the error. A "no" is reported, and the network stays in
  every table.
- **Effective walks (weights / ratio)**: how many independent observations the walk is worth.
- **Revisits**: share of steps that return to an already seen pair. **Pairs**: distinct pairs seen per walk.

| Network | Group | L | Shift | Observed share bias | Observed share SD | Design ratio bias | Design ratio SD | Check needed | Correctable | Effective walks (weights) | Effective walks (ratio) | Revisits | Pairs |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ad_memory_r1 | synthetic | 220 | 0.1484 | 0.1364 | 0.0187 | 0.0023 | 0.0533 | False | True | 97.6 | 122.1 | 0.236 | 168.2 |
| ad_memory_r2 | synthetic | 215 | 0.1562 | 0.1438 | 0.0199 | 0.0058 | 0.0532 | False | True | 93.4 | 118.1 | 0.246 | 162.1 |
| ad_memoryless_r1 | synthetic | 1080 | 0.0517 | 0.0456 | 0.0087 | -0.0001 | 0.0054 | False | True | 908.7 | 1043.2 | 0.075 | 998.6 |
| ad_memoryless_r2 | synthetic | 1014 | 0.0636 | 0.0555 | 0.0103 | -0.0002 | 0.0062 | False | True | 844.9 | 970.2 | 0.077 | 935.7 |
| copenhagen_bluetooth | real | 5895 | 0.4574 | 0.4233 | 0.0052 | 0.0009 | 0.0204 | True | True | 496.2 | 500.1 | 0.288 | 4195.4 |
| copenhagen_bluetooth__pwt | time-shuffled copy | 7651 | 0.2749 | 0.2699 | 0.0018 | 0.0004 | 0.0248 | True | True | 641.5 | 647.4 | 0.325 | 5163.9 |
| dar_a08_r1 | synthetic | 183 | 0.1446 | 0.1293 | 0.0239 | 0.0023 | 0.0581 | False | True | 82.3 | 112.6 | 0.274 | 132.9 |
| dar_a08_r2 | synthetic | 179 | 0.1517 | 0.1356 | 0.0243 | 0.0021 | 0.0619 | False | True | 78.5 | 108.0 | 0.278 | 129.3 |
| dar_a0_r1 | synthetic | 337 | 0.1991 | 0.1730 | 0.0274 | 0.0013 | 0.0316 | False | True | 196.9 | 234.4 | 0.158 | 283.8 |
| dar_a0_r2 | synthetic | 334 | 0.2024 | 0.1763 | 0.0287 | 0.0009 | 0.0322 | False | True | 192.1 | 229.3 | 0.160 | 280.7 |
| lkml_reply | real | 16436 | 0.4461 | 0.3028 | 0.1373 | -0.0119 | 0.0436 | True | True | 2720.6 | 5356.9 | 0.389 | 10045.4 |
| lkml_reply__pwt | time-shuffled copy | 14319 | 0.3791 | 0.3021 | 0.2148 | -0.0303 | 0.1345 | True | False | 2369.6 | 4574.1 | 0.367 | 9059.5 |
| nr_digg_reply | real | 11309 | 0.0040 | 0.0038 | 0.0014 | -0.0001 | 0.0006 | False | True | 6523.6 | 11232.6 | 0.262 | 8341.6 |
| nr_digg_reply__pwt | time-shuffled copy | 11315 | 0.0101 | 0.0087 | 0.0030 | -0.0002 | 0.0015 | False | True | 6541.8 | 11238.2 | 0.260 | 8370.3 |
| nr_radoslaw_email | real | 249 | 0.4033 | 0.3881 | 0.0197 | 0.0091 | 0.1077 | True | True | 30.1 | 30.9 | 0.278 | 179.8 |
| nr_radoslaw_email__pwt | time-shuffled copy | 312 | 0.1899 | 0.1853 | 0.0086 | 0.0048 | 0.1049 | True | False | 37.0 | 38.2 | 0.309 | 215.7 |
| reality_mining | real | 245 | 0.3586 | 0.3432 | 0.0197 | 0.0370 | 0.1841 | True | False | 16.0 | 16.4 | 0.356 | 157.7 |
| reality_mining__pwt | time-shuffled copy | 294 | 0.2015 | 0.2000 | 0.0048 | 0.0399 | 0.1786 | True | False | 18.3 | 18.7 | 0.383 | 181.3 |
| snap_collegemsg | real | 1519 | 0.1962 | 0.1488 | 0.0180 | 0.0000 | 0.0107 | True | True | 525.8 | 628.1 | 0.231 | 1168.2 |
| snap_collegemsg__pwt | time-shuffled copy | 1141 | 0.3383 | 0.3150 | 0.0136 | 0.0012 | 0.0268 | True | True | 399.4 | 470.1 | 0.204 | 908.5 |
| snap_email_eu | real | 1362 | 0.4391 | 0.4141 | 0.0096 | 0.0026 | 0.0407 | True | True | 152.8 | 159.2 | 0.342 | 895.6 |
| snap_email_eu__pwt | time-shuffled copy | 1537 | 0.2948 | 0.2852 | 0.0054 | 0.0025 | 0.0469 | True | True | 172.3 | 179.8 | 0.358 | 987.1 |
| snap_mathoverflow | real | 18219 | 0.1839 | 0.1525 | 0.0171 | -0.0004 | 0.0063 | True | True | 9614.6 | 11695.8 | 0.169 | 15144.3 |
| snap_mathoverflow__pwt | time-shuffled copy | 14498 | 0.3126 | 0.2862 | 0.0352 | -0.0007 | 0.0198 | True | True | 7748.9 | 9297.9 | 0.150 | 12317.8 |
| sp_highschool2013 | real | 674 | 0.4530 | 0.4014 | 0.0174 | 0.0082 | 0.0723 | True | True | 47.7 | 48.7 | 0.514 | 327.6 |
| sp_highschool2013__pwt | time-shuffled copy | 746 | 0.3252 | 0.3130 | 0.0080 | 0.0106 | 0.0810 | True | True | 53.2 | 54.3 | 0.519 | 358.8 |
| sp_hospital | real | 109 | 0.3121 | 0.2920 | 0.0488 | 0.0174 | 0.1271 | True | True | 17.8 | 18.7 | 0.272 | 79.4 |
| sp_hospital__pwt | time-shuffled copy | 116 | 0.1582 | 0.1559 | 0.0108 | 0.0300 | 0.1506 | True | False | 19.2 | 20.1 | 0.280 | 83.6 |
| sp_malawi | real | 178 | 0.4782 | 0.3629 | 0.1628 | 0.1208 | 0.2905 | True | False | 6.4 | 20.5 | 0.888 | 19.9 |
| sp_malawi__pwt | time-shuffled copy | 273 | 0.1637 | 0.1586 | 0.0151 | 0.0832 | 0.2008 | True | False | 7.9 | 25.7 | 0.903 | 26.5 |
| sp_workplace | real | 404 | 0.4341 | 0.3862 | 0.0282 | 0.0052 | 0.0506 | True | True | 46.7 | 47.7 | 0.341 | 266.4 |
| sp_workplace__pwt | time-shuffled copy | 414 | 0.3454 | 0.3340 | 0.0111 | 0.0093 | 0.0830 | True | True | 48.3 | 49.3 | 0.341 | 272.6 |
