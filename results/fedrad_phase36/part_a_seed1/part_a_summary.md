# Phase 3.6 Part A — Two-Probe Averaging

| Candidate | Estimator | Functional pct | Q-int P/S | Held-out overlap | Exact | Margin/K | Margin/Q std |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Full | Single 32/32 | 87.43 | 0.460/0.474 | 0.247 | 0.000 | 0.0090 | 0.0301 |
| Full | Average 2x32/32 | 92.08 | 0.532/0.541 | 0.280 | 0.000 | 0.0123 | 0.0408 |
| G | Single 32/32 | 87.37 | 0.502/0.505 | 0.227 | 0.000 | 0.0036 | 0.0359 |
| G | Average 2x32/32 | 92.63 | 0.575/0.578 | 0.247 | 0.000 | 0.0030 | 0.0298 |

Part B trigger: **True**

Both Full and G must improve mean Pearson and Spearman, not decrease mean held-out functional percentile, and improve mean overlap with held-out optimal assignment.
