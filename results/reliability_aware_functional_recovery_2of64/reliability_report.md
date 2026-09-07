# Reliability-Aware M2 Functional Recovery

## Controlled change

Both variants retain the accepted two independent raw-G probes, formal LocalTrainer, original TaskSpec reload, Hungarian matching, FedAvg, client sampling, and faithful FedPhoenix 2/64 setup. Only the matrix supplied to Hungarian changes:

- `half_se_lcb`: `Q = mean(G1,G2) - 0.5 * |G1-G2|/2`.
- `one_se_lcb`: `Q = mean(G1,G2) - |G1-G2|/2 = min(G1,G2)`.

These are fixed mild and strict reliability corrections, not a lambda sweep.

## Accuracy

| Variant | Peak | Peak round | Delta Clean | Delta M2 | Final | Last20 | Last50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Clean | 73.70% | 175 | +0.00 pp | -0.64 pp | 72.29% | 67.316% | 66.701% |
| M1 | 74.01% | 192 | +0.31 pp | -0.33 pp | 70.94% | 68.360% | 67.630% |
| M2 mean | 74.34% | 164 | +0.64 pp | +0.00 pp | 72.66% | 68.939% | 67.707% |
| **M2 half-SE LCB** | **74.42%** | 164 | +0.72 pp | +0.08 pp | 69.50% | 68.460% | 67.762% |
| M2 one-SE LCB | 73.88% | 175 | +0.18 pp | -0.46 pp | 71.93% | 68.675% | 67.746% |

The mild correction is the nominal new peak leader at 74.42%, but its gain over M2 is only 0.08 pp. The strict correction loses 0.46 pp versus M2, so stronger disagreement penalization is not supported.

## Assignment diagnostics

| Variant | Replicate overlap | Change rate | Margin | Penalty (all) | Penalty (selected) | Overlap with M2 |
|---|---:|---:|---:|---:|---:|---:|
| half_se_lcb | 64.05% | 90.25% | 0.01468 | 0.05085 | 0.04877 | 32.15% |
| one_se_lcb | 60.68% | 89.85% | 0.01658 | 0.10159 | 0.09227 | 28.95% |

## Stage-wise mean accuracy delta

### half_se_lcb

| Rounds | Delta Clean | Delta M2 |
|---|---:|---:|
| 1-40 | -0.199 pp | +0.082 pp |
| 41-80 | +0.376 pp | -0.190 pp |
| 81-120 | -0.122 pp | -0.411 pp |
| 121-160 | -0.209 pp | -0.664 pp |
| 161-200 | +0.761 pp | -0.085 pp |

### one_se_lcb

| Rounds | Delta Clean | Delta M2 |
|---|---:|---:|
| 1-40 | -0.407 pp | -0.126 pp |
| 41-80 | +0.459 pp | -0.107 pp |
| 81-120 | -0.065 pp | -0.354 pp |
| 121-160 | +0.461 pp | +0.006 pp |
| 161-200 | +0.686 pp | -0.160 pp |

## Decision

Retain `half_se_lcb` as the simple seed-1 peak leader, while describing the +0.08 pp improvement as marginal rather than robust. Stop the reliability-strength search: the strict analytic endpoint is clearly worse, and neither rule reduces the approximately nine-of-ten reassignment rate. The next assignment study, if pursued, should target structural stability of the permutation rather than apply a larger elementwise uncertainty penalty.
