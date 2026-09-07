# M2 Functional Recovery: Probe Horizon

## Controlled change

The experiment keeps M2 mean aggregation, two independent materialized support/query batches, raw functional recovery G, Hungarian assignment, formal LocalTrainer, FedAvg, and faithful FedPhoenix 2/64 unchanged. Only the number of SGD updates on each fixed Probe support batch changes.

## Accuracy

| Probe steps | Peak | Peak round | Delta Clean | Delta M2 | Delta 74.42 | Final | Last20 | Last50 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 (M2) | 74.34% | 164 | +0.64 pp | — | -0.08 pp | 72.66% | 68.939% | 67.707% |
| 2 | 74.28% | 164 | +0.58 pp | -0.06 pp | -0.14 pp | 71.82% | 68.781% | 67.782% |
| 3 | 73.63% | 184 | -0.07 pp | -0.71 pp | -0.79 pp | 70.63% | 66.521% | 66.690% |
| **5** | **74.70%** | 164 | +1.00 pp | +0.36 pp | +0.28 pp | 71.75% | 68.056% | 67.437% |

Five steps reaches 74.70% at round 164: +0.36 pp over M2 and +0.28 pp over the previous nominal record. Two steps is slightly below M2 and three steps is substantially worse, so the depth effect is non-monotonic.

## Probe and assignment behavior

| Steps | Score SE | Replicate-to-mean overlap | Reassignment rate | Assignment margin | Assignment overlap with 1-step M2 |
|---:|---:|---:|---:|---:|---:|
| 2 | 0.08964 | 64.43% | 91.30% | 0.01675 | 34.05% |
| 3 | 0.10168 | 61.30% | 90.65% | 0.01568 | 33.10% |
| 5 | 0.10877 | 62.10% | 90.65% | 0.02033 | 32.00% |

## Stage-wise mean accuracy delta

### 2-step

| Rounds | Delta Clean | Delta M2 |
|---|---:|---:|
| 1-40 | -0.614 pp | -0.333 pp |
| 41-80 | +0.290 pp | -0.276 pp |
| 81-120 | -0.353 pp | -0.641 pp |
| 121-160 | +0.240 pp | -0.214 pp |
| 161-200 | +0.781 pp | -0.064 pp |

### 3-step

| Rounds | Delta Clean | Delta M2 |
|---|---:|---:|
| 1-40 | -0.194 pp | +0.087 pp |
| 41-80 | +0.146 pp | -0.420 pp |
| 81-120 | +0.014 pp | -0.274 pp |
| 121-160 | +0.429 pp | -0.026 pp |
| 161-200 | -0.366 pp | -1.212 pp |

### 5-step

| Rounds | Delta Clean | Delta M2 |
|---|---:|---:|
| 1-40 | -0.649 pp | -0.368 pp |
| 41-80 | +0.179 pp | -0.387 pp |
| 81-120 | -0.527 pp | -0.816 pp |
| 121-160 | +0.637 pp | +0.183 pp |
| 161-200 | +0.349 pp | -0.497 pp |

## Decision

Retain 5-step M2 Functional Recovery as the new seed-1 peak leader and stop the Probe-depth sweep at the pre-specified success condition. The gain is a peak-specific result: its Final, Last20, and Last50 do not beat plain M2. The increased score uncertainty but larger assignment margin suggest that the benefit comes from measuring a different recovery horizon, not from denoising the Probe.
