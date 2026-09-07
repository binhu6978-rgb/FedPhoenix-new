# Consensus-Aware M2 Assignment

## Controlled variants

All variants retain M2's two independent raw-G probes and mean score. Only the one-to-one assignment feasible set changes:

- `consensus_lock`: lock edges shared by the two Probe-specific Hungarian permutations, then solve the remainder on mean G.
- `union_restrict`: allow baseline edges and edges selected by either Probe-specific Hungarian permutation.
- `bilateral_gain`: allow a reassignment edge only when both probes score it above that client's baseline edge.

## Accuracy

| Variant | Peak | Peak round | Delta Clean | Delta M2 | Delta 74.42 | Final | Last20 | Last50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Clean | 73.70% | 175 | — | -0.64 pp | -0.72 pp | 72.29% | 67.316% | 66.701% |
| M2 mean | 74.34% | 164 | +0.64 pp | — | -0.08 pp | 72.66% | 68.939% | 67.707% |
| half-SE | 74.42% | 164 | +0.72 pp | +0.08 pp | — | 69.50% | 68.460% | 67.762% |
| consensus_lock | 74.10% | 164 | +0.40 pp | -0.24 pp | -0.32 pp | 71.90% | 67.052% | 66.892% |
| union_restrict | 73.52% | 164 | -0.18 pp | -0.82 pp | -0.90 pp | 70.38% | 66.027% | 66.502% |
| bilateral_gain | 73.48% | 175 | -0.22 pp | -0.86 pp | -0.94 pp | 71.11% | 68.180% | 67.342% |

No structural variant beats M2 or the nominal half-SE peak. Consensus-Lock is best of this group but remains 0.24 pp below M2.

## Assignment behavior

| Variant | Reassignment rate | Shared Probe edges/10 | Final supported by both | Final supported by either | Reassigned edges with bilateral gain | Overlap with M2 |
|---|---:|---:|---:|---:|---:|---:|
| consensus_lock | 89.85% | 4.20 | 42.00% | 92.45% | 73.58% | 29.70% |
| union_restrict | 88.40% | 4.47 | 44.70% | 98.50% | 72.32% | 28.65% |
| bilateral_gain | 59.55% | 4.13 | 21.70% | 52.10% | 100.00% | 21.50% |

## Stage-wise mean accuracy delta

### consensus_lock

| Rounds | Delta Clean | Delta M2 |
|---|---:|---:|
| 1-40 | -0.194 pp | +0.087 pp |
| 41-80 | +0.589 pp | +0.023 pp |
| 81-120 | -0.304 pp | -0.593 pp |
| 121-160 | +0.391 pp | -0.063 pp |
| 161-200 | -0.191 pp | -1.037 pp |

### union_restrict

| Rounds | Delta Clean | Delta M2 |
|---|---:|---:|
| 1-40 | -0.389 pp | -0.108 pp |
| 41-80 | +0.296 pp | -0.270 pp |
| 81-120 | -0.507 pp | -0.796 pp |
| 121-160 | +0.455 pp | +0.001 pp |
| 161-200 | -0.767 pp | -1.613 pp |

### bilateral_gain

| Rounds | Delta Clean | Delta M2 |
|---|---:|---:|
| 1-40 | +0.191 pp | +0.473 pp |
| 41-80 | +0.214 pp | -0.352 pp |
| 81-120 | -0.182 pp | -0.471 pp |
| 121-160 | +0.435 pp | -0.020 pp |
| 161-200 | +0.281 pp | -0.564 pp |

## Decision

Stop the consensus/partial-Hungarian direction. The light constraint barely changes the global reassignment rate and loses peak accuracy; the stronger feasible-set restrictions reduce or certify reassignments as intended but degrade accuracy further. Return to plain M2 as the clean primary method. The 74.42% half-SE result remains only a nominal seed-1 peak, not sufficient evidence to complicate the main method.
