# M2 recovery measurement optimization

Only the recovery horizon or trajectory summary changes. All variants use two independent Probes, mean replicate aggregation, and unchanged Hungarian assignment.

| Variant | Peak | Round | Delta Clean | Delta M2 | Delta 5-step | Final | Last20 | Last50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 5-step terminal (reference) | 74.70% | 164 | +1.00 pp | +0.36 pp | — | 71.75% | 68.056% | 67.437% |
| 6-step terminal | 74.29% | 192 | +0.59 pp | -0.05 pp | -0.41 pp | 70.14% | 68.236% | 67.244% |
| 8-step terminal | 74.04% | 175 | +0.34 pp | -0.30 pp | -0.66 pp | 70.11% | 68.702% | 67.432% |
| 5-step trajectory mean | 74.20% | 184 | +0.50 pp | -0.14 pp | -0.50 pp | 71.28% | 68.665% | 67.734% |
| 5-step endpoints mean | 74.44% | 164 | +0.74 pp | +0.10 pp | -0.26 pp | 69.59% | 67.657% | 67.605% |

Best candidate: **5-step endpoints mean**, 74.44% at round 164 (-0.26 pp versus 5-step terminal).

| Variant | Score SE | Assignment margin | Assignment overlap with M2 |
|---|---:|---:|---:|
| 6-step terminal | 0.11018 | 0.01564 | 26.25% |
| 8-step terminal | 0.11243 | 0.01828 | 28.35% |
| 5-step trajectory mean | 0.09311 | 0.01702 | 32.30% |
| 5-step endpoints mean | 0.08588 | 0.01534 | 32.85% |

The analysis verified identical selected clients, task seeds, and local seeds against M2 for all 200 rounds.

## Decision

None of the horizon or trajectory candidates exceeds the 74.70% 5-step terminal reference. The deeper terminal horizons decline, while averaging the recovery path suppresses the useful late-horizon signal. Keep 5-step terminal as the peak leader and stop this optimization axis.
