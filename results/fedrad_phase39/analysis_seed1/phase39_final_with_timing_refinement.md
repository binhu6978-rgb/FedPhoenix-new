# Phase 3.9 Final: Matching Timing / Minimal Score Validation

All runs use CIFAR-10, ResNet18, beta=0.3, seed 1, 200 rounds, the frozen
64/32 one-step Probe, unchanged local training, and sample-size weighted FedAvg.

| Variant | Peak | Peak Round | Delta vs Clean | Final | Last-20 | Last-50 | Runtime |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Clean | 73.810% | 175 | +0.000pp | 71.490% | 68.282% | 67.575% | 68.5 min |
| Full-All | 74.440% | 175 | +0.630pp | 71.650% | 68.671% | 67.806% | 94.1 min |
| Full-Warm20 | 74.220% | 175 | +0.410pp | 71.930% | 68.684% | 67.366% | 111.9 min |
| **Full-Warm40** | **74.530%** | **175** | **+0.720pp** | 72.040% | 67.670% | 67.469% | 125.8 min |
| Full-Warm60 | 74.150% | 175 | +0.340pp | 70.970% | 68.372% | 67.534% | 97.8 min |
| Full-Warm80 | 73.950% | 164 | +0.140pp | 71.900% | 68.793% | 67.671% | 115.1 min |
| G-Warm40 | 74.480% | 175 | +0.670pp | 72.890% | 67.979% | 67.415% | 125.9 min |

## Validation

- Full-Warm20: rounds 1-20 exactly match Clean global hashes; zero warm-up Probe.
- Full-Warm40 and G-Warm40: rounds 1-40 exactly match Clean global hashes; zero warm-up Probe.
- Full-Warm60: rounds 1-60 exactly match Clean global hashes; zero warm-up Probe.
- Full-Warm80: rounds 1-80 exactly match Clean global hashes; zero warm-up Probe.
- V1/V2/V3 ran concurrently; Warm20/Warm60 ran concurrently. Their wall runtimes are not clean throughput benchmarks.

## Decision

- Best variant: **Full-Warm40**.
- Peak improvement over Clean: **+0.72pp**.
- Peak improvement over Full-All: **+0.09pp**.
- Warm20, Warm60, and Warm80 do not improve on Full-All; further timing-grid search is stopped.
- G-Warm40 is 0.05pp below Full-Warm40, so the G-only direction is stopped.
- Delaying matching to round 40 gives only a marginal peak improvement and worsens Last-20/Last-50 relative to Full-All. Full-Warm40 is retained only when peak accuracy is the primary objective.
