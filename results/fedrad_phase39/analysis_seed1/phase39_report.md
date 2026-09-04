# Phase 3.9: Matching Timing / Minimal Score Validation

| Variant | Peak | Peak Round | Delta vs Clean | Final | Last-20 | Last-50 | Runtime |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Clean | 73.810% | 175 | +0.000pp | 71.490% | 68.282% | 67.575% | 68.5 min |
| Full-All | 74.440% | 175 | +0.630pp | 71.650% | 68.671% | 67.806% | 94.1 min |
| Full-Warm40 | 74.530% | 175 | +0.720pp | 72.040% | 67.669% | 67.469% | 125.8 min |
| Full-Warm80 | 73.950% | 164 | +0.140pp | 71.900% | 68.793% | 67.671% | 115.1 min |
| G-Warm40 | 74.480% | 175 | +0.670pp | 72.890% | 67.979% | 67.415% | 125.9 min |

## Warm-up validation

- Full-Warm40: all 40 warm-up global hashes and replay fields equal Clean; post-start win ratio 50.6%, mean delta -0.107pp.
- Full-Warm80: all 80 warm-up global hashes and replay fields equal Clean; post-start win ratio 40.0%, mean delta -0.143pp.
- G-Warm40: all 40 warm-up global hashes and replay fields equal Clean; post-start win ratio 40.6%, mean delta -0.091pp.

## Decision

- Best variant: **Full-Warm40**.
- Peak improvement over Clean: **+0.720pp**.
- Peak improvement over current Full-All: **+0.090pp**.
- G-only direction stops: G-Warm40 did not exceed the best Full timing variant.
- Runtime for V1/V2/V3 is wall-clock time under concurrent GPU execution and is not a clean throughput benchmark.
