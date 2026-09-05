# Faithful FedPhoenix 2/64 Conditional Utility Validation

## Scope

Two independent formal 5-epoch LocalTrainer measurements were made for each client × reset-state pair on the frozen faithful 2/64 Clean trajectory. Primary utility is fixed before analysis: `V_full(i,j) = L_fed_ref(reset_j) - L_fed_ref(w_full(i,j))`, using a fixed, sample-size-weighted 1,600-example stratified training-set CE objective. No test samples, Probe, score, gate, Hungarian training policy, or FL trajectory was changed.

## Interaction reproducibility and noise

| Snapshot | Pearson (null percentile / p) | Spearman (null percentile / p) | mean interaction var. | interaction noise var. | I/noise |
| ---: | --- | --- | ---: | ---: | ---: |
| 20 | 0.050 (66.90% / 0.6568) | 0.012 (54.17% / 0.9138) | 0.00506938 | 0.00481859 | 1.052 |
| 40 | -0.257 (2.17% / 0.0395) | -0.359 (0.72% / 0.0164) | 0.0905183 | 0.111128 | 0.815 |
| 120 | 0.281 (97.92% / 0.0303) | 0.296 (97.24% / 0.0326) | 0.746804 | 0.537271 | 1.390 |
| 160 | 0.491 (99.79% / 0.0054) | 0.358 (99.75% / 0.0052) | 0.0503091 | 0.0268741 | 1.872 |

## Held-out Hungarian decision value

Each row selects Hungarian on one replicate and evaluates only on the other. Random percentiles and p-values are drawn directly from the held-out matrix's 10,000 one-to-one assignments.

| Snapshot | Direction | Oracle − Clean | Clean percentile | Cross-fit percentile / p |
| ---: | --- | ---: | ---: | --- |
| 20 | V1->V2 | 0.549683 | 8.44% | 80.94% / 0.1907 |
| 20 | V2->V1 | 0.176519 | 16.76% | 42.00% / 0.5800 |
| 40 | V1->V2 | -0.633419 | 54.92% | 21.00% / 0.7900 |
| 40 | V2->V1 | -0.134940 | 38.43% | 32.45% / 0.6755 |
| 120 | V1->V2 | -0.064692 | 79.38% | 77.24% / 0.2277 |
| 120 | V2->V1 | -1.457330 | 95.17% | 33.79% / 0.6621 |
| 160 | V1->V2 | 0.275160 | 67.47% | 86.40% / 0.1361 |
| 160 | V2->V1 | 0.585357 | 68.16% | 84.41% / 0.1560 |

## Assignment overlap and margins (auxiliary)

| Snapshot | V1/V2 overlap | overlap null percentile / p | V1 margin/K | V1 margin/std(I) | V2 margin/K | V2 margin/std(I) |
| ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 20 | 0.10 | 72.92% / 0.6284 | 0.001323 | 0.189 | 0.002600 | 0.359 |
| 40 | 0.00 | 36.86% / 1.0000 | 0.001263 | 0.057 | 0.002422 | 0.066 |
| 120 | 0.10 | 73.38% / 0.6295 | 0.000148 | 0.002 | 0.000885 | 0.010 |
| 160 | 0.10 | 73.06% / 0.6309 | 0.001080 | 0.042 | 0.000130 | 0.007 |

## Decision

**regime-dependent / heterogeneous**

The four frozen snapshots are not consistent: r20 is near its interaction null, r40 has negative interaction correspondence and negative held-out matching value, r120 has positive interaction correspondence but negative held-out matching value, and only r160 has both primary evidence streams positive. This does not support a stable, trajectory-wide conditional-utility assignment effect.

Primary evidence is replicate interaction correspondence relative to its structure-preserving null and bidirectional held-out matching utility. Noise, overlap, and margins are explanatory rather than standalone vetoes.
