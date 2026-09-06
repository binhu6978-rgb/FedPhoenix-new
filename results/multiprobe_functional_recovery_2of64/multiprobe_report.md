# Multi-Probe Functional Recovery

## Controlled change

The only method change is the number of independent short Functional Recovery probes. For each client and replicate, one concrete disjoint support/query batch is reused across all ten transient states. Different replicates use isolated seeds and independently materialized batches. The raw recovery matrices are averaged elementwise, then the existing one-to-one Hungarian assignment is applied:

`Q(i,j) = mean_m[L_reset_query(i,j,m) - L_adapted_query(i,j,m)]`

Formal local training still reloads the original TaskSpec state and uses the unchanged client-specific formal seed. Network, data, partition, reset, LocalTrainer, FedAvg, and evaluation are unchanged. The tested counts are M=2, M=3, and M=5; M=4 was omitted to keep the experiment small while covering weak, medium, and stronger averaging.

## Accuracy

| Variant | M | Peak | Peak round | Delta Clean | Delta M=1 | Final | Last20 | Last50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Clean | — | 73.70% | 175 | — | -0.31 pp | 72.29% | 67.316% | 66.701% |
| Full Ours | 1 | 74.01% | 192 | +0.31 pp | — | 70.94% | 68.360% | 67.630% |
| **Multi-Probe** | **2** | **74.34%** | **164** | **+0.64 pp** | **+0.33 pp** | **72.66%** | **68.939%** | 67.707% |
| Multi-Probe | 3 | 73.92% | 192 | +0.22 pp | -0.09 pp | 71.64% | 67.901% | 67.513% |
| Multi-Probe | 5 | 74.26% | 164 | +0.56 pp | +0.25 pp | 71.89% | 68.790% | **67.710%** |

M=2 is the new best variant. It improves peak by 0.33 percentage points over M=1 and 0.64 points over Clean. Its average difference from Clean is -0.281 pp in rounds 1–40, then +0.566, +0.289, +0.455, and +0.846 pp in the subsequent forty-round blocks. M=5 also exceeds M=1, but remains 0.08 pp below M=2. M=3 does not improve the peak.

## Reliability and assignment diagnostics

| M | Mean raw score variance | Mean pair std | Mean score standard error | Replicate-to-mean assignment overlap | Assignment change rate | Best-vs-second margin | Assignment overlap with M=1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 0.01795 | 0.10237 | 0.07238 | 65.55% | 90.25% | 0.01637 | 26.90% |
| 3 | 0.02379 | 0.13134 | 0.07583 | 60.25% | 89.90% | 0.01607 | 29.85% |
| 5 | 0.02752 | 0.14993 | **0.06705** | 58.27% | 90.55% | 0.01594 | 26.65% |

The uncertainty measures do not improve monotonically with M. M=5 has the lowest estimated standard error, as expected from stronger averaging, but M=3 is slightly worse than M=2. Raw within-round variance is not directly comparable as a pure M effect because each M changes assignments from round 1 and therefore follows a different model trajectory.

The averaged assignment differs substantially from M=1, with only 27–30% mean edge overlap. However, averaging does not make dispatch less aggressive: every variant still changes about nine of ten baseline pairs, and the best-vs-second assignment margin stays small. Therefore the accuracy evidence supports using exactly two probes, but does not yet support adding a variance penalty or claiming that assignment instability has been solved.

## Verification

- M=1 integration smoke exactly reproduced the pre-change two-round assignments, accuracies, task/local seeds, objectives, and global hashes.
- Replicate seeds and support/query samples are independent across replicates.
- A fixed client/replicate uses identical concrete support/query hashes for every state.
- Mean score aggregation and the resulting Hungarian bijection are tested directly.
- Additional probes do not alter client sampling or formal local seeds.
- Formal local training is asserted to start from the original TaskSpec hash.
- All three formal runs completed 200 rounds with the authoritative configuration and no runtime failure.
- Full regression suite: 67 tests passed.

## Decision

Retain **M=2** as the simplest and strongest version. The replicate-count sweep stops here. A future reliability-aware assignment study would need to address the persistent approximately 90% reassignment rate and small assignment margins, but this experiment does not add such a mechanism.
