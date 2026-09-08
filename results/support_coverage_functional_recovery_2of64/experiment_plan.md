# M2 5-step support coverage controls

Reference: legacy query semantics (reset eval, adapted train), M2 mean,
5-step terminal, Peak 74.70% at round 164. Reference execution remains the
default (`probe_support_coverage=fixed`).

Two candidates, maximum two concurrent trainers:

| Candidate | Batch IDs at steps 1 through 5 |
|---|---|
| fresh | 1, 2, 3, 4, 5 |
| refresh_once | 1, 1, 1, 2, 2 |

First support batch and held-out query exactly match the original seeded
materialization. Batch size remains min(64, client sample count // 2).
Additional support batches traverse the remaining shuffled client samples,
excluding the query; when exhausted, cycle through the same non-query pool.
Clients with insufficient samples therefore reuse samples. There are no
duplicates within an individual batch. This limitation is measured, not
hidden by calling all batches disjoint. Partial refresh uses up to 128 unique
examples, fresh up to 320; both retain five SGD steps and the same per-step size.

Materialize once per client/round/replicate and reuse across all states. The
two replicates keep their existing independent seeds. support_coverage.jsonl
records query indices/hash, support indices/hashes, step schedule and actual
unique support count. ProbeResult.support_hash covers the entire schedule;
the first support batch hash is available separately in the coverage log.

Network, partition, seed=1, rounds=200, K=10, reset_ratio=2/64, local training,
optimizer, FedAvg, Hungarian, query mode and evaluation are unchanged.

Launch using the shared two-process runner:
`python scripts/run_query_mode_experiments.py --support-coverage`
The launcher waits for process completion without reading live training logs.
After completion it checks sampling/task/local seeds against the authoritative
reference, verifies query disjointness and audit completeness, then writes
result_summary.json. No additional variants launch automatically.

Tests cover baseline first-batch/query preservation, independent reproducible
RNG, small-client wraparound, actual unique coverage, exact five-step batch
consumption and TaskSpec integrity. GPU smoke precedes formal launch.

Primary goal: Peak >74.70%. If neither candidate improves it, stop support
coverage tuning. Any explanation of the outcome remains a hypothesis unless
supported by additional evidence. This plan does not claim completed results.
