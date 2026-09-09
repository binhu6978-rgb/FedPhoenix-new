# M2 5-step query coverage results

The authoritative serial runs completed 200 rounds with exit code 0.
The initial parallel attempt was interrupted after round 1 and is excluded.

| Variant | Peak | Round | Delta vs 74.70 | Final | Last20 | Last50 |
|---|---:|---:|---:|---:|---:|---:|
| Query x1 reference | 74.70% | 164 | 0.00 pp | 71.75% | 68.0565% | 67.4368% |
| Query x2 | 74.50% | 175 | -0.20 pp | 71.32% | 67.8785% | 67.4680% |
| Query x4 | 74.48% | 192 | -0.22 pp | 71.34% | 68.5945% | 67.5814% |

Mean unique query coverage was 63.594 for x2 and 123.953 for x4. The
automatic audit verified identical selected clients, task seeds and local
seeds against the reference at all 200 rounds, complete coverage records, and
support/query disjointness.

Neither candidate improves the primary peak objective. Retain Query x1 and
stop query coverage tuning. Wider query coverage alone is insufficient to
increase peak in this experiment; the result does not establish why.

Validation before launch: 79 tests and two GPU smoke runs passed. Query
evaluation uses private model copies, preserving the five-step adaptation
state and separate 32-example BatchNorm semantics.

Machine-readable results: 20260908_165530/result_summary.json.
