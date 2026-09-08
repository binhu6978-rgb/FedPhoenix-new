# M2 5-step support coverage results

Both candidates completed 200 rounds, seed=1, exit code 0, with two concurrent
trainers. Reference: M2 5-step terminal, legacy query semantics, fixed support.

| Variant | Peak | Round | Delta vs 74.70 | Final | Last20 | Last50 |
|---|---:|---:|---:|---:|---:|---:|
| Fixed support reference | 74.70% | 164 | 0.00 pp | 71.75% | 68.0565% | 67.4368% |
| Fully fresh (1,2,3,4,5) | 73.92% | 164 | -0.78 pp | 72.11% | 68.6240% | 67.8632% |
| Partial refresh (1,1,1,2,2) | 74.00% | 192 | -0.70 pp | 72.55% | 68.4510% | 67.6358% |

Mean unique support count per client/round/replicate was 290.461 for fresh and
125.483 for partial refresh. Coverage is capped by the non-query sample pool
of each client; no duplicate samples occur within an individual batch.

The automatic checks passed for all 200 rounds: selected clients, task seeds
and local seeds match the authoritative 5-step reference; support/query sets
are disjoint, and the coverage audit contains 4,000 records per candidate.
Before launch, 77 tests and two GPU smoke runs passed. First support batch and
query are preserved from legacy materialization, with RNG isolation tested.

Decision: retain fixed-support M2 5-step terminal as the peak leader and stop
support coverage tuning. Both candidates have better final and terminal-window
averages than the reference, but neither improves the primary peak objective.
This does not establish that fresh support estimates compatibility less
accurately: no independent utility oracle was measured here. One possible
explanation is that changing samples changes the short-horizon optimization
path; wider coverage alone is not sufficient for higher peak accuracy.

Query coverage is a separate remaining candidate axis, not launched by this
experiment. No automatic additional training follows these results.

Machine-readable results: 20260908_131740/result_summary.json.
Per-round accuracy/assignment logs and configs are included in Git; large raw
Probe/Task logs and per-client support index audits remain available locally.
