# Fixed-support M2 5-step query coverage

Reference: 74.70% at round 164, legacy reset-eval/adapted-train query modes.
Candidates: 2 and 4 query batches per adaptation replicate, each of the same
size as the legacy query (32 unless the client has fewer available samples).
M2 continues to perform two independently seeded adaptations, not 4 or 8.

Keep the original support and first query exactly. Draw additional query
examples from the remaining shuffled client pool, exclude support throughout,
and cycle the non-support pool when exhausted. No duplicates within a batch;
small clients may reuse examples across batches. Record actual unique query
coverage in query_coverage.jsonl. Materialization is shared across states.

Each query runs on a private copy of the same reset/adapted model with RNG
restoration. Reset queries use eval, adapted queries use train. Compute mean
reset loss minus mean adapted loss, equivalent to mean per-batch recovery.
No query can mutate the support adaptation model or another query's BN state.
Global-reference diagnostics also use the same expanded query set in eval.

All other settings remain legacy M2, fixed support, 5-step terminal, mean
aggregation across replicates, Hungarian, CIFAR-10/ResNet18, beta=0.3,
seed=1, 200 rounds, K=10, reset_ratio=2/64 and unchanged formal training.

Validation: 79 tests passed, including sample preservation, small-client
wraparound, support disjointness, identical adaptation states and equality
of multi-query utility to individually measured and averaged recovery.
Two one-round GPU smokes must pass before formal launch.

Run `python scripts/run_query_mode_experiments.py --query-coverage`.
Updated user preference: run Query x2, then Query x4 serially. The initial
parallel attempt (20260908_165310) was interrupted after about one completed
round; retain its logs but exclude it from formal comparisons. Restart both
candidates from round 1 in a new timestamped directory.
The runner waits for exit without
reading live training logs, checks fairness seeds and coverage audit, then
writes result_summary.json. No Probe LR experiments launch automatically.

Primary comparison is peak vs 74.70%; preserve final and terminal-window
metrics as diagnostics. This plan does not claim completed formal results.
