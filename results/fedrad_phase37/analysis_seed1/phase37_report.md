# Phase 3.7 — Measurement Fix Validation & Gate Identifiability

> Scope override: per the latest user instruction, this report uses **seed 1 only**. The originally requested 3-seed mean/std and 30 held-out observations are therefore not applicable.

## A. Seed-1 40-round training

| Metric | Clean | FedRAD Full 64/32 | Delta |
|---|---:|---:|---:|
| **Peak (primary)** | **48.4600 (r36)** | **48.7200 (r36)** | **0.2600** |
| Top-5 rounds mean | 47.0920 | 47.2860 | 0.1940 |
| Final | 47.1800 | 48.0200 | 0.8400 |
| Last-5 mean | 43.5660 | 43.2820 | -0.2840 |
| Last-10 mean | 43.5320 | 43.8400 | 0.3080 |

This is a diagnostic single-seed comparison, not a paper-level result.

## B. Recovery dynamics (40 rounds)

| Metric | 64/32 mean | 32/32 historical mean |
|---|---:|---:|
| Gamma | 1.2446 | 1.2523 |
| changed_pairs | 8.9750 | 8.8000 |
| margin_per_K | 0.0088 | 0.0124 |
| margin_per_Q_std | 0.0289 | 0.0408 |
| Q_interaction_std | 1.2225 | 1.2569 |

Cross-probe Full reliability (ordered replicate pairs):

| Protocol | Q-int Pearson | Q-int Spearman | Pair overlap | Functional pct |
|---|---:|---:|---:|---:|
| 64/32 | 0.681 | 0.669 | 0.280 | 96.69 |
| historical 32/32 | 0.460 | 0.474 | 0.247 | 87.39 |

The representative round sets differ (64/32 uses round 30; historical 32/32 uses round 5), so this is diagnostic rather than paired inference. Detailed per-round G/A/D/C/Q statistics are in `dynamics_64_32.csv`.

## C. Primary → held-out validity

Five primary round states × two independent held-out probes = 10 observations.

| Metric | Mean | Positive/improvement rate |
|---|---:|---:|
| Delta G | 19.1939 | 100.0% |
| Delta A | 19.2004 | 100.0% |
| Delta adapted loss | -19.2004 | 100.0% |
| Functional percentile | 98.09 | — |

## D. Gamma validity

Verdict: **Gamma not predictive**.

| Outcome | Pearson | Spearman |
|---|---:|---:|
| delta_G | -0.0655 | 0.0492 |
| delta_A | -0.0653 | 0.0492 |
| adapted_loss_improvement | -0.0653 | 0.0492 |
| functional_percentile | -0.3736 | -0.2601 |

Gamma tail analysis (2/1/2 primary states because n=5):

| Group | Rounds | Functional pct | P(ΔG>0) | P(ΔA>0) | P(loss improves) |
|---|---|---:|---:|---:|---:|
| lower | [1, 40] | 98.53 | 100.0% | 100.0% | 100.0% |
| middle | [30] | 99.70 | 100.0% | 100.0% | 100.0% |
| upper | [10, 20] | 96.85 | 100.0% | 100.0% | 100.0% |

## E. Margin validity

| Outcome | Pearson | Spearman |
|---|---:|---:|
| delta_G | -0.8993 | -0.8370 |
| delta_A | -0.8998 | -0.8370 |
| adapted_loss_improvement | -0.8998 | -0.8370 |
| functional_percentile | -0.6385 | -0.7678 |

Both observation-level n=10 and non-duplicated round-mean n=5 correlations are preserved in JSON.

## F. Candidate tau

not estimated: current gate not calibratable from available evidence

## G. Full vs G-only (primary → held-out)

| Candidate | Interaction r | Pair overlap | Functional pct | Margin/K |
|---|---:|---:|---:|---:|
| Full | 0.698 | 0.250 | 98.09 | 0.00509 |
| G-only | 0.734 | 0.270 | 98.37 | 0.00199 |

This remains an offline diagnostic; the formal Full score was not changed.

## H. Cost

- Primary probe: 1993.3s; formal training: 1424.4s; probe/formal = 1.399×.
- Two held-out diagnostics at five rounds added 501.4s.
- End-to-end wall ratio vs Clean: 2.110×; peak GPU memory: 388.0 MiB.

## I. Final conclusion

**matching useful but gate invalid**

64/32 primary matching is strongly above random on held-out functional metrics, but Gamma does not identify which rounds are more reliable. Peak Full accuracy improves only slightly, and one seed cannot establish robustness.

Phase 4 was not started.
