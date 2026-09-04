# Phase 3.5 — Cross-Probe Validation & Score Simplification Diagnosis

## Protocol

- Reused the saved raw matrices from rounds 1, 5, 10, 20, and 40.
- Used three independent ProbeBatch replicates per round and all six ordered
  source-to-held-out replicate pairs.
- Compared every held-out assignment with the clean baseline and 1,000 random
  one-to-one permutations on the same held-out matrix.
- Used no test accuracy, no new forward passes, and no new training trajectory.
- Did not modify the formal score, component weights, gate, or tau.

The primary method ranking below prioritizes held-out functional recovery
percentile because the method is intended to recover tasks, not merely reproduce
an artificial score. Score-generalization and diagnostic-consensus ranks remain
available in `candidate_ranking.csv`.

## Candidate ranking

| Rank | Candidate | Q-int Pearson / Spearman | Pair overlap | Cross - baseline | Random percentile | Held-out G delta | Held-out A delta | Adapted-loss delta | Margin / K |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | Full | 0.460 / 0.474 | 0.247 | 6.326 | 91.1 | 16.018 | 16.019 | -16.019 | 0.0090 |
| 2 | G | 0.502 / 0.505 | 0.227 | 2.088 | 90.2 | 15.907 | 15.924 | -15.924 | 0.0036 |
| 3 | G+A | 0.469 / 0.473 | 0.213 | 3.980 | 87.1 | 15.902 | 15.922 | -15.922 | 0.0057 |
| 4 | A | 0.433 / 0.440 | 0.213 | 1.956 | 83.6 | 16.013 | 16.038 | -16.038 | 0.0030 |
| 5 | G+C | 0.539 / 0.541 | 0.280 | 2.578 | 94.4 | 12.495 | 12.448 | -12.448 | 0.0039 |
| 6 | A+C | 0.514 / 0.518 | 0.267 | 2.383 | 93.8 | 12.310 | 12.270 | -12.270 | 0.0031 |
| 7 | C | 0.750 / 0.739 | 0.293 | 5.053 | 99.1 | 1.901 | 1.807 | -1.807 | 0.0074 |

Raw held-out deltas are averaged over the 30 ordered comparisons and are
scale-sensitive across rounds. Random percentiles are the scale-comparable
quantities. The combined functional-quality percentiles are: Full 87.45, G
87.19, G+A 85.69, A 84.32, G+C 82.94, A+C 81.45, and C 48.49.

## Findings

1. Pair-specific recovery information exists. Full and G assignments evaluated
   on independent probes lie near the 91st and 90th random percentiles. This is
   inconsistent with Case C, although exact assignments remain unstable.
2. Case B is strongly supported. C is the most reproducible score signal and is
   above the random-score p90 in every ordered comparison, but its held-out
   functional recovery is at the 48.49th percentile. C captures stable gradient
   geometry without predicting easier functional recovery.
3. G is the best simplification candidate, but Case A is not yet decisive. G is
   more reliable than Full and almost ties Full on functional recovery, while
   Full remains ahead by only 0.25 functional-percentile points. G+C improves
   score reproducibility but reduces functional recovery relative to G.
4. G is preferable to A as an isolated signal: it has higher interaction
   reliability, cross-score percentile, functional percentile, and margin. G+A
   adds no evidence of benefit over G alone.
5. D is saturated and largely redundant, but not yet safely removable. Across
   the 15 matrices, `corr(A_int, -D_int)` is 0.965 and the median Full-versus-no-D
   overlap is 1.0. Removing D has median zero effect, but its mean held-out effect
   is slightly worse than Full (G -0.685, A -0.706, adapted loss +0.706), driven
   by a few changed assignments. This is insufficient evidence for deletion.
6. Every candidate has zero exact three-replicate assignment agreement. Mean
   pair overlap is only 0.213–0.293. Absolute second-best margins are only
   0.029–0.074 entry-standard-deviation units for most simplified candidates
   (Full 0.030), and margin/overlap correlations are weak or negative. The
   assignment problem remains in a near-tie regime relative to probe noise.

## Method-level decision

- Confirm **Case B**.
- Reject **Case C** for these saved probes: G and Full generalize above random.
- Treat **Case A as promising but not established**: G-only is the minimal
  candidate worth a later controlled confirmation, but this phase does not
  authorize a formal score change.
- Keep the formal G/A/D/C score and gate unchanged for now.
