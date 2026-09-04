# Phase 3.6 — Probe Stabilization

## Scope and controls

- Part A reused the saved 5 rounds x 3 Probe replicates and performed zero new
  forward passes.
- The targeted size diagnosis used only rounds 1, 20, and 40, with the same
  checkpoints, selected clients, reconstructed TaskBanks, and task seeds.
- All 30 selected client appearances had at least 128 local samples (minimum
  176), so every client used all four protocols.
- For each round/client/replicate, support32 is a prefix of support64 and query32
  is a prefix of query64. The complete support and query pools are disjoint.
- Probe steps, optimizer, LR, BN mode, formal score, lambda, gate, and tau were
  unchanged. No test accuracy or formal client training was used.

## A. Existing-data two-probe averaging

| Score | Estimator | Functional pct | Q-int Pearson/Spearman | Held-out overlap | Margin/K |
| --- | --- | ---: | ---: | ---: | ---: |
| Full | Single 32/32 | 87.43 | 0.460 / 0.474 | 0.247 | 0.00904 |
| Full | Average 2x32/32 | 92.08 | 0.532 / 0.541 | 0.280 | 0.01235 |
| G | Single 32/32 | 87.37 | 0.502 / 0.505 | 0.227 | 0.00359 |
| G | Average 2x32/32 | 92.63 | 0.575 / 0.578 | 0.247 | 0.00298 |

The round-block bootstrap 95% intervals for the functional-percentile gains are
Full `[+0.70, +8.46]` and G `[+1.26, +9.02]`. Reliability improved clearly.
Overlap improved only modestly, and G margin did not improve. These results were
sufficient to trigger the targeted size diagnosis, but averaging did not solve
the near-tie problem.

## B. Nested probe-size results

| Protocol | Score | Functional pct | Q-int Pearson/Spearman | Pair overlap | Margin/K | Seconds | Peak MiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 32/32 | Full | 87.43 | 0.493 / 0.496 | 0.256 | 0.00930 | 44.44 | 251.4 |
| 32/32 | G | 91.70 | 0.554 / 0.552 | 0.289 | 0.00227 | 44.44 | 251.4 |
| 32/64 | Full | 90.21 | 0.503 / 0.514 | 0.189 | 0.00979 | 45.28 | 251.8 |
| 32/64 | G | 92.48 | 0.568 / 0.572 | 0.211 | 0.00335 | 45.28 | 251.8 |
| 64/32 | Full | 96.89 | 0.714 / 0.689 | 0.411 | 0.01048 | 45.96 | 383.6 |
| 64/32 | G | 98.10 | 0.762 / 0.727 | 0.489 | 0.00199 | 45.96 | 383.6 |
| 64/64 | Full | 96.93 | 0.718 / 0.709 | 0.400 | 0.01116 | 47.02 | 383.5 |
| 64/64 | G | 99.03 | 0.769 / 0.748 | 0.433 | 0.00438 | 47.02 | 383.5 |

Full component interaction Pearson/Spearman reliabilities were:

| Protocol | G | A | D | C |
| --- | ---: | ---: | ---: | ---: |
| 32/32 | .55/.55 | .45/.46 | .39/.41 | .77/.76 |
| 32/64 | .57/.57 | .47/.48 | .38/.40 | .77/.76 |
| 64/32 | .76/.73 | .70/.66 | .60/.59 | .87/.86 |
| 64/64 | .77/.75 | .71/.68 | .60/.60 | .87/.86 |

## C. Single large batch versus averaging

| Method | Score | Functional pct | Q-int Pearson | Pair overlap | Margin/K | Seconds |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Single 32/32 | Full | 87.43 | .493 | .256 | .00930 | 44.44 |
| Average 2x32/32 | Full | 92.70 | .572 | .278 | .00881 | 88.87 |
| Single 64/64 | Full | 96.93 | .718 | .400 | .01116 | 47.02 |
| Single 32/32 | G | 91.70 | .554 | .289 | .00227 | 44.44 |
| Average 2x32/32 | G | 95.91 | .633 | .367 | .00317 | 88.87 |
| Single 64/64 | G | 99.03 | .769 | .433 | .00438 | 47.02 |

Single 64/64 dominates two-probe averaging on functional recovery, reliability,
overlap, and Full margin while taking only 52.9% of the averaging time. However,
64/32 already captures nearly all of the Full benefit and has better pair overlap
than 64/64.

## D. Support/query and cost conclusion

- Enlarging query only (32/64) changes Full functional percentile by +2.78 and
  G by +0.78, while pair overlap falls by 0.067 and 0.078.
- Enlarging support only (64/32) changes Full functional percentile by +9.45 and
  G by +6.40; Q-int Pearson rises by 0.221 and 0.208; overlap rises by 0.156 and
  0.200.
- Moving from 64/32 to 64/64 adds only +0.05 Full functional-percentile points,
  while Full overlap falls by 0.011. The additional query is not justified for
  the formal Full score.
- Relative times are 1.00x, 1.019x, 1.034x, and 1.058x for 32/32, 32/64,
  64/32, and 64/64. Timing excludes the small, shared one-time CPU tensor
  materialization. Increasing support raises peak allocated GPU memory from
  about 251 MiB to 384 MiB.

The dominant noise source is the one-step support gradient estimate, not query
loss estimation.

## E. Decision

The Phase 3.6 recommendation is **increase support only**: use 64/32 as the
minimal diagnostic measurement correction in a later explicitly authorized
stage. Do not change the current defaults in this phase.

G-only is functionally and correlationally stronger than Full under 64/32 and
64/64, but its assignment margin remains substantially smaller than Full. It
therefore fails the pre-specified requirement `margin >= Full`; formal score
simplification is not yet supported.
