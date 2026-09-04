# Phase 4.1 — Client–State Interaction

## Scope

This is an offline mechanism analysis. It reuses only the saved Phase 3.6 `P2_64_32` (support=64/query=32) Probe matrices at rounds 1, 20, and 40, with three independent Probe replicates per round. No model was loaded for training, no new Probe forward pass was run, and no 200-round trajectory was changed.

The primary pairwise utility is `H_ij = G_ij = reset_loss_ij - adapted_loss_ij`: one-step functional recovery gain for client `i` on transient reset state `j`. `A`, `D`, `C`, `-adapted_loss`, and `Q` are secondary sensitivity analyses only. The source NPZ archives contain client IDs, task IDs, global/reset/adapted losses, G/A/D/C, and Q.

## Estimands and noise accounting

For the mean of the three independent replicate matrices in a round, we use the exact balanced two-way decomposition `H_ij = mu + a_i + b_j + e_ij`. Reported variances use divisor `I*J`, so total variance equals the client-main, state-main, and interaction variances (up to numerical roundoff).

Probe noise is estimated directly in the relevant subspace: each replicate is first double-centered with the same two-way projection, then its per-cell between-replicate variance is averaged and divided by 3 for the replicate mean. This avoids treating shared client/state Probe variation as independent interaction noise and does not use an arbitrary percentage threshold. Raw cell-level replicate variance is also retained in the CSV/JSON as an auxiliary diagnostic.

## Primary result: G recovery utility

| Round | Total var | Client var | State var | Interaction var | Interaction / total | Expected interaction noise (mean of 3 reps) | Interaction / noise | Mean cross-replicate interaction r | Mean cross-state client rank rho | Stable reversal rate (among stable comparable) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.4444 | 0.4164 | 0.0060 | 0.0220 | 0.0494 | 0.0022 | 9.98 | 0.764 | 0.947 | 0.000 |
| 20 | 2050.9516 | 1306.7566 | 261.2335 | 482.9615 | 0.2355 | 34.7786 | 13.89 | 0.821 | 0.755 | 0.083 |
| 40 | 38536.5463 | 34709.7683 | 1887.1498 | 1939.6283 | 0.0503 | 260.8575 | 7.44 | 0.702 | 0.899 | 0.000 |

A stable ordering requires all three replicate client-pair differences to have the same nonzero sign **and** the conventional two-sided 95% t interval (df=2) for their mean to exclude zero. A stable reversal is an opposite stable order for the same client pair in two states. The analysis also reports its denominator, so a rate cannot hide a scarcity of stable orderings.

## Stable ranking reversals for G

| Round | All client-pair × state-pair candidates | Stable comparable | Stable comparable rate | Stable reversals | Reversal / all candidates | Reversal / stable comparable |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2025 | 111 | 0.055 | 0 | 0.000 | 0.000 |
| 20 | 2025 | 253 | 0.125 | 21 | 0.010 | 0.083 |
| 40 | 2025 | 386 | 0.191 | 0 | 0.000 | 0.000 |

## Secondary decomposition sensitivity

| Metric | Round | Interaction / total | Interaction / expected noise | Mean replicate interaction r | Mean cross-state rank rho |
| --- | ---: | ---: | ---: | ---: | ---: |
| A | 1 | 0.0253 | 4.93 | 0.579 | 0.973 |
| A | 20 | 0.2359 | 14.00 | 0.823 | 0.751 |
| A | 40 | 0.0503 | 7.44 | 0.702 | 0.899 |
| D | 1 | 0.0714 | 1.86 | 0.278 | 0.972 |
| D | 20 | 0.2359 | 14.00 | 0.823 | 0.751 |
| D | 40 | 0.0503 | 7.44 | 0.702 | 0.899 |
| C | 1 | 0.3566 | 50.86 | 0.944 | 0.280 |
| C | 20 | 0.6436 | 15.66 | 0.833 | 0.148 |
| C | 40 | 0.3947 | 15.01 | 0.828 | 0.030 |
| adapted_utility | 1 | 0.0422 | 4.93 | 0.579 | 0.978 |
| adapted_utility | 20 | 0.2349 | 14.00 | 0.823 | 0.751 |
| adapted_utility | 40 | 0.0502 | 7.44 | 0.702 | 0.899 |
| Q | 1 | 0.0660 | 5.16 | 0.626 | 0.939 |
| Q | 20 | 0.2235 | 13.38 | 0.810 | 0.725 |
| Q | 40 | 0.0549 | 7.25 | 0.705 | 0.902 |

## Answers to the Phase 4.1 questions

1. **Does client × state interaction exist?** Yes in these saved rounds: the G interaction component is nonzero after removing client and state main effects.
2. **Is it identifiable against Probe noise?** For all analysed rounds, observed interaction variance of the three-replicate mean exceeds the directly estimated expected measurement-noise variance.
3. **Do stable ranking reversals exist?** Yes: at least one client-pair/state-pair reversal meets the strict three-replicate definition.
4. **Does this support Q_t(i,j), rather than only Q_t(i)?** The evidence supports conditioning recovery utility on transient state in this controlled setting, because interaction remains after main effects and is not explained by the estimated Probe noise alone. It does not establish a universal or multi-seed claim.
5. **Weakest evidence.** Only three replicate Probes and three early/mid rounds are available, cells within a matrix are not independent observations, and the noise correction assumes replicate differences are measurement noise. This is a single-seed mechanism analysis, not an accuracy or generalization result.
6. **Phase 4.2 decision.** The mechanism evidence is sufficient to justify considering a narrowly scoped Phase 4.2 proposal, but no Phase 4.2 experiment is run here.

## Reproducibility

The exact source archive paths and SHA-256 digests are recorded in `phase41_sources.json`; machine-readable results are in `phase41_decomposition.csv`, `phase41_interaction_reliability.csv`, `phase41_rank_correlations.csv`, `phase41_reversals.csv`, and `phase41_summary.json`.

Source archive count: 9.
