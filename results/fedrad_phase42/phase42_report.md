# Phase 4.2 — Full-Training Utility Fidelity

## Scope

A single exact Clean FedPhoenix replay reconstructed pre-round snapshots at rounds 20, 40, 120, and 160. Each replayed round was checked against the original Clean run's selected clients, TaskBank seeds/hashes, local seeds, and post-round global hash. At each snapshot, every one of the original 10 clients was fully trained from every one of the original 10 reset TaskSpecs (100 pairs). No test-set data, test accuracy, aggregation trajectory, or score-weight search was used.

`V_full(i,j) = L_fed_ref(reset_j) - L_fed_ref(w_full(i,j))`, where `w_full` is produced by the unchanged 5-epoch LocalTrainer. `L_fed_ref` is a fixed sample-size-weighted stratified empirical cross-entropy over 16 deterministic training examples from each of all 100 federation clients (1,600 total); it uses no test samples and is analysis-only.

The fixed ratio candidate is `R = (reset_loss - adapted_loss) / (abs(reset_loss - global_loss) + 1e-12)`. This is recovery per absolute reset-damage unit; `1e-12` is only numerical stabilization.

## Snapshot replay

| Snapshot | Pre-round global hash verified | Original clients/tasks | Full pairs | Reference samples |
| ---: | --- | ---: | ---: | ---: |
| 20 | yes | 10 / 10 | 100 | 1600 |
| 40 | yes | 10 / 10 | 100 | 1600 |
| 120 | yes | 10 / 10 | 100 | 1600 |
| 160 | yes | 10 / 10 | 100 | 1600 |

## Correlation with full-training utility

Raw correlations are secondary. The primary interaction correlations double-center both score and `V_full`, removing client and state main effects that are constant over every bijection.

| Round | Score | Raw Pearson | Raw Spearman | Interaction Pearson | Interaction Spearman |
| ---: | --- | ---: | ---: | ---: | ---: |
| 20 | Legacy | -0.405 | -0.394 | -0.037 | -0.008 |
| 20 | GAD | -0.405 | -0.417 | -0.034 | -0.003 |
| 20 | R | -0.084 | -0.208 | 0.092 | 0.034 |
| 20 | AdaptedLoss | -0.409 | -0.420 | -0.034 | -0.002 |
| 20 | GOnly | -0.405 | -0.417 | -0.034 | -0.002 |
| 40 | Legacy | 0.301 | 0.418 | -0.034 | 0.213 |
| 40 | GAD | 0.291 | 0.409 | -0.058 | 0.220 |
| 40 | R | 0.159 | 0.355 | -0.029 | 0.151 |
| 40 | AdaptedLoss | 0.291 | 0.406 | -0.058 | 0.220 |
| 40 | GOnly | 0.291 | 0.409 | -0.058 | 0.221 |
| 120 | Legacy | 0.065 | 0.118 | -0.059 | -0.051 |
| 120 | GAD | 0.052 | 0.104 | -0.089 | -0.064 |
| 120 | R | -0.100 | -0.018 | -0.008 | -0.023 |
| 120 | AdaptedLoss | 0.053 | 0.103 | -0.089 | -0.064 |
| 120 | GOnly | 0.052 | 0.103 | -0.089 | -0.064 |
| 160 | Legacy | 0.019 | 0.280 | -0.103 | -0.071 |
| 160 | GAD | -0.003 | 0.289 | -0.007 | 0.018 |
| 160 | R | 0.045 | 0.154 | -0.051 | -0.155 |
| 160 | AdaptedLoss | -0.003 | 0.292 | -0.006 | 0.018 |
| 160 | GOnly | -0.003 | 0.289 | -0.007 | 0.018 |

## Assignment oracle evaluation

Each candidate's Hungarian-max assignment is evaluated on `V_full`, never on its own score. Random percentiles use 10,000 fixed-seed bijections per snapshot.

| Round | Candidate | V_full value | Random percentile | Regret to oracle | Improvement over Clean | Oracle-pair overlap |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 20 | Clean | -12.30912 | 99.22% | 0.25819 | 0.00000 | 0.20 |
| 20 | OracleV | -12.05093 | 100.00% | 0.00000 | 0.25819 | 1.00 |
| 20 | Legacy | -12.62067 | 68.22% | 0.56974 | -0.31155 | 0.10 |
| 20 | GAD | -12.74138 | 40.46% | 0.69045 | -0.43226 | 0.10 |
| 20 | R | -12.38741 | 97.13% | 0.33647 | -0.07828 | 0.20 |
| 20 | AdaptedLoss | -12.74138 | 40.46% | 0.69045 | -0.43226 | 0.10 |
| 20 | GOnly | -12.74138 | 40.46% | 0.69045 | -0.43226 | 0.10 |
| 40 | Clean | -15.92211 | 13.53% | 2.28533 | 0.00000 | 0.10 |
| 40 | OracleV | -13.63678 | 100.00% | 0.00000 | 2.28533 | 1.00 |
| 40 | Legacy | -14.25445 | 73.42% | 0.61767 | 1.66766 | 0.00 |
| 40 | GAD | -14.18447 | 80.51% | 0.54769 | 1.73764 | 0.00 |
| 40 | R | -15.12519 | 43.29% | 1.48840 | 0.79693 | 0.00 |
| 40 | AdaptedLoss | -14.18447 | 80.51% | 0.54769 | 1.73764 | 0.00 |
| 40 | GOnly | -14.18447 | 80.51% | 0.54769 | 1.73764 | 0.00 |
| 120 | Clean | -17.32484 | 14.08% | 3.05713 | 0.00000 | 0.00 |
| 120 | OracleV | -14.26771 | 100.00% | 0.00000 | 3.05713 | 1.00 |
| 120 | Legacy | -18.83955 | 9.69% | 4.57185 | -1.51472 | 0.00 |
| 120 | GAD | -18.82589 | 9.80% | 4.55818 | -1.50105 | 0.00 |
| 120 | R | -16.72453 | 21.68% | 2.45682 | 0.60031 | 0.00 |
| 120 | AdaptedLoss | -18.82589 | 9.80% | 4.55818 | -1.50105 | 0.00 |
| 120 | GOnly | -18.82589 | 9.80% | 4.55818 | -1.50105 | 0.00 |
| 160 | Clean | -14.49775 | 80.04% | 1.27216 | 0.00000 | 0.00 |
| 160 | OracleV | -13.22558 | 100.00% | 0.00000 | 1.27216 | 1.00 |
| 160 | Legacy | -14.36614 | 86.26% | 1.14056 | 0.13161 | 0.20 |
| 160 | GAD | -14.01932 | 97.26% | 0.79374 | 0.47843 | 0.30 |
| 160 | R | -17.91839 | 19.43% | 4.69280 | -3.42064 | 0.00 |
| 160 | AdaptedLoss | -14.01932 | 97.26% | 0.79374 | 0.47843 | 0.30 |
| 160 | GOnly | -14.01932 | 97.26% | 0.79374 | 0.47843 | 0.30 |

## Evidence-based conclusion

No fixed candidate has positive interaction Pearson correlation at all four snapshots: none.
No fixed candidate has positive interaction Spearman correlation at all four snapshots: none.
No score-derived Hungarian assignment improves on Clean at all four snapshots: none.
Accordingly, this diagnostic does not provide evidence that the present Probe scores faithfully recover the client-by-state interaction in full local-training utility. Isolated assignment gains must not be interpreted as validation of conditional utility when their interaction correlations are near zero or sign-unstable. On this evidence alone, Phase 4.3 state-conditioned assignment is not justified.

## Limits

The result is limited to four snapshots from one Clean seed and to a fixed 1,600-example federation-training reference subset. It excludes test data by design, and it does not establish that the full-training oracle itself is a complete downstream utility. These are the main remaining sources of uncertainty; they do not turn the observed lack of cross-snapshot score fidelity into positive evidence.

## Interpretation

This phase is a fidelity diagnostic, not an accuracy comparison. The report does not claim that any score improves 200-round test accuracy.

Machine-readable source/replay metadata, raw Probe matrices, full-training utilities, score matrices, correlation tables, assignments, and random distributions are saved beside this report.
