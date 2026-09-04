# Phase 3.8: Single-Seed Global Utility Validation

## 1. Accuracy

| Metric | Clean | FedRAD | Delta |
| --- | ---: | ---: | ---: |
| Round 40 | 47.400% | 47.310% | -0.090pp |
| Round 80 | 53.740% | 53.910% | +0.170pp |
| Round 120 | 64.710% | 64.010% | -0.700pp |
| Round 160 | 72.020% | 72.000% | -0.020pp |
| Round 200 | 71.490% | 71.650% | +0.160pp |
| Last-20 | 68.282% | 68.671% | +0.389pp |
| Last-50 | 67.575% | 67.806% | +0.231pp |
| 1-200 AUC/Mean | 56.223% | 56.131% | -0.092pp |
| Peak (diagnostic only) | 73.810% | 74.440% | +0.630pp |

Peak rounds: Clean 175; FedRAD 175.

## 2. Difference trajectory

| Segment | FedRAD > Clean | Mean delta |
| --- | ---: | ---: |
| Rounds 1-40 | 14/40 (35.0%) | -0.583pp |
| Rounds 41-80 | 19/40 (47.5%) | -0.063pp |
| Rounds 81-120 | 23/40 (57.5%) | -0.018pp |
| Rounds 121-160 | 21/40 (52.5%) | -0.091pp |
| Rounds 161-200 | 26/40 (65.0%) | +0.293pp |
| Rounds 101-200 | 56/100 (56.0%) | +0.005pp |

The per-round delta and trailing 10-round mean are saved in `accuracy_delta.csv` and `accuracy_delta.png`.

## 3. Matching dynamics

- Changed pairs: 8.96/10 on average; rounds 101-200: 9.01/10.
- Q interaction std: 0.9570 overall; late: 0.7641.
- Assignment margin: 0.0861 overall; late: 0.0597.
- Gamma: 0.9512 overall; late: 0.7499.
- Reset delta norm: 5.3224 overall; late: 2.2126. Active reset kernels remained 735.0 per round on average.
- Fixed lag-1/5/10 Pearson and Spearman results are saved in `phase38_report.json`; these are descriptive only.

## 4. Cost

- Clean wall runtime: 68.5 min; measured round-loop runtime: 68.4 min.
- Clean formal-training time was not separately instrumented in this completed run; its round timer also includes task construction, aggregation, evaluation, and logging.
- FedRAD wall runtime: 94.1 min; probe 50.9 min; formal training 32.2 min; matching 0.067s.
- FedRAD/Clean wall-runtime ratio: 1.375x; peak allocated GPU memory: 408.6 MiB.
- Engineering diagnostic: +0.374 final-accuracy pp and -0.216 trajectory-mean pp per additional GPU-hour.

## 5. Final conclusion

`global utility weak`

some utility indicators are positive, but the complete late-trajectory criteria are not all met. This is a single-seed controlled mechanism result, not a cross-seed robustness claim.
