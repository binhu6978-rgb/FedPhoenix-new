# M2 5-step Functional Recovery, 1/64, 1200 rounds

The serial seed-1 run completed all 1200 rounds with status `complete` and no
logged Traceback, RuntimeError or CUDA out-of-memory error.

| Metric | Original FedPhoenix | M2 5-step Ours | Ours - FedPhoenix |
|---|---:|---:|---:|
| Peak | 82.18% | 81.79% | -0.39 pp |
| Peak round | 1196 | 1064 | -132 rounds |
| Final | 73.96% | 80.89% | +6.93 pp |
| Last20 | 78.9095% | 79.1995% | +0.2900 pp |
| Last50 | 79.2248% | 79.0230% | -0.2018 pp |
| Last100 | 79.6904% | 79.2075% | -0.4829 pp |
| Top-10 mean | 82.0200% | 81.5370% | -0.4830 pp |
| Peak in rounds 1-200 | 74.20% @ 196 | 75.05% @ 175 | +0.85 pp |

The primary peak objective was not met. M2 improves the first-200-round peak
but does not overtake the original FedPhoenix full-budget peak. The large
positive final-round difference is not evidence of broad terminal stability:
Last50 and Last100 are slightly lower, while Last20 is slightly higher.

Configuration matches the supplied manifest on CIFAR-10, ResNet18, beta=0.3,
seed=1, 100 clients, 10 clients/round, local epochs=5, batch size=50, SGD
lr=0.01, momentum=0.5, weight decay=0, fp_conv_rounds=1000, ori_normal reset,
reset_ratio=0.015625 (1/64), and full-test evaluation every round. The supplied
baseline used its original method loop; this run used the clean FedPhoenix
implementation plus Functional Recovery. The baseline does not expose enough
per-round seed metadata to assert paired client/task trajectories.

Ours uses two independent Probes, fixed support, five repeated adaptation
steps, terminal recovery, legacy query modes, one query batch, mean replicate
aggregation, and Hungarian assignment. Summary diagnostics: mean Probe time
38.333 seconds/round, formal local training 10.218 seconds/round, score standard
error 0.05460, replicate assignment overlap 65.98%, assignment change rate
74.98%, and assignment margin 0.00543.

The earlier 2/64 1200-round attempts were incomplete or stopped and are not
used in this table. Their local logs remain preserved.

Decision: this run does not support claiming a peak improvement over the
supplied FedPhoenix result. It does support that M2 can improve early-budget
peak at the matched 1/64 reset ratio. Do not launch another experiment
automatically from this report.
