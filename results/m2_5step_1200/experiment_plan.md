# M2 5-step terminal: 1200-round serial run

The authoritative 200-round seed-1 peak leader is M2 5-step terminal at
74.70% (round 164). This run extends that exact method to 1200 rounds.

Configuration: CIFAR-10, ResNet18, beta=0.3, 100 clients, 10 clients/round,
seed=1, reset_ratio=2/64, fp_conv_rounds=1000, local epochs=5, batch size=50,
SGD lr=0.01, momentum=0.5, weight decay=0, and evaluation every round.
Functional Recovery uses two independent Probes, fixed support, five repeated
adaptation steps, terminal reset-minus-adapted query loss, legacy query modes,
one query batch, mean replicate aggregation, and one-to-one Hungarian.

The current serial Query x4 experiment is allowed to finish first. Its
controller PID is passed to the launcher, and the 1200-round process starts
only after that controller exits and no main_fedrad.py trainer remains. Thus
the full 1200-round trajectory runs alone on the training GPU.

Output: results/m2_5step_1200. The launcher writes stdout/stderr separately.
This plan records intent and configuration; it does not claim completion or
accuracy results.

First attempt status: the process ended externally after completing round 675,
without a summary and without a logged Python/CUDA exception. Partial peak was
79.38% at round 588. Preserve this incomplete run for diagnosis; exclude it
from completed 1200-round claims. Restart the same configuration from round 1.
