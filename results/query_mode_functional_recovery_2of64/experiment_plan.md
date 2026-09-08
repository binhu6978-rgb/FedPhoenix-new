# Functional Recovery query-mode controls

Reference: M2 5-step terminal, 74.70% at round 164. The reference computes
reset query loss in eval mode and adapted query loss in train mode (legacy).
The reference remains available without changing its execution path.

Two candidates, at most two simultaneous trainers:

| Candidate | Reset query | Adapted query |
|---|---|---|
| eval_eval | eval | eval |
| train_train | train | train |

Both use two independent Probes, five SGD updates on fixed support batches,
terminal reset-minus-adapted query loss, mean replicate aggregation and the
existing Hungarian assignment. Formal training starts from original TaskSpec.
CIFAR-10, ResNet18, beta=0.3, 100 clients, K=10, seed=1, 200 rounds,
reset_ratio=2/64, fp_conv_rounds=1000, local epochs=5, batch size=50,
SGD lr=0.01, momentum=0.5 and weight decay=0 remain fixed.

Each new query is evaluated on a private model copy inside a restoring Torch
RNG context. Train-mode reset queries therefore cannot update the adaptation
model's BN buffers. Eval-mode adapted queries use support-updated running
statistics. Support adaptation always remains in train mode.

Validation: 75 unit tests passed. Tests include BN/dropout query isolation,
the actual before/after query modes, and identical pre-query model states
after five support steps across legacy/eval_eval/train_train. GPU smoke and
formal run outcomes are recorded separately in timestamped directories.

Launch: `D:\software\minicoonda\envs\fd\python.exe scripts/run_query_mode_experiments.py`
The launcher blocks on subprocess completion without reading training logs,
then creates result_summary.json with completion status and accuracy metrics.
Do not infer completion or improvement from this plan.

Primary comparison: peak accuracy versus 74.70%. Keep final/last20/last50 as
diagnostics. Compare all rounds' client sampling, task seeds and local seeds
against the reference before final conclusions. Do not automatically launch
additional variants after this pair finishes.
