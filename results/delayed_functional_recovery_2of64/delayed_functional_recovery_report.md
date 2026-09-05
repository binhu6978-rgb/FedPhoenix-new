# Delayed Functional Recovery — Seed 1, Faithful FedPhoenix 2/64

## Scope

This experiment changed only the first round at which Functional Recovery is active. Before activation, dispatch is the original FedPhoenix pairing and Probe is skipped. From the activation round onward, the existing corrected Probe semantics, raw functional recovery score `Q_ij = reset_query_loss - adapted_query_loss`, and one-to-one Hungarian assignment are used without a gate.

All variants use CIFAR-10, ResNet18, beta 0.3, 100 clients, 10 clients per round, seed 1, 200 rounds, reset ratio 2/64, 5 local epochs, batch size 50, SGD learning rate 0.01, momentum 0.5, and weight decay 0.

## Why these activation rounds

The completed Full Functional Recovery trajectory trails Clean by an average of 0.832, 0.725, and 0.504 percentage points in rounds 1–40, 41–80, and 81–120. It turns positive in rounds 121–160 (+0.246 pp) and 161–200 (+0.496 pp). Therefore the controlled candidates were activation rounds 121, 141, and 161, covering the beginning, middle, and end of the observed positive regime without a broad sweep.

## Results

| Variant | Activation | Peak | Peak round | Delta vs Clean | Delta vs Full FR | Final | Last20 | Last50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Clean FedPhoenix | never | 73.70% | 175 | — | -0.31 pp | 72.29% | 67.316% | 66.701% |
| Full Functional Recovery | 1 | **74.01%** | 192 | +0.31 pp | — | 70.94% | **68.360%** | **67.630%** |
| Delayed FR | 121 | 73.84% | 164 | +0.14 pp | -0.17 pp | **72.94%** | 67.526% | 67.078% |
| Delayed FR | 141 | 73.44% | 164 | -0.26 pp | -0.57 pp | 72.27% | 67.743% | 67.163% |
| Delayed FR | 161 | 74.00% | 192 | +0.30 pp | -0.01 pp | 72.37% | 68.243% | 66.930% |

The best delayed variant is activation round 161. Its 74.00% peak nearly matches Full Functional Recovery but does not exceed 74.01%. Activation round 121 improves the mean accuracy over Clean by +0.746 pp during rounds 121–160, but the advantage falls to +0.042 pp during rounds 161–200 and its peak remains lower.

## Correctness checks

- Every run completed all 200 rounds with seed 1 and reset ratio 2/64.
- For every pre-activation round, selected clients, assignments, task seeds, task hashes, local seeds, global-state hash, and accuracy exactly match the authoritative Clean run.
- Probe time is zero and `matching_active=false` before activation.
- Every post-activation round reports `score_mode=functional`, `matching_active=true`, and `functional_recovery_hungarian`.
- No traceback, runtime error, assertion error, or CUDA out-of-memory error occurred. Stderr contains only the same CuBLAS deterministic-algorithm warning seen in this environment.

## Conclusion

Delayed activation successfully removes the early average deficit, but activation timing alone does not improve the peak beyond 74.01%. The result does not justify a denser activation-round sweep. The most useful next question is assignment reliability: Functional Recovery changes roughly nine of ten pairs even when the conditional part of the Probe matrix is weak, so a small reliability-focused intervention is better supported than changing Probe strength or adding another activation threshold. No such new mechanism is introduced in this experiment.
