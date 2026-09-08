# M2 5-step query-mode controls

Both controlled runs completed 200 rounds with exit code 0. Two trainers ran
concurrently. The existing mixed eval/train reference was reused.

| Query before/after | Peak | Round | Delta vs reference | Final | Last20 | Last50 |
|---|---:|---:|---:|---:|---:|---:|
| eval/train (reference) | 74.70% | 164 | 0.00 pp | 71.75% | 68.0565% | 67.4368% |
| eval/eval | 74.49% | 175 | -0.21 pp | 73.07% | 69.0190% | 68.1576% |
| train/train | 73.90% | 164 | -0.80 pp | 70.15% | 66.9450% | 67.1524% |

Eval/eval improves Final by 1.32 pp, Last20 by 0.9625 pp and Last50 by
0.7208 pp over the reference, but does not improve peak. Train/train also
fails to improve peak. Retain legacy 5-step as the seed-1 peak leader; keep
eval/eval as a measurement-consistent alternative with better terminal-window
metrics. Do not infer trajectory-wide superiority or a causal BN mechanism
from this single-seed comparison. No further variants are launched.

Implementation adds an opt-in probe_query_mode setting. The default legacy
execution path remains available. New measurements query private model copies
with restoring Torch RNG contexts, so reset-query BN updates cannot affect
support adaptation. LocalTrainer, FedAvg, Hungarian and evaluation are unchanged.

Validation: 75 unit tests passed before launch, including BN/dropout isolation
and identical five-step adaptation states across modes. Both GPU smokes exited
with code 0. Post-run checks verified identical selected_clients, task_seeds and
local_seeds against the 5-step reference at all 200 rounds for each candidate.

Detailed metrics and source run directories: 20260908_085550/result_summary.json.
Large probe_pairs.csv, tasks.jsonl, checkpoint and cache files are excluded from
the result commit. The runs remain on disk for further analysis.
