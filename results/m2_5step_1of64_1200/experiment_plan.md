# M2 5-step terminal, 1/64, 1200 rounds

Goal: compare Functional Recovery directly with the supplied original
FedPhoenix run whose peak is 82.18% at round 1196.

The supplied manifest fixes CIFAR-10, ResNet18, beta=0.3, seed=1, 100 clients,
10 clients/round, local epochs=5, batch size=50, SGD lr=0.01, momentum=0.5,
weight decay=0, fp_conv_rounds=1000, reset method ori_normal, and reset ratio
0.015625 (1/64). This run uses the same base settings.

Method: M2 two independent Probes, fixed support, five adaptation steps,
terminal Functional Recovery, legacy query semantics, one query batch, mean
replicate aggregation, and Hungarian assignment. Formal LocalTrainer, FedAvg
and full CIFAR-10 evaluation remain unchanged.

The active 2/64 1200-round attempt was stopped at the user's request and its
logs were preserved. This 1/64 run starts from round 1 and runs serially as the
only main_fedrad.py training process. A one-round 1/64 GPU smoke precedes it.

Output: results/m2_5step_1of64_1200. This plan does not claim completed results.
