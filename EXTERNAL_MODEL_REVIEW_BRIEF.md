# FedPhoenix-new：外部模型审阅说明（截至 Phase 3.9）

> 目的：这份文档是给后续参与分析的模型/研究者的项目交接说明。它优先说明**实际做过的实现、受控实验、结果、已排除的方向和未解决问题**，而不是把当前结果包装成论文结论。

## 一句话状态

在 CIFAR-10 / ResNet-18 / Dirichlet `beta=0.3` / 100 clients / 每轮 10 clients / `seed=1` / 200 rounds 的固定设定下，独立实现的 recovery-aware FedPhoenix（下称 **FedRAD**）最佳受控变体为 **Full-Warm40**：

| Method | Peak accuracy | Peak round | Delta vs Clean |
| --- | ---: | ---: | ---: |
| Clean FedPhoenix | 73.810% | 175 | 0.000 pp |
| Full-All | 74.440% | 175 | +0.630 pp |
| **Full-Warm40** | **74.530%** | **175** | **+0.720 pp** |
| G-Warm40 | 74.480% | 175 | +0.670 pp |

这是**单一 seed 的 200-round peak-accuracy 结果**，不是多 seed 稳健性结论，也没有进行 1200-round 正式训练。用户当前明确以相同 200-round budget 下的 peak full accuracy 为目标。

## 1. 初始任务与实现边界

项目原有代码包含旧的 FedPhoenix 实现。此次工作没有在旧训练逻辑上累积修改，而是在 [`fedrad/`](fedrad/) 中从头构建一条干净、可审计的 pipeline：

1. 保留原始 FedPhoenix 所需的 reset 思想和训练设定；
2. 新建确定性数据、随机数、TaskBank、本地训练、加权 FedAvg、评价、日志与 replay 接口；
3. 首先实现没有 Probe / score / matching / gate 的 **Clean FedPhoenix**；
4. 再只在“client-to-reset-task 的一对一配对”这一接口上接入 recovery-aware matching；
5. 不改变 reset、本地优化器、本地 epochs、聚合规则、客户端抽样、数据划分或模型。

因此，`Algorithm/` 等旧目录仍在仓库中，但本次的干净实验路径是 `fedrad/`、`main_fedrad.py`、相关 `tests/`、`scripts/` 与 `results/fedrad_phase*/`。

## 2. 固定实验设置

Phase 3.8/3.9 的主要对照固定为：

```text
dataset                 CIFAR-10（完整 10,000 样本 server test set）
model                   ResNet-18
non-IID split           Dirichlet beta = 0.3
clients                 100
participation           10 clients / round
seed                    1
communication rounds    200
local training          5 epochs, batch size 50, SGD lr 0.01, momentum 0.5
aggregation             sample-size weighted FedAvg
reset                   ori_normal, reset_ratio=1/64, fp_conv_rounds=1000
evaluation              每轮完整 server test evaluation
```

FedRAD active matching 时额外固定：

```text
candidate set           与每轮 Clean client sampling 相同的 10 个客户端
TaskBank                同一轮、同一 task seeds、同一 reset copies
Probe                   support=64, query=32, steps=1, lr=0.01
assignment              10 x 10 Hungarian maximum-weight one-to-one matching
formal local training   从被分配的 ORIGINAL TaskSpec/reset copy 开始
```

所有随机来源被拆分为独立 RNG streams：客户端抽样、TaskBank reset seeds、local training seeds、Probe batch seeds、evaluation seeds。每轮都记录 clients、task seeds、task hashes、assignment、local seeds、global hash；因此可以逐轮 replay。

## 3. 当前方法的精确定义

### 3.1 Clean FedPhoenix

每轮从当前 global state 构造 10 个确定性 reset tasks；按客户端抽样顺序进行 baseline pairing；每个客户端从对应的 reset copy 进行本地 SGD；以客户端样本数加权 FedAvg。它不执行 Probe、评分、Hungarian 或 gate。

关键代码：

- `fedrad/trainer.py` 中 `CleanFedPhoenixTrainer`
- `fedrad/task_bank.py`
- `fedrad/local_trainer.py`
- `fedrad/aggregation.py`

### 3.2 FedRAD matching

对于每个 client-task pair，在该 reset copy 上使用客户端私有的 support/query batch 做一次适应 Probe。产生四个矩阵（每轮均为 `K x K`，`K=10`）：

```text
G  functional recovery gain
A  advantage relative to the unreset model
D  residual damage
C  reset-gradient alignment
```

每个分量先对整个矩阵 z-score。Full score 为：

```text
Q = Z(G) + Z(A) - Z(D) + 0.25 * Z(C)
```

随后在 `Q` 上做 Hungarian assignment，并用该 assignment 进行与 Clean 完全相同的正式本地训练和 FedAvg。实现见：

- `fedrad/probe.py`
- `fedrad/scoring.py`
- `fedrad/assignment.py`
- `fedrad/trainer.py` 中 `FedRADTrainer`

注意：这些 score weights 是开发期默认值，不是已经校准或可声称为理论最优的权重。

### 3.3 Warm-up matching

在 `matching_start_round` 前，FedRAD **不运行 Probe**，直接使用 Clean pairing；开始轮及之后才运行 full Probe + Hungarian。为了防止“表面相同、实际 RNG drift”，延迟匹配配置必须指向一个已完成 Clean run 的 `rounds.jsonl`，训练时逐轮断言以下字段严格一致：

```text
selected clients, assignments, task seeds, task hashes,
local seeds, global state hash
```

对应测试：`tests/test_phase39_warmup.py`。Full-Warm40/G-Warm40 的 rounds 1–40 均已逐轮验证与 Clean hash 完全一致，而且 warm-up 期间 Probe 秒数为零。

## 4. 做过的工作与阶段结论

| 阶段 | 做了什么 | 结论/产物 |
| --- | --- | --- |
| Phase 1 | 建立 Clean FedPhoenix、TaskBank、独立 RNG、本地 SGD、weighted FedAvg、全量日志 | 2-round replay smoke 通过；Clean 路径不含 Probe/matching/gate。 |
| Phase 2 | 实现 Probe -> `G/A/D/C` -> `Q` -> Hungarian -> ORIGINAL TaskSpec -> local training；加入 unit tests | 2-round forced-Hungarian smoke/replay 通过。 |
| Phase 3 | 40-round 机制诊断：分量交互、assignment、probe replicate reliability、随机 assignment null | 发现 pair-specific signal，但原始 32/32 Probe 下 assignment 对采样敏感。 |
| Phase 3.5 | 复用保存的 raw matrices 做三重复 Probe 的 cross-probe held-out validation 和 score simplification diagnosis | Full/G 在 held-out functional recovery 上优于随机；exact assignment 不稳定，C 稳定但不预测 functional benefit。 |
| Phase 3.6 | 只做 measurement protocol 修正：2x32/32 averaging 与 32/32、32/64、64/32、64/64 nested Probe sizes | support 噪声是主因；选择最小修正 `64/32`，未改正式 score。 |
| Phase 3.7 | 在 64/32 下检查 held-out validity、Gamma 可识别性、Full vs G-only | Gamma 不预测 held-out benefit，停止 gate/tau 方向；G-only 仅保留为最小候选。 |
| Phase 3.8 | Clean vs Full-All 的 200-round、seed-1、严格控制训练 | Full-All peak +0.63 pp，但早期偏弱、总体轨迹均值 -0.092 pp；只可视为弱单-seed utility。 |
| Phase 3.9 | 四个原定 timing/score variants，并追加 Warm20/Warm60 的小范围 refinement | Full-Warm40 最佳 +0.72 pp；不继续 timing grid 或 G-only。 |

## 5. 关键诊断证据

### 5.1 Probe measurement 的发现（Phase 3.5–3.7）

- 32/32 下，Full 的 held-out functional percentile 为 87.43，G 为 87.37；这表示它们在独立 Probe 上比同矩阵随机一对一 assignment 更好，但 exact assignment 三重复一致率为 0，平均 pair overlap 仅约 0.25。
- C 的 score reproducibility 很高（interaction correlation 约 0.75），但 functional percentile 仅 48.49；稳定的 gradient geometry 并不等价于能选择功能性更好的恢复配对。
- `G` 与 `A` 强冗余，`D` 与 `A` 也高度相关；不过对删除任一分量的证据不足，因此 Full score 未被重写。
- 增大 support 是主要改进：Full 的 held-out functional percentile 从 32/32 的 87.43 升至 64/32 的 96.89；interaction Pearson 从约 0.49 升至约 0.71。单纯增大 query 的收益明显更小。
- 64/32 被选为最终 Probe protocol，因为比 64/64 少 query 计算、Full 的 held-out functional benefit 几乎相同且 pair overlap 更高。
- Gamma 与 held-out functional benefit 不相关/负相关（Phase 3.7 observation-level Pearson -0.37、Spearman -0.26），所以正式 200-round matching **不使用 Gamma gate，也不调 tau**。

### 5.2 200-round 轨迹（Phase 3.8）

| Metric | Clean | Full-All | Delta |
| --- | ---: | ---: | ---: |
| Round 40 | 47.400% | 47.310% | -0.090 pp |
| Round 80 | 53.740% | 53.910% | +0.170 pp |
| Round 120 | 64.710% | 64.010% | -0.700 pp |
| Round 160 | 72.020% | 72.000% | -0.020 pp |
| Final (round 200) | 71.490% | 71.650% | +0.160 pp |
| Last-20 | 68.282% | 68.671% | +0.389 pp |
| Last-50 | 67.575% | 67.806% | +0.231 pp |
| Mean / AUC, rounds 1–200 | 56.223% | 56.131% | -0.092 pp |
| Peak | 73.810% @ 175 | 74.440% @ 175 | +0.630 pp |

分段平均差异：rounds 1–40 为 `-0.583 pp`，41–80 为 `-0.063 pp`，81–120 为 `-0.018 pp`，121–160 为 `-0.091 pp`，161–200 为 `+0.293 pp`。这也是测试延迟 matching 的直接动机。

## 6. Phase 3.9 的完整比较（当前最重要结果）

除 variant 定义外，所有实验固定为上述设置和 64/32 Probe；每个 run 都是 `seed=1, 200 rounds`。

| Variant | Matching rule | Peak | Round | Final | Last-20 | Last-50 | Runtime |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Clean | Clean pairing, rounds 1–200 | 73.810% | 175 | 71.490% | 68.282% | 67.575% | 68.5 min |
| Full-All | Full Q + Hungarian, rounds 1–200 | 74.440% | 175 | 71.650% | 68.671% | **67.806%** | 94.1 min |
| Full-Warm20 | Clean 1–20; Full thereafter | 74.220% | 175 | 71.930% | 68.684% | 67.366% | 111.9 min |
| **Full-Warm40** | Clean 1–40; Full thereafter | **74.530%** | **175** | 72.040% | 67.670% | 67.469% | 125.8 min |
| Full-Warm60 | Clean 1–60; Full thereafter | 74.150% | 175 | 70.970% | 68.372% | 67.534% | 97.8 min |
| Full-Warm80 | Clean 1–80; Full thereafter | 73.950% | 164 | 71.900% | **68.793%** | 67.671% | 115.1 min |
| G-Warm40 | Clean 1–40; `Q=Z(G)` thereafter | 74.480% | 175 | **72.890%** | 67.979% | 67.415% | 125.9 min |

解释必须保持克制：

- 若**唯一主要目标是 peak**，Full-Warm40 是当前 best：相对 Clean `+0.72 pp`，相对 Full-All `+0.09 pp`。
- `+0.09 pp` 比 Full-All 的增益很小，且 Full-Warm40 的 Last-20/Last-50 低于 Full-All。因此不能说 Warm40 全面更好。
- G-Warm40 比 Full-Warm40 低 `0.05 pp`，所以没有证据支持用 G-only 替换 Full score；G-only 路线已经停止。
- Full-Warm20/60/80 都没有击败 Full-All peak；不再搜索更多 matching start round。
- 并发 run 的 wall time 不能当作严格的吞吐 benchmark；Phase 3.8 中 Full-All 对 Clean 的单 run wall-time ratio 为约 `1.375x`，Probe 是主要开销。

正式报告：

- `results/fedrad_phase39/analysis_seed1/phase39_final_with_timing_refinement.md`
- `results/fedrad_phase38/analysis_seed1/phase38_report.md`
- `results/fedrad_phase37/analysis_seed1/phase37_report.md`
- `results/fedrad_phase36/phase36_report.md`
- `results/fedrad_phase35/cross_probe_seed1/phase35_report.md`

## 7. 明确做过、但已停止的方向

以下方向是有意识地排除的；不要把缺失实现误解为遗忘。

1. **Gamma gate / tau tuning**：held-out functional benefit 不可识别，Phase 3.9 active matching 直接 Hungarian。
2. **G-only 的扩展 score search**：只验证了最小 G-Warm40；没有继续 A-only、C-only、G+C、A+C、G+A 等组合。
3. **更多 timing grid**：仅验证 20/40/60/80；最佳 40 的相对收益很小，停止。
4. **lambda tuning、Probe tuning、partial/soft matching、history/inertia、新 recovery loss、recovery-weighted aggregation**：均未做，以避免将有限的单-seed结论埋在更大的 hyperparameter search 中。
5. **多 seed 和 1200 rounds**：用户明确暂不进行；因此不能报告 mean/std、显著性或长程效果。
6. **修改 reset、optimizer、local epochs、aggregation、dataset 或 model**：严格禁止，未发生。

## 8. 代码/数据产物地图

```text
fedrad/
  config.py           实验配置与 warm-up/replay 约束
  rng.py              独立确定性 RNG streams
  task_bank.py        可重建的 reset tasks 和 state hashes
  probe.py            support/query one-step recovery probe
  scoring.py          G/A/D/C、z-score、Full/G-only Q
  assignment.py       Hungarian 与 assignment diagnostics
  local_trainer.py    正式本地 SGD（Clean/FedRAD 共用）
  aggregation.py      sample-size weighted FedAvg（共用）
  trainer.py          CleanFedPhoenixTrainer 与 FedRADTrainer
  logger.py           JSONL/CSV/NPZ 产物写入

tests/
  test_*.py           replay、TaskBank、Probe、score、assignment、aggregation、warm-up 回归测试

scripts/
  analyze_phase*.py   各阶段分析/汇总脚本
  run_regression_tests.py

results/fedrad_phase35...phase39/
  原始 run configs、tasks、assignments、per-round metrics、scores 和阶段报告
```

权重 `.pt` 已从工作区和仓库中永久删除，`.gitignore` 已防止之后提交 `.pt`。因此所有结论应从保存的 logs/JSON/CSV/NPZ 与可复现实验脚本审阅，而不是依赖 checkpoint。

## 9. 文档与默认配置的注意事项

根目录 `readme.md` 包含历史说明和通用默认命令，其中可能显示 `1200 rounds`、旧 gate 描述或其他非 Phase 3.9 控制配置。它**不应**被用作当前实验的真相来源。

以本文件、每个 run 目录的 `config.json`、`summary.json`、`rounds.jsonl`、`assignments.jsonl` 以及第 6 节所列阶段报告为准。特别是 Phase 3.9：200 rounds、64/32、无 Gamma gate、以及固定 warm-up replay。

## 10. 建议外部模型优先审阅的问题

1. `G/A/D/C` 的数学定义、符号方向、Probe 实现与 `Q` 中正负号是否自洽？
2. 当前 Full score 是否存在明显重复分量或规模/normalization 问题？注意：现有证据不足以自动删除分量。
3. 64/32 Probe 是否存在训练/评估泄漏、batch reuse、BN 状态污染或不公平的 RNG 差异？
4. warm-up replay 断言是否足以支持“round 1–40 严格等价于 Clean”的主张？
5. 200-round peak 的 +0.72 pp 是否更可能是偶然峰值、算法信号，还是在受限预算下值得继续验证的候选？
6. 若恢复多 seed 或扩大预算，最小且信息量最大的下一实验应是什么？请避免建议无约束的大规模调参。
7. 如何改善文档、repo hygiene（如许可证、依赖锁定、去除可再生 `__pycache__`）并让独立复现更可靠？

## 11. 当前结论与诚实表述

可以说：在一个严格对齐、可 replay 的单-seed controlled experiment 中，recovery-aware assignment 对 200-round peak 有小的正向信号；在所测 timing 中，round-40 delayed Full matching 的 peak 最高。

不能说：该方法已在多随机种子上稳定优于 Clean FedPhoenix；任何 score component 已被理论或统计地证明最优；Gamma gate 有效；或者 Full-Warm40 在 final/late stability/compute 上全面占优。

下一阶段尚未自动开始。是否做多 seed、是否跑 1200 rounds、是否改变 score 或协议，均应在外部审阅后重新授权。
