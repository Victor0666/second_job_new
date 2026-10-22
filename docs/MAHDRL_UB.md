# MAHDRL-UB：TA 未来价值上界测试变式

算法 ID：`mahdrl_ub`。训练、评测入口直接从原 MAHDRL 复制，原 MAHDRL、MAHDRL-OE 和 MAHDRL-TS 的代码及模型均保留。独立模型目录、日志、评测 CSV 和逐实例测量均使用 `mahdrl_ub` 名称。

## 唯一的学习规则变化

仅 TA 使用专用 `TAUpperBoundAgent`。仍由在线网络选择下一合法动作 `a*`，由目标网络评价该动作，但训练标签中的未来价值增加零上界：

```text
原 TA：y = r + gamma * (1 - done) * Q_target(s_next, a*)
新 TA：y = r + gamma * (1 - done) * min(Q_target(s_next, a*), 0)
```

当前 TA 使用 tardiness 模式，真实阶段奖励全部非正，因此真实未来折扣回报也不应大于0。新 TA 拒绝正奖励转移，避免奖励定义改变后继续默默使用不适用的上界。非终止状态必须有合法下一动作；终止回报没有未来项；原掩码、经验复制与决策边界保持不变。

只约束 TD 标签的未来项。当前 Q、网络输出、贪心 argmax、epsilon 探索、动作空间和环境奖励都不裁剪。该约束不保证网络输出始终非正，也不保证动作排序、负方向稳定性或最终调度表现，不能仅凭 TD 目标非正宣称修复完成。

| 设置 | 原 MAHDRL | MAHDRL-UB |
|---|---:|---:|
| TA 学习率 | 0.0003 | 0.0003 |
| TA 目标网络 tau | 0.01 | 0.01 |
| TA gamma | 0.95 | 0.95 |
| TA batch / buffer / 预热 | 256 / 60000 / 1800 | 相同 |
| TA 观测 / 动作维度 | 15 / 243 | 相同 |
| TA 网络、Huber、梯度裁剪10及epsilon退火 | 原设置 | 相同 |
| HA/VA 更新与全部参数 | 原实现 | 相同 |

HA/VA直接使用未修改的 `common.d3qn_agent_EXDehid.D3QNAgent`，环境直接复用 `environments.mahdrl_fcfs`。没有叠加TS参数、OE观测/奖励、动作精简或决策频率调整。

## 约束诊断

沿用 `f11_diagnostics.jsonl`，增加以下 TA 信息：

| 位置 | 字段 | 含义 |
|---|---|---|
| `numerics.TA.profiles` | `Q_selected` | 当前动作的原始Q，没有约束 |
| 同上 | `Q_next_raw` | 约束前、所选下一动作的目标网络Q，仅非终止且有合法动作的样本 |
| 同上 | `Bootstrap_raw` | 约束前的折扣未来项；终止样本为0 |
| 同上 | `Bootstrap_projected` | 实际用于学习的约束后折扣未来项 |
| 同上 | `TD_target_raw` | 未施加上界时本来会得到的TD目标，仅用于诊断 |
| 同上 | `TD_target`、`TD_error`、`loss` | 实际用于更新的目标、误差和损失 |
| `learners.TA.ta_future_upper_bound` | `eligible_samples`、`projected_samples` | 累计可应用约束的样本数、其中原始未来Q大于0的样本数 |
| 同上 | `cumulative_projection_ratio` | 累计触发比例，分母不含终止样本 |
| 同上 | `last_batch_eligible`、`last_batch_projected`、`last_batch_projection_ratio` | 最近一批次的样本数及触发比例 |

原数值诊断分位数来自最近256个样本，历史 `max_abs` 及累计触发比例不是近期值。每回合触发比例可由相邻日志累计计数之差计算。原 `future_abs_mean` 现在描述实际约束后的未来项；应同时查看新增的 `Bootstrap_raw`，避免把被强制压低的目标误认为原始价值已经恢复正常。

`F11_DIAGNOSTICS=0` 或 `F19_DIAGNOSTICS=0` 只影响统计，不取消实际TA约束。

## 运行指令

在项目根目录 PowerShell 中运行：

```powershell
# 只预览场景与模型路径
.\run.cmd algorithms/mahdrl_ub/train.py --task S --resource M --deadline Tight --episodes 50 --dry-run
.\run.cmd algorithms/mahdrl_ub/evaluate.py --task S --resource M --deadline Tight --dry-run

# 50回合早期筛查
.\run.cmd algorithms/mahdrl_ub/train.py --task S --resource M --deadline Tight --episodes 50

# 600回合完整训练（与上面的50回合二选一，或先归档本变式自身输出再运行）
.\run.cmd algorithms/mahdrl_ub/train.py --task S --resource M --deadline Tight --episodes 600

# 冻结模型后按当前协议评测；不用于反复筛选修改方案
.\run.cmd algorithms/mahdrl_ub/evaluate.py --task S --resource M --deadline Tight --seeds 700-729

# 泛化到M任务，使用S任务训练的本变式模型
.\run.cmd algorithms/mahdrl_ub/evaluate.py --task M --resource M --deadline Tight --train-task S --seeds 700-729
```

也可使用 `python -m algorithms.mahdrl_ub.train ...` 与 `python -m algorithms.mahdrl_ub.evaluate ...`。公共缓存、数据划分和归一化配置沿用当前协议，无需为本变式重新生成。

训练不自动续训，仍拒绝覆盖非空模型/日志目录。50与600回合使用同一场景目录，不能直接接着运行；没有改变公共的多模型种子目录命名规则。

700–729已在先前方案讨论中查看过；上面的评测命令表示当前协议入口，不能将其重新视为完全未见测试集。后续方案筛选应使用训练和验证数据，最终核验需统一预留未参与方案选择的实例。

## 输出与模型身份

- 模型：`outputs/ckpts/mahdrl_ub_task.../`。
- 训练日志：`outputs/logs/mahdrl_ub_task.../`。
- 评测 CSV：`outputs/run/<输出版本>/<学习协议>/<权重协议>/mahdrl_ub_wfInfo__HVrn_<截止期>/eval_mahdrl_ub_task...csv`。
- 逐实例测量：`outputs/run/<测量版本>/mahdrl_ub/<输出版本>/<场景>/`。

TA checkpoint在 `normalization.ta_future_value_target` 标记 `mahdrl_ub_ta_future_upper_v1` 及零上界。评测会拒绝原MAHDRL、OE、TS或缺少该标记的TA权重，即使网络尺寸一致也必须独立训练。评测还检查加载后的TA学习率和tau仍为原配置，避免加载恢复优化器状态后混入TS参数。HA/VA的更新规则和模型格式保持原版。

## 验证

```powershell
.venv\Scripts\python.exe -X utf8 -B -m unittest common.test_mahdrl_ub common.test_scenarios common.test_f17 common.test_f11 common.test_mahdrl_ts -q
```

新增11项测试覆盖：正未来项归零、负未来项与原版更新一致、Double DQN及非法动作处理、终止零未来项、累计/批次触发比例、原始贪心动作、正奖励拒绝、诊断开关不影响更新、旧模型拒绝，以及临时目录内2工作流/小网络的真实更新、保存、加载和评测。正式网络的稳定性、能耗与成功率尚需实验验证。

本次上述相关回归共52项通过；训练与评测入口的dry-run、语法及空白检查通过。原MAHDRL、OE、TS及其环境/共享学习与诊断实现共17个Python文件在修改前后SHA256一致。未启动50或600回合正式训练。
