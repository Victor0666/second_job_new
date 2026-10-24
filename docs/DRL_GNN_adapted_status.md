# DRL-GNN 公共问题适配版：当前提交状态

日期：2026-10-25。

本提交保存当前实现进度，不代表完整复现验收完成，也没有正式训练权重或性能结论。

## 实现依据与保留内容

作者公开仓库：<https://github.com/kdsuneraavinash/workflow-cloudsim-drlgnn>。
固定提交：`be67762c3dd22c05733ebb808083375de3bd7d62`。

- 保留原生 task4 / VM3 前缀，追加公共状态为 task8 / VM9；保留 raw 输入和单图 BatchNorm。
- hidden64 / embedding16 / 3 层 GIN，单向依赖及兼容边、VM 顺序边、共享虚拟首尾节点。
- 独立 Actor/Critic、全边打分后截取兼容边、联合任务—VM 分布。
- PPO、GAE、价值裁剪、minibatch 优势归一化、单 Adam 和原超参数。
- 运行时仅使用既有 PyTorch，无新增 PyG 依赖。

作者源码与论文描述存在差异，包括论文的 32 维嵌入与源码的 16 维，以及任务特征和部分 BN 设置；本实现选择源码，不声称逐式完全复现论文。

## 必要的问题适配

接入公共在线到达、真实 Ready×idle 动作条件、通信、Host 任务窗口能耗和任务截止期。奖励改为公共全局能耗—超期奖励；没有保留原式(20)的 makespan/活动能耗估计奖励。

按本次要求补充公共截止期、数据量及资源状态，字段见下表。预计完成时间仍使用公共执行函数；它不再声明为严格下界。未加入 GE 引导探索、PF 性能筛选、OE 五项摘要、内在奖励或新的图网络机制。

| 节点 | 字段顺序（从左到右） |
| --- | --- |
| Task8 | 已分配、Ready、工作量 MI、预计完成绝对时刻 s、任务子截止时间剩余量 s、工作流截止时间剩余量 s、逻辑输入 bits、输出 bits |
| VM9 | VM 可用绝对时刻 s、CPU 能力倒数、原生活动能耗系数、带宽 Mb/s、剩余忙碌时间 s、所属主机当前负载、所属主机当前功率 W、所属主机空闲功率 W、所属主机编号 |

- 截止时间剩余量分别为 `task_baseline_finish - current_time` 和 `wf.deadline - current_time`，保留逾期负值，不减预计执行时长。所有真实任务使用其所属已到达工作流的状态；两个虚拟首尾节点的新增字段均为零。
- 数据量直接读取公共 `task_in_bits` / `task_out_bits`，与 MAHDRL 基础观测使用相同逻辑数据口径；不将其替换成某个候选 VM 上扣除本地通信后的传输量。
- VM 剩余忙碌时间为 `max(0, vm_available_at - current_time)`。主机负载为当前仍忙碌 VM 的 CPU 能力之和除以主机总 CPU 能力，忙碌判断沿用 `available_at > now + 1e-12`；当前和空闲功率直接使用公共 `host.power(load)` / `host.power(0)`。
- 所属主机编号使用公共 `vm.host_id` 原始整数，作为明确标识直接加入 VM 特征；同一主机的 VM 共享编号、负载和功率。此编号不是大小有物理含义的资源量，不新增可学习主机嵌入或主机节点。固定 9 维 VM 输入可处理不同主机数量。
- 新增物理量沿用 raw 输入，不引入额外 F19 缩放或校准。任务输入层改为 `8→64`，VM 输入层改为 `9→64`；Actor/Critic 两套网络合计增加 1280 个权重，其余层宽、GIN/PPO、奖励、训练超参数与选模规则保持不变。
- 输入版本为 `public_raw_task8_vm9_bn_v2`，模型版本为 `drl_gnn_source_be67762_public_state_v2`，检查点和日志路径带新输入版本，加载时核对版本与完整字段顺序。旧 task4/VM3 权重不兼容，需要重新训练；入口参数不变。

这属于已声明的公共观测扩展适配，不再是原生 task4/VM3 数值复现，也不声明与 MAHDRL_OE_GE_PF 的所有人工构造特征完全一致。

训练入口遵循 F22：实例种子 0–599、600 Episodes；验证种子 600，每完成 Episode 验证一次，按能耗选模；测试种子 700–729。四个采样环境共用这 600 个训练实例，并非各自训练 600 回合。验证、测试使用隔离副本及 argmax，保留作者单图 BN 语义。

## 已检查

- 环境测试包含真实 Montage 双工作流与 MAHDRL 固定分配序列重放，核对时刻、VM 可用时间、截止期及最终能耗。
- 历史 task4/VM3 版与作者源码、隔离 PyG 2.6.1 的 CPU 数值对照已完成：116 项 state_dict 键一致，相同 seed 初始权重、logits、value 和参数梯度误差均为 0。该证据只适用于历史原生输入网络，不能用于声称本次扩展输入后的数值仍与作者一致。
- 新增观测检查覆盖逾期负值、输入输出数据、忙碌时间、同主机负载/功率、归属编号、后续到达工作流及历史快照；小图测试覆盖新增输入权重的 PPO 更新、训练尾批、验证、保存加载一致及旧版本拒绝。
- 提交前检查 Python 语法及 train/evaluate `--dry-run`。

## 尚未完成及已知限制

1. **显存处理尚未修改。**当前 PPO 仍同时保留一个 minibatch 中多个完整图的反向计算图，正式规模可能超显存。拟改为等价逐图梯度累计，保持 minibatch、损失、一次梯度裁剪和一次 Adam 更新；需补梯度等价测试。
2. 已覆盖小图训练、验证、保存加载和尾批处理；未完成正式规模验收，没有运行正式 600 回合训练或正式 30 种子测试。
3. 当前每步额外计算下一状态 Critic 值，存在重复前向开销；训练硬件元数据记录也待补齐。
4. 当前检查点用于评估，不支持从 rollout 中途恢复训练；文件明确记录 `resume_supported=False`。
5. 本版补充公共 Deadline/数据/资源状态，但没有复制 MAHDRL 的全部人工派生特征与机制，不能将适配结果单独解释为层次结构优劣。相同 Episode 数也不代表相同更新次数或计算开销。

在上述验收完成前，不应将本提交当作已验证的正式对比基线。

## 检查命令

```powershell
.\.venv\Scripts\python.exe -X utf8 -B -m unittest common.test_drl_gnn_environment common.test_scenarios common.test_experiment_measurements -q
.\run.cmd algorithms/drl_gnn/train.py --dry-run
.\run.cmd algorithms/drl_gnn/evaluate.py --dry-run
```
