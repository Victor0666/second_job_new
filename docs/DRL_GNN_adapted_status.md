# DRL-GNN 公共问题适配版：当前提交状态

日期：2026-10-23。

本提交保存当前实现进度，不代表完整复现验收完成，也没有正式训练权重或性能结论。

## 实现依据与保留内容

作者公开仓库：<https://github.com/kdsuneraavinash/workflow-cloudsim-drlgnn>。
固定提交：`be67762c3dd22c05733ebb808083375de3bd7d62`。

- 原生 task4 / VM3 特征、raw 输入和单图 BatchNorm。
- hidden64 / embedding16 / 3 层 GIN，单向依赖及兼容边、VM 顺序边、共享虚拟首尾节点。
- 独立 Actor/Critic、全边打分后截取兼容边、联合任务—VM 分布。
- PPO、GAE、价值裁剪、minibatch 优势归一化、单 Adam 和原超参数。
- 运行时仅使用既有 PyTorch，无新增 PyG 依赖。

作者源码与论文描述存在差异，包括论文的 32 维嵌入与源码的 16 维，以及任务特征和部分 BN 设置；本实现选择源码，不声称逐式完全复现论文。

## 必要的问题适配

接入公共在线到达、真实 Ready×idle 动作条件、通信、Host 任务窗口能耗和任务截止期。奖励改为公共全局能耗—超期奖励；没有保留原式(20)的 makespan/活动能耗估计奖励。

不新增 Deadline/Slack、Host 负载观测、启发式筛选、内在奖励或新的网络机制。预计完成时间使用公共执行函数；它不再声明为严格下界。

训练入口遵循 F22：实例种子 0–599、600 Episodes；验证种子 600，每完成 Episode 验证一次，按能耗选模；测试种子 700–729。四个采样环境共用这 600 个训练实例，并非各自训练 600 回合。验证、测试使用隔离副本及 argmax，保留作者单图 BN 语义。

## 已检查

- 环境测试包含真实 Montage 双工作流与 MAHDRL 固定分配序列重放，核对时刻、VM 可用时间、截止期及最终能耗。
- 作者源码与隔离 PyG 2.6.1 的 CPU 数值对照已完成：116 项 state_dict 键一致，相同 seed 初始权重、logits、value 和参数梯度误差均为 0。该证据只证明网络对照，不代表完整 PPO 或性能复现。
- 提交前检查 Python 语法及 train/evaluate `--dry-run`。

## 尚未完成及已知限制

1. **显存处理尚未修改。**当前 PPO 仍同时保留一个 minibatch 中多个完整图的反向计算图，正式规模可能超显存。拟改为等价逐图梯度累计，保持 minibatch、损失、一次梯度裁剪和一次 Adam 更新；需补梯度等价测试。
2. 端到端训练、验证、保存加载、尾批处理和完整回归验收尚未完成；没有运行正式 600 回合训练或正式 30 种子测试。
3. 当前每步额外计算下一状态 Critic 值，存在重复前向开销；训练硬件元数据记录也待补齐。
4. 当前检查点用于评估，不支持从 rollout 中途恢复训练；文件明确记录 `resume_supported=False`。
5. 原生状态不含 Deadline，不能将适配结果单独解释为层次结构优劣。相同 Episode 数也不代表相同更新次数或计算开销。

在上述验收完成前，不应将本提交当作已验证的正式对比基线。

## 检查命令

```powershell
.\.venv\Scripts\python.exe -X utf8 -B -m unittest common.test_drl_gnn_environment common.test_scenarios common.test_experiment_measurements -q
.\run.cmd algorithms/drl_gnn/train.py --dry-run
.\run.cmd algorithms/drl_gnn/evaluate.py --dry-run
```
