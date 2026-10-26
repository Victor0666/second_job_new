# MARL：独立 Local 与动作条件 Q-Critic 适配

> 本文保留 v1–v3 历史实现和验收记录。当前正式版本为 `marl_q_v4`，见 [V4 实现说明](MARL_Q_v4_implementation.md)。

2026-10-24 更新。正式入口仍为 `algorithms/marl/train.py` 和 `evaluate.py`，算法版本为 `marl_q_v3`。此次将 Global/Local 后继 Q 从完整策略期望改为单样本估计，Global 仅查询抽中的一个 Local；其他已确定的适配方案保留。

## 实际采用的方案

- 每个 Host 有完整独立的 Local Actor、Critic，以及各自的 Actor/Critic Adam 优化器；参数、优化器状态、经验后继均不共享。
- Local 观测移除 Host one-hot。程序按 Global 选择的 Host 路由到对应 Local，Host 编号不输入 Local 网络。
- Global 仍选 Host，Local 仍选该 Host 内的空闲 VM；任务仍为 FCFS。
- Global Critic 读取 Global 观测、Host 动作和本次 Local 的 VM 动作；Local Critic 读取本地观测和 VM 动作。
- Critic 的动作采用 one-hot 编码。这是动作输入，不是已移除的 Local Host 身份特征。
- 熵正则和梯度裁剪均关闭。配置拒绝非零 `ent_coef` 或 `grad_clip`，代码没有执行梯度裁剪。
- 保留固定源训练尺度 F18/F19 归一化、原 Local 预测特征及全部原 Local 非身份字段。
- 保留现有奖励、物理执行、通信、能耗、到达与截止期规则。

## Global 观测

Global 参照 MAHDRL HA 的字段定义，使用 `task(12) + queue(10) + host(5H)`：当前任务 MI/输入/输出、I/O ratio、等待、子节点数、上下行 Rank、全局空闲 VM 下的最佳/平均预计执行时长、Slack、同工作流 Ready 占比；队列统计；逐 Host 资源能力、负载、空闲 VM 比例、可用延迟和运行比例。

相同物理状态与相同当前任务下，新 Global 观测与 MAHDRL HA 数值对齐。它不要求两算法选择相同任务或动作。S/M/L 资源的 Global 维度分别为 37/52/67。

Local 的字段定义保持原版本：仍使用所选 Host 内的候选 VM 预测，子节点数/上下行 Rank 三项仍为零，队列总 MI 原占位仍为零。仅删除末尾 Host one-hot。其维度为 `27 + 5*maxVM`，当前三个资源规模均为72。没有把 Global 新增字段反向加入 Local。

## 原文未完整规定部分的最小实现

原文印刷页609式(12)、图3与610页Algorithm 1/2明确使用动作条件 Q 与局部邻域动作共享。Algorithm 1 第10/11步、Algorithm 2 第7/8步均先更新 Critic，再更新 Actor；当前按此执行，Actor 重新计算并读取该条经验更新后的 Q。

原文明确采用 TD 学习，但未给出完整 TD 目标和损失。这里对真实后继决策状态采用更新前当前策略的一步单样本 Q 估计；这属于原文未规定部分的工程实现，不能称为作者唯一指定的 TD 公式。Global 按合法动作掩码抽取一个 Host，只查询该 Host 的 Local 并抽取一个合法 VM；Local 只从自己对应 Host 的合法 VM 中抽样。

```text
Global: Q_g(o_g, a_host, a_vm)
Local h: Q_h(o_h, a_vm)

# 仅为 TD 估值抽样，不作为实际执行动作：
next_host ~ pi_g(.|next_o_g, legal_host_mask)
next_vm ~ pi_next_host(.|next_o_next_host, legal_vm_mask)
next_Q_g = Q_g(next_o_g, next_host, next_vm)
next_local_vm ~ pi_h(.|next_o_h, legal_vm_mask_of_h)
next_Q_h = Q_h(next_o_h, next_local_vm)
y_g = r_g + gamma * next_Q_g
y_h = r_h + gamma * next_Q_h
critic_loss = MSE(Q_current, stop_gradient(y))
# 执行 Critic 优化器后，重新读取同一观测与实际动作的 Q：
actor_loss = -log pi(actual_action | observation) * stop_gradient(Q_after_critic_step)
```

Actor/Critic 使用独立网络。`record()` 只保存实际观测、动作与奖励。真实后继决策出现后，先进行一次后继估值抽样并读取 Q，以本条奖励更新 Critic，再按式(12)更新 Actor，最后才从更新后的 Actor 独立采样实际执行动作。终止经验不进行后继抽样，以零未来项更新 Critic 后同样更新 Actor。每条经验两个优化器各执行一次。

Global 连接下一次真实 Global 决策；Local h 只连接自己下一次接到任务的决策。Local A→B→A 时，A 等到第二个 A 决策才 Bootstrap；B 不会触发 A 的更新。在记录经验与提交该经验之间，同一个 Actor 没有其他更新，因此重算历史动作的策略概率仍对应其执行策略，不需要保留跨优化步骤的计算图。

已移除 v2 的全 Host Local 策略查询和合法 Host–VM 组合枚举。每次 Global 非终止目标只查询一个 Local，并对一个动作组合读取一次 Critic；每次 Local 非终止目标也只读取一个动作的 Q。`sample_target_action()` 与行为接口 `select_action()` 分开：估值抽样只消耗策略随机流，不推进环境、不写入经验、不更新网络，也不复用为随后实际执行的动作。它使用更新前策略，实际动作使用更新后策略；终止时两者没有后继估值抽样。

固定状态、策略和网络时，单样本估计与 v2 策略期望的均值相同，但增加动作抽样波动并减少计算量，不保证性能升降。该变更用于收窄训练信息访问范围；不能宣称单样本 TD 是原文唯一指定的实现。评测不调用后继估值接口，仍使用实际 Actor 的合法动作 argmax。

折扣按各学习器自身的一次决策计一次，gamma=0.95；不是按物理秒数或其他 Host 的分配次数折扣。原文没有指定这类异步时间折扣细节，这里作固定工程约定。Episode 终止时，所有尚未提交的 Global/Local 经验以零未来项结清；显式截断单独记为 cutoff，不跨回合连接。

未增加 V 基线、优势归一化、熵、裁剪、回放、目标网络、PPO、GAE 或专家动作。原网络宽度 `1024→1024→512→512→256`、ReLU、Adam、lr=3e-4、gamma=0.95 保留；Actor/Critic 分别优化，Critic 使用未加额外权重的 MSE。

## 奖励与比较口径

```text
D = clip(max(0, task_finish-task_subdeadline)/300, 0, 1)
r_g = -0.75*D - 0.25*0.001*delta_system_host_window_energy
r_h = -0.75*D - 0.25*0.001*delta_selected_host_window_energy
```

Local h 只接收原环境在选择 Host h 的分配步骤所返回的 `r_local`。其他步骤没有收到的本 Host 能耗，不因独立 Local 改造而额外补发；也不重新分摊所有 Host 的能耗。此处保持用户选定的 MAHDRL HA/VA 对应奖励范围与结算口径。每条实际返回奖励只提交一次。

环境八个物理/奖励/FCFS 核心方法与改造前 Git 内容的 AST 相同。固定动作逐步对照59次任务分配，新旧 Local 观测除末尾 Host one-hot 外逐项相同，逐步奖励、资源可用时刻及最终指标完全相同，见 `MARL_Q_preservation.json`。

## 模型和运行

显示名仍为 **MARL-Q**，其中 Q 表示动作条件价值函数；文件标识升为 `marl_q_v3`。这次改变了训练估计器，故 v2（含长名 `marl_independent_q_v2`）checkpoint 显式拒绝加载，需要按 v3 重训。既有文件不搬移、不改写；历史验收 JSON 保留当时的版本名称。

训练、评测 CLI 不变。模型/日志名称使用 `marl_q_v3`，模型初始化种子也进入名称。正式评测和指标记录进入新版本目录。旧模型、缓存和校准文件不覆盖；旧 A2C、v1、v2 checkpoint 显式拒绝加载，需要重训。

结果按独立对比算法标记：显示名 `MARL-Q`，机器标识 `marl_q_v3`。统一定义在 `algorithms/marl/identity.py`；场景模型名、控制台、CSV 的 `algorithm`、训练/评测测量 JSON 共用此标识。原先评测控制台的 `MARL0.75-MIX`、CSV 文件名 `run_marl0p75_style...` 不再用于新输出。

默认输出如下（路径相对项目根目录）：

```text
outputs/ckpts/marl_q_v3_task.../best_global.pth
outputs/ckpts/marl_q_v3_task.../best_local_actor_critics.pth
outputs/logs/marl_q_v3_task.../train_metrics_marl_q_v3.csv
outputs/run/marl_q_v3/<归一化/协议/能耗版本>/from_<训练场景>_model<模型种子>/eval_<评测场景>/
    evaluation_marl_q_v3.csv
    measurements/episodes/seed700.json
    measurements/workflows/seed700.csv
    measurements/decisions/seed700.json
```

例如源训练场景为 `SS_T`、模型种子为1，评测场景为 `SS_Tight` 时，末两级为 `from_SS_T_model1/eval_SS_Tight`。训练来源、模型种子、评测场景均参与目录名，避免不同模型的评测互相混入。CSV 同时记录模型种子和训练来源场景。`--dry-run` 显示完整路径。

旧 MARL/A2C/v1/v2 的其他目录不会触发新版输出检查，不会被自动搜索或删除。同一新版训练目录非空时，沿用 `FileExistsError` 拒写；训练过程中同一次运行仍正常刷新本次最佳模型。同一评测 CSV 已存在（即使为空），或该评测附属目录非空时，评测在模型加载与日志写入前拒绝追加/覆盖。直接调用逐种子评测接口时，测量器也会拒绝覆盖已有的该种子 episode/workflow/decision 文件。

另一次评测使用 `--out <新的CSV路径>`；附属明细自动进入该文件旁的 `<CSV主名>_marl_q_v3_artifacts/`。显式指定已存在的旧 CSV 同样会被拒绝。训练和评测入口脚本名仍为 `algorithms/marl/train.py`、`evaluate.py`，原命令可以继续使用。

最佳验证模型仍按公共能耗选模规则保存。一次保存整套 Global 与所有 Local，不为不同 Host 单独挑选最佳回合再拼装。Local checkpoint 保存各 Host 的 Actor、Critic、两个优化器；加载检查版本、归一化、输入/动作维度、Host 数及配置。

`algorithms/marl/agents.py` 的旧 A2C 类仅供历史 F11/F19 兼容测试；正式训练、评测、生产探针均调用 `q_agents.py`。新机制的验收测试位于 `common/test_marl_q.py`。

## 验证与边界

以下1–5为历史 v2 验收记录，不作为单样本 v3 目标的正确性证据；当前 v3 验收列在其后。

1. 更新顺序修复时，`python -B -X utf8 -m unittest common.test_marl_q -q`：当时11项通过。覆盖 Critic→Actor 顺序、Actor 使用更新后的 Q、同条终止奖励影响 Actor、每条经验只更新一次、合法动作策略期望、Global 条件 Local 策略、无额外采样、Host 独立链及 v1 模型拒载。
2. `python -B -X utf8 -m unittest common.test_f11 common.test_f18 common.test_f19 common.test_scenarios common.test_energy common.test_communication common.test_task_deadlines common.test_paired_instances common.test_f22 -q`：92项通过，与前组共103项不同测试。
3. `common.smoke_marl_q.run_smoke((1024,1024,512,512,256))`：正式隐藏层宽度 CPU 短程训练、验证、整套最佳模型保存/重载和确定性评测通过。59次分配，Global 两个优化器各59步；三个 Local 两个优化器分别各25/19/15步。每条经验恰好更新两个网络一次，下一动作采样前已完成待处理更新，奖励余额均为0，评测不更新参数，损失和梯度均有限。结果见 `MARL_Q_v2_smoke.json`。
4. `python -B -X utf8 -m common.probe_F11 --out docs/MARL_Q_v2_probe.json`：生产宽度有限更新探针通过；其中 MARL 调用统一的 Critic→Actor 更新入口。Global 探针的 Local 条件概率使用固定合法动作均匀分布，仅用于有限性测试；正式训练使用各独立 Local 的实际策略概率。`git diff --check` 通过。
5. 输出身份整理后，`python -B -X utf8 -m unittest common.test_marl_q common.test_scenarios common.test_f11 common.test_experiment_measurements -q` 分两组执行，共49项通过。新增4项输出隔离/拒写测试；MARL 测试现为15项。小网络真实训练、完整评测 CSV 入口及测量 JSON 身份校验通过；重复运行拒写且已有文件内容不变，见 `MARL_Q_v2_output_smoke.json`。该项验证没有重跑600回合训练。

当前 v3 验收：

- `python -B -X utf8 -m unittest common.test_marl_q -q`：17项通过。验证单个后继 Q 而非期望/最大值、策略概率抽样与合法掩码、Global 仅查询一个 Local/读取一次 Q、无后继梯度、Critic→Actor、同条终止奖励影响 Actor、终止不抽样、Host 独立链以及 v1/v2 模型拒载。
- `python -B -X utf8 -m unittest common.test_f11 common.test_f18 common.test_f19 common.test_scenarios common.test_experiment_measurements common.test_energy common.test_communication common.test_task_deadlines common.test_paired_instances common.test_f22 -q`：99项通过，与前组共116项不同测试。
- 正式网络宽度 `common.smoke_marl_q.run_smoke((1024,1024,512,512,256))` CPU 临时训练/验证/重载/完整评测通过。59次分配，Global 非终止估值58次、恰好查询58次 Local；Local 非终止估值56次。Global Actor/Critic 各59步，三个 Local Actor/Critic 分别各34/11/14步。奖励余额为0，实际动作调用仍为每次分配一组，评测无 TD 抽样且不更新参数；重复训练/评测拒写，已有文件不变。损失和梯度有限，结果见 `MARL_Q_v3_smoke.json`。
- `python -B -X utf8 -m common.probe_F11 --out docs/MARL_Q_v3_probe.json`：生产宽度有限更新探针通过，MARL 使用单样本目标和 Critic→Actor 更新。探针仅对被抽中的 Local 使用合法 VM 均匀抽样；正式训练使用该 Local 实际策略。`git diff --check` 通过。

历史 v1/v2 的 `MARL_Q_smoke.json`、`MARL_Q_probe.json`、`MARL_Q_v2_*.json` 保留原内容，不作为 v3 正确性的证据。`MARL_Q_preservation.json` 对应此前环境与奖励适配验证，本次没有改动环境与奖励。

未运行600回合正式训练、性能消融或CUDA验证；短程有限性不能证明长期收敛。独立Actor/Critic使参数量与内存、计算开销随Host数量增加，这是原文结构恢复带来的成本，未通过共享参数规避。

保留的统一奖励、预测特征、公开全局队列信息及Host/VM物理模型均属于新问题适配，因此仍不能称为原论文原始系统与实验的完全复现。
