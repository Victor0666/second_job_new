# MARL-Q V4

> 本文记录历史版本 `marl_q_v4`。当前入口已升级为 [MARL-Q V4.1](MARL_Q_v4_1_implementation.md)，训练和评测请使用 V4.1 命令文件。

正式版本为 `marl_q_v4`，显示名 MARL-Q。入口仍为 `algorithms/marl/train.py`、`algorithms/marl/evaluate.py`。本次按用户确定的方案修改 MARL；PD3QN-DAG-FE 只提供纯选模函数，其他算法的资源约束、学习机制和选模规则不改。

## 调度与物理执行

Global 选择 Host，独立 Local 选择该 Host 内任意真实 VM，包含忙 VM。掩码只屏蔽不存在的槽位。只分配依赖已满足的 Ready 任务，按 Ready 时间和任务 ID 的 FCFS 顺序处理。每 VM 按分配顺序 FIFO、非抢占执行，不增加迁移、重排、等待动作或队列长度限制。

分配时预约区间：`start=max(now, vm_queue_tail)`，`finish=start+exec_time_components(...)`。`vm_available_at` 保存预约队尾，`vm_running_task` 保存实际执行任务。排队任务为 Queued，开始事件将其变为 Running，完成事件将其变为 Finished。同刻先处理完成再开始，并在决策前处理该时刻的全部事件和工作流到达。

排队不增加 Host 运行算力。在线能耗只按真实运行负载积分，记录区间为真实执行区间；继续使用公共 Host 任务窗口口径，包含窗口内部空闲间隔。回合必须等待所有任务实际完成；显式 horizon 截断不会把预约完成时间冒充真实完成。

## 精简输入

- 任务4维：MI、总输入 bits、输出 bits、工作流截止期减当前时间。
- Global：任务4维、Ready 数量1维、每 Host 的平均 VM 算力与实际算力利用率2维。
- Local：任务4维、每 VM 的算力/带宽/实际空闲标志3维，无全局队列、Host 聚合或身份 one-hot。

S/M/L Global 为11/17/23维；Local 当前均为31维。真实 VM 数不足9时只为动作布局保留结构性 padding；删除旧的四项固定零特征。

移除 I/O ratio、任务等待和关联比例、拓扑 Rank/子节点数、执行时长预测、预测 Slack、队列派生统计、VM 可用延迟和预计完成时间。保留底层执行时长、DAG 依赖、父任务位置和 LF 子截止期计算。F19 使用源训练场景的冻结尺度；不改公共尺度文件和其他算法的预处理。

## 实际后继动作 TD

每个决策点只采样一次实际 Host 和 VM，并缓存。两个后继 Q 都在网络更新前读取。Global 的前一条经验以本次实际 Host/VM 作为后继动作；Local h 只以 h 自己下一次接任务时的实际 VM 作为后继动作。随后每条待处理经验执行 Critic→Actor，再执行缓存动作，绝不二次抽样或静默替换。

```text
y_g = r_g + gamma * Q_g(next_obs_g, actual_next_host, actual_next_vm)
y_h = r_h + gamma * Q_h(next_obs_h, actual_next_vm_of_h)
terminal/cutoff: y = r
critic_loss = MSE(Q_current, stop_gradient(y))
actor_loss = -log pi(actual_action|obs) * stop_gradient(Q_after_critic_step)
```

这是 SARSA 式目标的 Actor-Critic 适配，不是原论文唯一规定的 TD 公式。缓存动作来自更新前策略，对应一次更新的策略滞后；不增加重要性修正、回放、目标网络、基线或优势归一化。

每个 Host 的 Actor/Critic 和两个 Adam 优化器独立。隐藏层保持1024/1024/512/512/256，ReLU，lr=3e-4，gamma=.95。熵正则与梯度裁剪继续为0。

## 奖励与选模

奖励保留：`D=min(1,max(0,finish-task_LF)/300)`；Global 为 `-.75*D-.25*.001*delta_system_energy`，Local 为 `-.75*D-.25*.001*delta_selected_host_energy`。完成时间包含 VM 排队等待，不另加等待惩罚或重新分配 Local 能耗。

直接复用当前 PD3QN-DAG-FE 的 `select_sr099_minenergy_v1`：每回合在固定验证集确定性评估，总成功工作流数除以总工作流数得到 SR；未完成计失败，验证必须覆盖配置工作流数。SR>=.99 为可行，可行优先；可行候选先比平均实例总能耗，再比更高 SR；无可行候选时先比更高 SR，再比更低能耗。只有严格更优才替换，完全平局保留较早模型。不计算 ESP。

当前 seed600、50工作流的验证实例需要50/50按时完成才可行。未找到可行模型仍保存最佳备用模型，并标记 `feasible_model_found=false`。测试700–729不参与选择。

验证直接返回选模明细，不依赖可关闭的测量器；验证不更新参数和学习计数，并恢复 Python/NumPy/Torch CPU/CUDA 随机状态及各网络模式。选中时整套保存 Global 和全部 Local，以相同 pair_id、回合、验证记录和政策绑定，评测拒绝错配。

公共 F22 JSON 和校准绑定不修改；MARL 有效协议、dry-run、检查点、日志和总结明确覆盖新选模声明。新增 `validation_selection.jsonl`、`training_summary.json` 和版本化评测 summary。目录包含 v4、源场景、模型种子和选择规则；旧模型拒载，已有输出继续拒绝覆盖，需要重训。

## 验收入口

`python -B -m unittest common.test_marl_q common.test_marl_v4 -q` 检查动作条件 Q、Critic→Actor、Host 独立链、FIFO、依赖释放、真实负载/能耗、截断、精简输入、选模边界、配套检查点和验证隔离。

`python -B -m common.smoke_marl_q` 临时训练、验证、保存/重载、完整评测及重复写入保护。正式宽度烟测可调用 `run_smoke((1024,1024,512,512,256))`，不代表600回合收敛或性能消融。

2026-10-27 已完成58项针对性测试：V4/Q、F11、通信、受影响的 F18/F19 布局和 F22 验证检查。三回合真实训练测试在测量关闭时仍保存第2回合的可行模型，拒绝第3回合低能耗但不可行的替换。

正式隐藏层宽度 CPU 端到端验收见 `MARL_Q_v4_smoke.json`：59次分配、58次 Global 后继 Q、56次 Local 后继 Q；实际策略调用每次分配各一组，奖励余额为0，检查点重载、无学习评测和重复输出拒写通过。

运行命令见 [600回合训练](Linux_CUDA_MARL_Q_v4_600代训练指令.txt)和[27场景评测](Linux_CUDA_MARL_Q_v4_27场景评测指令.txt)。未执行600回合正式训练或CUDA验收。
