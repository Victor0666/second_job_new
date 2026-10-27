# MARL-Q V4.1

版本为 `marl_q_v4_1`，显示名为 MARL-Q4.1。入口保持 `algorithms/marl/train.py` 和 `algorithms/marl/evaluate.py`。在 V4 上增加每 VM 等待队列容量屏蔽、Local 的可用等待时间和执行时长，保留 Global 基本输入以及现有学习与选模设置。

## 队列与动作约束

每 VM 默认最多等待 10 个任务，不包含正在运行的任务；即最多 1 个 Running 加 10 个 Queued。允许选择忙 VM，只有队列满或槽位不存在时屏蔽 Local 动作。Host 内至少一台 VM 的等待队列未满时 Global 才能选择该 Host。

`vm_waiting_tasks` 记录逐 VM FIFO 等待队列，`vm_available_at` 仍是预约队尾，`vm_running_task` 记录实际运行任务。任务立即开始时不占等待名额；运行任务结束后，等待队首立即开始并释放一个等待名额。完成事件只为实际运行任务登记，连续零时长任务也按 FIFO 依次处理。

所有队列满时，Ready 任务继续保留，物理时间推进并结算能耗，直到等待名额释放再决策。Queued 不增加运行负载；继续使用实际执行区间和公共 Host 任务窗口能耗口径。非法动作报错，不替换、不丢弃任务。

队列长度仅用于内部调度和动作屏蔽，不额外输入网络。容量默认值、实际环境和训练/评测配置保持一致，元数据记录容量以及“仅统计等待任务”的定义。

## 网络观测

任务4项保持：计算量 MI、总输入 bits、输出 bits、工作流截止时间减当前时间。Global 继续为任务4项、Ready 数量1项、每 Host 平均 VM 算力与实际算力利用率2项；S/M/L 资源维度为11/17/23。

Local 为任务4项加每 VM 5项，顺序为：

| VM 特征 | 定义 | 缩放 |
|---|---|---|
| 算力 | VM 计算能力 | `vm_pc_ref` |
| 带宽 | VM 带宽 | `vm_bw_ref` |
| 实际空闲标志 | 当前无运行任务为1，否则0 | 不缩放 |
| 可用等待时间 | `max(0, vm_available_at - now)`，包括运行剩余时间和已预约队列 | `T_ref` |
| 当前任务执行时长 | `_task_duration(task, vm)`，包含通信影响，不包括排队等待 | `T_ref` |

Local 当前最大9个 VM 槽位，维度为 `4+5*9=49`。没有真实 VM 的结构性槽位继续 padding；不恢复固定零占位块。执行时长复用实际分配的物理计算器及父任务位置，同 VM 数据免传输规则不变；Global 不增加执行预测循环。全部使用源训练场景的冻结 F19 尺度，不重新拟合校准文件。

不单独输入预计完成时间；不计算最佳/平均预计时长或预测 Slack。不恢复上下行 Rank、子节点数、任务等待时间、同工作流 Ready 占比、V3 整组队列派生统计、额外 Host 汇总和 I/O ratio。

## 学习、选模与版本隔离

独立 Local Actor/Critic/优化器、Global Critic 的实际 Local 动作输入、复用实际后继动作的一步 TD、Critic→Actor 顺序、奖励公式及权重均与 V4 相同。隐藏层1024/1024/512/512/256、Adam lr=3e-4、gamma=.95、熵正则与梯度裁剪为0，保持不变。验证和评测保持确定性最大概率动作。

选模继续复用 PD3QN-DAG-FE 的 `select_sr099_minenergy_v1`，总体工作流成功率达到0.99后按平均验证实例总能耗择优；无可行模型时先比较成功率，再比较能耗。固定验证集、F22 划分、验证随机状态隔离、成套 Global/Local 保存校验保持不变。

模型、日志、评测 CSV 和 summary 进入 `marl_q_v4_1` 身份对应的新目录。观测标签为 `task4_ready1_host2_local_vm5_wait_exec_v1`，调度标签为 `vm_fifo_queue_cap10_v1`；检查点写入并核对观测、调度和队列容量信息。拒绝加载 V4 及更早版本，需要重新训练；历史结果文件不迁移或覆盖。

## 验证与运行

针对性入口：`python -B -m unittest common.test_marl_v4_1 common.test_marl_v4 common.test_marl_q common.test_communication -q`。覆盖10个等待任务的边界、满队列拒绝后状态不变、Global/Local 掩码联动、全部队列满后的时间推进、零时长 FIFO、等待与执行时长分离、通信语义、输入维度及检查点绑定。

端到端烟测：`python -B -m common.smoke_marl_q`。正式网络宽度可调用 `run_smoke((1024,1024,512,512,256))`，记录见 `MARL_Q_v4_1_smoke.json`。烟测覆盖真实训练、验证选模、检查点重载、无学习评测和重复输出保护，不代表600回合收敛或成功率恢复。

2026-10-27 已通过134项回归测试，覆盖 MARL-Q/V4/V4.1、通信、能耗、任务子截止期、F18/F19/F22、配对实例、场景和 DAX。正式隐藏层宽度 CPU 烟测完成59次真实分配，Global/Local 后继 Q 读取58/56次，奖励记录余额均为0；检查点重载、无学习评测及重复输出保护通过。训练与跨任务评测 CLI dry-run 的源模型路径一致，新命令文件通过 Bash 语法检查。未运行600回合训练或 CUDA 实测。

运行命令见 [600回合训练](Linux_CUDA_MARL_Q_v4_1_600代训练指令.txt)和[27场景评测](Linux_CUDA_MARL_Q_v4_1_27场景评测指令.txt)。训练自动选择 CUDA，命令文件先验证 CUDA 可用；评测支持 `--device cuda`。
