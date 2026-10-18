# F11 实现返修与验收记录

修改分支：`codex/f11-real-transitions`。

修改前 Git Commit：`012619f6beab40f586b8491b1d8007a979f415ab`。开始 F11 时，工作区已有上一任务尚未提交的 F07 修改；创建分支时完整保留。因此当前分支同时承载原有 F07 工作区改动与本次 F11 改动，不能把相对 HEAD 的全部差异都算作 F11。本次没有提交或推送 Git，也没有覆盖历史缓存、模型、日志或结果。四份无关的未跟踪 Word 报告未修改。

## 1. 问题与修复范围

检查确认：MAHDRL 及四种消融中保留的资源学习层、PD3QN、MARL Local 存在固定终止经验；IRWS PPO 的非终止 Rollout 尾部原先使用零价值。它们分别退化为即时奖励目标或错误截断未来价值。

本次只修改真实决策转移、必要的奖励收尾与终止语义、学习诊断和结果版本隔离。没有调整任务负载、DAX 文件模型、动态到达、资源、Deadline、alpha、功耗积分、通信共置规则、网络层数/维度、优化器、学习率、gamma、目标网络更新规则和原有最佳模型选择标准。

`F11_configuration_check.json` 通过 AST 对照记录了网络类定义和八个训练入口选定超参数与上述 Git Commit 一致。F07 的文件模型修改继续沿用，没有在 F11 中重新修改。

## 2. MAHDRL 的真实决策单位

- TA：一次真实排序 Phase 的权重增量选择。Phase 后获得环境推进与已完成任务的阶段奖励，下一状态对应下一次排序前的 Manager 状态。保留原有经验、更新阈值、探索和目标网络机制；没有将 Phase 结束当成整个 Episode 终止。
- HA：每一次为实际待分配任务选择合法 Host。上一条 HA 经验等到下一次 `get_host_state_for_next_assignment()` 返回有效状态后才能补全。
- VA：在本次 HA 已选择 Host 后，为当前任务选择实际 VM 槽。上一条 VA 经验等到下一次 Host 已确定、`get_vm_state_for_current_task()` 返回有效状态后补全。下一 Host 可以与当前不同。
- TL：没有 Host 学习层，直接全局 VM 选择；下一状态来自其下一次真实全局 VM 决策。

公共 `PendingDecision` 保存当前观测、掩码、动作、奖励和 Host 的独立快照。`advance()` 只有在真实有效下一决策出现时才提交非终止经验；`finish()` 关闭真正终止或已明确记录的无后续状态截断。不能用空掩码完成普通非终止经验，不能覆盖尚未补全的经验。

`DQNDecisionTrace` 复用原 `remember()`、warmup 比例和 `update()`。更新从“分配后立即存伪终止样本”移到“下一真实决策到达、样本完整后”；终止时提交最后一条。这一必要的提交时机变化没有改变更新阈值或折扣配置。

TA 现有跳过零分配 Phase 的配置保留；正常环境 reset/推进后产生实际可调度任务，排序准备函数处理全部 Ready 任务。短程运行未出现有奖励却被跳过的空 Phase。HA/VA 的待处理经验跨空 Phase 保留，不构造虚假资源动作；额外时间推进只对已有待处理经验结算实际新增奖励，无先前动作却出现非零奖励会明确报错。

## 3. 延迟能耗奖励与守恒

同一物理时刻连续分配任务时，没有经过时间的动作能耗增量可以为零。最后一个触发推进的动作返回从旧时刻到下一有效决策点的能耗奖励，其中可能包含之前动作启动的并行任务成本。这是事件驱动的延迟团队信号，不是最后一个任务的独占边际成本。

保持论文现有式 (30)、(31) 的权重和范围：

- HA 使用该推进区间的全系统能耗增量与当前任务迟延；
- VA 使用该区间当前选定 Host 的能耗增量与当前任务迟延；
- VA 的局部信号不是全系统能耗的另一份完整积分，不能把所选 Host 的能耗和全部主机总能耗强行要求相等；
- 较早的同刻分配通过非终止后继状态的价值项学习后续区间信号。本次没有平均分配阶段能耗，没有新增任务边际能耗或额外奖励项。

`vm_assign()` 已推进时，`finish_phase_and_advance()` 只复用阶段统计供 TA 使用，不再增加物理能耗。新字段 `lower_energy_already_emitted` 区分该情况。如果 Phase 收尾确实进行了额外推进，`finish_lower_phase()` 才把尚未交给下层的能耗按原权重补入最后待处理转移；VA 使用该转移记录的 Host 对应分解。任务迟延在分配时已按确定的结束时间计算，收尾不再重复发放。最后一次分配后的运行尾部由环境推进至实际完成，并由同一机制结算。

`LowerRewardAudit` 记录阶段物理能耗、下层区间能耗、所选 Host 局部能耗，以及按原公式计算的 HA/VA 奖励差额。八入口验证中，主算法和四种消融的物理能耗差额均为 0，下层区间能耗差额绝对值小于 `9e-13`，HA/VA 奖励公式差额均为 0。所有待处理奖励的记录值与提交值相等。

这些检查证明计量与结算守恒，并不证明单动作的因果归因或最优信用分配。HA 仍是延迟团队信号，VA 仍是论文规定的所选 Host 局部信号；这是本次保留的奖励语义。

## 4. D3QN 与 PD3QN

两个 D3QN 实现的目标仍为：

```latex
\begin{align}
 a^* &= \arg\max_{a':m'(a')=1} Q_{\theta}(s',a'),\\
 y &= r + \gamma(1-d)Q_{\theta^-}(s',a^*).
\end{align}
```

Online 网络在合法下一动作中选择，Target 网络估值；终止时只使用即时奖励。非终止的空下一掩码会被拒绝，不能悄悄变成 bandit 样本。Replay 的当前/下一观测与掩码均复制保存。

PD3QN 训练直接使用 `step()` 返回的 `obs2` 和下一合法动作掩码，保留 Dueling、Double DQN、PER 抽样、重要性权重和按新 TD-error 更新优先级。环境会在暂时没有当前任务时重新观察/推进到真实决策，不以 `_cur_task_gid is None` 伪造终止。F01 回归仍验证 Running/Finished、前驱完成后才释放后继及 Episode 尾部实际完成。

可手算测试：`r=-2, gamma=0.9, Q_next=-10`，终止目标 `-2`，非终止目标 `-11`。Online 与 Target 有不同最优动作，并设置最大的非法动作，测试仍得到正确 Double DQN 目标。PER 的优先级使用对应新 TD-error。三次分配奖励 `[0,0,-2]` 的测试验证前两个非终止目标可分别包含 `-1.62` 与 `-1.8` 的后续价值。

## 5. MARL Global/Local

Global 保留真实一步 A2C 更新。Local 保留共享 Actor、按 Host 独立 Critic；当前状态由当前 Host Critic 估值，下一状态由下一次实际 Global 选择的 Host Critic 估值。更新接口新增可选 `next_host_index`，非终止 Local 更新必须明确提供它。

Local 待补全经验在下一次 Global 选择 Host 后、下一 Local 选择 VM 前完成更新。测试覆盖同 Host 连续决策、Host 1→Host 3，以及最后一条 Terminal。

这是按真实 Global→Local 联合调度顺序构成的条件化 Local 决策轨迹，而不是每个 Host 完全独立的轨迹。跨 Host 的价值衔接是本次按任务要求明确采用的实现修正，不能宣称原论文已明确规定该实现。

短程训练进一步复现了一个直接影响真实 Local 决策的接口错误：`get_local_state()` 原先仅在 Host 队列非空时返回状态，但训练/评测在 `step()` 入队之前请求 Local 状态，因此常得到全零状态/掩码和空 VM 映射，继而使用默认动作。必要修复让该接口在 Global 选定 Host 后即可为当前 FCFS 任务返回真实条件化状态，不改变队列调度或任务排序。新增回归断言该预览状态、掩码与 `step()` 入队后实际使用的值完全一致。训练不再为无实际 Local 动作的状态生成策略样本。

## 6. IRWS PPO 与 GAE

保留任务选择器、内在奖励、PPO 网络、Clip、GAE、更新阈值 2048、四个更新 Epoch 及全部训练超参数。

非终止 Rollout 达到阈值后：先推进到下一有效决策，Task Selector 选定下一任务，获取该任务的真实 VM 状态；用更新前 Critic 计算 `last_value`；执行上一批 PPO 更新；再用更新后的策略选择下一批的第一个 VM 动作。没有把尚未选定任务的任意中间状态当作末端状态。

```latex
\begin{align}
 \delta_t &= r_t + \gamma(1-d_t)V(s_{t+1})-V(s_t),\\
 A_t &= \delta_t + \gamma\lambda(1-d_t)A_{t+1}.
\end{align}
```

批次最后的 `V(s_{t+1})` 由真实下一决策提供；正常 Terminal 为零。`update()` 对非终止批次缺少末端价值直接报错并保留缓冲区。修复原先单条尾部经验被直接清空的问题；单样本优势标准差使用零，正常多样本保持原标准差计算。未更改 GAE lambda 或 PPO 损失结构。

实际 LS/seed=1 短程回合有 2940 次 VM 决策，在第 2048 条后发生一次非终止更新，末端 Critic 值为 `0.05522283911705017`；末尾 Terminal 更新使用零。PPO 共完成 48 次小批量梯度更新，最后缓冲区为空。

## 7. Terminal、Phase End、No Decision 与 Truncation

| 情况 | 学习处理 |
|---|---|
| 所有已请求到达的工作流与实际运行任务完成 | 真正 Terminal，未来项为零 |
| 当前 Phase 结束，但 Episode 继续 | 继续等待同一学习器的下一真实决策，非终止 |
| 暂无 Ready 任务/可用 VM，仍有未来事件 | 推进环境，不构造虚假动作或终止样本 |
| 未完成任务且已无任何到达/完成事件 | 明确报错；不能依赖保险计数器伪造成功终止 |
| 时间/步数截断，有有效返回决策 | PD3QN 和 PPO 使用真实下一价值，随后断开 Episode |
| 截断且环境无法提供真实后续决策 | 显式无后继 cutoff，停止 Bootstrap 并单独计入 `truncation_cutoffs` |

MAHDRL/MARL 现有层级接口的 horizon 结束没有返回下一实际下层决策，因此使用明确的 cutoff 策略。`done` 在 Replay 中表示“停止未来价值项”；诊断中的正常 Terminal 与 cutoff 分开。达到训练硬步数上限时也会关闭并记录下层待处理样本。正常完整短程回合的 cutoff 数全部为零。

PD3QN 原先“无当前任务就 Terminal”的逻辑已改为先寻找真实决策。MAHDRL、MARL、IRWS 原先无事件仍可能强制置终止的保护分支改为报错。这些都是必要的终止语义保护，不修改正常完成事件或依赖释放规则。

## 8. 消融与启发式

| 名称 | 实际入口 | 保留学习器 | 本次处理 |
|---|---|---|---|
| MAHDRL | `algorithms/mahdrl/train.py` | TA、HA、VA | TA 核验；HA/VA 真后继 |
| TL | `algorithms/ablation_nohost/train.py` | TA、全局 VM VA | 直接 VM 真后继，不恢复 HA |
| NTA | `algorithms/ablation_mgrrand/train.py` | HA、VA | 保留随机合法 Manager 增量替代策略 |
| NHA | `algorithms/ablation_hostrand/train.py` | TA、VA | 保留随机合法 Host，不新增 HA |
| NVA | `algorithms/ablation_vmrand/train.py` | TA、HA | 保留随机 VM，不新增 VA |
| ERTF-H-FIFS | `algorithms/ertf_fifs/evaluate.py` | 无 | 源码未修改；做公共环境兼容验证 |

所有保留的资源学习层使用同一真实决策经验机制，各自动作空间、掩码和状态构造不变。单 VM 情况下参考 FCFS/FIFS 与启发式使用环境的 Makespan/能耗一致，现有公共环境回归全部通过。

## 9. F11 诊断、模型版本与历史保护

新增 `common/f11.py`，统一版本为 `f11_real_decisions_v1`。诊断写入训练日志目录下的 `f11_diagnostics.jsonl`，可通过以下方式关闭统计与文件记录：

```powershell
$env:F11_DIAGNOSTICS = '0'
```

日志为累计学习统计，含有效/非终止经验数和比例、采样目标数、实际非零未来项比例、未来项绝对均值、目标均值、梯度更新次数、最后/最大绝对 Loss、正常终止/截断计数、PPO 非终止 Rollout 数和末端值、待补全数、奖励公式及物理能耗差额。损失在反向传播前检查有限性，诊断关闭也不会放行 NaN/Inf Loss。

`bootstrap_nonzero_ratio` 的分母是实际计算过目标的采样项，而非环境经验数；DQN 有 Replay 重采样，不能和非终止经验比例直接等同。一个合法非终止样本可以恰好有零未来价值，日志不把它视为非法状态。非终止空掩码则明确报错，不能伪装成合法的零 Bootstrap。

所有学习算法的默认模型名追加 `_f11_real_decisions_v1`，Checkpoint、训练日志和评测结果默认路径与历史实验隔离。训练拒绝复用非空输出目录；重新运行时应先自行保存/另行管理已有该版本目录，不自动删除或覆盖。最佳 Checkpoint 仍按各入口原先的最低评测能耗规则选择。

Checkpoint 保存新增 `learning_model` 元数据，网络参数键和形状保持兼容。底层 `load()` 默认仍可读取历史模型；正式评测使用 `require_learning_model=True` 或对 IRWS 联合模型做等价检查，拒绝把无 F11 标记的模型作为修复后的模型评测。历史模型不能只复制到新目录就充当已修复模型。显式历史加载接口只用于兼容读取，不代表完成重训。

## 10. 测试与短程训练

实际执行：

```powershell
.\.venv\Scripts\python.exe -m compileall -q common algorithms environments
.\.venv\Scripts\python.exe -m unittest common.test_f11 -v
.\.venv\Scripts\python.exe -m unittest discover -s common -p 'test_*.py' -v
.\.venv\Scripts\python.exe -m common.smoke_F11 --out docs/F11_smoke.json
```

最终全量回归：75 项通过，其中 F11 专项 23 项（T01–T20 加三项真实接口/终止保护回归）。完整记录见 `F11_tests.txt`。保存加载测试包含 D3QN、PER、Global A2C、Local A2C、PPO；新模型严格检查可加载，历史格式可兼容读取但被 F11 正式模型检查拒绝。

`smoke_F11.py` 实际调用八个原有 `train()` 入口并完成一个训练 Episode、原入口评测、梯度更新和临时 Checkpoint 保存，全部写入临时目录。七个算法使用 SS/seed=1 的 50 个工作流；IRWS 使用 LS/seed=1 的 50 个工作流以实际跨过 2048 的 Rollout 阈值。临时缓存包含 seed=1、2，以兼容原训练循环最终 reset 的行为。

为使验证轻量，只有该验证脚本通过构造器替换将隐藏层改为 `(16,8)`，DQN batch=4、buffer=32，CPU 单线程；生产训练源码的网络和超参数完全保留。它不是原配置的正式收敛训练，也不产生可用于论文的新训练结果。静态一致性证据另见 `F11_configuration_check.json`。

实际逐学习器诊断见 `F11_diagnostics.md` 与完整 `F11_smoke.json`。所有学习器均有非终止样本、实际非零未来项和实际梯度更新；所有 Episode 末端待处理经验/缓冲区为零。

执行中遇到并修正：启发式对照测试误传基础环境专用参数；MARL Local 状态请求时序错误；初次失败的临时验证在 Windows 下因未关闭 CSV 句柄导致清理异常。测试配置已纠正，真实接口已修复，训练成功结束及验证异常退出均关闭验证 CSV 句柄。以上失败不计入通过项，最终回归全部通过。

## 11. 论文 Algorithm 1、Algorithm 2 的同步建议

依据仓库 `docs/paper.pdf` 第 7–8 页：Algorithm 1 已写出 Online 选动作、Target 估值的多步目标；本次主要修复经验收集，使其实际能使用该目标。Algorithm 2 第 18–20 行应明确“待下一次相应层真实决策再完成经验”，不能暗示分配后任意中间状态就是下一状态。

可直接用于论文的英文 LaTeX 说明：

```latex
For each learning tier, a transition connects two consecutive actual
 decision epochs of that tier. After a task assignment, the host-selection
 and VM-selection transitions are retained as pending records until their
 respective next valid decision observations and action masks become
 available. The next VM observation is conditioned on the host selected at
 that next decision, which may differ from the previous host. A sequencing
 phase boundary does not by itself terminate these transitions. Pending
 transitions are closed without bootstrapping only at an episode terminal,
 or under an explicitly identified truncation policy when no valid
 successor decision can be obtained.

Energy rewards are settled over the physical time intervals advanced by
 the event-driven environment. Assignments made at the same simulation
 time may therefore receive zero immediate energy increments, followed by
 a delayed interval reward. These interval costs are counted once in the
 environment and propagate to preceding decisions through bootstrapped
 state values; no uniform redistribution of phase energy is introduced.
 The host-selection reward retains the system-wide energy increment,
 whereas the VM-selection reward retains the increment of the selected
 host, using the existing energy--tardiness weights.
```

Algorithm 2 的伪代码调整建议：在 HA/VA 真实决策前分别完成上一条 pending transition；分配后只保存当前记录及奖励；Phase 结束保留 pending；Episode Terminal 时将最后一条经验以停止 Bootstrap 标志提交。Algorithm 1 的网络结构、gamma、目标网络更新和动作权重不变。

MARL/IRWS 的实现说明（本次修正，不宣称原文已规定）：

```latex
In the corrected MARL implementation, local transitions follow the actual
 global-to-local scheduling sequence. The current local value is evaluated
 by the critic associated with the current host, while the bootstrap value
 is evaluated by the critic associated with the next selected host. This
 defines a host-conditioned decision trajectory with a shared actor,
 rather than independent per-host trajectories.

For IRWS, a nonterminal PPO rollout is bootstrapped using the critic value
 of the next VM decision after the task selector has selected its task.
 This tail value is computed before the PPO update. At a true episode
 terminal, the tail value is zero and GAE propagation is stopped.
```

仓库无可编辑论文源文件，本次只新增返修说明，没有修改 PDF。

## 12. 独立问题与验收边界

1. 短程验证中，HA/VA 与 MARL Local 的未来项和 Loss 量级较大。它们在本次验证中均有限、梯度更新可执行，但这不构成收敛性证明。现有局部观测含较大的原始 I/O 比值特征；其归一化和正式训练稳定性应另行评估。本次按要求不缩放状态、不调整奖励或学习率。
2. IRWS 训练入口未见 Task Selector 与内在奖励网络的优化器/训练调用；F11 保留其现状，没有重新设计这两部分。是否符合原 IRWS 定义需独立审查。
3. 原论文 Algorithm 1 写 MSE，而现有 D3QN 使用 Huber Loss；本次未变更既有损失函数。这是已有的论文/实现表述差异。
4. F07 已记录的工作量/处理速度单位差异与 SIPHT 表述问题仍独立存在。
5. 没有执行八种学习算法的生产网络全程训练、完整 27 组合重跑或统计显著性实验。旧模型/旧训练结果不因代码修复而成为新结果。

F11 的状态转移、Bootstrap、终止、奖励守恒与接口兼容验收在上述范围内通过。正式重训仍需先生成完整 F07 新版 Deadline Cache，并使用本次 F11 输出命名空间重新训练。全量训练的性能、数值量级与收敛表现仍需在正式运行中检验。


## 13. 最终验收表

| 算法/模块 | 原问题 | 修改文件与函数 | 修复状态 | 测试结果 |
|---|---|---|---|---|
| MAHDRL-TA | 真实多步转移核查 | `mahdrl/train.py::train`、公共 D3QN Replay 校验 | 保留并核验 | 380/381 非终止，378 次更新 |
| MAHDRL-HA | 固定 Terminal | `mahdrl/train.py`、`f11.py::DQNDecisionTrace` | 已修复 | 1363/1364 非终止，1361 次更新 |
| MAHDRL-VA | 固定 Terminal | 同上，下一 Host 已选择后取得 VM 状态 | 已修复 | 1363/1364 非终止，1361 次更新 |
| MAHDRL 奖励 | 延迟能耗归因 | `mahdrl_heft.py::finish_phase_and_advance`、`f11.py::finish_lower_phase/LowerRewardAudit` | 保留原目标，补收尾审计 | 物理/奖励守恒，未重复计量 |
| PD3QN | 仅即时奖励目标 | `pd3qn/train.py`、`d3qn_agent_PER.py`、`pd3qn.py::step` | 已修复 | 1363/1364 非终止，PER 目标/优先级通过 |
| MARL-Global | 真实转移核查 | `marl/train.py`、`marl/agents.py::A2CAgent` | 保留并核验 | 1364 次更新，未来项非零 |
| MARL-Local | 固定 Terminal | `marl/train.py`、`agents.py::LocalSharedActorIndependentCriticA2C.update_one_step`、环境 Local 状态接口 | 已修复 | 跨 Host Critic、真状态、末端关闭通过 |
| IRWS-PPO | Rollout 末端零 Bootstrap | `irws/train.py`、`vm_agent.py::value/update` | 已修复 | 非终止尾值 0.0552228391，48 次梯度更新 |
| MAHDRL-TL | 保留学习层同步 | `ablation_nohost/train.py` | 已修复 | TA、直接 VM VA 均有未来项 |
| MAHDRL-NTA | 保留学习层同步 | `ablation_mgrrand/train.py` | 已修复 | HA/VA 均更新，未新增 TA |
| MAHDRL-NHA | 保留学习层同步 | `ablation_hostrand/train.py` | 已修复 | TA/VA 均更新，随机 Host 保留 |
| MAHDRL-NVA | 保留学习层同步 | `ablation_vmrand/train.py` | 已修复 | TA/HA 均更新，随机 VM 保留 |
| ERTF-H-FIFS | 兼容性核查 | 启发式源码未修改；公共环境测试 | 已核验 | 单 VM 对照与现有环境回归通过 |


## 14. 生产网络尺寸补充验证

另执行 `.\.venv\Scripts\python.exe -m common.probe_F11 --out docs/F11_production_probe.json`，使用真实环境的连续决策状态与生产网络隐藏层 `(1024,1024,512,512,256)`，验证 MAHDRL VA D3QN、PD3QN PER、MARL Global/Local、IRWS PPO 的实际参数更新。权重与采样随机流在该验证脚本内设为 1；环境实例 seed=2。DQN/PPO 只为构造一个批次重复真实样本，因此这是网络尺寸/有限性检查，不是完整轨迹的正式训练，也不替代八入口短程验证。五项更新的 Loss 和更新后所有参数均有限。结果如下：

| 学习器 | 参数数 | 更新 Loss | 结果 |
|---|---:|---:|---|
| MAHDRL_VA_D3QN | 2308874 | 34053.348 | PASS：Loss/参数有限 |
| PD3QN_PER | 2190618 | 0.065795012 | PASS：Loss/参数有限 |
| MARL_Global | 1980676 | -0.008432202 | PASS：Loss/参数有限 |
| MARL_Local | 2049292 | 1.3051583e+09 | PASS：Loss/参数有限 |
| IRWS_VM_PPO | 2136858 | 91710010 | PASS：Loss/参数有限 |
