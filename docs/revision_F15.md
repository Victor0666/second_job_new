# F15：TA 五维精简启发式修正

本次采用方案 A，特征版本记为 `ta_five_ready_minmax_v1`（文档标识）。只改变 MAHDRL 共享 TA 排序方法及其说明；未修改 F07/F11、IRWS、Deadline Cache 或其他算法的独立排序。当前分支仍为 `codex/f11-real-transitions`，HEAD 为 `012619f6beab40f586b8491b1d8007a979f415ab`；此前未提交的修改完整保留。本次未提交或推送 Git。

## 修改文件

- `environments/mahdrl_heft.py`：`_compute_task_heuristics_for_ready()`、`_phase_prepare_tasks()`，以及头部 EDF/五维属性说明。
- `common/test_f15.py`：T01–T14 专项测试。
- `docs/revision_F15.md`：本记录与论文替换建议。
- `docs/F15_order_comparison.json`：真实固定场景的新旧特征、分数和排序。
- `docs/F15_tests.txt`：实际完整回归输出。

相对 HEAD，`mahdrl_heft.py` 中另外两个方法 `finish_phase_and_advance`、`_advance_until_decision_energy_only` 的差异属于此前 F11，本次没有再次修改。不得将当前工作区所有差异归入 F15。

## 修改前的实际计算

固定顺序一直是 `[FCFS,SJF,MCF,HUR,EDF]`，但含义为：

| 维度 | 原始实际计算 | 旧缩放 |
|---|---|---|
| FCFS | `max(0, current_time-ready_time)` | normalize=True 时除 Horizon |
| SJF | `1/(estimated_time+1)` | 倒数 |
| MCF | 直接子节点数量 | normalize=True 时除全部任务平均子节点数 |
| HUR | Upward Rank | normalize=True 时除计算时长构成的 time_scale |
| EDF | `1/(max(0, workflow_deadline-now-estimated_time)+1)` | 工作流 Slack 倒数，负 Slack 被截断 |

旧排序使用综合分数降序，平局交由 NumPy argsort，没有显式 ReadyTime/任务 ID 次序。

## 新定义与实际数据源

设当前 Ready 集合为 R、当前环境时间为 t。B 为所有 VM 带宽的算术均值，换算为 bits/s；P 为所有 VM 计算能力的算术均值，单位 MI/s。

`ProcessingTime(i) = task_in_bits[i]/B + task_mi[i]/P + task_out_bits[i]/B`。

只用于任务排序，不替代实际 VM 执行时长；此时尚未选择资源，不按实际同 VM 放置减免估计数据量。InputBits 使用公共数据生成器确定的外部输入加全部父任务输入；F07 的完整父输出模型自然传入，无独立随机数据或 DAX 原始文件大小。

| 固定位置 | 标签 | 新原始属性 | 实际数据源 | 优先方向 |
|---|---|---|---|---|
| 0 | FCFS | ReadyTime(i) | `task_ready_time[gid]` | 越小越优先 |
| 1 | SJF | ProcessingTime(i) | `task_in_bits/task_mi/task_out_bits` 与当前 VM 平均资源 | 越小越优先 |
| 2 | MCF | UpwardRank(i) | `task_up_rank[gid]` | 越大越优先 |
| 3 | HUR | TaskSubDeadline(i)-t-ProcessingTime(i) | `task_baseline_finish[gid]`、环境当前时间、同一 ProcessingTime | 越小越优先 |
| 4 | EDF | TaskSubDeadline(i) | `task_baseline_finish[gid]` | 越小越优先 |

FCFS 是 Ready Queue First Come First Served，不是工作流 Arrival Time 排序。入口任务进入 Ready 时记录当前就绪时刻；子任务仅在 `unReady` 且所有前驱完成后首次解锁时写入 Ready 时间，持续等待期间不会覆盖。只选 `task_state == 'Ready'` 的任务，非 Ready 即使误留在列表也不参与属性集合。

Upward Rank 已由 `Workflow.compute_upward_ranks()` 在当前工作流到达、公共任务负载生成后计算：任务自身输入/计算/输出预计时长加上后继最大 Rank，反映通往出口的预计最长剩余路径成本。因此以它替代 ChildCount，可区别“后继多”与“关键路径长”；HA/VA 的 ChildCount 状态仍保留。

`task_baseline_finish` 来自共享 `task_subdeadlines`，其 Latest-Finish 返回值已加 arrival_time，是绝对子截止时刻，与 `vm_assign()` 计算迟延惩罚使用的是同一数组及全局任务索引。F15 不再推导或修改 Deadline。HUR 保留负 Slack，实际是 Signed-Slack-Based Urgency，不是执行时间/剩余时间的比率；EDF 只看截止时刻，因此与 HUR 不同。

## 逐属性 Ready 集合 Min-Max

令各属性在 R 中最小值为 m_k、最大值为 M_k。

- FCFS/SJF/HUR/EDF：`f_ik=(M_k-x_ik)/(M_k-m_k)`。
- MCF：`f_ik=(x_ik-m_k)/(M_k-m_k)`。
- 同列所有值相等：该列全为 0。
- 所有输出在 [0,1]；高分统一表示高优先级。

实际计算先以每列最大绝对值（至少为 1）缩放，再相减，数学上与上述 Min-Max 等价，防止较大但有限的正负值相减溢出。计算使用 float64；特征形状仍为 N×5，TA 状态和网络维度不变。带宽/计算均值非正或非有限、任务缺子截止时刻、属性或综合分数非有限时明确抛出 ValueError；不把异常值替换成误导性的合法分数。

`normalize=False` 保留无量纲缩放关闭的用途：输出 `[-ReadyTime,-ProcessingTime,+UpwardRank,-SignedSlack,-TaskSubDeadline]`，即原始属性乘方向符号。这样仍然高分优先，但与原来一样不保证不同量纲可直接比较；正式归一化场景使用 Ready-local Min-Max。没有使用倒数、Horizon 或 ChildCount 均值缩放。

Ready 为空时返回 0×5 空数组、空排序；只有一个任务时归一化各列为 0，唯一任务直接排入队列。当前 Ready 集合变化时重新计算各列边界，不能沿用上一 Phase 的 min/max。

## 加权排序、动作与共享调用链

`Score(i) = sum_k combo_weights[k] * f_ik`，按分数降序，平局依次按 `ReadyTime` 升序、全局任务 ID 升序。

5 维非负权重、和为 1 的归一化、raw 权重增量、动作枚举和 mask 均保持原样。`MANAGER_ACTION_TABLE` 仍是 `{-0.1,0,+0.1}^5`，顺序与 `itertools.product` 原实现相同，数量 243。

MAHDRL 训练与评测、NHA/hostrand、NVA/vmrand、TL/nohost 都继承同一个 `mahdrl_heft` 排序方法。NTA/mgrrand 虽然使用随机合法权重动作，也继承该同一排序方法，不改变其随机替代定义。本次没有恢复任何被消融的智能体。

ERTF-H-FIFS 虽共享主环境，但用单一 FCFS 权重；旧等待时间与新 ReadyTime 在同一时刻的优先方向相同，仍是 Ready 先到先服务，新平局规则为确定性任务 ID。未修改其排序策略或代码。PD3QN、MARL、IRWS 使用独立排序；预计算的 base_fcfs/base_heft 保留原代码，没有重建 Cache。HA/VA 状态、奖励和实际资源执行代码没有 F15 改动。

## 固定场景前后差异

相同当前工作区 F07 实例与环境状态：任务/资源 SS、seed=2、单工作流 `CyberShake_30.xml`、`deadline_mode='none'`、初始决策时刻 `1.298611360039865` 秒，五维权重均为 0.2，Ready 集合 `[2,13]`。

- 旧特征综合排序：`13 → 2`。
- F15 特征综合排序：`2 → 13`。

完整五列特征与分数见 `F15_order_comparison.json`。此对照是同一个未调度状态的属性排序比较；none 模式使用大值占位截止期，不代表正式 Cache/Deadline 场景性能。旧函数仅在一次性审计脚本中从 HEAD 提取用于计算，没有作为生产回退代码保留。

## 测试记录

实际执行：

```powershell
.\.venv\Scripts\python.exe -m unittest common.test_f15 -v
.\.venv\Scripts\python.exe -m unittest discover -s common -p 'test_*.py' -v
```

完整回归：89 项通过，用时 24.382 秒。专项测试 T01–T14 全部通过，覆盖：五种独立方向、SJF 秒单位、MCF 与 ChildCount 相反案例、负/零/正 Slack、EDF 与 Slack 顺序不同、Ready 集合逐列手算 Min-Max、相同属性、异常输入、极大有限值、空/单任务、五权重映射、243 动作枚举顺序与 mask、权重非负及归一化、手算综合分数、确定性平局、真实 DAG 完整调度、实际训练/评测模块继承方法身份。

初次 T14 使用了错误的评测环境类名，评测入口实际将类导入为别名；修正测试为检查模块实际导入的环境子类后通过。没有为使测试通过而更改训练/评测入口。

真实环境测试核对入口 Ready 时刻、等待期间不覆盖、子节点首次解锁时刻、Rank 与任务对象一致、罚时 Deadline 与数组一致，完成全部任务后再核对子 Deadline 原公式以及共同窗口能耗复算。未执行全量正式训练。

## Section IV-B2 可用英文 LaTeX

```latex
For each ready task $i$ at scheduling time $t$, the task-sequencing agent
uses five attributes in the fixed order FCFS, SJF, MCF, HUR, and EDF:
\[
\mathbf{x}_i =
\left[r_i,\;\widehat{p}_i,\;\operatorname{rank}_u(i),\;
 d_i-t-\widehat{p}_i,\;d_i\right],
\qquad
\widehat{p}_i = \frac{S_i^{\rm in}}{\overline{B}}
              + \frac{W_i}{\overline{P}}
              + \frac{S_i^{\rm out}}{\overline{B}},
\]
where $r_i$ is the time at which the task first becomes ready, $d_i$ is
its absolute task sub-deadline, $\overline{B}$ is the mean VM bandwidth
in bits per second, and $\overline{P}$ is the mean VM processing capacity
in MI per second. FCFS therefore refers to first come first served in
the ready queue, and SJF prioritizes estimated processing time including
both computation and input/output communication. MCF uses the upward
rank to represent the estimated longest remaining path to a workflow
exit. HUR uses signed-slack-based urgency; negative slack is retained.
EDF prioritizes the task sub-deadline rather than workflow-level slack.

Each attribute is independently min--max normalized over the current
ready set $R$. For FCFS, SJF, HUR, and EDF, smaller raw values receive
higher scores, $f_{ik}=(M_k-x_{ik})/(M_k-m_k)$; for MCF, larger values
receive higher scores, $f_{ik}=(x_{ik}-m_k)/(M_k-m_k)$, where $m_k$ and
$M_k$ are the minimum and maximum over $R$. When $M_k=m_k$, all scores
in that dimension are set to zero. Tasks are ordered by descending
$\sum_{k=1}^{5}w_k f_{ik}$, with ties resolved by earlier ready time and
then smaller task identifier. The five nonnegative weights sum to one,
and the existing $3^5=243$ weight-increment actions are unchanged.
```

符号请与论文其他章节对齐；本段只给出本次实现的精确定义，不主张采用其他 Urgency Ratio。仓库论文源文件未修改。

## 实验兼容与验收

虽然权重维度不变，五列含义及尺度已变化。旧 MAHDRL Checkpoint 不可当作 F15 正式结果，需重新训练；保留 TA 的 NHA、NVA、TL 需同步重训。NTA 的随机权重排序也改变，应重新运行其训练/评测。HA/VA 参数的张量形状兼容不代表旧模型与新排序状态分布相容。未使用本机制的独立对比算法不因 F15 单独变更。

本次没有修改统一的 F11 Checkpoint 命名/元数据或加载接口；因此代码不会自动识别“F11 已训练、F15 未训练”的历史模型，正式实验需另行标记 F15 模型目录，不能把仅通过 F11 标记检查视为 F15 已重训证明。历史文件不删除、不覆盖。若共同环境还在修订，应按共同环境版本另行统一组织实验。

| 项目 | 预期结果 | 完成情况 |
|---|---|---|
| FCFS | Ready Time | 完成 |
| SJF | Estimated Processing Time | 完成 |
| MCF | Upward Rank | 完成 |
| HUR | Signed Slack | 完成，负值保留 |
| EDF | Task Sub-deadline | 完成，绝对时刻 |
| Min-Max | Ready 集合逐列 | 完成 |
| TA 特征维度 | 5 | 保持 |
| TA 动作数量 | 243 | 保持，顺序一致 |
| TA 权重更新 | 原实现 | 未修改 |
| TA 综合排序 | 五维加权降序 | 完成，平局确定性 |
| HA/VA | 不改算法定义 | 未修改，输入任务顺序按预期改变 |
| Deadline | 不改生成及约束 | 未修改 |
| 能耗 | 不改物理模型 | 未修改，复算通过 |
| 训练评测 | 共享定义 | 方法身份检查通过 |
| F15 专项测试 | 全部通过 | 14/14 |

当前无待解决的五项属性定义歧义；历史模型的实验版本管理与其他独立问题不属于本次排序修复。完整回归的实际结果见 `F15_tests.txt`。
