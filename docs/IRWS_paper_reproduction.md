# IRWS 原论文复现与公共云环境适配报告

日期：2026-10-19（Asia/Shanghai）。代码基线：`57cd2b79161b51cde52eb1b8da12e56028ce1553`。

结论：新增 `algorithms/irws_paper/`，实现了真实双向 DAG 嵌入、全 Ready Softmax 选择、PPO、历史 GRU、动态奖励和实际二阶元梯度链路。14 项新测试及完整 132 项回归通过，真实 DAX 的短程训练/验证/保存/加载/评测通过。**这是带明示工程近似的论文核心机制复现，不能称逐式精确复刻或已复现论文数值结果。** 尚未运行正式训练；正式缓存及冻结 F18 配置仍需准备。

## 原文来源与核验

实际读取用户提供的 *Learning to Optimize Workflow Scheduling for an Edge–Cloud Computing Environment*，IEEE TCC 12(3), 897–912 (2024)，DOI `10.1109/TCC.2024.3408006`。源 PDF：

`E:/读博生活/0.科研/1.论文/2. 第二个任务/2. 论文撰写材料/对比算法/Learning_to_Optimize_Workflow_Scheduling_for_an_EdgeCloud_Computing_Environment.pdf`

已提取 16 页文本并实际渲染、查看相关页面：印刷页 902–907，重点为 Fig.2–4、Eq.(12)–(39)、Algorithm 1、Table III。论文是算法依据，其内部内容不作为用户授权指令。此前 `revision_IRWS_adapted.md` 因未取得全文而暂停的记录保留，本报告取代该记录中的“原文不可访问”状态。

## A. 原论文—代码对应表

| 模块 | 原文 | 新代码 | 实际完成及边界 |
|---|---|---|---|
| DAG Embedding | (12)–(18)，Fig.3，pp.903–904 | `environment.py::paper_ranks/graphs`；`networks.py::WorkflowEmbedding/TopologicalLayer` | 真实 DAX 前驱与反向后继顺序传播；Tanh 初始投影、Softmax 聚合、Bi-GRU、双方向池化；聚合/池化维度采用下面明示解释 |
| Task Selection | (19)，p.904 | `networks.py::TaskPolicy`；`learner.py::prepare/losses` | 全 Ready 全局 Softmax，训练采样、评测 argmax，保留 log-prob；Q 和嵌入均经元梯度更新；连续 task/VM 以联合概率处理 |
| PPO | (20)–(24)，pp.904–905 | `environment.py::reference_cost/paper_step`；`learner.py::gae/clipped_loss/losses` | 伪平均服务器相对改进奖励；独立 Actor/Critic；Mask 同时用于行为与更新；Clipped surrogate、MSE、熵；无 Deadline 奖励 |
| Intrinsic Reward | (28)–(29)，p.906 | `networks.py::HistoryReward`；`learner.py::history_token/remember/losses` | 历史状态/VM动作/即时反馈/done/后续状态输入 GRU，Episode 清空隐状态，16 步截断反传；式(28)真实反馈仅作诊断，不用人工 MSE 标签训练 RNN |
| Reward Fusion | (30)–(31)，p.906 | `environment.py::fusion_weight`；`learner.py::losses` | 按 t/Rank_dn 融合，取消固定 0.1；入口零分母明示补全；保留原式允许的 >1 值并统计，不能称凸组合 |
| PPO Update | (32)–(33)，p.906 | `learner.py::update` | 实际优化用 Adam；元梯度虚拟内步采用论文线性近似的一步可微梯度，不声称展开整个 Adam 历史 |
| Intrinsic Meta-gradient | (34)–(36)，p.906 | `learner.py::losses/update` | RNN→融合奖励→虚拟 PPO 更新→实际物理长期成本的有限轨迹似然比估计；梯度非零、有限，并通过有限差分；不是逐项照抄(35)/(36)的无穷和 |
| Embedding Meta-gradient | (37)–(39)，p.906 | 同上 | η/Q→内步→更新后融合 PPO 外层目标，保留二阶计算图；使用截断代理目标，不能称(37)自引用方程的唯一精确解 |
| Algorithm 1 | p.907 | `runner.py::run_episode/train/evaluate` | 收集、PPO/元更新、验证选模、保存与确定性评测闭环均实际运行；云事件决策与连续 rollout 替代原 time-slot/随机抽样内循环 |

## 原文参数与工程解释

Table III：gamma=0.2，batch=64，Actor LR=1e-5，Critic LR=1e-4，Intrinsic LR=1e-5，Embedding LR=1e-6，Adam，嵌入维度=5，轨迹长度=16，Actor 隐层=(128,64)，Critic 隐层=(64,64,32)，目标 alpha=0.5。Clip=0.2 取 p.905 公式(23)后的文字。旧 IRWS 使用的大网络、gamma=0.99 和 Deadline 奖励没有复制到新实现。

原文未提供以下完整实现，代码作出显式、固定的工程选择，没有搜索超参数：

1. **式(16)聚合**：其式子没有写出邻居向量的归约维度。按相邻文字，将旧 self-state 与已按拓扑算出的直接邻居状态放入集合，逐特征 Softmax 加权求和。不能将该解释称唯一官方实现。
2. **式(17)Bi-GRU**：以 `[旧 self-state, 聚合结果]` 为双向 GRU 的两项输入，两个方向隐藏表示均值作为节点更新；原文未给序列布局。前向 DAG 与反向 DAG 各按实际拓扑顺序计算，非独立 MLP。
3. **式(18)池化**：按 Fig.3 的任务级输出与下标 v，对每个任务的两个方向表示均值池化后 FC。没有把整个工作流所有任务池化成同一个向量，否则无法实现任务区分。L 未给定，默认一次完整拓扑传播。
4. **历史 RNN**：GRU 隐宽采用已有嵌入宽度 5。每步 token 包含 `(s, VM one-hot, r_step, duration, energy, done, s_next)`；已选任务通过 s 中的嵌入表示，Ready 集合、任务身份和 task log-prob 另行保存。历史数值跨 rollout 保留，每 16 步 detach 截断梯度，每 Episode 重新置零。duration 除源 T_ref，energy 除固定的 T_ref×源集群满载总功率，只处理 RNN 输入单位，不改变实际奖励。
5. **TD/熵/更新轮次**：默认 GAE lambda=0，对应式(31)的一步 TD；保留可验证的 GAE 实现。原文未给熵系数和 S，采用仓库 PPO 已有熵系数 0.01、一次更新轮次。没有给 IRWS 额外交互预算。
6. **精度**：二阶元梯度可能小于 float32 参数分辨率，所有新网络计算使用 float64。未提高论文学习率、改变优化器 epsilon、注入参数扰动或增加损失标签来制造参数变化。

### 动态融合的原文不确定性

式(13)入口节点没有前驱，通常空集合最大值取 0；式(30)因而可能除零。代码保留原始 Rank_dn，并在分母**恰为零**时采用入口 intrinsic-only 权重 1（零工作量同时为零时取 0）。这是未定义边界的工程补全，不能声称原文已明确规定。

式(13)还可能令后继任务的 t(v) 大于 Rank_dn(v)。代码保留原式有限比值，不静默裁成 [0,1]，因此此时是线性外推而非凸组合。短程实际轨迹中，融合系数范围约为 0.1161–2.9680；54 次分配中 8 次超出 [0,1]，26 次触发入口零 Rank 边界。此差异是原式与“权重”文字的真实歧义，需要作者代码/勘误才能确认是否应采取另一种解释，不能擅自反转 Rank 或新增常数。

### 元梯度估计与实际路径

可微虚拟内步为：

`theta_fast = theta - actor_lr * grad_theta(L_PPO(theta; eta, phi))`。

Phi 的外层目标是实际 Makespan–Energy 成本到达回报的策略似然比估计，eta 的外层目标是更新后策略的融合 PPO surrogate；两者均经 theta_fast 反传，保留 `create_graph=True`。外层状态/advantage 在适当位置 stop-gradient，使被检验的链路经过虚拟学习更新，而不是用直接监督损失旁路训练。

这里对无穷时间目标采用已观察有限 rollout 的折扣成本；未观察的外层未来成本作截断，不伪造完成时间或测试反馈。实际 Actor/Critic 及两个元优化器仍按 Table III 用 Adam 更新。**一步虚拟 SGD、有限轨迹、截断 RNN、联合 task/VM likelihood 和原式未规定的池化等均属于工程近似，不是“公式(34)–(39)逐式完全一致”。** 其优点是链路可微且可检查，不构成新的算法创新声明。

连续 task 与 VM 决策的行为概率为 `P(task|Ready,DAG) × P(VM|task,resources)`，两个 log-prob 都保存并在更新时重算。Rollout 非终止边界提前固定实际下一任务与 VM 决策，使用更新前 Critic 的真实后继值；被固定的行为 log-prob 保留，不在更新后伪造为新策略概率。

在当前无抢占、固定 VM 速率的仿真中，环境返回此次提交的确定性执行区间和后续事件反馈。公式(28)的完成差仅为诊断；没有用未来未到达工作流或未来测试结果训练 RNN。

## B. 公共物理环境与公平性

| 项目 | 处理及证据 |
|---|---|
| DAX、MI、文件与到达 | 继承原 `IRWSEnvRouteA`，读取同一公共生成器与实例哈希，不复制物理模拟器 |
| Deadline 与子截止期 | 原 Cache、配对 Alpha、公共 task_subdeadlines 保留；新奖励和策略状态没有增加 Deadline 惩罚 |
| VM/Host 与功率 | 原公共规格、主机功率曲线和积分函数保留；VM 所属 Host 由环境确定，无 Host Agent |
| 通信 | 复用 exec_time_components，同 VM 减免，同 Host 不同 VM 仍传输 |
| Ready 集合 | 新适配层绕过历史候选截断，全部合法 Ready 任务参与一个 Softmax；R_ref 不作容量限制 |
| VM 动作 | 仅真实空闲 VM；没有空闲 VM 时推进事件，绝不把最后一个忙 VM 伪造为合法 |
| 结束条件 | 已到达工作流和实际任务全部完成才结束；短程断言所有任务为 Finished |
| F18 | 正式入口必须读取既有 `f18_shared_v1`。同单位/定义的量用同一尺度；论文 Rank 与旧公共 Rank 定义不同，保留论文定义并按共享 T_ref 的秒尺度缩放，不错误套用旧 Rank P95 |
| 内部目标 | Makespan–Energy；没有复制 MAHDRL Deadline 奖励。伪平均 VM 采用平均 PC/BW，参考能耗在公共主机曲线上做只读反事实增量积分并取均值 |
| 外层长期代价 | 对每工作流完成上包络的增量及全主机能耗增量作论文式(25)任务均值。完成上包络保证并行乱序完成时仍累计为实际 workflow makespan，不引入新 Deadline shaping |
| 最终指标 | 调用原环境 get_episode_metrics，输出 success/violation、lateness/tardiness、makespan、energy 等；与内部 reward 分离，跨种子汇总均值与样本标准差 |

### 固定轨迹核验

新适配层对比原 IRWS 公共模拟器的合成分叉/汇合流程逐步一致。另用真实 DAX、相同 seed=32 的两个工作流，与 MAHDRL 公共环境重放同样的 54 次 task–VM 分配：

- 执行/通信各分量最大差为 0；任务开始和结束一致。
- 工作流绝对 Deadline、任务子 Deadline 一致；Host 负载与 VM 可用时刻一致。
- 逐时刻公共积分窗口能耗和最终总能耗最大差约 `1.46e-11 J`，属于浮点加法顺序误差。

IRWS 的内部 reservation 能量账本与 MAHDRL 按当前时刻结算的中间账本不同，因此比较中间能耗时使用双方同一已发生时间窗口的公共积分，不能把未来已预约能量误当当前已消耗能量。最终实际能耗一致。

## 公共实验协议与历史兼容

新增 `configs/paired_cloud_protocol_v1.json` 与 `common/experiment_protocol.py`：

- Train：32–631，600 个种子；默认预算仍 600 Episode，每回合 50 工作流。
- Validation：632，一个种子，保持原有每轮单种子验证的开销。
- Test：2–31，保留原最终测试种子，不修改测试样本。
- 组间严格无交集；配置中的训练种子显式列出。选模仍按最小验证总能耗，不增加 Deadline 联合分数。
- F18 正式校准示例 101–130 属于该训练分区，不接触验证/测试；这些尺度仍须统一冻结。

MAHDRL、MARL、PD3QN 及其四个既有消融入口只新增了**显式协议的首训练种子和验证种子读取**，网络、奖励、优化器、任务/资源动作均未改。未运行这些消融实验。原六个 IRWS 文件一字未改，新正式 IRWS 使用新目录。

历史入口不传 `--experiment-protocol` 时仍按旧 seed=1 起训/验证1 运行，保留复现行为，但这种历史重叠协议**不能与新 IRWS 混合报告为无泄漏正式比较**。正式配对重跑必须为所有算法传同一协议。新协议模型/日志名增加协议哈希并正确显示 seed32；显式加载错协议 Checkpoint 也会被拒绝。旧 F18 模型不带协议字段时，历史模式仍可读。

新 paper Checkpoint 保存 task 模块（含 embedding/Q）、Actor、Critic、Intrinsic、四个优化器、Episode/steps/updates、RNG、原文超参数、环境、源训练归一化与哈希、实验协议、基线 Git Commit、数值精度和 RNN 状态。只在真实回合/rollout 结清后保存；断点恢复从 `final.pth` 继续，恢复优化器和 RNG。旧 IRWS 文件不能作为新模型读取。验证和测试不更新任何网络。

## C. 实际验证结果

专项自动测试 14 项通过，覆盖用户要求的 DAG、Selector、PPO、Intrinsic、元梯度、融合、候选、物理环境、数据隔离和 Checkpoint 等 10 类检查；额外覆盖全部 15 个真实 DAX 模板的不同任务数、双方向拓扑影响、元梯度有限差分、共享协议穿透既有入口等。完整回归 **132 项通过，0 失败，0 错误**，保留此前 F01/F02/F03/F04/F05/F06/F07/F08/F11/F15/F17/F18 的原测试。详见 `IRWS_paper_tests.txt`。

真实 DAX 短程集成：默认 Table III 网络、学习率、gamma、batch 和轨迹参数均未缩小；只将正式 600 Episode/50 工作流缩为验证用 **1 Episode/2 工作流**。使用训练 seed32、验证632、测试2；校准101在验证协议保留的训练分区内。正式缓存与模型只在临时目录生成和运行，没有覆盖历史结果。实际 54 次分配，完成 Episode、验证、Checkpoint 重载、确定性测试；五组参数最大变化：

| 参数组 | 最大绝对变化 |
|---|---:|
| DAG embedding | 9.5345e-7 |
| Task Selector Q | 2.5358e-9 |
| PPO Actor | 1.0000e-5 |
| PPO Critic | 1.0000e-4 |
| Intrinsic RNN | 9.9977e-6 |

所有损失/梯度有限，Phi/eta 链路非零。Selector 的二阶信号较弱，非零不代表已经学到有用排序或达到论文性能。有限差分单元测试为数值可分辨性单独放大虚拟步长，实际上述短程流程使用原文 1e-5，无参数扰动。完整记录见 `IRWS_paper_smoke.json`，包含融合分布、原始损失、梯度、物理核验和配置。

未执行：600 回合正式训练、正式跨种子性能比较、任何 IRWS 消融或超参数搜索。没有用短程能耗或单测试种子结果证明性能优越。官方实现/勘误未提供，原文歧义不能据这些测试消除。

## D. Windows PowerShell 运行说明

```powershell
# 代码/数学及物理正确性验证；不启动正式训练
.\run.cmd common/test_irws_paper.py
.\run.cmd common/smoke_irws_paper.py --out docs/IRWS_paper_smoke.json
.\.venv\Scripts\python.exe -m unittest discover -s common -p "test_*.py" -v

# 配置预览（已实际验证可运行）
.\run.cmd algorithms/irws_paper/train.py --task S --resource S --deadline Tight --dry-run
.\run.cmd algorithms/irws_paper/evaluate.py --task M --resource S --train-task S --deadline Tight --dry-run

# 正式训练模板：仅给出，没有自动执行
.\run.cmd algorithms/irws_paper/train.py --task S --resource S --deadline Tight --episodes 600 --experiment-protocol configs/paired_cloud_protocol_v1.json --normalization-config outputs/f18_normalization_calibration/f18_shared_normalization_v1.json

# 从最后完整回合恢复；路径替换为该 run 的真实 final.pth
.\run.cmd algorithms/irws_paper/train.py --task S --resource S --deadline Tight --episodes 600 --experiment-protocol configs/paired_cloud_protocol_v1.json --normalization-config outputs/f18_normalization_calibration/f18_shared_normalization_v1.json --resume-checkpoint PATH_TO_FINAL.pth

# 正式评测模板；默认读取新目录下按验证能耗选出的 best.pth
.\run.cmd algorithms/irws_paper/evaluate.py --task S --resource S --deadline Tight --seeds 2-31 --experiment-protocol configs/paired_cloud_protocol_v1.json --normalization-config outputs/f18_normalization_calibration/f18_shared_normalization_v1.json

# 跨任务规模泛化仍读取源 S 训练尺度与模型，不拟合 M 测试尺度
.\run.cmd algorithms/irws_paper/evaluate.py --task M --resource S --train-task S --deadline Tight --seeds 2-31 --experiment-protocol configs/paired_cloud_protocol_v1.json --normalization-config outputs/f18_normalization_calibration/f18_shared_normalization_v1.json

# 其他正式算法也必须启用同一协议，例如：
.\run.cmd algorithms/mahdrl/train.py --task S --resource S --deadline Tight --episodes 600 --experiment-protocol configs/paired_cloud_protocol_v1.json --normalization-config outputs/f18_normalization_calibration/f18_shared_normalization_v1.json
.\run.cmd algorithms/marl/train.py --task S --resource S --deadline Tight --episodes 600 --experiment-protocol configs/paired_cloud_protocol_v1.json --normalization-config outputs/f18_normalization_calibration/f18_shared_normalization_v1.json
.\run.cmd algorithms/pd3qn/train.py --task S --resource S --deadline Tight --episodes 600 --experiment-protocol configs/paired_cloud_protocol_v1.json --normalization-config outputs/f18_normalization_calibration/f18_shared_normalization_v1.json
```

资源可选 S/M/L、Deadline 可选 Loose/Medium/Tight，组成 SS/SM/SL 九场景。不同资源规模分别训练各自输入/输出维度，候选 mask 可变不等于一个固定全连接网络可直接跨 VM 数量迁移。

当前工作区仍缺少 F07 新版正式 Deadline Cache 和有效的完整 F18 九场景配置。请先按 `F18_shared_state_normalization.md` 准备三组 S 任务/资源缓存并运行九场景校准，不能把本轮临时 smoke 数据当正式配置。更大目标任务规模还需对应目标场景的新版物理缓存。程序缺配置时明确报错，无静默 Legacy 回退。

新模型位于 `outputs/ckpts/irws_paper_cloud_v1/<source_scene>_<protocol_hash>_<normalization_hash>/`，日志在该 run 的 `training.jsonl`；评测结果位于对应 `outputs/run/irws_paper_cloud_v1/.../<target_scene>/`。独立于旧 `algorithms/irws` 的模型、日志和指标目录。

## E. 修改范围与最终状态

新增：`algorithms/irws_paper/{__init__,config,environment,networks,learner,runner,train,evaluate}.py`；`common/experiment_protocol.py`、`common/test_irws_paper.py`、`common/smoke_irws_paper.py`；公共协议 JSON；本报告与测试/短程记录。

修改：`common/scenarios.py`（可选显式协议、源模型路径标识和新入口参数）；`common/f18_state_normalization.py`（仅增加可选实验协议 Checkpoint 指纹，历史缺字段仍兼容）；七个既有训练入口的首训练/验证种子读取；README 与临时文件 ignore；旧审计文档补充状态指向。

未改：旧 IRWS 六个源文件，所有网络结构与训练目标的旧定义，公共 workflow/data/resource/Deadline/通信/能耗文件，MAHDRL F15/F17 排序与权重转移，原历史 Checkpoint、结果及四个无关 Word 文档。本轮未提交或推送。

正式论文应标注“IRWS adapted to the common cloud workflow environment”，说明原方法优化 Makespan–Energy、并非专门针对严格 Deadline；披露统一观测预处理和工程近似。**可以确认核心机制与可运行闭环已经实现并验证；不能声称消除了原文歧义、与作者代码逐项完全一致或已经取得正式 IRWS 对比结果。**
补充数值/结构说明：Actor、Critic 和任务打分 Q 的隐层使用 Tanh（原文未指定，固定工程选择）；Q 为 5→5→1。新版本所有参数使用 float64，正式验收仅在 CPU 上进行，未验证 CUDA 后端二阶导兼容性。保存的 `paper_config` 记录原文数值及已参数化的工程设置，Checkpoint 另记 `numerical_precision`，未据验证/测试性能搜索这些设置。

最终保护性检查：旧 IRWS 六个文件、MAHDRL/FCFS/MARL/PD3QN 物理环境、公共 TA 权重、workflow/data/resource/deadline 文件及现有 D3QN/PER/MARL 网络实现，共 18 个关键文件与基线内容一致。`git diff --check` 和新增模块编译通过；最终公共协议单种子验证的配置穿透检查另外通过。