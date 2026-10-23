# PD3QN-DAG：论文学习步骤与最小云工作流适配

更新：2026-10-24。该云 DAG 工作流适配版作为独立对比算法命名为 **PD3QN-DAG**，算法ID为 `pd3qn_dag`，模型版本为 `pd3qn_dag_v1`。独立入口为 `algorithms/pd3qn_dag/train.py` 和 `evaluate.py`，复用 `algorithms/pd3qn` 的计算实现。单位版本仍为 `pd3qn_units_v1`，观测布局仍为 `task6_vm4_parentdata_host2_v3`。数学计算沿用此前v5：式(29)的target最大值、Host历史窗口字段、五层共享网络、论文学习参数及best选模均保留。新旧输出目录、文件和模型标记分离，本次不修改已有结果。

原文：Chi 等，*Task Offloading via Prioritized Experience-Based Double Dueling DQN in Edge-Assisted IIoT*，IEEE TMC 23(12)，14575–14591，2024，DOI：10.1109/TMC.2024.3452502。[此前核查报告](PD3QN_原文算法与参数差异核查_2026-10-23.md)记录的是修改前实现，本文件说明修改后状态。

## 恢复的论文步骤

| 内容 | 新默认行为 | 依据 |
|---|---|---|
| 损失 | 无权 MSE，不使用 Huber | p.14583 式(30) |
| PER | 按 `(|TD|+1e-6)^alpha` 比例抽样，学习后更新优先级 | p.14584 式(33)–(35) |
| 新经验入池 | 用自身当前 TD error 计算优先级，不继承最大优先级 | Algorithm 1 第8–12行 |
| IS 校正 | 不计算 IS 权重，不做 beta 退火 | 原文未记载，省去额外机制 |
| 梯度裁剪 | 不裁剪，保留有限值检查与只读诊断 | 原文未记载裁剪 |
| 目标网络 | 按回合内决策步每60步完整复制，不使用 Polyak | Algorithm 1 第23–24行，p.14587 选择60 |
| TD 目标 | 非终止时 `r+gamma*max_legal Q_target(s',a)` | p.14583 式(29)、p.14585 步骤说明；终止时仅 r |
| epsilon | 原有线性衰减按实际训练决策推进；验证不推进 | 式(31)–(32)及逐决策流程 |
| 回放预热 | 经验达到一个完整 batch（128条）后更新 | 论文未交代不足 batch 的边界，采用最小等待条件 |
| 学习率、gamma | 0.01、0.95 | p.14585；学习率不再与 MAHDRL 的0.0003强行统一 |
| Batch、回放容量 | 128、10000 | p.14585 |
| 评测模型 | `best_pd3qn_dag.pth`，验证总能耗最低模型 | 按用户要求与 MAHDRL/F22 一致；同时保留 final 文件 |

Algorithm 1 的 step 从0开始，同步条件为 `step % 60 == 0`，即回合内第1、61、121、181……次决策的当次学习之后。未凑满 batch 时不更新网络。每回合调用 `begin_episode()` 重置局部时钟；全局 epsilon 决策计数不重置。

训练、评测和 PD3QN 生产尺寸探针共用 `algorithms/pd3qn/config.py::agent_kwargs`，避免分别硬编码。

## 未披露细节的最小补全

- 图2画一个共享 Hidden Layer，未公布宽度。按用户授权保留五个共享隐层 `(1024,1024,512,512,256)`，与 MAHDRL HA/VA 的共享层相同；每个 V/A 分支为一个256维隐层再输出。共享网络既扩大又加深，这是明确的架构调整，不能称为图2逐层复现。
- 原文只写 Optimizer，沿用现有 Adam。
- PER 指数原文仅定义范围，沿用0.6；正数 epsilon 取1e-6。
- 原文未给探索率端点及具体递减量，沿用1到0.05、80000个训练决策的固定线性规则。
- v5 按式(29)及 p.14585 可执行步骤，使用 target 网络自行选择下一状态最大合法 Q；当前执行动作仍由 online 的 epsilon-greedy 选择。式(28)描述当前状态的动作，不能据此推导 online 在下一状态选 TD 动作。论文名称和“分离选择与估值”的概念与具体式(29)存在矛盾；v5 优先遵循公式及步骤，不把它称为标准 Double 目标。
- 式(30)的网络下标写 T，但正文和 Algorithm 1 第22行要求更新 E-NET；当前仍对 online 的 Q(s,a) 做无权 MSE，目标无梯度。这是使更新主体与步骤相符的解释。

v5 目标为 `y_29=r+(1-d)*gamma*max_A Q_T(s',a)`。终止或空后继动作集先把未来项设为0，再形成目标，避免 `0*(-inf)`。新经验入池、batch 更新和 PER 优先级均通过同一目标函数计算。

历史 v1–v4 采用标准 Double：online 在 s' 选动作，target 估值。online 为 `[5,2]`、target 为 `[1,10]`、r=-2、gamma=0.95 时，v5 为7.5，历史 Double 为-1.05。原版 MAHDRL 仍采用标准 Double，本轮没有更改其他算法。PD3QN 的显式历史 shared/legacy 模式也保留原有 Double 行为。

## 固定输入单位

新模式不读取校准文件，不运行参考策略采集 P95/P99，不使用在线统计、整体 L2 或 Ready 比例截断。原观测前缀保持原顺序并做固定线性换算，依次追加每 VM 父数据量与每 Host 历史窗口字段：

| 字段 | 编码 |
|---|---|
| 当前时间、任务 Slack、VM 剩余占用时间 | 秒 / 1000 |
| 任务工作量 | MI / 100 |
| 输入、输出量 | bits / (8×1024³)，即 GiB |
| VM 算力 | 保持 MI/s 原值 |
| VM 带宽 | Mb/s / 1000，即 Gb/s |
| Ready 数 | 保留原计数 |
| 各 VM 上的父任务输入数据量 | 对当前任务的 `parent_in_bits` 按已完成父任务的 VM 位置求和，bits / (8×1024³) |
| 每 Host 的使用标记 | 对应能耗账本非空时为1，否则为0；零服务长度/零负载记录不会开启能耗窗口 |
| 每 Host 的历史空闲间隔 | `max(0,now-window_end)/1000`；从未使用时0，仍有运行任务时0 |

负 Slack 保留，数值允许大于1。所有场景使用相同常数，不按测试数据拟合。常数及版本存入 checkpoint，防止单位误用。公共 F18/F19 实现及其他算法默认值保持兼容。

布局为 `原任务6维 + VM三字段(3*N_VM) + 父数据(N_VM) + Host历史(2*N_Host)`，即 `6+4*N_VM+2*N_Host`。SS 场景25台 VM、3台 Host，v4的106维变为112维。Host 按资源创建顺序排列，每个 Host 的两个字段依次为 used、idle_gap。父数据原位置保持不变；无父任务或终止时父数据为0，终止 Host 字段保留真实历史，bootstrap被关闭。

父数据 L_j 与总输入 I 可恢复候选 VM j 的有效传输输入 `I-L_j`，对应同 VM 免传输规则。父位置来自执行所用的同一个映射，不跨工作流混用局部任务编号。Host 的 window_end 直接读取 `_host_segments` 已有有效服务窗口的最晚端点，不采集统计量，不更改账本。没有加入候选成本估计、动作筛选、额外DAG特征或图网络。

已复现的历史混叠案例：t=100 同一新任务到达，Host0前次窗口结束于1或90秒，v4前缀完全相同，同动作新增能耗分别1619.2和204.1 J；v5的 Host0 idle_gap 分别0.099和0.010，能区分两历史。此补充只修复已确认的即时能耗信息缺口，不代表完整DAG过程已完全可观测。其他 Ready 任务属性和完整后继结构仍未输入，作为压缩观测的边界保留。

## 问题适配边界

原文研究 IIoT 原子任务在本地 ED 和边缘 MS 间卸载，本项目研究云 DAG 工作流，不能宣称相同 MDP。本次保留既有 FCFS/ERTF、空闲 VM 合法动作、DAG 完成事件、真实后继 Bootstrap、公共子截止期、父子通信及 Host 任务窗口能耗。

既有公共云奖励保持：

```text
r = -0.75 * clip(max(0, finish-task_subdeadline)/300, 0, 1)
    -0.00025 * delta_host_window_energy
```

它不是原文 ED 时间—能耗归一化成本，而是当前实验问题的适配目标。没有虚构 ED 电量、本地计算或无线位置，也没有只给 PD3QN 换物理问题。准确称呼为“按论文恢复学习步骤的最小云工作流适配版”，不是原始 IIoT 系统逐式复现。

F22 训练0–599、验证600、测试700–729的划分保留。每回合按相同验证集评估，以最低验证总能耗保存 best；正式评测只加载 best，不按测试集选模。元数据为 `checkpoint_selection=minimum_validation_total_energy`，与 MAHDRL 一致。

## 公共实验参数与算法学习参数

| 类型 | 当前约定 |
|---|---|
| 公共任务与资源 | 相同场景选项、DAX任务/输入输出数据、到达实例、每回合50个工作流、Host/VM资源及算力/带宽档位 |
| 公共物理目标 | 相同父子通信规则、Host任务窗口能耗、公共截止期缓存及 Tight/Medium/Loose 混合概率 |
| 公共奖励系数 | 延迟0.75、能耗0.25、能耗缩放0.001、延迟分母300秒 |
| 公共训练与评测 | 默认600回合、相同训练/验证/测试实例划分和模型种子控制、验证最低总能耗选best、统一物理指标评测 |
| PD3QN 论文学习参数 | lr=0.01、gamma=0.95、batch=128、回放容量10000、硬同步步长60、无权MSE及PER流程 |
| 论文未披露参数 | 沿用已记录的Adam/ReLU、PER alpha=0.6、epsilon端点与衰减安排；网络宽度使用用户指定值，不称为论文原值 |

公共设置统一不表示每层逐步回报完全相同。MAHDRL 的区间/阶段结算、MARL 的全局/局部奖励、IRWS 的内在融合属于现有算法结构，本次没有重写。具体差异见 [奖励设置核查](Algorithm_reward_settings_2026-10-23.md)。PD3QN 的固定单位观测保留，v5目标优先按式(29)执行；五层网络属于用户授权的架构例外。

## 与原版 MAHDRL 的参数、网络及奖励比较

这里比较 `algorithms/mahdrl`，不包括 OE/GE/PF 等变式。公共实验设置对齐；PD3QN 学习率已按最新要求恢复论文0.01，选模、父位置观测和指定共享隐层宽度保留。

| 项目 | PD3QN-DAG | MAHDRL |
|---|---|---|
| 学习率 | 0.01，论文值 | HA/VA/TA 均0.0003 |
| 优化器、激活 | Adam、ReLU | Adam、ReLU |
| gamma | 0.95 | 三层均0.95 |
| 共享隐层 | `(1024,1024,512,512,256)` | HA/VA 相同；TA `(512,512,256,128)` |
| V/A 分支 | 各一个256维隐层后输出 | HA/VA 各两个256维隐层；TA 各两个128维隐层后输出 |
| Batch、回放容量 | 128、10000 | 三层batch256；HA/VA容量120000，TA容量60000 |
| PER | alpha=0.6，无 IS | 不使用 PER，均匀回放 |
| epsilon 起止 | 1到0.05 | 三层均1到0.05 |
| epsilon 衰减时钟 | 80000次训练决策 | HA/VA 80000次成功更新；TA 40000次成功更新 |
| 目标网络 | 回合内每60个决策步硬复制 | HA/VA tau=0.005，TA tau=0.01 的软更新 |
| TD 目标 | 式(29)：target 取最大合法值 | 标准Double：online选下一动作，target估值 |
| 损失、裁剪 | MSE、不裁剪 | Huber、梯度范数上限10 |

SS 场景在线网络参数量如下，不计各自同尺寸的 target 网络。输入/输出维度来自当前场景，其他资源规模会改变这些数字。

| 网络 | 输入/动作数 | 在线参数量 |
|---|---|---:|
| PD3QN | 112/25 | 2,222,362 |
| MAHDRL HA | 37/3 | 2,271,492 |
| MAHDRL VA | 72/9 | 2,308,874 |
| MAHDRL TA | 15/243 | 532,596 |
| MAHDRL 三层合计 | — | 5,112,962 |

共享隐层宽度已相同，但分支层数、输入/输出维度及网络数量仍不同，参数量不完全相同。宽度、Adam、ReLU、PER 指数和探索率细节均不是论文已公布的具体参数；其中 Adam/ReLU 与 MAHDRL 相同，PER 不同，epsilon 端点相同但衰减时钟不同。

奖励权重和单位相同：延迟0.75、能耗0.25、能耗缩放0.001、延迟分母300秒。令 `D_i=clip(max(0,C_i-d_i)/300,0,1)`：

- PD3QN：`-0.25*0.001*本次分配带来的Host窗口总能耗增量 -0.75*D_i`，包括窗口延伸的空闲能耗。
- MAHDRL HA：按实际时间推进区间的全局能耗结算，延迟使用相同 `D_i`。
- MAHDRL VA：按实际时间推进区间的所选 Host 能耗结算，延迟使用相同 `D_i`。
- MAHDRL TA：按整个阶段能耗与阶段内新完成任务的超期总和/300结算，延迟和不截断到1。

因此奖励函数并非逐步完全一致。同一时刻连续分配时 MAHDRL 下层可能先收到0能耗项、在时间推进时集中结算；PD3QN 按每次分配的窗口能耗增量结算。即使物理能耗口径相同，结算时点、范围及折扣回报仍可能不同。本次未修改奖励函数。

## 使用及历史兼容

新默认无需归一化校准，但仍需公共物理模型的截止期缓存：

```powershell
# 预览新默认参数及独立模型目录
.\run.cmd algorithms/pd3qn_dag/train.py --task S --resource S --deadline Tight --dry-run
# 新版训练不带旧的 --normalization-mode f19_shared_v1 参数
.\run.cmd algorithms/pd3qn_dag/train.py --task S --resource S --deadline Tight --episodes 600
# 加载新目录中的验证最优模型
.\run.cmd algorithms/pd3qn_dag/evaluate.py --task S --resource S --deadline Tight --seeds 700-729
```

新模型/日志目录以 `pd3qn_dag_` 开头，checkpoint为 `best_pd3qn_dag.pth`、`pd3qn_dag_final.pth`，训练CSV为 `train_metrics_pd3qn_dag_cost.csv`。评测集中在 `outputs/run/pd3qn_dag/<版本>/<目标场景_源训练场景_模型种子>/`，主CSV以 `eval_pd3qn_dag_` 开头并包含场景、模型种子和测试种子，逐种子明细放在同目录的 `measurements` 下。测量 algorithm 为 `pd3qn_dag`，诊断 learner 为 `PD3QN-DAG`。公共格式的JSON/逐工作流CSV保留通用文件名，目录与内容均标识新算法。

评测版本为 `pd3qn_dag_v1_task6_vm4_parentdata_host2_v3_...`，独立于所有旧PD3QN及v5结果。已有旧PD3QN文件不会使新算法首次运行报错，也不会被覆盖；不自动搜索或加载旧模型。即使显式指定旧目录，也必须存在 `best_pd3qn_dag.pth` 且模型实现标记为 `pd3qn_dag_v1`，旧v5权重不会被冒充为新算法，需在独立目录重新训练。

独立 `pd3qn_dag` 入口只允许新版观测模式。旧 `algorithms/pd3qn_v5` 入口作为别名指向新算法，不再生成v5文件；原 `algorithms/pd3qn` 入口默认也生成PD3QN-DAG。显式历史shared/legacy模式仍走旧学习实现与旧命名。

重复训练同一场景/模型种子且新目录非空时，在写入前拒绝。重复评测在主CSV或所选测试种子的任一测量明细已存在时，于加载模型和打开CSV前拒绝；底层测量器也拒绝覆盖，避免CSV追加与明细重写混杂。模型不存在时不创建空CSV。不同模型种子或不同源/目标训练场景使用不同目录；部分测试集合重叠时同种子的既有明细仍受保护。公共deadline缓存是只读共享输入，不属于算法输出。

为保留旧实验可读性，显式选择 `--normalization-mode f19_shared_v1`、`f18_shared_v1` 或 `legacy`，会同时选择匹配的**历史学习实现**，沿用旧模型目录和 `best_pd3qn.pth`。它们是历史兼容入口，不是新版算法的归一化消融开关。例如：

```powershell
.\run.cmd algorithms/pd3qn/evaluate.py --task S --resource S --deadline Tight --seeds 700-729 --normalization-mode f19_shared_v1
```

旧 checkpoint 缺少新增版本/网络元数据时，仍可由历史模式加载。新旧算法交叉加载会在修改权重前拒绝，网络维度、输入单位、观测布局及能耗版本也继续检查。

## 验证

- 当前PD3QN-DAG：[121项回归](PD3QN_DAG_tests.txt)全部通过，其中24项聚焦论文目标、观测与新旧隔离、已有训练/评测防覆盖、禁止旧名回退。新train/eval独立脚本dry-run均通过。
- [PD3QN-DAG闭环](PD3QN_DAG_smoke.json)：2回合各50工作流训练，2735条经验、2608次更新；新名best正式加载、评测CSV与新标识生成、重复评测拒绝均通过，权重和训练时钟在评测中保持不变。loss、评测能耗和回报与此前v5烟测相同；算法计算未变。
- 当前 `verify_package.py --imports-only`：200个模块导入，0失败；`git diff --check` 通过。
- [聚焦及既有回归输出](PD3QN_paper_tests.txt)：116项全部通过，其中19项聚焦PD3QN，覆盖式(29)/历史Double区分、非法动作与终止空掩码、入池/训练/PER同一目标、Host历史反例/未使用/运行/共享窗口/重置、父位置、物理等价、best加载及旧版本拒绝。
- 历史命名调整证据：[v5的36项测试](PD3QN_v5_naming_tests.txt)与[v5闭环](PD3QN_v5_naming_smoke.json)原样保留，不能当作当前独立命名与覆盖保护的验证。
- 历史v5命名时的导入检查：184个模块导入、0失败。
- [v5 真实闭环烟测](PD3QN_paper_v5_smoke.json)：默认五层网络、学习率0.01、输入112维、在线参数2,222,362。在临时目录完成2个50工作流训练回合，共2735条经验和2608次更新；保存best及final，经正式评测加载器恢复best，评测seed700的1370个任务，能耗重算差5.24e-10 J，评测不改变权重或训练时钟。末次MSE为0.0232577510；best恰为最后一回合（episode=1），另有加载路径测试约束必须读取best。该检查验证运行流程和有限性，不证明长期收敛或性能优劣。
- [历史 v4 大网络检查](PD3QN_network_v4_check.json)：学习率0.01、输入106维、一次更新和保存恢复，保留历史证据；不是当前 v5 目标和观测的验证。此前 [v3 检查](PD3QN_network_v3_check.json) 使用0.0003。
- [历史 v2 闭环烟测](PD3QN_paper_smoke.json)：此前128维共享层结果，不代表当前v5网络。历史记录原样保留。
- [历史分支一致性](PD3QN_legacy_equivalence.json)：与修改前 Git HEAD 中 agent 固定种子对照，8次决策、7次更新的动作、损失、优先级及 online/target 参数逐项完全相同。

```powershell
.\.venv\Scripts\python.exe -m unittest common.test_pd3qn_paper common.test_pd3qn common.test_f11 common.test_f18 common.test_f19 common.test_scenarios common.test_paired_instances common.test_task_deadlines common.test_f22 common.test_communication common.test_energy -v
.\run.cmd common/smoke_pd3qn_paper.py --out docs/PD3QN_DAG_smoke.json
```

未启动正式600回合训练，未按短程结果调参；烟测只验证运行与有限性，不证明长期稳定或性能优劣。
