# 2SD-GAT 公共问题适配实现

算法 ID：`two_stage_gat`，论文展示名：**2SD-GAT-Adapted**。实现位于 `algorithms/two_stage_gat/`。原版 MAHDRL、已有 DRL-GNN/IRWS 及其输出独立保留。

## 固定来源与范围

来源为 [2sd-gat-cloud-scheduler](https://github.com/kdsuneraavinash/2sd-gat-cloud-scheduler/tree/aae1397f11e8e75d5dcb70c9b38608dcf05ef773)，固定提交 `aae1397f11e8e75d5dcb70c9b38608dcf05ef773`。本版本保留两阶段任务选择/全局 VM 选择、GAT、任务相关 VM 编码和联合 PPO；没有 Host Actor、排序权重学习、内在奖励、偏好搜索或额外状态字段。

它是当前能耗/截止期问题的适配基线，不是原三目标 Pareto 实验或原文数值结果的完整复现。当前目录中的 `drl_gnn` 是另一条 GIN/单 pair 策略，不能代替本方法。

## 调度与物理规则

- 图只包含已到达工作流。已到达 DAG 中尚未 Ready 的后继可以参与编码，未来到达列表、缓存中后续工作流不能进入策略。
- Task Actor 选择真实 Ready 任务；Ready 仍要求所有父任务**实际完成**。这是论文/公共在线环境语义，区别于作者源码在父任务已分配时就释放子任务。
- VM Actor 可以选择全部真实兼容 VM，**包括当前忙碌 VM**。采用队尾追加预留：`start=max(dispatch_time, vm_tail, parent_finish)`，再将执行时长加到 VM 队尾。没有插队、抢占或迁移。
- 任务保留 dispatch/start/end；未来开始的任务状态为 `Queued`，实际开始后变为 `Running`，结束后变为 `Finished`。无 Ready 时按到达、预留开始、任务结束事件推进，不会跳过中途到达。
- 只有一个 `(task, VM)` 物理分配和一个 reward。原工作流依赖之外，加入同 VM 前一分配任务到当前任务的顺序边。
- 执行调用公共 `exec_time_components`，包括输入、计算、输出。只有同 VM 父子数据免除对应网络时间，同 Host 不同 VM 仍传输。
- Host 负载和增量能耗复用 `IRWSEnvRouteA` 的公共积分实现。奖励结算已经承诺的任务区间，Episode 终止必须等所有预留实际完成。总能耗采用 `host_task_window_v1`，主机窗口内部空闲计入，未用主机和窗口外不计。

MAHDRL 保留空闲 VM 即时分配，2SD-GAT 保留未来预留；这属于方法调度机制的差异，不能宣称动作权限完全一致。

## 6 维任务和 4 维任务–VM 状态

| 任务槽位 | 定义 |
|---|---|
| 1 | 是否真实 Ready |
| 2 | 是否已经分配，包括 Queued/Running/Finished |
| 3 | 预计完成时刻 / 源场景 T_ref；已分配使用已知完成时刻 |
| 4 | 未分配时各兼容 VM 主动能耗代理的最小值，已分配取所选 Host 代理，乘 0.001 |
| 5 | **带符号预计 Slack** `(task_subdeadline - estimated_finish) / T_ref` |
| 6 | 共同问题没有优先等级，真实任务统一为 1 |

| 任务–VM 槽位 | 定义 |
|---|---|
| 1 | VM 已排定队尾完成时刻 / T_ref |
| 2 | 兼容性，当前没有内存/核数限制，真实 VM 为 1 |
| 3 | 公共完整执行时长 / T_ref |
| 4 | 原主动能耗代理乘 0.001 |

主动能耗代理为 `MI_i * (P_host(1)-P_host(0)) / host_total_PC`。它保留作者近似观测形式，不升级为包含当前负载、窗口内部空闲的精确边际能耗。实际奖励/指标始终用公共真实积分。

未分配任务的预计完成时刻，按工作流拓扑顺序，在**所有兼容 VM（包括忙碌 VM）**上取最早估计值；输入未知父任务位置时不假定同 VM。预计开始不早于当前在线时钟。辅助预测不声称严格下界。

F5 替换、优先级统一、时间/能耗输入缩放均为适配改动。F5 保留正负值，避免所有预计未超期任务都压为 0。没有添加 Ready 等待时间、剩余工作流截止期、Host 当前负载、Rank、PC/BW 等新输入。测试和泛化使用源训练场景冻结尺度，不重新拟合。

## 网络和 PPO

| 项目 | 固定设置 |
|---|---|
| Task GAT | 6→64→64→8，3 层、4 heads、concat=False、自环、ReLU、dropout=0、无 norm/residual |
| VM encoder | 4→64→64→64→8 |
| Task/VM decoder | 各自 24→64→64→64→1 |
| Critic | 独立 GAT/VM encoder，16→64→64→64→1 |
| 共享 | 两阶段共用 Actor 编码器，两个 decoder 独立；Critic 与 Actor 不共享参数 |
| 池化 | 真实已到达任务/真实 VM 的 Mean Pooling；批状态之间不混合上下文 |
| 初始化/精度 | PyTorch/PyG 默认初始化、网络 float32，公共物理计算保留原精度 |
| 优化器 | 单 Adam，lr=2.5e-4，eps=1e-5，默认 betas |
| 折扣/GAE | gamma=.99，GAE lambda=.95 |
| Rollout | 4 个逻辑环境，各 128 次真实分配，完整批次 512 |
| 更新 | 4 epochs，每 epoch 4 minibatches，完整 minibatch=128 |
| PPO | 联合 logprob 相加、一个 ratio/advantage、clip=.2 |
| 熵 | Task 熵＋所选任务对应 VM 熵，系数 .01 |
| Value loss | clipped，范围 .2；内部 .5×MSE，再乘 vf_coef=.5 |
| 其他 | minibatch advantage 标准化，梯度总范数 clip=.5；不做 KL 早停或额外 reward/return 标准化 |

原论文表 6 的 MLP 两层、正文 Decoder 共享与发布代码不一致；本版本采用固定源码实际的三隐藏层、独立 Decoder。GAT heads、联合 logprob 等由源码补全。论文写评测取最高概率，但源码评测采样；本版本遵循论文及公共评测的**两阶段依次 argmax**。

源码固定 1000/50 容量 padding 的直接均值改为真实元素均值，保持论文按真实节点数量定义的池化；稀疏批图排除 padding。源码 Decoder 的跨批池复制索引按所属 graph 修正，确保同状态单独/批量推理一致。这些是明确披露的实现差异，不引入新学习模块。

## 共同目标与实验协议

```
r = -0.25 * 0.001 * delta_energy_global
    -0.75 * clip(max(0, task_finish - task_subdeadline) / 300, 0, 1)
```

不保留原 Makespan/二元优先级 QoS 三目标和估计差分归一化奖励，也不训练 26 个偏好模型。每场景一个共同目标模型。该奖励是截止期约束问题的学习代理，不保证概率约束。

默认实例划分 F22：训练 0–599、验证 600、测试 700–729；每 Episode 50 工作流，到达率 .03，模型 seed=1。训练使用相同数据、VM/Host、alpha、子截止期和通信。600 是四环境合计；环境 r 取 `seeds[r::4]`。没有额外预训练或 200,000 步提前停止。

Rollout 的 GAE 按环境和真实终止边界隔离。末批实际有多少条就更新多少条，不填充重复样本。总更新轮数按每环境总分配量/128 的向上取整最大值确定，学习率按这些更新轮次线性衰减。未来模板计数仅用于预算元数据，不输入策略。

每个 Episode 实际完成即验证当时参数，不等待后续采样/PPO 更新。若恰好到正常完整 Rollout 或全训练预算终点，先进行该次正常更新，再验证；不会为中间 Episode 强制增加更新。最终预算更新后保存 final；没有额外最终选模机会。验证不进入缓存、不更新参数/计数，并恢复训练 RNG。相同参数版本重复验证不会增加模型候选。

相同实例及物理分配预算不等于相同计算量；日志记录真实分配、optimizer step、样本呈现量和同步耗时。原 MAHDRL 的层级信用分配、状态和更新机制保留。其他算法仍使用原最低验证能耗选模；本算法从 2026-10-25 起采用下述独立选模规则，比较时须披露这一差别。

### 2SD-GAT 独立选模 v2（2026-10-25）

依据当前研究论文 `docs/paper.pdf` 第4页式(12)、式(16)及第9页实验设置：工作流按时率门槛 Gamma=0.92，满足 `SR>=Gamma` 的运行概率目标 beta=0.99。beta 不是工作流按时率门槛。

- 每个验证实例根据 `wf_success_count/wf_count` 计算 SR，再计算验证实例中 `SR>=0.92` 的比例（验证 ESP）。验证 ESP>=0.99 的候选视为经验达标。
- 有经验达标候选时，只在达标者中选择平均能耗最低的完整两阶段策略；能耗相同选择 SR 更高者，完全相同保留更早模型。
- 没有达标候选时，依次比较验证 ESP、汇总工作流 SR、平均能耗。仍保存 `best.pth`，但明确记录 `feasible=false`，不宣称已达标。
- 沿用训练 0-599、验证 600、测试 700-729，不增加验证实例。当前验证只有一个50工作流实例，因此经验达标等价于至少46/50按时（SR>=0.92）。单个验证实例不能证明真实满足概率>=0.99，checkpoint 元数据明确记录 `population_probability_guaranteed=false`。
- 本次不改变奖励、PPO参数、状态、网络、未来预留或其他算法。共享 F22 文件继续作为数据/标定绑定，2SD-GAT 实际选择规则通过独立 `checkpoint_selection` 及有效协议元数据覆盖基础选模声明。
- 新模型版本为 `two_stage_gat_source_aae1397_selection_v2`，模型/日志目录增加 `select_sr092_esp099_v2`；旧能耗选模模型不能作为 v2 模型自动加载。原始历史日志不能恢复已覆盖的 checkpoint，需要重新训练产生 v2 候选。
- 日志新增当前验证 SR、ESP、feasible、是否更新 best；checkpoint、训练总结和评测汇总均保存选模政策及所选模型的验证结果。最终测试只做指标评估，不参与选择。
- 本次独立快照回归327项通过（153.712秒），其中包含选模边界、未达标回退、达标后拒绝低能耗不达标候选、checkpoint/元数据与旧模型隔离；记录见 `docs/Two_stage_GAT_selection_tests.txt`。小规模真实DAX、CUDA及源码权重对照见 `docs/Two_stage_GAT_selection_smoke.json`；训练/跨规模评测入口 dry-run 通过。未执行正式600回合训练。

## 运行与输出

先安装共同依赖（新增 PyG 2.6.1，不更换 PyTorch）：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\run.cmd algorithms/two_stage_gat/train.py --task S --resource S --deadline Tight --dry-run
.\run.cmd algorithms/two_stage_gat/train.py --task S --resource S --deadline Tight --episodes 600 --model-seed 1 --device cuda
.\run.cmd algorithms/two_stage_gat/evaluate.py --task S --resource S --deadline Tight --model-seed 1 --device cuda
.\run.cmd algorithms/two_stage_gat/evaluate.py --task M --resource S --deadline Tight --train-task S --device cuda
```

Linux 使用 `python -m algorithms.two_stage_gat.train ...` 和 `python -m algorithms.two_stage_gat.evaluate ...`。正式场景及源模型映射应与本轮 MAHDRL 完全对应，不单独增加训练场景。

模型/日志分别位于 `outputs/ckpts`、`outputs/logs` 下独立 `two_stage_gat_source_aae1397_..._select_sr092_esp099_v2` 目录，包含输入版本、模型 seed、归一化、数据协议哈希、能耗和选模标识。非空训练输出目录拒绝覆盖。本版本不提供续训，best/final 均可独立评测。

评测输出位于 `outputs/run/two_stage_gat_source_aae1397_selection_v2/`，主 CSV/JSON 文件名包含任务、资源、截止期、模型种子、测试实例数量及必要的源场景后缀；测量目录包含逐实例、逐工作流和分配时延。Checkpoint 检查算法来源、状态/调度/奖励版本、选模政策、协议、源场景和冻结尺度，不回退其他方法/旧目标模型。

### 单场景显存测量

报告为 `docs/Two_stage_GAT_memory_profile.json`。本机 RTX4060Ti 8GiB、PyTorch 2.5.1+cu121、PyG 2.6.1，使用真实50工作流的末期图，训练 seed551 的1431个任务节点，分别25/50/75个VM。重复真实状态构造完整128图 minibatch，执行两次前向/反向/Adam更新，以包含优化器状态；只将探针的 epochs/minibatches 设为1，不改变网络宽度，不执行正式600回合训练。

| 当前S任务训练场景 | PyTorch峰值分配 | PyTorch峰值保留 |
|---|---:|---:|
| SS（25VM） | 5717.8 MiB（5.58 GiB） | 6446 MiB（6.29 GiB） |
| SM（50VM） | 5709.7 MiB（5.58 GiB） | 6508 MiB（6.36 GiB） |
| SL（75VM） | 5696.9 MiB（5.56 GiB） | 6538 MiB（6.38 GiB） |

LL评测末期图（seed723，3921节点、10045边、75VM），batch=1、不反向，实测分配峰值58.1MiB、保留峰值206MiB。上述数字不包含CUDA上下文和其他进程，不是服务器RTX5090上完整600回合进程的实测峰值。实际各 minibatch 的节点/边数不同；本探针采用末期大图重复构造负载，通常高于较早期批次，但不保证覆盖所有轨迹的最坏边数。

按当前S规模训练协议，建议单训练进程预留8-10GiB，评测进程按1-2GiB规划（含上下文余量的估算）。显存峰值主要由总图节点/边数及128图 minibatch 决定，不随600回合线性增长；若改为训练M/L规模，不能直接套用S规模训练数值。可在目标服务器复测：

```bash
python -m common.profile_two_stage_gat_memory --out outputs/two_stage_gat_memory_profile.json
```

探针仅输出显存报告、不生成正式训练模型；对该探针进程设置显存分配上限为设备容量的80%，不足时明确报告失败，不把失败峰值当成完整128图成功测量。

## 验证

```powershell
.\.venv\Scripts\python.exe -X utf8 -B -m unittest common.test_two_stage_gat -v
.\.venv\Scripts\python.exe -X utf8 -B -m common.smoke_two_stage_gat --out docs/Two_stage_GAT_smoke.json
.\.venv\Scripts\python.exe -X utf8 -B -m unittest discover -s common -p "test_*.py" -q
```

专项测试覆盖未来预留/父依赖/中途到达、非法动作原子性、执行和能耗守恒、与 MAHDRL 相同空闲分配的物理重放、公共实例一致、全 512 条 Rollout 与末批、联合概率/GAE、拓扑敏感性、真实梯度更新、单状态/批状态一致、验证隔离、保存加载及旧协议拒绝。

独立 smoke 使用真实 DAX、原网络宽度、4 个逻辑环境，仅 1 个训练 Episode、2 个工作流及临时模型目录。另运行 CUDA 小批更新。若 `tmp/two_stage_gat_source/models_gnn_agent.py` 中存在已审核固定源码，先严格检查固定 SHA256，再复制原权重对照单/批网络输出；仅修正已披露的池复制索引，不联网自动执行来源代码。报告记录源码 SHA256 和实际差值；源码不存在时明确记录该项未执行。

这些验证不能替代正式 600 回合训练与 30 实例性能比较。本次没有运行正式性能实验。

## v1 首次复现验收结果（历史记录）

- 新增专项 12 项通过，完整 `common` 测试集 **275 项通过**，完整回归用时 147.429 秒；完整记录为 `docs/Two_stage_GAT_tests.txt`。
- 专项集成使用 8 个真实配对实例、每实例 3 个工作流，共 657 次分配，覆盖完整 512 条 Rollout 和最后不足整批的数据。训练实例恰好 0–7，未增加或重复实例。
- 独立 smoke 使用 1 个训练实例、2 个工作流、原网络宽度，共 59 次分配。强制同 VM 重放出现 58 次未来预留，全部完成；VM 无重叠、派发前父任务已经完成。执行时长最大差 `3.41e-13 s`，能耗重积分差 0，奖励累计守恒差约 `1.42e-14`。
- Actor GAT、VM encoder、两个 Decoder、Critic GAT/VM/value 七组参数均有实际更新，损失和梯度有限。保存加载后确定性测试一致；验证不改变模型、训练计数或 RNG。
- 固定作者权重、相同真实节点/VM 的单/批网络数值对照通过，最大绝对差 `1.1920928955078125e-7`。对照仅修正来源代码中已经披露的批次池索引，不代表整个原问题实验数值复现。
- NVIDIA GeForce RTX 4060 Ti 上的 GAT 前向、反向和 PPO 小批更新通过。运行时保留公共 PyTorch 2.5.1+cu121，PyG 2.6.1。
- 完整回归最初发现已有 frozen-TA 测试导入诊断工具会关闭其他测试的日志；仅在该测试导入处隔离环境变量。IRWS-restored 历史清单中的注册表整文件哈希检查改为物理场景/划分语义检查，以允许独立算法登记；历史来源哈希清单及原算法源码未改写。
- 未执行正式 600 回合训练、正式 30 实例性能比较、额外调参或消融，未提交或推送 Git。
