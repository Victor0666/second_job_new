# F18：固定训练参考尺度与共享状态归一化

基线提交：`21c16d04919df9adb47e5bb2286111322025f2f3`。本次只处理观测预处理及其校准、版本和加载验证，不修改 F15/F17 权重与排序、F11 转移、物理执行、奖励、通信、能耗或截止期生成。没有重新核查原论文 PDF；英文修订建议对应用户给定的公式编号。

## 审查结果与修改清单

MAHDRL 的队列分母是理论任务容量，等待/Slack/资源时间以 Horizon 缩放，任务与资源能力使用动态均值；MARL 使用固定 64 及 Horizon；PD3QN 对包含时间、MI、bits、能力的整个向量执行 L2；IRWS 使用 3000、1000、10、10000 等固定尺度。四个消融继承 MAHDRL，其中 nohost 另建全局 VM 特征，需要单独接入。

新增公共校准与归一化模块，修改上述观测构造及八种算法的训练/评测配置，增加 Checkpoint 元数据检查。`horizon`、`max_ready_tasks`、奖励尺度等原参数保留原用途。低层环境默认 `legacy` 保留历史、参考模拟及现有调用行为；所有正式学习训练/评测入口通过 Scenario **明确默认启用 `f18_shared_v1`**。配置缺失、场景未完成校准、版本或单位不符均报错，不静默回退。

## 九场景与参考策略

场景顺序固定：SS_L、SS_M、SS_T、SM_L、SM_M、SM_T、SL_L、SL_M、SL_T；L/M/T 分别为 Loose/Medium/Tight。均使用 Scenario 的 S 任务模板混合、S/M/L 资源、Alpha 概率、每回合 50 工作流、arrival_lambda=0.03，与现有正式入口一致。

预采样直接复用 `environments.mahdrl_fcfs` 的事件与调度流程，以及正式 ERTF-FIFS 的 FCFS 权重、Host 最先空闲选择、VM 最先空闲选择函数。F15 的 FCFS 列随 ReadyTime 递减，同分按 ReadyTime/任务 ID，等价于 Earliest Ready Time First；每次参考分配另断言所选任务等于真实 Ready 集合中 `(ReadyTime, ID)` 最小者。Host 在合法主机中按最早 VM_available_at、动作编号选择，VM 在所选 Host 内按同样规则选择。没有修改正式 ERTF 评测策略，没有加载网络/Checkpoint、训练或反向传播。

在到达/完成事件处理后、参考决策前采样，按不同 `current_time` 去重。同一时刻的多次分配不重复加权。统计非空、可决策的真实 Ready 集合；空/无分配机会的观察次数单独记录。所有任务状态数组只包含已到达任务，按全局 ID 去重；Running 和 Finished 不计入 Ready/Blocked。参考分母不会截断队列。每个任务物理属性仅采集一次，每个工作流的 `deadline-arrival` 仅采集一次。

使用 `numpy.percentile(..., method='linear')`，记录 P50/P95/P99/Max。R_ref=max(1,ceil(Ready P99))；T_ref=相对工作流 Deadline P95。MI、逻辑输入/输出 bits、up/down rank 使用任务等权 P95；VM PC/BW 使用公共资源配置在该源场景中的最大规格，Host PC 使用最大主机总能力。Ready 统计为每个不同决策时刻等权，非时间加权。

Rank 沿用 `Workflow.compute_upward_ranks/compute_downward_ranks` 的平均 PC/BW 处理/通信代价定义，以秒计；MAHDRL/IRWS 已使用这一公共实现。MARL 的 Rank 占位零保持零，PD3QN 不添加 Rank 维度。没有把不同数学定义的预测/Slack 合并。

## 固定尺度与状态映射

共享参数单位：R_ref=tasks，T_ref=s，workload_mi_ref=MI，input_bits_ref/output_bits_ref=bit，up_rank_ref/down_rank_ref=s，vm_pc_ref/host_pc_ref=MI/s，vm_bw_ref=Mb/s。没有将 bits 与 Byte 或 Mb/s 与 bit/s 混用。逻辑输入量保留同 VM 通信减免之前的数据量；实际传输仍由原物理模型决定。

| 字段 | 原始定义及 F18 处理 | 适用输入 |
|---|---|---|
| RR | min(1,真实 Ready 数/R_ref)，无容量截断 | MAHDRL TA/HA/VA、MARL Global/Local、PD3QN queue |
| URR | Blocked/max(1,Unfinished)，无未完成任务时为 0 | MAHDRL、MARL 的现有队列字段 |
| AWT/MWT | Ready 任务等待时长均值/最大值，再除 T_ref | MAHDRL、MARL |
| 工作流 MSL/ASL | Ready 任务的 workflow_deadline-now-mean(duration(vm))，取最小/均值，再除 T_ref | MAHDRL TA/HA/VA、MARL Global/Local |
| Task Waiting | max(0,now-ready_time)/T_ref | MAHDRL、MARL、IRWS |
| OSL | workflow_deadline-now-best_predicted_execution，除 T_ref | MAHDRL、MARL；best 的算法候选范围保留 |
| Task Slack | 任务子截止期-now，除 T_ref | PD3QN、IRWS |
| Estimated Slack | 任务子截止期-predicted_finish，除 T_ref | IRWS VM |
| Estimated Execution | 原算法预测时长/T_ref | MAHDRL、MARL、IRWS |
| EAL/Availability | max(0,available_at-now)/T_ref | Host/VM 状态 |
| Predicted Finish Duration | max(0,predicted_finish-now)/T_ref，修正原绝对完成时刻字段的含义 | MAHDRL、MARL、nohost VM |
| Episode progress | 自仿真起点 0 开始的 elapsed time/T_ref | PD3QN 的首字段 |
| MI/Input/Output | 原物理值除对应共享参考 | 各算法已有字段 |
| Up/Down Rank | 各自现有公共 Rank 值除相应共享参考 | MAHDRL、IRWS |
| VM PC/BW、Host PC | 除共享能力尺度 | 各算法已有字段 |
| Ready 总 MI | 总 MI/workload_mi_ref，保留总量含义 | MAHDRL queue |
| 比例、one-hot、拓扑计数、IO/MI | 保留现有含义，不重复缩放 | 各算法独有字段 |

负 Slack 保留；时间与任务量可以超过参考值，不裁剪为 [0,1]。只有 RR 按定义饱和至 1；URR 按真实集合计数自然位于 [0,1]。

状态字段顺序、维度和网络均未改。SS 默认尺寸：TA=15、HA=12+10+5H=37、VA=12+10+5+5maxVM=72；nohost VA=22+8NVM=222；MARL Global=10、Local=12+10+5+5maxVM+H=75；PD3QN=6+3NVM=81；IRWS task=7、VM=7+6NVM=157。资源规模造成的原有维度差异保留，不承诺现有网络可跨资源尺寸直接加载。

PD3QN 的 F18 版本取消整体 L2，按字段分别缩放，属于**统一观测预处理后的实验版本**，不能宣称与原始预处理完全相同。Legacy 模式保留原整体 L2。IRWS 的候选上限与截断顺序保持原样，不能将该上限替换为 R_ref。

## 校准程序与输出

`common/f18_calibrate_normalization.py` 支持 `--all-nine`、单场景、`--seeds`、`--out`、`--reference-policy ERTF-FIFS`、`--dry-run`、`--resume`、`--summary-only`、`--raw-records`。

输出目录默认为 `outputs/f18_normalization_calibration/`：

- `ready_percentiles_9scenarios.csv`
- `deadline_percentiles_9scenarios.csv`
- `f18_normalization_summary_9scenarios.csv`
- `f18_shared_normalization_v1.json`
- `f18_calibration_manifest.json`
- `f18_calibration_per_seed.csv`
- `episodes/<scene>_<seed>.json`：断点恢复及重新汇总所需的每回合样本；逐决策详细记录仅在 `--raw-records` 时加入。

三张场景表始终包含九行。未采集/失败的场景百分位留空并标记 incomplete；公共配置仅包含完整成功的场景，加载缺失场景报错。Resume 校验场景、种子、缓存哈希和采样代码/拓扑指纹；仅跳过成功回合，失败回合可重试，缓存缺失时明确给出预计算命令。Summary-only 不运行模拟。正式缓存始终只读。

Manifest 保存基线提交、采样代码/拓扑指纹、日期、Scenario 配置、种子、DAX、参考策略、50 工作流、缓存路径/哈希/meta、公共 Alpha 规则、单位、百分位算法、成功/失败/未生成数量及结果路径。正式配置包含单位、生成规则与实际种子。正式配置生成后应冻结，不能训练中重拟合。

### Windows PowerShell

```powershell
# 首先检查，不写缓存、不训练
.\run.cmd common/f18_calibrate_normalization.py --all-nine --seeds 101-130 --dry-run

# 本工作区缺少新版正式缓存。仅当对应文件尚不存在时生成；程序拒绝覆盖已有缓存。
foreach ($resourceSize in 'S','M','L') {
    .\run.cmd common/precompute_deadlines.py --task S --resource $resourceSize --seeds 0-1000
    if ($LASTEXITCODE -ne 0) { throw "Deadline cache preparation failed" }
}

# 一条命令完成九场景。当前已有缺失诊断清单，应使用 --resume。
.\run.cmd common/f18_calibrate_normalization.py --all-nine --seeds 101-130 --resume
# 首次使用新目录时可省略 --resume；后续续跑使用它。
.\run.cmd common/f18_calibrate_normalization.py --task S --resource S --deadline Tight --seeds 101-130 --out outputs/f18_single
.\run.cmd common/f18_calibrate_normalization.py --summary-only

# 校准完成后正式入口默认 f18_shared_v1，可显式指定冻结配置
.\run.cmd algorithms/mahdrl/train.py --task S --resource S --deadline Tight --normalization-config outputs/f18_normalization_calibration/f18_shared_normalization_v1.json
# 从 SS 模型在 MS 评测，继续使用 SS 参数，不使用 MS 测试统计
.\run.cmd algorithms/mahdrl/evaluate.py --task M --resource S --deadline Tight --train-task S --normalization-config outputs/f18_normalization_calibration/f18_shared_normalization_v1.json
# 历史 F17/F11 观测复现必须显式选择 legacy
.\run.cmd algorithms/pd3qn/evaluate.py --normalization-mode legacy --checkpoint-dir PATH_TO_HISTORICAL_F11_MODEL
```

若新版缓存文件存在但版本不符，先用 `--out` 写入另一个不存在的文件并核验，不要直接覆盖历史缓存。已有 F07/F02/F03/F06 的缓存和配对校验继续有效。

### Linux/Bash

```bash
python launch.py common/f18_calibrate_normalization.py --all-nine --seeds 101-130 --dry-run
for resource in S M L; do
  python launch.py common/precompute_deadlines.py --task S --resource "$resource" --seeds 0-1000 || exit 1
done
python launch.py common/f18_calibrate_normalization.py --all-nine --seeds 101-130 --resume
python launch.py algorithms/mahdrl/train.py --task S --resource S --deadline Tight --normalization-config outputs/f18_normalization_calibration/f18_shared_normalization_v1.json
```

`run.cmd`/`launch.py` 已支持新增脚本，无需修改。

## 种子与源训练配置

现有训练按 1、2、3……递增，600 回合对应 1–600；训练内验证使用 1，默认正式评测 2–31。校准默认 101–130 来自训练范围，与已知验证/评测集合不重叠，程序拒绝 1–31 或超出 1–600 的校准种子。本次不改变既有训练/评测划分；其训练与验证/测试种子重叠是另一个需要实验设计确认的问题，不能称已完全独立划分。如自行改用其他最终测试种子，须先核对并选择互不重叠的训练校准种子，不能复用最终测试样本拟合尺度。

正式训练从 task/resource/deadline 定位配置。评测从 `--train-task`、可选 `--train-resource`/`--train-deadline` 定位源训练配置；目标任务规模不改变尺度。MS/LS 从 SS 模型泛化时继续使用 SS 参数。不存在对应小任务源配置时明确拒绝，不在 M/L 测试集上拟合新参考。

Checkpoint 的 `normalization` 元数据包含 normalization_version、source_train_scene、normalization_config_path、normalization_config_hash、R_ref、T_ref。D3QN、MARL Global/Local 和 IRWS joint 均在写入时保存、读取网络权重前检查。配置允许迁移路径，但文件字节哈希、源场景及尺度必须相同。旧格式只有显式 Legacy 入口可按旧观测读取；F11 原有检查仍有效。Legacy 模式不会恢复已删除的 F17 Raw 权重机制。

新版 checkpoint/log 名称追加 `_f18_shared_v1`，评测进入独立版本子目录。旧文件保留。需重训：MAHDRL、MARL、PD3QN、IRWS、ablation_hostrand、ablation_mgrrand、ablation_vmrand、ablation_nohost 的九种源训练场景模型。ERTF-FIFS 无需强化学习重训；F18 不改变已有物理 Deadline Cache。

每回合原有 `f11_diagnostics.jsonl` 新增 normalization 字段，记录 RR 饱和次数、观察次数、比例及配置指纹。对相同 Ready 状态的重复观测去重，分母是非空 Ready 的不同观测状态，非时间加权。校准统计按不同参考决策时刻计数，与实际学习策略下的观测饱和诊断分开解释；不改变奖励。

## 正式九场景结果：当前缺失条件

实际尝试运行九场景 `--seeds 101-130`，但以下三个新版正式缓存不存在：

- `data/deadlines/dynamic_full_parent_output_v1/ddl_fcfs_taskS_resS_seed0-1000.json`
- `data/deadlines/dynamic_full_parent_output_v1/ddl_fcfs_taskS_resM_seed0-1000.json`
- `data/deadlines/dynamic_full_parent_output_v1/ddl_fcfs_taskS_resL_seed0-1000.json`

没有自动生成、修改或覆盖正式缓存。已输出九行缺失诊断；正式百分位和参数尚不可用。不得将空配置用于训练，也不得将临时 smoke 校准当成 30 种子的正式结果。

| 场景 | Ready P50 | P95 | P99 | Max | R_ref |
|---|---|---|---|---|---|
| SS_L | 未生成 | — | — | — | — |
| SS_M | 未生成 | — | — | — | — |
| SS_T | 未生成 | — | — | — | — |
| SM_L | 未生成 | — | — | — | — |
| SM_M | 未生成 | — | — | — | — |
| SM_T | 未生成 | — | — | — | — |
| SL_L | 未生成 | — | — | — | — |
| SL_M | 未生成 | — | — | — | — |
| SL_T | 未生成 | — | — | — | — |

| 场景 | 相对 DDL P50 | P95 | P99 | Max | T_ref |
|---|---|---|---|---|---|
| SS_L | 未生成 | — | — | — | — |
| SS_M | 未生成 | — | — | — | — |
| SS_T | 未生成 | — | — | — | — |
| SM_L | 未生成 | — | — | — | — |
| SM_M | 未生成 | — | — | — | — |
| SM_T | 未生成 | — | — | — | — |
| SL_L | 未生成 | — | — | — | — |
| SL_M | 未生成 | — | — | — | — |
| SL_T | 未生成 | — | — | — | — |

九场景各公共参数 R_ref、T_ref、workload_mi_ref、input_bits_ref、output_bits_ref、up_rank_ref、down_rank_ref、vm_pc_ref、vm_bw_ref、host_pc_ref 均待正式采样，不填入人工默认值。

## 验证与限制

测试覆盖 T01–T17，含实际参考调度可复现、缓存字节不变、Ready/Blocked/Running 统计、负 Slack、逐字段尺度、状态维度、源场景选择、Checkpoint 保存/加载及错配拒绝、缺失汇总、数值边界，以及四种真实环境在固定动作下的完整 50 工作流物理轨迹对照。完整回归 **118 项通过、0 失败**，包含既有 101 项和 F18 新增 17 项；最新运行证据见 `docs/F18_tests.txt`。

T18 的 `common/smoke_F18.py` 使用临时新模型缓存，实际采集九场景各 seed=101 的 50 工作流，生成仅供验证的尺度，然后运行八个原训练入口各 1 episode（含原训练内验证），并通过原评测入口加载所保存的 checkpoint。仅验证脚本使用小网络 `(16,8)`、batch=4、buffer=32；生产配置、优化器和网络结构未改。八个入口、训练内验证和评测 checkpoint 加载均通过；结果见 `docs/F18_smoke.json`。IRWS 的 S 场景 50 工作流少于原有 2048 步 Rollout 阈值，本次 smoke 经终止回合更新，未人为修改生产阈值；非终止 Rollout 的旧 F11 测试仍通过。临时缓存与模型不写入正式路径。

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s common -p "test_*.py" -v
.\.venv\Scripts\python.exe -m common.smoke_F18 --out docs/F18_smoke.json
```

此前修复的逻辑通过原有回归保护。固定动作对照验证的是相同动作下物理结果不变；真实策略读到不同输入后可以选择不同动作，这正是需要重训的原因。没有运行 600 回合正式训练，不据短程测试声称收敛或性能提升。

未解决的其他公平性事项包括：IRWS 的任务候选截断差异、既有 seed 划分重叠，以及此前 F12/F13 等论文适配审查中仍未核实的条目；F18 不修改这些问题，也不代表其他算法适配已经完整验证。

## IEEE TSC 英文 LaTeX 修改建议

```latex
% Replace the normalization definitions in (17)--(20), (23)--(24), (27), (29).
We use fixed reference scales estimated exclusively from training-calibration
instances. Let $N_{\mathrm{ready},t}$ denote the number of arrived tasks whose
predecessors have all completed, and let $N_{\mathrm{blocked},t}$ denote arrived,
unassigned tasks blocked by unfinished predecessors. The unfinished set includes
ready, blocked, running, and other genuinely unfinished arrived tasks. We define
\begin{align}
RR_t &= \min\{1,N_{\mathrm{ready},t}/R_{\mathrm{ref}}\}, \\
URR_t &= \frac{N_{\mathrm{blocked},t}}{\max\{1,N_{\mathrm{unfinished},t}\}}.
\end{align}
The reference $R_{\mathrm{ref}}$ is an observation scale, not a queue-capacity
constraint. For each source training scenario, ERTF--FIFS reference scheduling
provides nonempty ready-queue samples at distinct decision timestamps.
Each timestamp receives equal weight. We compute the 50th, 95th, and 99th
percentiles and maximum using linear interpolation, and set
$R_{\mathrm{ref}}=\max\{1,\lceil P_{99}(N_{\mathrm{ready}})\rceil\}$.
For each generated workflow, its relative deadline $D_w-A_w$ is recorded once,
and $T_{\mathrm{ref}}=P_{95}(D_w-A_w)$.

All time-valued observation features are divided by this fixed
$T_{\mathrm{ref}}$, including mean and maximum waiting times, minimum and mean
workflow slack, task waiting time, predicted execution duration, and resource
availability delay. In the task-ranking and host-selection observations,
workflow slack is computed as $D_w-t-\overline{\tau}_i$, where
$\overline{\tau}_i$ is the mean predicted execution duration over VMs.
For the task's best-execution slack, we retain the original candidate set and
use $(D_w-t-\tau_i^{\mathrm{best}})/T_{\mathrm{ref}}$.
Task-subdeadline slack remains distinct from workflow-deadline slack.
Negative slack is retained, and time features may exceed one.
For the VM agent, availability delay and predicted remaining completion duration
are respectively $\max\{0,A_v-t\}/T_{\mathrm{ref}}$ and
$\max\{0,F_{i,v}-t\}/T_{\mathrm{ref}}$; an absolute finish timestamp is not
interpreted as a remaining duration. The simulation clock, deadline generation,
reward scales, communication, and energy accounting are unchanged.

For controlled baseline comparisons, MAHDRL, MARL, PD3QN, IRWS, and the associated
ablations share the same source-scenario reference file for common physical
quantities. Workload, logical input/output size, and ranks with the same
mathematical definition use their training-calibration 95th percentiles;
resource capacities use the maximum corresponding source-resource specifications.
We retain each algorithm's network architecture, action space, feature order,
and algorithm-specific feature definitions. In this controlled preprocessing
variant, PD3QN uses attribute-wise scaling instead of whole-vector L2
normalization. Scales are frozen before learning and reused during validation
and cross-task-scale evaluation, without fitting on target test instances.
The percentile choices are empirical scaling conventions, not theoretically
optimal parameters. Checkpoints record the normalization version, source training
scenario, reference values, and configuration hash for consistency checks.
```

正式采样、重训及统计检验完成之前，不应把上述实验设置建议写成已得到正式实验结果的陈述，也不要填写尚未生成的数值。

## 实际临时采样数值（仅验证，不能用于正式结果）

以下来自 T18 临时缓存及真实动态工作流，九组各仅使用训练校准 seed 101、50 工作流。它们不是所要求的 101–130 正式校准表，也没有写入正式配置路径；正式表仍为上面的缺失状态。

| scene | ready_sample_count | ready_P50 | ready_P95 | ready_P99 | ready_Max | R_ref |
|---|---|---|---|---|---|---|
| SS_L | 317 | 1.000000 | 21.000000 | 45.880000 | 71.000000 | 46 |
| SS_M | 317 | 1.000000 | 21.000000 | 45.880000 | 71.000000 | 46 |
| SS_T | 317 | 1.000000 | 21.000000 | 45.880000 | 71.000000 | 46 |
| SM_L | 592 | 1.000000 | 7.450000 | 21.000000 | 39.000000 | 21 |
| SM_M | 592 | 1.000000 | 7.450000 | 21.000000 | 39.000000 | 21 |
| SM_T | 592 | 1.000000 | 7.450000 | 21.000000 | 39.000000 | 21 |
| SL_L | 678 | 1.000000 | 7.000000 | 21.000000 | 21.000000 | 21 |
| SL_M | 678 | 1.000000 | 7.000000 | 21.000000 | 21.000000 | 21 |
| SL_T | 678 | 1.000000 | 7.000000 | 21.000000 | 21.000000 | 21 |

| scene | workflow_count | deadline_P50 | deadline_P95 | deadline_P99 | deadline_Max | T_ref |
|---|---|---|---|---|---|---|
| SS_L | 50 | 1118.730467 | 1872.579135 | 2169.471615 | 2215.947569 | 1872.579135 |
| SS_M | 50 | 1036.686018 | 1711.412711 | 1867.912942 | 1925.073798 | 1711.412711 |
| SS_T | 50 | 824.341978 | 1448.843714 | 1841.646819 | 1925.073798 | 1448.843714 |
| SM_L | 50 | 881.582875 | 1707.048306 | 2417.461034 | 2676.482217 | 1707.048306 |
| SM_M | 50 | 826.682707 | 1707.048306 | 1969.729979 | 2147.867559 | 1707.048306 |
| SM_T | 50 | 704.512907 | 1347.847527 | 1969.729979 | 2147.867559 | 1347.847527 |
| SL_L | 50 | 953.089128 | 1662.371363 | 1931.654968 | 2029.799231 | 1662.371363 |
| SL_M | 50 | 859.061828 | 1504.960936 | 1798.289922 | 1829.504817 | 1504.960936 |
| SL_T | 50 | 753.759620 | 1218.194794 | 1563.626233 | 1765.800950 | 1218.194794 |

临时公共参数（单位及生成方法见前文；仅用于上述 smoke）：

| scene | R_ref | T_ref | workload_mi_ref | input_bits_ref | output_bits_ref | up_rank_ref | down_rank_ref | vm_pc_ref | vm_bw_ref | host_pc_ref |
|---|---|---|---|---|---|---|---|---|---|---|
| SS_L | 46 | 1872.5791 | 88.563099 | 7.3817505e+10 | 2.1384954e+10 | 284.81507 | 204.22662 | 8 | 8000 | 34 |
| SS_M | 46 | 1711.4127 | 88.563099 | 7.3817505e+10 | 2.1384954e+10 | 284.81507 | 204.22662 | 8 | 8000 | 34 |
| SS_T | 46 | 1448.8437 | 88.563099 | 7.3817505e+10 | 2.1384954e+10 | 284.81507 | 204.22662 | 8 | 8000 | 34 |
| SM_L | 21 | 1707.0483 | 88.563099 | 7.3817505e+10 | 2.1384954e+10 | 284.81507 | 204.22662 | 8 | 8000 | 34 |
| SM_M | 21 | 1707.0483 | 88.563099 | 7.3817505e+10 | 2.1384954e+10 | 284.81507 | 204.22662 | 8 | 8000 | 34 |
| SM_T | 21 | 1347.8475 | 88.563099 | 7.3817505e+10 | 2.1384954e+10 | 284.81507 | 204.22662 | 8 | 8000 | 34 |
| SL_L | 21 | 1662.3714 | 88.563099 | 7.3817505e+10 | 2.1384954e+10 | 284.81507 | 204.22662 | 8 | 8000 | 34 |
| SL_M | 21 | 1504.9609 | 88.563099 | 7.3817505e+10 | 2.1384954e+10 | 284.81507 | 204.22662 | 8 | 8000 | 34 |
| SL_T | 21 | 1218.1948 | 88.563099 | 7.3817505e+10 | 2.1384954e+10 | 284.81507 | 204.22662 | 8 | 8000 | 34 |

## 实际修改文件清单

- `README.md`
- `algorithms/ablation_hostrand/evaluate.py`
- `algorithms/ablation_hostrand/train.py`
- `algorithms/ablation_mgrrand/evaluate.py`
- `algorithms/ablation_mgrrand/train.py`
- `algorithms/ablation_nohost/evaluate.py`
- `algorithms/ablation_nohost/train.py`
- `algorithms/ablation_vmrand/evaluate.py`
- `algorithms/ablation_vmrand/train.py`
- `algorithms/irws/evaluate.py`
- `algorithms/irws/train.py`
- `algorithms/mahdrl/evaluate.py`
- `algorithms/mahdrl/train.py`
- `algorithms/marl/agents.py`
- `algorithms/marl/evaluate.py`
- `algorithms/marl/train.py`
- `algorithms/pd3qn/evaluate.py`
- `algorithms/pd3qn/train.py`
- `common/d3qn_agent_EXDehid.py`
- `common/d3qn_agent_PER.py`
- `common/f18_calibrate_normalization.py`
- `common/f18_state_normalization.py`
- `common/scenarios.py`
- `common/smoke_F11.py`
- `common/smoke_F17.py`
- `common/smoke_F18.py`
- `common/test_f18.py`
- `docs/F18_shared_state_normalization.md`
- `docs/F18_smoke.json`
- `docs/F18_tests.txt`
- `environments/ablation_nohost.py`
- `environments/irws.py`
- `environments/mahdrl_heft.py`
- `environments/marl.py`
- `environments/pd3qn.py`

`common/smoke_F11.py` 增加可选的新归一化/评测加载检查，默认仍显式运行历史 Legacy 验证；`common/smoke_F17.py` 的历史验证夹具显式选择 Legacy 观测，F17 权重转移测试逻辑不变。未触碰现有四个 Word 文档。未提交或推送 F18。

最终保护性核验：对比基线提交的 Python 语法树，`set_manager_combo`、`apply_manager_delta`、`get_manager_action_mask`、`_compute_task_heuristics_for_ready`、`_phase_prepare_tasks`、`vm_assign`、`finish_phase_and_advance`、`_task_duration` 均完全相同。`common/ta_weights.py`、`common/workflow_opt.py`、`common/read_xml_opt_Tsize.py`、`common/resource_opt.py`、`common/precompute_deadlines.py`、`environments/mahdrl_fcfs.py` 与基线内容一致。最终完整回归 118 项通过，编译及 `git diff --check` 通过。