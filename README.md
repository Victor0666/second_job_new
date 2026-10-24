# MAHDRL 实验代码

新增独立对比算法 `two_stage_gat`（2SD-GAT-Adapted）：保留作者两阶段 GAT–PPO 与忙碌 VM 队尾未来预留，接入共同能耗—截止期问题，使用原 6＋4 维观测的目标适配。原 MAHDRL 与已有算法保持独立。训练/评测入口为 `algorithms/two_stage_gat/train.py`、`evaluate.py`；参数、源码差异、未来排程及公平性边界见 [2SD-GAT 实现说明](docs/Two_stage_GAT_implementation.md)。

`algorithms/` 按算法分目录，每个算法集中保存统一训练/评测入口及专用智能体；环境集中在 `environments/`。场景由命令行参数选择。

新增独立变式 `mahdrl_oe`（MAHDRL-OE），实施截止期/功耗/候选主机观测增强与按分配结算能耗奖励，保留原 MAHDRL。用 `run.cmd algorithms/mahdrl_oe/train.py` 或 `run.cmd algorithms/mahdrl_oe/evaluate.py` 启动，场景参数相同，模型与输出独立；详见 [MAHDRL-OE 说明](docs/MAHDRL_OE.md)。

新增 TA 稳定性测试变式 `mahdrl_ts`（MAHDRL-TS），基于原 MAHDRL，仅将 TA 学习率从 `0.0003` 改为 `0.0001`、目标网络软更新系数从 `0.01` 改为 `0.005`。训练、模型、日志和评测目录独立，原 MAHDRL 与 MAHDRL-OE 保持不变；详见 [MAHDRL-TS 说明](docs/MAHDRL_TS.md)。

新增未来价值上界测试变式 `mahdrl_ub`（MAHDRL-UB），基于原 MAHDRL，仅在 TA 的 TD 目标中使用 `min(Q_next, 0)`，选动作仍使用原始 Q；TA 学习率 `0.0003`、tau `0.01` 及 HA/VA 保持原配置。模型和输出独立，保留约束前估值与触发比例；详见 [MAHDRL-UB 说明](docs/MAHDRL_UB.md)。

```text
algorithms/                    每个算法一个目录
  mahdrl/                      train.py、evaluate.py
  mahdrl_ts/                   train.py、evaluate.py（TA 稳定性测试）
  mahdrl_ub/                   train.py、evaluate.py、agent.py（TA 未来价值上界测试）
  marl/                        train.py、evaluate.py、agents.py
  irws/                        train.py、evaluate.py、task_ranker.py、vm_agent.py、intrinsic_reward.py
  pd3qn/                       train.py、evaluate.py
  ablation_hostrand/           train.py、evaluate.py
  ablation_mgrrand/            train.py、evaluate.py
  ablation_vmrand/             train.py、evaluate.py
  ablation_nohost/             train.py、evaluate.py
  ertf_fifs/                   evaluate.py
environments/                  所有环境实现
  mahdrl_fcfs.py / mahdrl_heft.py
  marl.py / irws.py / pd3qn.py
  ablation_nohost.py
  base_fcfs.py / base_heft.py
common/                        场景配置、共享智能体、工作流、资源、日志、预计算及测试
data/dax/                      工作流 XML
data/deadlines/                截止期基准缓存
plots/                         绘图/统计脚本与 Excel 输入
outputs/                       本地模型、日志、评测及图表，不上传
docs/                          论文、环境版本、配置修正说明及验证记录
```

## 安装

Windows 下先运行 `setup.cmd` 建立 `.venv`，依赖见 `requirements.txt`，验证环境见 `docs/runtime_versions.json`。其他电脑需有可用的 Python；PyTorch/CUDA 版本应与设备兼容。不要直接复制其他电脑的虚拟环境。

## 训练与评测

在项目根目录的 PowerShell 中运行：

```powershell
# 训练：小任务、小资源、紧截止期，默认 600 回合
.\run.cmd algorithms/mahdrl/train.py --task S --resource S --deadline Tight

# 切换资源和截止期
.\run.cmd algorithms/mahdrl/train.py --task S --resource L --deadline Medium

# 只训练一回合（会写入该场景的模型与日志）
.\run.cmd algorithms/mahdrl/train.py --task S --resource S --deadline Tight --episodes 1

# 评测：默认使用该资源和截止期下以 S 任务训练的模型，种子 2..31
.\run.cmd algorithms/mahdrl/evaluate.py --task S --resource S --deadline Tight

# 泛化评测：M 任务、M 资源、Medium；以 S 任务训练的模型；只跑种子 700
.\run.cmd algorithms/mahdrl/evaluate.py --task M --resource M --deadline Medium --seeds 700

# 如果模型以 M 任务训练，明确指定 --train-task M
.\run.cmd algorithms/mahdrl/evaluate.py --task L --resource M --deadline Medium --train-task M --seeds 700,705,708

# 预览参数与模型目录，不运行实验
.\run.cmd algorithms/mahdrl/evaluate.py --task L --resource L --deadline Loose --dry-run
```

将路径中的 `mahdrl` 换成其他算法目录名，即可使用相同参数。启发式使用：

```powershell
.\run.cmd algorithms/ertf_fifs/evaluate.py --task M --resource S --deadline Medium --seeds 700
```

| 参数 | 可选值/默认值 | 含义 |
|---|---|---|
| `--task` | S / M / L，默认 S | 工作流规模分布 |
| `--resource` | S / M / L，默认 S | 3 / 6 / 9 个主机，25 / 50 / 75 个 VM |
| `--deadline` | Tight / Medium / Loose，默认 Tight | ddlAlpha=2.0 的概率为 0.8 / 0.5 / 0.2，其余为 3.0 |
| `--episodes` | 正整数，默认 600 | 训练回合数，仅训练入口 |
| `--seeds` | 如 `700-729` 或 `700,705,708`，默认 700-729 | 评测种子，范围含两端 |
| `--train-task` | S / M / L，默认 S | 被评测模型的训练任务规模，独立于评测任务 |
| `--checkpoint-dir` | 目录路径 | 显式选择评测模型目录 |
| `--dry-run` | 开关 | 只打印所选配置 |

S 任务仅使用约 30 节点的五类工作流；M 任务按 30/50 节点组的 3:7 混合；L 任务按 30/50/100 节点组的 2:3:5 混合。每回合仍为 50 个工作流。统一配置位于 `common/scenarios.py`。

MARL 额外保留 `--deadline-cache`、`--ckpt-global`、`--ckpt-local`、`--out` 和 `--device`，可用 `--help` 查看。其余算法自动选择 CUDA 或 CPU。

MARL 正式入口现为 `marl_q_v3`：独立 Local Actor/Critic、动作条件 Q、Global 参照 MAHDRL HA 补全观测；同一条经验先更新 Critic，再用更新后的 Q 更新 Actor，随后采样实际动作。原文未明确 TD 公式的部分使用一步单样本后继 Q：Global 只查询抽中的一个 Local，Local 只抽自己的合法 VM；估值抽样与执行动作分开。熵和梯度裁剪关闭，保留公共归一化与奖励。旧 A2C、v1、期望 Q 的 v2 模型不能加载，需重训，训练/评测命令不变，输出进入独立版本目录。实现与验收见 [MARL Q 适配说明](docs/MARL_Q_implementation.md)。

新 MARL 按独立对比算法标记为 **MARL-Q**，CSV 和测量 JSON 的 `algorithm` 均为 `marl_q_v3`。训练日志名为 `train_metrics_marl_q_v3.csv`；默认评测结果位于 `outputs/run/marl_q_v3/<协议与归一化版本>/from_<训练场景>_model<模型种子>/eval_<评测场景>/`，其中 CSV 为 `evaluation_marl_q_v3.csv`，逐种子明细在 `measurements/`。旧 MARL/A2C/v1/v2 的目录不阻挡新实验；同一新版训练目录非空或同一评测目的地已有结果时明确报错，防止追加或覆盖。另一次评测可通过 `--out <新的CSV路径>` 同时隔离 CSV 和附属明细。`--dry-run` 会打印实际目的地。

训练输出在 `outputs/ckpts` 和 `outputs/logs`，评测 CSV 在 `outputs/run`。MARL 的重复写入规则见上段；其他算法沿用各自的输出规则。评测不会自动搜索并替换为其他场景的模型；缺失模型时会明确报错。

## 截止期缓存、绘图与检查

```powershell
# 统一预计算入口：默认生成指定任务/资源的种子 0..1000 缓存
.\run.cmd common/precompute_deadlines.py --task M --resource L

# 快速检查一个种子，另存到 outputs，不覆盖正式缓存
.\run.cmd common/precompute_deadlines.py --task S --resource S --seeds 2 --out outputs/deadline_check.json

# 入口导入与根目录检查，不需要权重
.\run.cmd verify_package.py --imports-only

# 全场景配置回归测试：不训练、不加载权重
.\.venv\Scripts\python.exe -m unittest common.test_scenarios

# 绘图：启动器会切换到 plots，读取相邻 Excel
.\run.cmd plots/algorithm_boxPlot_Merge.py
```

预计算只选择 `--task` 和 `--resource`；基准 makespan 缓存与 Tight/Medium/Loose 无关。非完整种子的输出会使用独立文件名，或由 `--out` 指定，避免把正式的 0..1000 缓存替换成不完整文件。

不带参数运行 `run.cmd` 会检查导入，并对九种算法各评测一个 SS/Tight 回合；报告在 `outputs/verification_result.json`。GitHub 不包含权重，克隆后可先检查导入，再训练或另行获取已有权重。

## 实验版本说明

场景整合后保留 17 个统一训练/评测入口和一个截止期预计算入口。当前算法恢复为每个算法一个目录；八个环境实现（含两个基础环境）继续集中在 `environments/`。场景仍通过参数切换，已移除的重复场景脚本保持合并。

此前平铺算法的版本可通过 Git 提交 `be8b6ed` 查看，旧场景入口可通过 `31d7110` 恢复。历史打包记录保留在本地 `docs/archive`，不进入当前 GitHub 文件列表。

此前场景整合也按用户要求修正了若干历史场景配置，因此部分新旧实验不可直接等同。具体修正及对照规则见 [场景修正说明](docs/scenario_changes.md)。已有模型、原始数据和历史实验输出未被批量改写；部分历史绘图仍使用旧汇总数据，重新出图时应核对实验版本。

## 统一能耗计量

所有算法的 episode 能耗统一为 `host_task_window_v1`：每台主机只在自身第一个任务开始到自身最后一个任务完成之间积分功耗。完全未使用的主机不计；该主机首个任务之前、最后一个任务之后不计；窗口内部的任务间空闲仍计入。

窗口按 Host 而不是 VM 划定。公共函数 `energy_from_records(records, hosts, start_time=..., end_time=...)` 的显式端点仅裁剪各主机自身窗口；零负载记账记录不能扩展窗口。HA/VA 的全局或局部奖励范围保持原设计，但物理能耗均采用此口径。PD3QN、IRWS 在新增任务扩展主机窗口时补计内部空闲；MARL、MAHDRL 及消融在实际时间推进时结算，后续使用同一主机时补计被纳入窗口的间隔。

能耗窗口回归检查：`.\.venv\Scripts\python.exe -m unittest common.test_energy`。新模型与评测路径追加 `host_task_window_v1`，Checkpoint 校验此标识，旧全系统窗口模型需要重训。任务、通信、截止期与物理轨迹规则不变，不要求重建截止期缓存。详细例子见 [主机级能耗说明](docs/Host_task_window_energy.md)。此前四算法甘特图及其数据保留为全系统窗口历史示例，不会因本次修改自动变成新口径结果。

## 统一父子通信规则

全部算法只在父子任务分配到同一 VM 时忽略父子数据传输；同主机的不同 VM 和不同主机均计入传输。外部输入和任务输出仍按原规则计费。实际执行、VM 完成时间预测、HEFT 参考调度和 FCFS/FIFS 截止期基准均使用这一规则。

本次通信模型变更会改变截止期基准 makespan。旧 JSON 缓存没有 `meta.communication_model = "same_vm_only_v1"` 标记，环境会拒绝加载并提示重新生成。旧缓存文件保留，未自动覆盖；正式重跑之前，请在项目根目录为九种任务/资源组合重新预计算：

```powershell
foreach ($taskSize in 'S', 'M', 'L') {
    foreach ($resourceSize in 'S', 'M', 'L') {
        .\run.cmd common/precompute_deadlines.py --task $taskSize --resource $resourceSize
        if ($LASTEXITCODE -ne 0) { throw "截止期缓存生成失败" }
    }
}
```

上面的预计算在当前数据模型的版本目录创建各场景正式 `seed0-1000` 缓存；已有文件会拒绝覆盖。仅 Tight/Medium/Loose 不同的实验共用同一任务/资源基准。回归测试：`.\.venv\Scripts\python.exe -m unittest common.test_communication`。

## 统一任务子截止期

MARL、MAHDRL（含消融和启发式）、PD3QN、IRWS 的缓存截止期环境共同调用 `common.workflow_opt.task_subdeadlines`。每个任务先计算在各个 VM 上的输入传输、计算和输出传输时长，再对 VM 时长取算术平均，使用 Latest-Finish 从后继向前逆推。汇点子截止期等于工作流截止期；不执行 `max(LF, local_EF)` 下界修正，也不替换成平均 PC、平均带宽下的时长。

此修改保持工作流截止期与 alpha 采样方式不变。任务到达时尚未确定 VM 放置，子截止期估算包含全部父任务输入；实际执行仍使用仅同 VM 减免的通信规则。验证命令：`.\.venv\Scripts\python.exe -m unittest common.test_task_deadlines`。

## 配对截止期实例

所有实验算法使用 `common.workflow_opt.sample_deadline_alpha`，由 episode 种子、工作流编号和同一概率分布确定 alpha；不消耗环境随机流。公共采样器保留 MAHDRL 的派生种子公式与逆 CDF 规则，统一偏移量为 `20260316`。相同种子和工作流编号的重复或换序查询始终返回相同 alpha；Tight、Medium、Loose 中 alpha=2 的概率仍分别为 0.8、0.5、0.2。

基准缓存锁定模板、到达时间和参考 makespan，payload 种子统一为 `episode_seed + wf_id`，工作流绝对截止期统一为 `arrival + alpha * makespan_ref`。MARL 使用 reset 的实际 episode 种子；各训练入口从种子 1 开始逐回合加 1，MARL 缓存缺少训练种子时直接报错。PD3QN 不再使用独立偏移量，旧配置中显式传入 `20260317` 会被拒绝，应删除该参数。

验证命令：`.\.venv\Scripts\python.exe -m unittest common.test_paired_instances`。该测试对 SS、seed=2 的三组截止期分布，逐一核对九种算法的 50 个工作流的模板、到达、payload 种子与数据、alpha、绝对截止期和全部任务子截止期。

## 缓存模板名称匹配

模板标识统一为不区分大小写的文件名（`basename_casefold_v1`）；例如 `Ligo_50.xml` 和 `LIGO_50.xml` 指向同一模板，30 节点与 50 节点模板仍严格区分。MARL 只从配置的 `dax_paths` 中解析缓存名称，加载缓存时检查未知模板，并拒绝多个不同路径映射到同一标识的歧义；任务分布使用的同一路径重复权重保留。日志输出使用实际配置文件的名称。

其他缓存环境使用相同的名称比较规则，严格模式下真实名称不匹配会报错，纯大小写差异不会再被警告为模板不同。新预计算缓存记录 `meta.dax_template_id_scheme` 和现有通信模型标识。大小写兼容不改变工作流或基准数值；此前通信模型变更所要求的缓存重新生成仍然适用。

验证命令：`.\.venv\Scripts\python.exe -m unittest common.test_dax_templates`，覆盖 MS/MM/ML、未知/歧义模板，以及九组历史缓存的模板抽样序列。


## F07 文件依赖模型与重跑

文件模型统一为 `dynamic_full_parent_output_v1`：每个任务生成 1–3 个输出文件，仅入口任务生成 1–3 个外部输入文件。每个直接子任务共享父任务的全部输出文件；非入口输入数不限，外部输入为零，父输出不按出度均分。同 VM 仍仅减免网络时间，逻辑边数据量保持正值。

新截止期缓存默认保存在 `data/deadlines/dynamic_full_parent_output_v1/`。旧目录文件保留，缺少新模型及参数标记的缓存会被拒绝。预计算写入工作流实例哈希，各算法到达时校验。必须重新生成九种任务/资源组合的基准后再训练；已有缓存不会被覆盖，历史权重和结果应与新实验分开管理。

详细论文 LaTeX 替换段落、验收表、15 个模板检查、逐工作流修复前后对照及命令见 [F07 返修记录](docs/revision_F07.md)。测试：`.\.venv\Scripts\python.exe -m unittest discover -s common -p "test_*.py" -v`。


## F11 真实决策转移与重训

MAHDRL 及四种消融中保留的资源学习器、PD3QN、MARL Local 已恢复真实下一决策 Bootstrap；IRWS PPO 在非终止 Rollout 边界使用更新前 Critic 对下一实际 VM 决策估计的价值。Phase 结束不等于 Episode 终止，待处理经验在真正终止或明确的截断策略下关闭。具体奖励结算、算法差异、论文修改与验证边界见 [F11 返修记录](docs/revision_F11.md)。

新模型/日志名称追加 `_f11_real_decisions_v1`，学习型评测结果进入 `outputs/run/f11_real_decisions_v1/`。默认评测拒绝无 F11 标记的旧 Checkpoint；旧模型保留，但必须重新训练才能报告 F11 结果。训练拒绝复用非空输出目录。最佳模型选择规则保持不变，F07 的新版缓存路径保持不变。

诊断写入对应训练日志目录的 `f11_diagnostics.jsonl`；设置 `$env:F11_DIAGNOSTICS = '0'` 可关闭诊断统计和写入。专项测试：`.\.venv\Scripts\python.exe -m unittest common.test_f11 -v`。八个实际训练入口的轻量验证：`.\.venv\Scripts\python.exe -m common.smoke_F11 --out docs/F11_smoke.json`，仅在该验证脚本内使用小网络和临时输出，不替代正式训练。


## F17 TA 归一化权重与重训

TA 直接执行 `Norm(w + delta)`，只维护一套归一化决策权重；默认构造与 reset 为 `[0.2]*5`，替换旧随机 Raw 初始化。Mask 与执行共用合法性检查，非法动作拒绝，全零增量精确保持权重。F15 排序属性、243 个动作与 HA/VA 算法保持不变。

MAHDRL 及四种消融的新模型/日志名称追加 `_f17_normalized_weights_v1`，评测写入独立 F17 子目录。旧 checkpoint 张量虽可兼容读取，不能作为 F17 正式结果，必须重新训练。F17 不要求额外重建缓存，但此前 F07 等版本要求仍有效。验证命令、英文论文补充、审稿回复和完整验收表见 [F17 返修记录](docs/revision_F17.md)。

## F18 共享状态归一化与训练前校准

正式学习算法与四种消融默认使用 `f18_shared_v1`：从固定训练参考配置加载 R_ref/T_ref 和物理属性尺度，配置缺失时明确报错。PD3QN 的新版输入改为逐属性缩放，取消整体 L2；F15/F17 排序与权重转移、奖励、通信、能耗和 Deadline 生成保持原有逻辑。历史观测复现须显式使用 `--normalization-mode legacy`。

先准备三个 S 任务规模、S/M/L 资源规模的新版正式 Deadline Cache，再一条命令完成九场景启发式校准（不进行强化学习训练）：

```powershell
.\run.cmd common/f18_calibrate_normalization.py --all-nine --seeds 101-130 --dry-run
.\run.cmd common/f18_calibrate_normalization.py --all-nine --seeds 101-130 --resume
```

首次在新输出目录运行时省略 `--resume`；本工作区已保留缺失缓存的诊断清单，应使用 `--resume`。默认 101–130 来自训练种子，避开现有验证 seed 1 和默认测试 2–31；如自定义测试集，先核对隔离。禁止在测试集上重拟合尺度。

Linux/Bash 等价命令为 `python launch.py common/f18_calibrate_normalization.py --all-nine --seeds 101-130 --resume`。结果位于 `outputs/f18_normalization_calibration/`；支持单场景、断点续跑、重新汇总及按需详细记录。当前缺少新版正式缓存，九场景正式百分位和归一化参数尚未生成，不能直接开始 F18 正式训练。

所有新模型/日志名追加 `_f18_shared_v1`，Checkpoint 保存源训练场景与配置哈希；评测校验一致性。MS/LS 等泛化继续使用源 S 训练配置。四算法及四消融需要重训，旧模型与结果不覆盖。预计算命令、字段映射、物理轨迹对照、测试边界和英文论文公式修改建议见 [F18 完整说明](docs/F18_shared_state_normalization.md)。轻量集成验证：`.\.venv\Scripts\python.exe -m common.smoke_F18 --out docs/F18_smoke.json`。
## IRWS 公共环境适配版本

正式适配实现位于 `algorithms/irws_paper/`，原 `algorithms/irws/` 和历史模型保留。当前 v2 使用公共能耗—超期外部奖励及相同元学习物理成本，保留双向 DAG/Bi-GRU、全 Ready Softmax、PPO、历史 GRU与元梯度；动态内在奖励采用有界加法补充，任务和VM策略增加当前截止期上下文。共同问题适配与当前验收见 [R03 说明](docs/R03_IRWS_common_objective.md)；[原文对应报告](docs/IRWS_paper_reproduction.md)记录此前阶段，不能据短程验证声称精确复制原论文数值结果。

以下为 IRWS 初次适配时的历史协议，已由下文 F22 默认协议替代。历史公共无重叠协议位于 `configs/paired_cloud_protocol_v1.json`：训练 32–631（600 回合）、验证 632、最终测试仍 2–31。当时新 IRWS 默认使用它；需要复现该历史协议时，各学习算法通过 `--experiment-protocol configs/paired_cloud_protocol_v1.json` 显式启用同一协议。当前不传该参数时，所有学习入口默认使用 F22 协议，不能将两份协议结果混作正式配对比较。选模规则仍为验证总能耗最小。

```powershell
# 最小完整流程：临时缓存、1 Episode、2 工作流；无正式训练或消融
.\run.cmd common/smoke_irws_paper.py --out docs/R03_IRWS_smoke.json
# 正式参数预览
.\run.cmd algorithms/irws_paper/train.py --task S --resource S --deadline Tight --dry-run
```

正式重跑之前，仍须生成新版 Deadline Cache 与有效的 F18 校准配置；当前短程数据不能替代它们。新旧输出目录分离，Checkpoint 校验模型版本、源尺度和实验协议，评测仅使用公共物理指标。

## F22 统一实例划分

所有学习算法、四个 MAHDRL 消融及启发式评测默认使用 `configs/f22_split_v1.json`：训练 0–599（600 Episode）、验证仅 600、测试 700–729（30 个实例）。所有学习入口的 `--model-seed` 默认统一为 1，与实例种子分离；新结果保存协议与实例哈希，旧 Checkpoint 不会被标记为 F22。历史种子评测需显式 `--legacy-split --seeds 2-31`。正式启动会严格审计新版 Deadline Cache；当前九组缓存缺失、F18 校准场景为空，尚不能直接启动正式训练。

验收、修改清单及各算法 PowerShell 命令见 [F22 报告](docs/F22_data_split.md)。仅检查缓存：`.\run.cmd common/f22_cache.py --all`；仅预览配置：`.\run.cmd algorithms/mahdrl/train.py --dry-run`。本次没有运行正式训练或 30 种子正式测试。


## F19 输入尺度与数值诊断

正式学习入口默认采用 `f19_shared_v1`，将 MAHDRL/MARL 及相关消融保留的 I/O ratio 除以 `(input_bits_ref + output_bits_ref) / workload_mi_ref`。参考尺度继续复用 F18 校准文件和源训练场景，未改变网络维度、物理模型或奖励。显式 `--normalization-mode f18_shared_v1` 可复现原 F18 编码，新旧模型版本隔离。

新增按网络字段统计的有界分位数、损失和裁剪前梯度诊断。完整说明、实际输入样本及测试见 [F19 验收报告](docs/F19_numerical_scaling.md)。只运行临时真实输入检查：`.\run.cmd common/smoke_f19.py --out docs/F19_real_input_smoke.json`；不训练策略、不写正式缓存。正式九场景缓存和校准仍需准备。


### TL_2 消融

新增 `algorithms/ablation_tl2/`，在原 TL 两层结构基础上将 VA 的能耗奖励改为全局能耗；原 TL 保留。模型和评测结果独立保存。设计、泛化关系和训练评测命令见 [TL_2 说明](docs/TL_2_ablation.md)。


### 新版 IRWS 的共同目标适配

新版 `irws_paper` 已按公共问题使用能耗—超期外部奖励及同一元学习物理成本，保留图嵌入、PPO、动态内在奖励和元梯度；新增动态截止期上下文，采用独立 `irws_cloud_energy_tardiness_v2` 模型目录。旧目标权重必须重训。设计、类似问题排查和验证见 [R03 适配说明](docs/R03_IRWS_common_objective.md)。


### R16 校准与测试名单检查

默认训练0–599、校准101–130、验证600、测试700–729保持不变。正式评测的 `--seeds` 必须属于所选协议的测试名单；可用其中的子集调试，正式对比应统一使用完整名单。扩展测试集需新建独立版本的协议，不能在原协议中临时换题。

校准新增 `--experiment-protocol`，默认仍为F22。校准、续跑、仅汇总均核对同一协议；校准种子只能来自该协议的训练名单。训练和评测会检查校准文件记录的协议与种子，缺少记录或不匹配时拒绝加载。旧校准文件需在新输出目录重新生成，不能只改标签。命令和验证边界见 [R16说明](docs/R16_protocol_binding.md)。


### 审稿实验记录

训练和评测默认启用 `review_metrics_v1` 记录：训练/验证/保存耗时、各优化器更新和样本预算、验证成功率及超期量；评测另存统一结果、逐工作流明细和决策延迟。Makespan统一为最后完成时刻减首次到达时刻。奖励、随机种子、学习及选模规则不变；计时同步与日志有运行开销。输出路径、离线统计与对照验证见 [记录说明](docs/Review_metrics_logging.md)。

### IRWS 完成提示与可下载文件名

新版 `irws_paper` 训练完成后打印 `[IRWS train completed]`、最佳/最终模型和训练日志的绝对路径；评测完成后打印 `[IRWS eval completed]`、主 CSV、汇总 JSON 和逐种子明细目录。主文件直接携带场景信息，例如 `train_irws_paper_taskS_resS_Tight_model1.jsonl`、`eval_irws_paper_taskS_resS_30seeds_Tight_model1.csv` 和 `summary_irws_paper_taskS_resS_30seeds_Tight_model1.json`。泛化评测在文件名中追加 `_from_taskS_resS_Tight` 等源训练场景信息。目录及 `best.pth` / `final.pth` 名称保持现有规则，已训练的模型仍可加载。

逐种子 CSV 刷新保持不变，汇总和完成提示只在全部种子成功后出现。历史简化 `algorithms/irws` 同步增加完成提示及含算法、场景和模型种子的主日志/CSV 名称。旧 `episodes.csv`、`summary.json`、`training.jsonl` 不自动改名或迁移；更新后新生成的主文件使用新名称。详见 [IRWS 输出说明](docs/IRWS_output_files.md)。

### MAHDRL 性能引导探索与候选筛选

新增两个独立变式：`mahdrl_ge` 在 HA/VA 的 epsilon 探索中以 50% 概率选择预计最快候选；`mahdrl_pf` 在训练和评估中保留预计时长不超过该层最快候选 1.25 倍的动作。HA 按主机内最快可用 VM 比较，VA 按所选主机内的候选 VM 比较，时长复用原计算与通信模型。原 MAHDRL 的算法、模型和结果保持不变，两变式也不互相叠加。设计、输出隔离、检查与训练评估命令见 [GE/PF 说明](docs/MAHDRL_GE_PF.md)。

### MAHDRL 的 OE＋GE 与 OE＋GE＋PF 组合

新增独立的 `mahdrl_oe_ge` 和 `mahdrl_oe_ge_pf`。前者结合 OE 观测/逐分配能耗奖励与 GE 探索；后者再加 PF，按筛选后的候选计算 OE 摘要、执行 GE，并一致限制 TD 后继动作。原 MAHDRL 及 OE/GE/PF/TS/UB 各单独变式保持原样；两组合的模型、日志和评测结果独立，拒绝错版本权重。实现、验证和 50/600 代训练评测命令见 [组合说明](docs/MAHDRL_OE_GE_combinations.md)。

另新增独立的 `mahdrl_oe_pf`，保留 OE 观测/逐分配能耗奖励与 PF 筛选，使用原 epsilon-greedy，不加入 GE 引导探索。原版、各单独变式及已有两个组合保持不变；本版模型与输出独立。实现、检查和训练评估命令见 [OE＋PF 说明](docs/MAHDRL_OE_PF.md)。

### PD3QN-DAG 独立对比算法

新版作为独立对比算法命名为 **PD3QN-DAG**（DAG工作流适配），算法ID为 `pd3qn_dag`、模型版本为 `pd3qn_dag_v1`，入口为 `algorithms/pd3qn_dag/train.py` 和 `evaluate.py`。算法计算沿用此前v5：batch/回放容量/lr/gamma为论文128/10000/0.01/0.95，硬同步60、MSE、PER，无IS/裁剪；TD按式(29)由target取最大合法Q。共享隐层保留指定的1024、1024、512、512、256，输入含父数据和Host历史窗口，SS为112维。公共物理成本、奖励系数、数据划分及best选模不变。

原 `algorithms/pd3qn/train.py`、`evaluate.py` 作为兼容入口保留。显式传入旧 `--normalization-mode f19_shared_v1`、`f18_shared_v1` 或 `legacy` 会选择历史学习实现及旧 best 模型，用于复查已有结果；它们不是新版的归一化消融开关。公共云 DAG、奖励和物理模型保留适配，不能称原始 IIoT 问题完全复现。原文歧义、未披露参数的固定取值、命令及验证见 [PD3QN 实现说明](docs/PD3QN_paper_implementation.md)。

新模型和日志目录前缀为 `pd3qn_dag_`，文件为 `best_pd3qn_dag.pth`、`pd3qn_dag_final.pth`、`train_metrics_pd3qn_dag_cost.csv`；评测输出集中在 `outputs/run/pd3qn_dag/`，主CSV以 `eval_pd3qn_dag_` 开头，明细放在同场景/模型目录内。旧PD3QN和v5文件不动、不自动加载；新训练/评测同名输出已存在时，写入前报错，避免追加混合或覆盖。新旧checkpoint实现标记隔离，需在新目录重训。`pd3qn_v5`入口保留为新算法别名。命令及覆盖规则见[实现说明](docs/PD3QN_paper_implementation.md)。

各算法实际奖励的公式、能耗结算时点/范围、MAHDRL 变式与 IRWS 融合差异见 [奖励设置核查](docs/Algorithm_reward_settings_2026-10-23.md)。

### IRWS-R 独立对比算法

此前修改的IRWS现在使用独立入口`algorithms/irws_r/train.py`、`evaluate.py`，显示名称为**IRWS-R**，算法ID为`irws_r`，模型版本为`irws_r_v1`。新模型使用`best_irws_r.pth`、`final_irws_r.pth`；训练、评测主文件均带`irws_r`前缀，目录按模型种子隔离。新训练及评测入口拒绝覆盖已有结果。旧IRWS目录及结果保留；本次独立命名不改变此前v2的算法计算。命令与文件示例见[IRWS-R输出说明](docs/IRWS_R_output_identity.md)。
