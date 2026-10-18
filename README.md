# MAHDRL 实验代码

`algorithms/` 按算法分目录，每个算法集中保存统一训练/评测入口及专用智能体；环境集中在 `environments/`。场景由命令行参数选择。

```text
algorithms/                    每个算法一个目录
  mahdrl/                      train.py、evaluate.py
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

# 泛化评测：M 任务、M 资源、Medium；以 S 任务训练的模型；只跑种子 2
.\run.cmd algorithms/mahdrl/evaluate.py --task M --resource M --deadline Medium --seeds 2

# 如果模型以 M 任务训练，明确指定 --train-task M
.\run.cmd algorithms/mahdrl/evaluate.py --task L --resource M --deadline Medium --train-task M --seeds 2,5,8

# 预览参数与模型目录，不运行实验
.\run.cmd algorithms/mahdrl/evaluate.py --task L --resource L --deadline Loose --dry-run
```

将路径中的 `mahdrl` 换成其他算法目录名，即可使用相同参数。启发式使用：

```powershell
.\run.cmd algorithms/ertf_fifs/evaluate.py --task M --resource S --deadline Medium --seeds 2
```

| 参数 | 可选值/默认值 | 含义 |
|---|---|---|
| `--task` | S / M / L，默认 S | 工作流规模分布 |
| `--resource` | S / M / L，默认 S | 3 / 6 / 9 个主机，25 / 50 / 75 个 VM |
| `--deadline` | Tight / Medium / Loose，默认 Tight | ddlAlpha=2.0 的概率为 0.8 / 0.5 / 0.2，其余为 3.0 |
| `--episodes` | 正整数，默认 600 | 训练回合数，仅训练入口 |
| `--seeds` | 如 `2-31` 或 `2,5,8`，默认 2-31 | 评测种子，范围含两端 |
| `--train-task` | S / M / L，默认 S | 被评测模型的训练任务规模，独立于评测任务 |
| `--checkpoint-dir` | 目录路径 | 显式选择评测模型目录 |
| `--dry-run` | 开关 | 只打印所选配置 |

S 任务仅使用约 30 节点的五类工作流；M 任务按 30/50 节点组的 3:7 混合；L 任务按 30/50/100 节点组的 2:3:5 混合。每回合仍为 50 个工作流。统一配置位于 `common/scenarios.py`。

MARL 额外保留 `--deadline-cache`、`--ckpt-global`、`--ckpt-local`、`--out` 和 `--device`，可用 `--help` 查看。其余算法自动选择 CUDA 或 CPU。

训练输出在 `outputs/ckpts` 和 `outputs/logs`，评测 CSV 在 `outputs/run`。训练会更新所选场景的模型和日志；需要保留旧实验时，先备份这些目录。评测不会自动搜索并替换为其他场景的模型；缺失模型时会明确报错。

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

所有算法的 episode 能耗统一为：从首个工作流到达到最后一个任务完成，对全部主机积分功耗。未使用主机、各主机任务执行前后的空闲时段，以及工作流之间的空闲间隔均计入；首个到达之前不计入。

公共函数 `energy_from_records(records, hosts, start_time=..., end_time=...)` 接收明确的共同窗口。主算法及消融的总能耗奖励和分主机奖励使用同一窗口；IRWS、PD3QN 的步骤能耗奖励包含分配增量及推进期间全部主机的空闲能耗，`info.r_energy_total` 对应该分量。原有 `r_energy_host` 保留作所选主机分配增量的诊断字段。

能耗窗口回归检查：`.\.venv\Scripts\python.exe -m unittest common.test_energy`。历史模型、日志和统计图表不应与修正后的新结果混用。

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
