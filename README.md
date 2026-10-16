# MAHDRL 实验代码

训练、评测与专用智能体统一平铺在 `algorithms/`，通过算法名前缀区分；环境单独集中在 `environments/`。场景由命令行参数选择。

```text
algorithms/                    算法入口与专用智能体，目录内平铺
  mahdrl_train.py              MAHDRL 训练
  mahdrl_evaluate.py           MAHDRL 评测
  marl_train.py / marl_evaluate.py
  marl_agents.py
  irws_train.py / irws_evaluate.py
  irws_task_ranker.py / irws_vm_agent.py / irws_intrinsic_reward.py
  pd3qn_train.py / pd3qn_evaluate.py
  ablation_*_train.py / ablation_*_evaluate.py
  ertf_fifs_evaluate.py
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
.\run.cmd algorithms/mahdrl_train.py --task S --resource S --deadline Tight

# 切换资源和截止期
.\run.cmd algorithms/mahdrl_train.py --task S --resource L --deadline Medium

# 只训练一回合（会写入该场景的模型与日志）
.\run.cmd algorithms/mahdrl_train.py --task S --resource S --deadline Tight --episodes 1

# 评测：默认使用该资源和截止期下以 S 任务训练的模型，种子 2..31
.\run.cmd algorithms/mahdrl_evaluate.py --task S --resource S --deadline Tight

# 泛化评测：M 任务、M 资源、Medium；以 S 任务训练的模型；只跑种子 2
.\run.cmd algorithms/mahdrl_evaluate.py --task M --resource M --deadline Medium --seeds 2

# 如果模型以 M 任务训练，明确指定 --train-task M
.\run.cmd algorithms/mahdrl_evaluate.py --task L --resource M --deadline Medium --train-task M --seeds 2,5,8

# 预览参数与模型目录，不运行实验
.\run.cmd algorithms/mahdrl_evaluate.py --task L --resource L --deadline Loose --dry-run
```

将文件名中的 `mahdrl` 换成 `marl`、`irws`、`pd3qn` 或相应消融算法前缀，即可使用相同参数。启发式使用：

```powershell
.\run.cmd algorithms/ertf_fifs_evaluate.py --task M --resource S --deadline Medium --seeds 2
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

场景整合后保留 17 个统一训练/评测入口和一个截止期预计算入口。本轮将算法文件按前缀平铺，并将八个环境实现（含两个基础环境）集中到 `environments/`，同时移除已由统一命令覆盖的旧 MARL 场景入口。算法与环境不再按算法各自建目录。

原按算法分目录的版本可通过 Git 提交 `6519152` 查看，旧场景入口可通过 `31d7110` 恢复。历史打包记录保留在本地 `docs/archive`，不进入当前 GitHub 文件列表。

此前场景整合也按用户要求修正了若干历史场景配置，因此部分新旧实验不可直接等同。具体修正及对照规则见 [场景修正说明](docs/scenario_changes.md)。已有模型、原始数据和历史实验输出未被批量改写；部分历史绘图仍使用旧汇总数据，重新出图时应核对实验版本。
