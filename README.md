# MAHDRL 实验代码

项目按职责组织。每个算法的环境、专用智能体、训练和评测入口放在同一个目录中。

```text
algorithms/                算法实现与实验入口
  mahdrl/                  主算法
  ablation_hostrand/       随机主机选择消融
  ablation_mgrrand/        随机管理器消融
  ablation_vmrand/         随机虚拟机选择消融
  ablation_nohost/         去掉主机层的消融
  marl/                    MARL 环境、智能体与入口
  irws/                    IRWS 环境、任务选择器、PPO 与入口
  pd3qn/                   PD3QN 环境与入口
  ertf_fifs/               启发式评测入口
common/                    共享 D3QN、基础环境、工作流、资源、日志与截止期预计算
data/                     实验输入（下方两项）
  dax/                     DAX 工作流 XML
  deadlines/               训练和评测使用的基准 makespan 缓存
plots/                     绘图与统计脚本、Excel 输入表格
outputs/                   本地模型 ckpts、日志 logs、评测 run、图表 figures（不上传）
docs/                      论文、环境版本、目录迁移映射与历史归档 archive
```

## 安装与运行

Windows 下先运行 `setup.cmd` 建立虚拟环境；其他电脑需有可用的 Python，并按 `requirements.txt` 安装依赖。PyTorch/CUDA 版本应与本机设备兼容；当前验证环境见 `docs/runtime_versions.json`。

在项目根目录的 PowerShell 中运行：

```powershell
# 扫描当前全部共享模块、训练和评测入口，验证导入与根目录定位
.\run.cmd verify_package.py --imports-only

# 主算法训练（原设置：600 回合）
.\run.cmd algorithms/mahdrl/train_taskS_resS_Tight.py

# 主算法评测（原设置：30 个种子，需要对应已有模型）
.\run.cmd algorithms/mahdrl/eval_taskS_resS_Tight.py

# 重新生成 SS 场景的截止期缓存（较耗时）
.\run.cmd common/precompute_ddl_fcfs_ss.py

# 绘图示例；启动器自动切换到 plots，以读取相邻 Excel
.\run.cmd plots/algorithm_boxPlot_Merge.py
```

同一算法目录内，`train_*.py` 是训练入口，`eval_*.py` 是评测入口。文件名中 `taskS/M/L` 和 `resS/M/L` 分别表示工作流和资源规模，`Tight/Medium/Loose` 表示截止期设置。保留原有场景入口，避免在目录整理中改变实验参数。

不带参数运行 `run.cmd` 会执行导入检查，并对九种算法分别评测一个 SS/Tight 回合；报告写入 `outputs/verification_result.json`。此检查需要本地 `outputs/ckpts` 下的已有权重。GitHub 中不包含权重，克隆后可先执行 `--imports-only` 检查，再训练或另行获取权重。

训练和评测会更新对应的本地结果，运行前请保留需要对比的模型。部分绘图脚本使用历史汇总值或指定日志，需按实际数据选择脚本与参数。

## 目录整合

- 原 `b_hrl`、`b_hrl_legacy_ave`、`b_hrl_legacy_fcfs` 合并到 `common`。
- 原 `b_marl_FCFS`、`b_irws_FCFS` 并入 `algorithms/marl`、`algorithms/irws`。
- 原 `b_*` 实现与 `c_run` 评测按算法合并到 `algorithms`，不再分散存放。
- 输入统一到 `data`，运行产物统一到 `outputs`；原有模型和缓存内容不变。
- 原提取清单、校验和、名称映射和审计报告在本地 `docs/archive` 保留，不进入当前 GitHub 文件列表；旧版本也可从 Git 历史查看，其路径与哈希不代表当前布局。

逐文件的新旧位置见 `docs/structure_migration.json`。原始版本保存在 Git 提交 `0ef88a2`。算法、奖励、训练超参数与已有权重不因这次目录整理而修改；目录与导入检查通过不等同于严格复现论文。
