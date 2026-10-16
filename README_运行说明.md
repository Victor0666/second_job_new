# MAHDRL 论文相关代码可运行提取版

这是从原项目提取的独立副本，包含 MAHDRL 主算法、四种消融、PD3QN/MARL/IRWS/ERTF-H-FIFS 对比、DAX、截止期缓存、已有模型、相关日志/评测结果和绘图数据。未包含 Cluster/DRRN 扩展和 sh 述职图表。

## 在当前电脑运行

1. 双击 `setup.cmd`，建立本目录的 `.venv`。当前电脑使用已安装依赖的 `D:\Python3.12\python.exe`，虚拟环境通过 system-site-packages 复用它；不依赖原项目文件夹或原项目的空虚拟环境。
2. 双击 `run.cmd`，执行模块导入检查并对主算法、四种消融、三种学习基线和启发式基线各跑一个 SS/Tight 评测回合（每回合 50 个工作流、seed=2）。GPU 可用时自动使用 GPU，预计需要数分钟。
3. 详细结果在 `verification_result.json`；完整复制校验在 `copy_verification.json`。

PowerShell 中可执行：

```powershell
.\run.cmd verify_package.py --imports-only
.\run.cmd c_run/mahdrl/eval_taskS_resS_Tight.py
.\run.cmd b_mahdrl/train_taskS_resS_Tight.py
```

第二条执行原评测入口的 30 个种子，第三条执行原训练入口的 600 回合。重新训练会按原脚本规则更新本副本中的日志和模型，请先备份需要保留的结果。其他规模和 Tight/Medium/Loose 入口位于相同目录。

绘图可使用 `run.cmd e_draw_plot/对应脚本.py`；启动器会为绘图切换至脚本目录，以读取同目录 Excel 文件。绘图/统计脚本有历史版本和硬编码汇总数据，未逐一重新生成全部图表。

## 文件组织

- `b_mahdrl`：论文主分支。
- 同名前缀的 `_MGRRAND/_HOSTRAND/_VMRAND/_NoHOST`：四种消融。
- `b_pd3qn`、`b_marl`、`b_irws`：所选学习对比算法。
- `b_hrl`、`b_marl_FCFS`、`b_irws_FCFS` 及少量旧环境目录：实际导入依赖，不能删除。
- `a_set`、`a_dax`、`a_deadlines`：共享模型、数据和九种规模的截止期预计算。
- `c_run`：主算法、消融、对比算法的评测入口。
- `d_output/deadlines`：运行所需的基准 makespan 缓存。
- `d_output/ckpts` 等：所选入口引用的已有模型；日志和历史评测结果一并保留。
- `e_draw_plot`：实验图表、Excel 数据和统计脚本。
- `docs/paper.pdf`：用户指定的论文。
- `output/paper_code_audit`：原项目核对报告和全项目分类清单；这是原项目审计记录，部分链接指向未纳入本副本的历史/扩展文件。

## 运行依赖与可迁移性

`runtime_versions.json` 记录验证所用的 Python 3.12 和包版本。`requirements.txt` 用于新机器重建依赖。当前验证使用 PyTorch 2.5.1+cu121；新机器的 CUDA/PyTorch 版本须与设备兼容。`setup.cmd` 在依赖缺失时会尝试 pip 安装；当前电脑已有依赖，不需要下载。

虚拟环境仍依赖本机的基础 Python 安装，这是正常的 Windows venv 行为。搬到其他电脑时重新运行 setup，勿直接搬运 `.venv`。

## 仅为可运行搬迁做的改动

`packaging_changes.json` 记录 54 个评测脚本的两类路径修复：PD3QN 根目录识别加入实际使用的 nr 目录；ERTF-H-FIFS 的 ROOT_DIR 向上定位两层。算法公式、奖励、训练样本和权重均未修改。

`extraction_manifest.json` 记录从原项目提取的文件，`package_checksums.json` 记录交付包文件的 SHA-256。验证入口检查模块和根目录必须位于本副本中，防止误读原项目。

“可运行”不等同于“严格复现论文”。此前指出的下层 done=1、排序特征、能耗统计窗等算法偏差仍保留，详见核对报告。验证不重新执行全部 600 回合训练，也不复算全部 27 场景×30 次实验。


## 目录精简（2026-10-16）

目录、导入及路径引用已同步修改，新旧对应如下：

- `a_precompute_ddl_fcfs_fifs` → `a_deadlines`
- `b_hrl_tri_mgc_ave` → `b_hrl_legacy_ave`
- `b_hrl_tri_mgc_ave_ddlFCFS` → `b_hrl_legacy_fcfs`
- `b_hrl_tri_mgc_ave_ddlFCFS_mgrn_ddlMix_HVrn` → `b_mahdrl`
- `b_hrl_tri_mgc_ave_ddlFCFS_mgrn_ddlMix_HVrn_HOSTRAND` → `b_ablation_hostrand`
- `b_hrl_tri_mgc_ave_ddlFCFS_mgrn_ddlMix_HVrn_MGRRAND` → `b_ablation_mgrrand`
- `b_hrl_tri_mgc_ave_ddlFCFS_mgrn_ddlMix_HVrn_NoHOST` → `b_ablation_nohost`
- `b_hrl_tri_mgc_ave_ddlFCFS_mgrn_ddlMix_HVrn_VMRAND` → `b_ablation_vmrand`
- `b_irws_mix_nr` → `b_irws`
- `b_marl_FCFS_ddlMix` → `b_marl`
- `b_pd3qn_ddlFCFS_alpha075_ddlMix_nr` → `b_pd3qn`
- `c_run/FCFS_FIFS_ddlMix` → `c_run/ertf_fifs`
- `c_run/hrl_tri_ddlFCFS_alpha075_FCFS_MIX_Alpha_HVrn` → `c_run/mahdrl`
- `c_run/hrl_tri_hostRand_MIX_Alpha_HVrn` → `c_run/ablation_hostrand`
- `c_run/hrl_tri_mgrRand_MIX_Alpha_HVrn` → `c_run/ablation_mgrrand`
- `c_run/hrl_tri_moHost_MIX_Alpha_HVrn` → `c_run/ablation_nohost`
- `c_run/hrl_tri_vmRand_MIX_Alpha_HVrn` → `c_run/ablation_vmrand`
- `c_run/irws_FCFS_Mix_Alpha_rn` → `c_run/irws`
- `c_run/marl_FCFS_Mix_Alpha` → `c_run/marl`
- `c_run/pd3qn_FCFS_Mix_Alpha_rn` → `c_run/pd3qn`
- `e_draw_plot/ablation_main_heatmap_and_supp_boxplots` → `e_draw_plot/ablation_plots`
- `e_draw_plot/ablation_main_heatmap_and_supp_boxplots_new` → `e_draw_plot/ablation_plots_new`
- `e_draw_plot/scenario_boxplots_from_excel_adaptive_y` → `e_draw_plot/boxplots_adaptive`
- `e_draw_plot/d_output/plots/pd3qn_taskS_resL_warmup_sl` → `e_draw_plot/d_output/plots/pd3qn_training`

改名前清单中已有 5079 个文件缺失，包括模型和历史输出，本次未补回。旧验证报告是历史记录。本次结果见 `docs/folder_rename_validation.json`，原文件备份见 `docs/folder_rename_backup.zip`，完整映射见 `docs/folder_renames.json`。提取清单保留原始哈希；package_checksums.json 对应精简后实际文件。


## 恢复文件后的名称精简（2026-10-16）

已处理找回的 d_output 模型、日志和评测结果，并精简训练、评测、环境及绘图文件名。当前入口、缩写含义、完整对照与备份见 [名称精简说明](docs/名称精简说明.md)。上节缺失文件数是恢复前的历史状态，当前缺失情况见 docs/name_shortening_validation.json。
