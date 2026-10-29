# PD3QN-FE：可行性优先的 PD3QN 选模变式

PD3QN-FE（算法 ID：`pd3qn_fe`）沿用旧 PD3QN 的学习方法，只改变每个训练回合结束后的检查点选择。所有实现都位于新增目录 `algorithms/pd3qn_fe/`；没有修改或注册到既有算法入口、公共配置、环境或学习器。变式不调用旧训练/评测入口，也不替换它们的全局变量。

## 学习配置与数据

直接复用 `algorithms.pd3qn.config.agent_kwargs` 的历史配置、`common.d3qn_agent_PER.D3QNAgent`、`environments.pd3qn.CloudWorkflowEnv_PD3QN_Cost` 和公共 F11 经验结算。默认 `f19_shared_v1`、`paper_mode=False`；允许显式使用 `f18_shared_v1` 或 `legacy` 观测，拒绝 PD3QN-DAG 的 `pd3qn_units_v1`。

历史配置为学习率 0.0003、折扣 0.99、batch 256、回放容量 120000、隐层 1024/1024/512/512/256、tau 0.005、梯度裁剪 10，沿用 PER 和探索计划。warmup 为 `max(batch_size, int(buffer_size * 0.03))`，即 3600 条经验。训练动作、真实下一决策经验和每步更新顺序沿用旧 PD3QN。奖励系数、物理能耗、截止期和每实例 50 个工作流均保持一致。

默认数据协议为 `configs/f22_split_v1.json`：训练种子 0–599，验证种子 600，测试种子 700–729，600 个训练回合。不允许重叠的 legacy 数据划分。观测模式 `legacy` 与数据划分无关。

## 单成功率门槛选模（2026-10-29）

每回合结束后，在固定验证集上执行确定性贪心策略，不更新模型、回放或探索计数；验证前后恢复 Python、NumPy、Torch CPU/CUDA 随机状态及网络模式。

总体工作流成功率 SR 为所有验证实例按时完成数之和除以工作流总数之和，未完成工作流计为失败。使用公共 `episode_metrics`，完成时间不晚于截止期（容差 1e-12）视为成功。总体 `SR >= 0.92` 时模型可行，不计算 ESP，也不要求每个实例分别达到门槛。平均能耗是实例总能耗的算术平均。

可行模型永远优于不可行模型。可行模型按平均能耗更低、总体成功率更高排序；不可行模型按总体成功率更高、平均能耗更低排序。只在严格更优时更新，所以完全相同保留较早检查点。规则版本为 `select_sr092_minenergy_v1`。这些是验证经验指标，不构成总体概率保证。

当前只有 seed600 一个含 50 个工作流的验证实例，46/50=0.92 达标，45/50=0.90 不达标；删除只有 0/1 两种取值的冗余 ESP。历史双 99% 规则被新规则替代，学习配置、奖励、预算和数据划分保持。此修改仅作用于 `pd3qn_fe`，PD3QN-DAG-FE、R090、R010 及其他算法的选模规则不随之改变。

## 独立产物与加载

检查点与日志目录使用 `pd3qn_fe_..._select_sr092_minenergy_v1`，包含训练场景、模型种子、观测模式、数据协议哈希和能耗版本。训练拒绝复用非空目录。评测源模型路径同样绑定新规则，旧双 99% 规则检查点会因选模元数据不匹配而拒绝加载，需要按新规则重新训练；旧结果保留。

- `best_pd3qn_fe.pth`：最优可行模型；全程不可行时为最优备用模型。
- `pd3qn_fe_final.pth`：最后回合的模型，不用于默认测试。
- `train_metrics_pd3qn_fe_cost.csv`：逐回合训练指标、验证能耗、总体成功率、可行状态、是否选中、最佳回合（从 0 开始），不再包含 `eval_ESP`。
- `validation_selection.jsonl`：每回合的逐实例验证明细、候选与当前最佳指标，可独立复算选模。
- `training_summary.json`：选模规则、实际划分、最佳回合和 `feasible_model_found`。

复用现有 Agent 保存器，将变式身份、规则、划分、模型种子、源训练场景、回合、检查点角色和验证明细放入检查点的 `normalization.pd3qn_fe`。专属加载器先验证这些信息，再由公共加载器验证学习实现和归一化。原 PD3QN、PD3QN-DAG 或错误规则的模型不能被此入口误认。

评测结果位于 `outputs/run/pd3qn_fe/<源模型标识>/<目标场景与种子>/`，主CSV名包含目标场景、模型种子和测试种子，例如 `eval_pd3qn_fe_taskS_resS_Loose_model1_seeds700-729.csv`；同目录还包含 `summary_pd3qn_fe.json` 及公共工作流/时间测量明细。模型源场景与目标场景分开标识，已有非空评测目录拒绝覆盖。测量日志通过 `EXPERIMENT_METRICS=0` 关闭时，选模仍直接从环境计算。

不支持从检查点恢复训练；旧 PD3QN 的历史 best 不能恢复未保存的逐回合模型。正式比较应在新目录重训。

## 命令

九个源模型（SS/SM/SL × Tight/Medium/Loose）的9条训练命令和9条分组评测命令见 [训练和评测指令](Linux_CUDA_PD3QN_FE_九场景训练和评测指令.txt)。每场景训练600回合；每条分组评测命令依次执行任务规模S/M/L，得到1个原训练规模结果和2个泛化结果，合计27个目标场景，各评测固定测试种子700–729。可单独使用 [27场景共用评测指令](Linux_CUDA_PD3QN_FE_27场景共用评测指令.txt)。评测显式绑定对应的源训练场景，三种目标结果保存在不同目录。

Windows 训练和评测（仓库根目录执行）：

```powershell
.\run.cmd algorithms/pd3qn_fe/train.py --task S --resource S --deadline Tight --episodes 600 --model-seed 1 --normalization-mode f19_shared_v1
.\run.cmd algorithms/pd3qn_fe/evaluate.py --task S --resource S --deadline Tight --train-task S --seeds 700-729 --model-seed 1 --normalization-mode f19_shared_v1
```

Linux/CUDA：

```bash
python -u launch.py algorithms/pd3qn_fe/train.py --task S --resource S --deadline Tight --episodes 600 --model-seed 1 --normalization-mode f19_shared_v1 --device cuda
python -u launch.py algorithms/pd3qn_fe/evaluate.py --task S --resource S --deadline Tight --train-task S --seeds 700-729 --model-seed 1 --normalization-mode f19_shared_v1 --device cuda
```

将 `--task`、`--resource`、`--deadline` 换成所需场景。跨场景评测时显式提供 `--train-task`、`--train-resource`、`--train-deadline`，或通过 `--checkpoint-dir` 指定模型目录；仍校验模型的源场景、模型种子与规则。默认共享观测需要已有的训练校准文件与截止期缓存，复用原项目的生成流程。

上述训练/评测命令添加 `--dry-run` 可只打印身份、配置、划分与输出路径。默认 CUDA 可用时使用 CUDA，否则使用 CPU；显式 `--device cuda` 不会静默回退。

## 验证

```powershell
.\.venv\Scripts\python.exe -m unittest common.test_pd3qn_fe -v
```

测试覆盖包含等号的 92% 门槛、总体 SR 聚合、能耗与备用模型排序、平局、错误数据、路径和协议隔离、DAG 模式拒绝、随机状态恢复、分区约束，以及真实环境中的两回合训练、梯度更新、保存加载、旧双 99% 规则拒绝、无 ESP 日志和 seed 700 测试。烟雾检查在临时目录中使用测试专属的小网络（16/16）、batch 8 和容量 128，以快速验证完整流程；正式入口仍使用上述完整历史配置。烟雾检查不证明收敛或正式实验的可行性。

2026-10-26 验证结果：变式专属 16 项测试全部通过；原有 `common.test_pd3qn`、`common.test_pd3qn_paper`、`common.test_scenarios` 共 33 项回归测试全部通过。默认 F19 校准与缓存预检查通过；训练和跨场景评测入口的 dry-run 通过。实施前后 466 个既有受版本控制文件的 SHA-256 全部一致。本次只新增变式、专属测试和本文档，没有执行 600 回合正式实验。
