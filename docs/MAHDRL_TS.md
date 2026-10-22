# MAHDRL-TS：TA 稳定性测试变式

算法 ID：`mahdrl_ts`。训练、评测入口从原 `algorithms/mahdrl` 复制，基线与 MAHDRL-OE 的代码和输出均不覆盖。

## 本次修改

| 参数 | 原 MAHDRL | MAHDRL-TS |
|---|---:|---:|
| TA 学习率 `lr_mgr` | 0.0003 | 0.0001 |
| TA 目标网络软更新系数 `tau_mgr` | 0.01 | 0.005 |

本变式对应讨论中的组合方案 C。TA 的 15 维观测、243 个动作、奖励、决策边界、网络、gamma=0.95、batch=256、buffer=60000、预热 1800 条、epsilon 退火、Huber 损失和梯度裁剪 10 均沿用原实现；HA/VA 不变。直接复用原 `environments.mahdrl_fcfs` 和 `common.d3qn_agent_EXDehid`。

没有启用未来价值非正约束，没有改变动作空间、就绪队列、到达率、资源、观测或奖励，也没有叠加 OE 修改。调参是否能抑制异常并降低能耗，需要正式实验验证。

## 训练与评测指令

从项目根目录的 PowerShell 运行：

```powershell
# 预览配置与模型目录，不执行训练
.\run.cmd algorithms/mahdrl_ts/train.py --task S --resource M --deadline Tight --episodes 50 --dry-run

# 短程筛查：SM / Tight，50 回合
.\run.cmd algorithms/mahdrl_ts/train.py --task S --resource M --deadline Tight --episodes 50

# 完整训练：SM / Tight，600 回合（与上面的短程训练二选一，见下方说明）
.\run.cmd algorithms/mahdrl_ts/train.py --task S --resource M --deadline Tight --episodes 600

# 完整训练：SM / Loose，600 回合
.\run.cmd algorithms/mahdrl_ts/train.py --task S --resource M --deadline Loose --episodes 600

# 评测：SM / Tight，默认加载本变式的 best 模型，30 个测试实例
.\run.cmd algorithms/mahdrl_ts/evaluate.py --task S --resource M --deadline Tight --seeds 700-729

# 泛化评测：MM / Tight，使用 S 任务训练的本变式模型
.\run.cmd algorithms/mahdrl_ts/evaluate.py --task M --resource M --deadline Tight --train-task S --seeds 700-729
```

Linux 或已激活 Python 环境时，可用 `python -m algorithms.mahdrl_ts.train ...` 和 `python -m algorithms.mahdrl_ts.evaluate ...`，参数相同。截止期缓存、归一化和实例划分继续使用当前公共协议，无需为该变式重新生成缓存。

50 回合与 600 回合训练使用同一个场景目录，不能依次直接运行。训练沿用原有“拒绝覆盖非空输出目录”的保护；短程结束后，应先将 **MAHDRL-TS 自身的模型和日志目录**另行归档，再从头进行正式训练。此入口不支持续训；本次没有改变公共的多模型种子目录命名规则。

## 输出隔离

- 模型：`outputs/ckpts/mahdrl_ts_task.../`。
- 训练日志：`outputs/logs/mahdrl_ts_task.../`，测量元数据中的算法名为 `mahdrl_ts`。
- 评测 CSV：`outputs/run/<版本>/<学习协议>/<TA权重协议>/mahdrl_ts_wfInfo__HVrn_<截止期>/eval_mahdrl_ts_task...csv`。
- 逐实例测量：`outputs/run/<测量版本>/mahdrl_ts/<输出版本>/<场景>/`。

完整模型和日志目录可通过 `--dry-run` 预览。评测默认只查找本变式目录，不回退到原 MAHDRL 或 OE。显式使用 `--checkpoint-dir` 时，也会检查加载后 TA 的学习率和 tau，拒绝把原配置模型误标为 TS 结果。网络结构相同不代表旧模型已经经历新的训练过程。

## 诊断与验证

保留 `f11_diagnostics.jsonl` 中 TA 的近期 `Q_selected`、`TD_target`、`TD_error`、loss 和梯度统计。梯度裁剪比例及历史 max 是累计统计，应结合近期分位数和相邻回合计数差判断，不能仅看历史最大值。`measurement_metadata.json` 与 checkpoint 保存实际学习率、tau；验证种子用于选模型，最终测试实例不用于调参。

回归命令：

```powershell
.venv\Scripts\python.exe -X utf8 -B -m unittest common.test_mahdrl_ts common.test_scenarios common.test_f17 common.test_mahdrl_oe -q
```

新增测试在临时目录内，以 2 个工作流、`(16, 8)` 小网络、1 回合训练检查三层真实梯度更新、TA 参数差异、原输出保留、模型保存/加载、评测及日志隔离，并拒绝加载原配置 TA 模型。该测试只验证实现链路，不代表真实网络已经稳定收敛。

本次上述回归共 25 项通过。修改前后对原 MAHDRL、MAHDRL-OE、对应环境及共享更新/诊断实现共 14 个 Python 文件核对 SHA256，内容一致。未执行正式 50/600 回合实验。
