# MAHDRL-GE 与 MAHDRL-PF

两个独立变式均从原 MAHDRL 建立，分别检验性能引导探索与性能候选筛选。没有建立两者叠加版本，没有叠加 OE、TS、UB。原算法、物理执行、通信、奖励、TA 排序、网络维度及训练超参数保留。

| 算法 ID | 训练时 | 验证和独立评估时 |
|---|---|---|
| `mahdrl_ge` | 仅 HA/VA 的 epsilon 探索分支增加性能引导 | 原贪心策略，保留全部原合法动作 |
| `mahdrl_pf` | HA/VA 使用性能候选掩码，原 epsilon-greedy 在掩码内选择 | 使用同样的性能掩码，由网络贪心选择 |

## 性能指标

性能依据为当前任务在可用 VM 上的预计完成延迟，即 `max(now, available_at) - now + _task_duration(task, vm)`。`_task_duration` 复用原计算与通信公式，包含输入/输出传输、带宽差异，以及父任务同 VM 的数据复用。只读取当前已知状态，不查询未来到达任务、不执行候选任务、不推进时钟。

- HA：一个合法主机的性能成本为其当前空闲 VM 的最短预计完成延迟。使用原 HA 的可用性阈值和主机索引映射。
- VA：一个合法 VM 槽位的性能成本为该 VM 的预计完成延迟。忙碌 VM 和 padding 始终保持不可选。

没有新增观测维度；性能成本用于探索或动作掩码，不追加进网络状态。不单纯按 PC 大小排序，也不将性能值与 Q 值直接相加。

## GE：性能引导探索

默认引导概率为 **0.5**，仅作用于 epsilon 分支：

```text
概率 1-epsilon：按原 Q 值贪心选择全部合法候选。
概率 epsilon：
    50% 从预计最快的合法候选中选一个；
    50% 从全部合法候选中均匀随机选择。
```

最快候选并列时均匀随机选取。TA 不接受性能参数，沿用原动作选择；三层 epsilon 仍只按成功优化器更新次数衰减。验证和评估使用 `deterministic=True`，完全绕过性能引导，不改变原合法掩码。

## PF：性能候选筛选

默认每层保留 `T <= 1.25 * T_min` 的合法候选，再使用原 epsilon-greedy/贪心选择。等价于允许比该层最快候选慢 25%；保留边界和并列最优，不会把原非法动作置为合法。原 mask 非空时至少保留最快动作。

HA 以各主机最快 VM 比较；VA 以已选主机内的最快 VM 比较。**1.25 是分别作用于两层的阈值，不是最终 VM 相对全局最快 VM 的单一上界。**

筛选在环境状态构造时产生，并同时用于实际动作、回放经验当前 mask 和后继 mask，因此 Double DQN 的后继 argmax 也受相同限制。直接调用环境执行被筛掉的动作会报错。观测、物理积分和奖励仍复用原实现。

## 代码与模型身份

- 两套入口：`algorithms/mahdrl_ge/{train,evaluate}.py`、`algorithms/mahdrl_pf/{train,evaluate}.py`。
- 共用性能计算、引导选择和模型身份检查：`common/mahdrl_performance.py`。
- PF 环境：`environments/mahdrl_pf.py`；GE 直接使用原 `mahdrl_fcfs` 环境。
- `GUIDED_POLICY`、`FILTER_POLICY` 保存上述固定实验参数；本版未增加命令行调参选项。
- 三层 checkpoint 都记录独立 `performance_policy` 标记和具体参数。新入口拒绝原 MAHDRL、另一性能变式或缺少标记的权重，即使网络维度相同也须重新训练。参数也写入 `measurement_metadata.json` 的各 agent `config`。

模型与日志目录分别以 `mahdrl_ge_task...`、`mahdrl_pf_task...` 开头。评测主 CSV 分别为 `eval_mahdrl_ge_task...csv`、`eval_mahdrl_pf_task...csv`，位于各自 `mahdrl_ge_wfInfo__HVrn_<deadline>` / `mahdrl_pf_wfInfo__HVrn_<deadline>` 子目录；统一逐实例测量使用各自算法 ID。不会回退查找原 MAHDRL 模型。

仍拒绝覆盖非空训练目录，不支持在此入口续训。若修改固定实验参数，需要归档该变式自己的旧模型/日志，独立重训；旧 checkpoint 参数不匹配会被拒绝。

## Windows 命令

从项目根目录运行，沿用现有公共缓存、F19 归一化和 F22 划分，无需为这两项改动重建缓存。

```powershell
# 只预览，不训练
.\run.cmd algorithms/mahdrl_ge/train.py --task S --resource M --deadline Tight --dry-run
.\run.cmd algorithms/mahdrl_pf/train.py --task S --resource M --deadline Tight --dry-run

# 两个独立版本，各训练 600 回合
.\run.cmd algorithms/mahdrl_ge/train.py --task S --resource M --deadline Tight --episodes 600
.\run.cmd algorithms/mahdrl_pf/train.py --task S --resource M --deadline Tight --episodes 600

# 加载各自 best 模型评估
.\run.cmd algorithms/mahdrl_ge/evaluate.py --task S --resource M --deadline Tight --seeds 700-729
.\run.cmd algorithms/mahdrl_pf/evaluate.py --task S --resource M --deadline Tight --seeds 700-729

# 泛化示例：使用 SM/Tight 训练模型，评估 MM/Tight
.\run.cmd algorithms/mahdrl_ge/evaluate.py --task M --resource M --deadline Tight --train-task S --seeds 700-729
.\run.cmd algorithms/mahdrl_pf/evaluate.py --task M --resource M --deadline Tight --train-task S --seeds 700-729
```

Linux/已激活 Python 环境等价入口：

```bash
python -m algorithms.mahdrl_ge.train --task S --resource M --deadline Tight --episodes 600
python -m algorithms.mahdrl_pf.train --task S --resource M --deadline Tight --episodes 600
python -m algorithms.mahdrl_ge.evaluate --task S --resource M --deadline Tight --seeds 700-729
python -m algorithms.mahdrl_pf.evaluate --task S --resource M --deadline Tight --seeds 700-729
```

原算法比较命令仍使用 `algorithms/mahdrl/`。方案选择使用训练/验证材料；700–729 已在既有分析中查看，后续最终核验应统一预留未参与选方案的实例。

## 验证范围

```powershell
.\.venv\Scripts\python.exe -X utf8 -B -m unittest common.test_mahdrl_performance common.test_scenarios -q
```

专项检查覆盖引导/随机探索分支、贪心策略保持原样、主机/VM 索引映射、筛选边界/并列/空 mask/padding、checkpoint 往返及错版本拒绝、两个工作流完整配对轨迹的物理能耗与奖励一致性，以及临时目录内的原版/GE/PF 小网络真实训练、三层梯度更新、保存加载、评估输出和原输出保护。

小网络测试只说明实现链路可用。没有启动正式 600 回合训练，不据此宣称节能或成功率提升。

2026-10-22 验证：首轮专项与场景测试共 12 项通过；最终完整回归共 235 项通过（147.261 秒），四个真实 CLI 入口的 dry-run、语法与空白检查通过。原 MAHDRL、物理环境、共享学习/奖励及已有 OE/TS/UB 入口共 20 个受保护源码文件的 SHA256 与实施前完全一致。

完整测试发现既有未提交 `common/evaluate_frozen_ta.py` 在导入时全局设置 `F11_DIAGNOSTICS=0`、`F19_DIAGNOSTICS=0`、`EXPERIMENT_METRICS=0`，导致普通全目录 discovery 的多项诊断/日志测试失败。最终全量验证在 discovery 后、测试执行前，仅在该测试进程中恢复上述开关为 1；未改动该工具或原算法。完整通过记录为 `tmp/mahdrl_performance_regression_enabled.txt`。上面的专项命令不导入该工具，可直接运行。
