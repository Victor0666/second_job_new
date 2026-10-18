# F17：TA 归一化权重转移修复

基线提交：`b0018b16003275e38eccb8e9f936b2e1f1390d63`。修改仅涉及 MAHDRL 与共享 TA 环境的四种消融，不改变其他对比算法。本报告依据用户给定的 Algorithm 2 规则；未重新核对原论文 PDF。

## 状态及转移

旧实现持续累加 `manager_raw_w`，将负数截断后再计算排序用权重；相同归一化观测可能因隐藏 Raw 总量不同而产生不同转移。现在删除实际学习环境中的 Raw 字段及 `_normalize_manager_raw()`，唯一可变决策权重是 `combo_weights`。

令 `u = w + delta`，仅当候选各维非负、总量正且有限、归一化结果有限时执行 `w_next = u / sum(u)`。Mask 和执行接口共同调用 `common.ta_weights.weight_candidates`。负数容差统一为 `1e-8`，仅该范围内的舍入负值归零；归一化总和校验使用 float64 累加，绝对容差 `1e-7`。实质负权重明确抛出 `ValueError`，权重和当前 phase 均不变。

例：均匀权重增加第一维 0.1，得到 `[0.272727, 0.181818, 0.181818, 0.181818, 0.181818]`；第一维 0.05 减少 0.1 被 Mask 和执行接口同时拒绝。全零增量保留原有编号，精确保留当前 float32 权重，不增加第 244 个动作。

默认构造和每次 reset 都使用 `[0.2]*5`。这明确替换了旧 reset 的固定随机 Raw 向量。显式构造权重经严格检查并归一化，保存为不可变的 reset 配置；这不是另一套可变决策状态。`set_manager_combo` 只设置当前权重，不改变构造时的 reset 配置。非法、全零、非有限或负数外部输入报错；归一化先按最大值缩放，避免大数溢出。

TA 状态返回独立数组；动作、Mask、排序都读取同一权重。保持原有 phase 失效机制、243 个动作及枚举顺序、0.1 步长、状态维度、D3QN 和奖励不变。F15 的五项属性、Ready 集合归一化与平局规则保持不变。HA/VA 策略及物理仿真代码未改；调度结果可以随 TA 决策变化。

## 调用链与文件

- `common/ta_weights.py`：版本标识、外部权重归一化、Mask/执行共用的候选检查。
- `environments/mahdrl_heft.py`：构造、reset、setter、delta、Mask；状态与排序继续读取 `combo_weights`。
- `common/scenarios.py`：仅主算法及四种 TA 消融的模型与日志名称追加 F17 标识。
- `algorithms/{mahdrl,ablation_hostrand,ablation_mgrrand,ablation_vmrand,ablation_nohost}/evaluate.py`：评测结果增加 F17 子目录。
- `common/test_f17.py`：T01–T11 与版本/共享环境检查，共 12 项；`common/test_f15.py`：仅替换 T10 过时的 Raw Mask 测试条件，F15 属性断言不变。
- `common/smoke_F17.py`：原训练入口短程验证；`docs/F17_tests.txt`、`docs/F17_smoke.json`：实际运行证据；`README.md`：运行说明。

仓库搜索覆盖权重读写、训练、评测、消融和保存/恢复调用。五种训练及评测入口均继承此修复；随机 TA 消融也从合法增量选择，固定 FCFS 启发式策略仍保持其固定排序。`environments/base_heft.py` 中的历史 Raw 接口保留：该文件用于参考/预计算路径，不是上述 TA 学习环境，修改它会扩大本次范围。F17 不修改缓存生成、工作流、截止期、通信或能耗。

## 模型与结果隔离

版本标识为 `f17_normalized_weights_v1`。默认 checkpoint 和训练日志名称在 F11 名称后追加该标识；五种算法的评测结果进入 `outputs/run/f11_real_decisions_v1/f17_normalized_weights_v1/`。其他算法路径不变，旧权重和结果均保留。

网络输入输出及 checkpoint 张量格式未变。现有保存接口不保存环境 Raw 状态，也未发现训练入口恢复独立 Raw 环境状态。显式指定旧格式兼容 checkpoint 仍可能读取网络参数（原有 F11 检查继续有效），但这不代表它经过 F17 训练，不能作为 F17 正式结果或等价续训。正式结果应从新环境重新训练；不能继续沿用旧 Raw 状态转移。

## 验证结果

命令：

```powershell
.\.venv\Scripts\python.exe -m unittest common.test_f17 common.test_f15 -v
.\.venv\Scripts\python.exe -m unittest discover -s common -p "test_*.py" -v
.\.venv\Scripts\python.exe -m common.smoke_F17 --out docs/F17_smoke.json
```

专项及 F15 共 26 项通过；完整回归 101 项通过、0 失败，耗时 19.500 秒。覆盖 300 次连续合法转移、19 个状态逐项枚举 243 个动作、状态快照、数值边界、非法输入、显式初始化和 reset，以及真实动态工作流训练和评测。

五个原始 train() 入口各运行 1 episode，使用真实 SS 动态实例（50 工作流），临时生成 seed 1–2 的缓存，包含入口原有的训练内验证。仅验证脚本缩小网络为 `(16,8)`、batch 为 4、buffer 为 32，并使用单线程 CPU；生产参数未改。这是流程正确性检查，不证明收敛或性能提升。所有保留学习器均发生梯度更新，pending 均为 0，未出现全非法 Mask 或死循环。

| 原训练入口 | 训练及验证权重动作检查次数 | reset 检查次数 | 最终 pending |
|---|---:|---:|---:|
| mahdrl | 691 | 4 | 0 |
| ablation_hostrand | 728 | 4 | 0 |
| ablation_mgrrand | 722 | 4 | 0 |
| ablation_vmrand | 699 | 4 | 0 |
| ablation_nohost | 676 | 4 | 0 |

## Section IV-B2 英文 LaTeX 建议

```latex
The task agent maintains a single normalized weight vector
$\mathbf{w}_t\in\{\mathbf{w}\in\mathbb{R}_{\geq 0}^{5}:
\sum_{i=1}^{5}w_i=1\}$, initialized to $w_i=0.2$.
Each of the $3^5=243$ actions specifies an increment
$\Delta\mathbf{w}(a)\in\{-0.1,0,0.1\}^{5}$.
For each action, we first form
$\mathbf{u}(a)=\mathbf{w}_t+\Delta\mathbf{w}(a)$.
An action is admissible only if every component of $\mathbf{u}(a)$
is nonnegative and its total mass is positive, with finite candidate
and normalized values. Inadmissible actions are excluded by an action mask.
For an admissible selected action, the update is
\begin{equation}
\mathbf{w}_{t+1}=\frac{\mathbf{w}_t+\Delta\mathbf{w}(a_t)}
{\sum_{i=1}^{5}(w_{t,i}+\Delta w_i(a_t))}.
\end{equation}
The observation, action mask, and task-ranking score use this same weight
vector. The zero-increment action is always admissible and leaves the
weights unchanged; hence, the admissible action set is never empty.
This guarantees feasibility of the weight update, rather than optimality
of the resulting task order or improvement in energy consumption at every
step. D3QN uses learned Q-values and subsequent environment feedback to
learn a weight-adjustment policy. The implementation uses a tolerance of
$10^{-8}$ for roundoff-sized negative candidate components.
```

## Reviewer 1 Comment 3 英文回复草稿

Thank you for requesting clarification of invalid-increment filtering and edge cases in weight normalization. We have revised the implementation and propose the following clarification for Section IV-B2. The task agent maintains a single nonnegative weight vector with unit sum. For each of the 243 candidate actions, the mask checks the actual candidate obtained by adding the increment to the current normalized weights. A candidate is excluded if it produces a negative component beyond the numerical tolerance, nonpositive total mass, or nonfinite candidate or normalized values. Mask construction and action execution share the same validation routine; invalid direct calls are rejected without changing the weights.

The original action set includes the zero-increment action. This action always preserves a valid weight vector and therefore ensures that at least one admissible action exists, including at boundary states. We distinguish this feasibility guarantee from optimization performance: an admissible normalization can still yield a suboptimal task order, and neither the mask nor normalization guarantees a reduction in energy consumption at each step. D3QN uses learned Q-values and subsequent scheduling feedback to learn which admissible adjustments are useful. We have also corrected the implementation to update the current normalized vector directly, rather than a hidden accumulated raw vector, and added boundary, sequential-update, mask-consistency, and short training/evaluation tests. Full experiments must be rerun using the corrected environment; the short checks do not establish improved scheduling performance.

## 最终验收

| 检查项 | 预期结果 | 实际结果 |
|---|---|---|
| 持久化权重 | 单一归一化权重 | 通过：仅 combo_weights 为可变决策状态 |
| 默认初始值 | 五维均为 0.2 | 通过：构造与 reset 一致 |
| 权重更新 | Norm(w+Δw) | 通过：公式及数值例测试 |
| Raw 隐藏状态 | 不再参与后续更新 | 通过：学习环境无 Raw 依赖 |
| Action Mask | 检查实际候选合法性 | 通过：与执行共用函数 |
| 非法动作 | 拒绝，不静默截断 | 通过：ValueError，状态不变 |
| 全零动作 | 始终合法 | 通过：原动作，精确不变 |
| TA 观测 | 同一归一化权重 | 通过：独立快照 |
| 任务排序 | 同一归一化权重 | 通过：feats @ combo_weights |
| 动作数量 | 243 | 通过：枚举及顺序不变 |
| 增量步长 | 0.1 | 通过：动作表不变 |
| 连续更新测试 | 通过 | 通过：300 次 |
| 浮点边界测试 | 通过 | 通过：零、近零、近一、非有限及大/小外部输入 |
| 真实环境短程测试 | 通过 | 通过：五个原训练入口及验证 |
| HA/VA 行为 | 保留原有算法 | 通过：未修改网络、奖励和分配算法 |

F17 当前无已知未解决项，具备开展正式重新训练的代码条件；仍须满足此前 F07 等修改所要求的新版缓存准备条件。此结论不等于其他审查项全部解决。没有运行全量正式训练，没有删除旧结果，本次尚未提交或推送。
