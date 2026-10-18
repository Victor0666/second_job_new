# F19 剩余问题修复验收

F19 代码修复完成，正式学习入口默认采用 `f19_shared_v1`。F22 数据划分保持训练 0–599、单验证 600、测试 700–729；全部学习入口默认模型种子为 1。未启动正式训练，未改写正式缓存、校准参数、历史模型或实验结果。

## 归一化修复

原字段 `r=(in_bits+out_bits)/max(mi,1.0)` 的单位为 bits/MI。MAHDRL、MARL 和复用该字段的消融现在使用：

```text
ratio_ref = (input_bits_ref + output_bits_ref) / workload_mi_ref
normalized_ratio = r / ratio_ref
```

共享参考来自源训练场景的 F18 冻结尺度，没有使用验证/测试实例重新估计。无需新增校准参数，未增加 log 变换、裁剪或修改原分母规则。ratio_ref 不是 ratio 的 P95，结果也不要求在 [0,1]。

MAHDRL 和 MARL 的任务块仅第 4 个字段改变，其他字段顺序、维度及网络不变。PD3QN 和旧 IRWS 延用已有逐字段缩放；paper IRWS 不新增观测字段。调度、奖励、学习公式、更新频率、截止期、物理执行、通信及能耗规则保持原实现。

## 版本与历史兼容

新编码版本为 `f19_shared_v1`，参考尺度文件仍可使用兼容的 F18 配置。Checkpoint 同时记录编码版本、尺度文件版本/hash、ratio 公式标识和参考值，加载时校验。新模型和结果目录与 F18 隔离；paper IRWS 显式使用 F18 时保留原路径规则。

历史模型应显式传入 `--normalization-mode f18_shared_v1` 或其实际 Legacy 编码，同时匹配原数据协议、源场景和模型目录。不能将旧模型改标签作为 F19 模型。

## 数值诊断

新增 `common/f19_numerics.py`，读取 detached 张量，不消耗 RNG、不参与反向传播、不修改输入或梯度。输入 hook 返回 None。

| 范围 | 记录内容 |
|---|---|
| 各网络实际输入 | 每字段 P1/P50/P95/P99、max_abs、非有限值计数 |
| I/O ratio | 原始 bits/MI 与归一化值 |
| DQN | Online/Target 输入、选定 Q、TD target/error、损失及裁剪前梯度范数 |
| MARL Global/Local | 输入、Value、Return、Advantage、损失及裁剪前梯度范数 |
| 旧 IRWS PPO | 输入、Value、Return、原始/实际使用的 Advantage、损失及梯度 |
| 旧 IRWS Task/Intrinsic | 实际 forward 输入，不新增旧实现原本没有的学习更新 |
| paper IRWS | Rank、Selector embedding、Actor/Critic/RNN 输入、Return/Advantage、四目标损失和各组梯度 |

分位数采用最近 256 行的有界窗口，不是全训练集精确分位数；max_abs、非有限值和行数从诊断对象创建或 reset 起累计。字段索引对应原网络输入顺序，包含实际零填充。网络输入统计可包含更新估值及验证前向，不是独立实例计数。

梯度在原裁剪前读取，阈值和裁剪调用不变。clipped_ratio 表示超过已有阈值的比例；无裁剪的 paper IRWS 阈值为 null，未新增裁剪。

既有算法写入 `f11_diagnostics.jsonl` 的独立 numerics 字段；ratio 位于 normalization.numerics。旧 IRWS Task/Intrinsic 使用独立日志键。paper IRWS 每回合 reset 统计，在回合结果中记录完整 profile，更新结果保留原损失/梯度并补充 Critic 范数。

可用 `F19_DIAGNOSTICS=0` 关闭新增统计；未单独设置时沿用 F11 开关。GPU 数据读取会有诊断开销，未做 CUDA 性能测试。

## 测试与实际证据

- `F19_regression_tests.txt`：完整回归 157 项通过，75.116 秒。
- `F19_tests.txt`：末次 F19 边界及 checkpoint 聚焦测试 12 项通过。
- 真实 MAHDRL/MARL 特征仅 ratio 字段改变，显式 F18 保留原值。
- 关闭/开启诊断时，DQN、MARL Global/Local、旧 PPO 和 paper IRWS 四组网络的更新后参数逐项相等。
- 固定任务–VM动作时，F18/F19 的任务结束时间、能耗一致。

`common/smoke_f19.py` 只在临时目录为训练 seed=0 生成兼容参考缓存，用 50 个工作流估计 SS_T 临时参考，再对两个工作流固定重放。小型未训练网络只做前向，零次学习更新。结果保存在 `F19_real_input_smoke.json`：

| 指标 | 结果 |
|---|---:|
| 相同分配次数 | 59 |
| 原 ratio P99 | 2,984,345,235.282889 bits/MI |
| 原 ratio 最大值 | 3,146,389,930.377271 bits/MI |
| 无量纲 ratio P99 | 2.7684047208 |
| 无量纲 ratio 最大值 | 2.9187242260 |
| F18/F19 最终能耗差 | 0 J |

这是临时单种子检查，不能替代九场景正式校准或证明训练收敛。未根据该短程结果调整模型超参数。

## 可执行检查命令

```powershell
# 临时真实输入检查，不训练策略、不写正式缓存
.\run.cmd common/smoke_f19.py --out docs/F19_real_input_smoke.json
# 自动检查
.\.venv\Scripts\python.exe -m unittest common.test_f19 -v
# 正式配置预览
.\run.cmd algorithms/mahdrl/train.py --task S --resource S --deadline Tight --dry-run
# 缓存只读检查；缺失时按 F22 报告补齐
.\run.cmd common/f22_cache.py --all
# 缓存就绪后，使用训练 seed 完成原 F18 校准
.\run.cmd common/f18_calibrate_normalization.py --all-nine --seeds 101-130 --resume
```

当前正式校准配置的 scenes 仍为空，新版正式缓存尚待准备。缺失配置会明确报错，不回退旧编码。代码修复完成，但正式长程数值稳定性仍需在缓存和校准准备后验收。

## 修改范围

公共归一化/配置：`common/f18_state_normalization.py`、`common/scenarios.py`；任务字段：`environments/mahdrl_heft.py`、`environments/marl.py`。

诊断：`common/f19_numerics.py`、`common/f11.py`、两份公共 D3QN agent、MARL agents、旧 IRWS Task/Intrinsic/PPO/train、paper IRWS learner/runner。

验证：`common/test_f19.py`、`common/smoke_f19.py`、相关报告。`common/smoke_irws_paper.py` 同步使用实际 model_seed 做初始参数对照，避免此前统一初始化 seed 后仍与 seed=32 的网络比较。

README 和 F22 文档同步新编码说明。先前默认初始化 seed=1 的未提交修改保留，用户 Word 文件未改。本次未提交或推送远程。
