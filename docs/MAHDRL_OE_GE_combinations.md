# MAHDRL-OE-GE 与 MAHDRL-OE-GE-PF

两个组合均为独立算法 ID，原 `mahdrl`、`mahdrl_oe`、`mahdrl_ge`、`mahdrl_pf`、`mahdrl_ts`、`mahdrl_ub` 及其模型、日志和结果保留。

| 算法 ID | 组成 | 训练动作选择 | 验证与独立评估 |
|---|---|---|---|
| `mahdrl_oe_ge` | OE 观测/逐分配能耗奖励 + GE | HA/VA 在 epsilon 分支内，50% 选择预计最快候选、50% 从全部合法候选随机选择 | 在全部合法候选中按 Q 值贪心选择 |
| `mahdrl_oe_ge_pf` | OE + GE + PF | 先筛选，再在保留候选中执行同样的 GE 探索 | 保留 PF 筛选，由 Q 值贪心选择 |

## 保留的设置

- OE 的任务子截止期、主机身份/功率、候选主机时长与能耗摘要，以及按任务分配即时反馈的精确新增排程能耗，均复用原实现。
- 原能耗缩放、延迟项与权重、网络隐藏层、学习率、gamma、batch、buffer、epsilon 退火和目标网络更新参数不变。输入维度与 OE 相同，不额外扩大网络观测。
- TA 的观测、243 个动作、排序、phase 奖励与学习规则不变。组合身份仍写入 TA checkpoint，避免把不同实验的三层模型混用。
- 最终物理能耗仍按 `host_task_window_v1` 统计。HA/VA 的差分奖励只发放一次，phase 保留 OE 的异常强制分配补结算；没有新增重复能耗奖励。
- 沿用现有公共缓存、F19 归一化与 F22 划分，不需要因这两项组合重新生成缓存。

## 三合一的候选范围

性能时长复用原 `candidate_durations`：计算时间、实际通信与同 VM 数据复用均包含在内。

1. HA 按每台主机当前最快可用 VM 的预计时长比较主机，保留不超过最快主机成本 1.25 倍的合法主机。
2. VA 在已选主机内保留预计时长不超过本地主机最快 VM 1.25 倍的合法 VM。阈值分别应用于两层。
3. 训练中的 GE 只能在上述保留候选内探索。验证与评估没有 GE 随机探索，PF 仍生效。
4. 当前 mask 和后继 mask 同时进入回放经验，Double DQN 的目标动作也受筛选限制。

OE 的 HA 摘要按本主机通过 PF 的 VM 重新计算：最低能耗和它对应的时长不会来自已被排除的慢 VM。被 HA 筛掉的主机，其追加候选摘要置零；VA 的追加候选超期量/功率变化也对不可选槽位置零。原基础资源状态仍保留为上下文，不改变观测维度。

候选预览不分配任务、不推进时钟、不消耗随机数；不修改 OE 的原候选缓存或能耗账本。掩码保留并列最优及至少一个最快合法动作，不启用忙 VM 或 padding；直接执行被筛掉的动作会报错。

## 代码与输出隔离

- 入口：`algorithms/mahdrl_oe_ge/{train,evaluate}.py`、`algorithms/mahdrl_oe_ge_pf/{train,evaluate}.py`。
- 组合 agent 与模型身份：`common/mahdrl_combinations.py`，直接复用 GE 的动作选择、OE 的角色校验和原 D3QN 更新。
- 环境：`environments/mahdrl_oe_ge.py`、`environments/mahdrl_oe_ge_pf.py`，复用 OE 奖励及 PF 掩码执行逻辑。
- phase 结算直接导入 `algorithms.mahdrl_oe.reward`，没有复制一套不同的奖励规则。

模型与日志目录分别以 `mahdrl_oe_ge_task...`、`mahdrl_oe_ge_pf_task...` 开头。主评测 CSV 分别以 `eval_mahdrl_oe_ge_task...`、`eval_mahdrl_oe_ge_pf_task...` 开头，结果及逐实例测量目录使用对应算法 ID。

三层 checkpoint 同时记录 `performance_policy` 和带角色的 `algorithm_variant`，后者包含组合组件、OE 基础协议、50% 引导概率、筛选阈值及摘要范围。新组合拒绝原版、任一单独变式、另一组合、错角色或缺少身份信息的权重；必须各自从头训练。`algorithm_variant.json` 同时写入模型、日志和评测目录，测量记录保存各 agent 的实际性能配置。

## 训练与评测命令

从项目根目录、已激活的 Python 环境执行。以下使用 SM / Tight；固定模型种子 1，其他公共参数沿用默认 F22/F19。

```bash
# OE + GE：50代短程训练，或600代正式训练，二选一
python -u launch.py algorithms/mahdrl_oe_ge/train.py --task S --resource M --deadline Tight --episodes 50 --model-seed 1
python -u launch.py algorithms/mahdrl_oe_ge/train.py --task S --resource M --deadline Tight --episodes 600 --model-seed 1

# OE + GE + PF：50代短程训练，或600代正式训练，二选一
python -u launch.py algorithms/mahdrl_oe_ge_pf/train.py --task S --resource M --deadline Tight --episodes 50 --model-seed 1
python -u launch.py algorithms/mahdrl_oe_ge_pf/train.py --task S --resource M --deadline Tight --episodes 600 --model-seed 1

# 两个训练预算共用评测命令，分别加载当前对应组合的best模型
python -u launch.py algorithms/mahdrl_oe_ge/evaluate.py --task S --resource M --deadline Tight --train-task S --seeds 700-729 --model-seed 1
python -u launch.py algorithms/mahdrl_oe_ge_pf/evaluate.py --task S --resource M --deadline Tight --train-task S --seeds 700-729 --model-seed 1

# 泛化示例：同一源SM模型，评估MM
python -u launch.py algorithms/mahdrl_oe_ge/evaluate.py --task M --resource M --deadline Tight --train-task S --seeds 700-729 --model-seed 1
python -u launch.py algorithms/mahdrl_oe_ge_pf/evaluate.py --task M --resource M --deadline Tight --train-task S --seeds 700-729 --model-seed 1
```

Windows PowerShell 可将 `python -u launch.py` 替换为 `.\run.cmd`。任一指令追加 `--dry-run` 只预览配置。

50 与 600 代使用同一组合/场景的模型与日志目录，不支持续训，入口拒绝覆盖非空训练目录。若先跑 50 代，应先评测并归档该组合自己的模型、日志和评测结果，再从头训练 600 代。不同组合之间目录独立。700–729 是现有已查看的评测名单，组合方案选择使用训练/验证材料，最终核验应统一预留未参与选方案的实例。

## 验证范围

```powershell
.\.venv\Scripts\python.exe -X utf8 -B -m unittest common.test_mahdrl_combinations common.test_scenarios common.test_mahdrl_performance common.test_mahdrl_oe -q
```

覆盖内容：

- 相同动作下与 OE 的逐分配奖励、TA phase 奖励及最终物理能耗一致，并与单独 PF 的掩码逐步一致。
- 被筛掉的最低能耗慢 VM 不进入候选摘要；原候选数据、物理状态与 RNG 不变。
- 先筛选后 GE、并列/合法动作限制、贪心评估以及筛选后继动作的 TD bootstrap。
- 原强制分配补结算仍只发奖一次；完整回合差分能耗和最终物理能耗守恒。
- 原版、单变式、两组合和角色之间的 checkpoint 隔离。
- 临时目录内的 OE/两组合小网络真实训练、三层实际梯度更新、保存加载、评测及原输出保护；正式网络参数与 OE 一致。

本次不启动正式 50/600 代训练，也不据测试宣称组合的能耗优于单独变式。

2026-10-23 验证：首轮组合/场景/单独变式专项共 27 项通过；最终全量回归共 243 项通过（125.654 秒）。两组合的 50/600 代训练与评估共六条真实 CLI dry-run、语法和空白检查通过。新增前已存在的算法、环境与共享学习源码共 126 个受保护 Python 文件的 SHA256 保持一致；公共文件仅增补两个算法注册及对应场景测试，README 增加入口说明。

完整回归沿用此前做法：既有未提交冻结 TA 工具在导入时全局关闭 F11/F19/测量开关，因此在 discovery 后仅为测试进程恢复这三个开关，再执行全部测试；没有修改该工具。通过记录位于 `tmp/mahdrl_combinations_regression.txt`。专项命令不导入该工具，可直接运行。
