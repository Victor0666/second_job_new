# MAHDRL-AS-Fast

独立入口为 `algorithms/mahdrl_oe_ge_apf_score_fast/`，环境为
`environments/mahdrl_oe_ge_apf_score_fast.py`。当前运行身份是 `mahdrl_as_fast_v2`。
该版本是 MAHDRL-AS 的执行优化，不是改变目标函数的新学习方法。

## 保持一致的内容

直接复用 AS 的网络、参数形状、APF 参数、GE 概率、学习率、batch=256、
经验容量、warmup、epsilon 衰减、目标网络更新与梯度裁剪设置。
TA/HA/VA 三层奖励、任务顺序、物理执行、通信模型和 OE 能耗账本均不变。
仍用 `random.sample(deque, batch_size)`，不引入额外随机调用。
训练、验证、评测与 TD 后继动作共用同一 APF 规则。

科学配置 `algorithm_variant` 保持 AS 身份，执行版本单独写入
`implementation.json` 和 checkpoint 的 `normalization.implementation`。
网络形状、角色、训练源归一化、实验协议和 TA/HA/VA 保存组仍须全部匹配。
v2 评测还要求新选择规则的来源证据；缺少证据的原 AS/Fast v1 权重不能作为
新规则 best 加载。历史权重沿用提交 `220879a` 的 v1 入口评测。
定长网络、不同 APF 参数及混合保存组不能加载。没有增加断点续训入口。

模型与日志路径使用独立的 `mahdrl_as_fast_v2_..._select_sr092_minenergy_v1` 名字，评测路径使用
`outputs/run/mahdrl_oe_ge_apf_score_fast/mahdrl_as_fast_v2_.../`。
训练、评测均拒绝覆盖非空输出，已有 v1 输出保留。此次模型选择变更不修改
其他算法、公共注册表、共享 F22 文件、环境或学习更新公式。

## v2 模型选择（2026-10-29：总体成功率至少 92%）

规则标识为 `select_sr092_minenergy_v1`。每个训练回合结束后，仍在原验证种子上
进行一次确定性验证。用总成功工作流数除以总工作流数得到成功率 S，
用各验证实例的 total_energy 算术平均得到 E。成功指工作流在自身截止期内完成，
不是任务成功率，也不是要求每个验证实例分别达到阈值。

- S >= 0.92 的候选优先；其中 E 最低者为 best，E 相同时选 S 更高者。
- 尚无达标模型时，选 S 最高者，S 相同时选 E 最低者。
- 两项完全相同则保留较早回合，不重复替换。
- 默认 seed600 的 50 个验证工作流，46/50=0.92 达标，45/50=0.90 未达标。
  该阈值仅描述验证集，不能保证未知实例的总体成功概率。

TA/HA/VA 的 best 在选中回合一起保存，共享 checkpoint_group。每层 checkpoint 的
`normalization.mahdrl_as_fast` 记录算法版本、角色、选择规则、协议、训练源、
验证种子、当前回合及其验证记录、历史 best 回合及其验证记录。加载时重算两组
指标，检查三层保存组、回合和证据一致。`final` 始终是最后一回合的权重，
历史 best 单独记录；将 final 文件改名为 best 也不能通过新评测校验。

`validation_selection.jsonl` 逐回合记录候选、历史 best 和是否发生替换；
`training_summary.json` 增加 selected_episode、best_validation_selection、
final_validation_selection 和 feasible_model_found。episode 从 0 开始，
completed_episodes 表示已完成回合数。选择不依赖可关闭的测量/诊断日志。
运行元数据局部声明新规则，共享 F22 划分文件及其哈希保持不变，归一化尺度和
截止期缓存可继续使用。显式 legacy 调试仍标记为 legacy，不宣称其数据集互斥。

新模型/日志路径及评测版本绑定新选模标识；旧 `select_sr099_minenergy_v1` 检查点因规则不同拒绝加载。
PD3QN-DAG-FE 和 DRL-GNN-Fast 继续使用 99% 门槛，11 个消融仍严格使用 `SR > 0.92`；完整版保留两级 APF。
需要按新规则重新训练；已有旧 best/final 权重仅能在现存权重之间重新比较，验证日志不能恢复未保存
回合的模型，也不能据此声称找到了整个训练过程按新规则的最优模型。

## 加速措施

1. 经验入库完整校验当前/后继观测、掩码、动作、奖励和终止标志；
   保存自有只读数组和只读结构，外部原数组的修改不会污染回放。
   采样时同长度直接 `np.stack`，混合长度按长度分组填充，保留原采样顺序，
   再检查整个 batch 的有限性、形状、二值掩码及合法性。
2. 每次决策复用 HA/VA 平面观测、公共 block、时长预测与 APF 结果。
   `host_select` 和 `vm_assign` 复用已构建的掩码。
   分配、时间推进、完成事件、到达、重置及 TA 权重变更会使缓存失效。
   缓存只覆盖一个决策；直接更改内部任务属性的外部调试代码应调用
   `_invalidate_decision()`，当前子截止期和 VM 可用时刻另有缓存键检查。
3. APF 使用 float64 NumPy 数组、`np.minimum.at`、布尔索引及批量
   `np.isclose` 实现主机/VM 最短时长、能耗保留、最终候选摘要和保留统计。
   保持原时间容差、能耗容差、所有并列最优项、候选 ID 与顺序。
   复杂任务时长和账本预览仍使用原标量公式，结果在决策中复用；
   任务执行和能耗账本写入仍逐分配进行。
4. 详细输入/Q/TD/梯度诊断默认采样第一次及每 100 次更新/策略调用；
   在 GPU 索引、分位数统计与 CPU 回传之前判断是否采样。
   验证/评测不采集这些详细诊断。每次更新都检查 loss 和梯度有限性。
   全部 transition、update、sampled-target 计数保留；目标汇总在 GPU
   累计并在日志边界回传。详细分位数和裁剪比例只代表采样更新，日志注明范围。
5. 默认每 100 次更新/环境决策采样计时，记录 batch 拼装、H2D、三个前向、
   反向/裁剪/优化器、目标更新、诊断、观测/APF、分配和阶段推进。
   CPU 使用排除子阶段的 wall time；CUDA 使用事件并在回合边界统一收集。
   两种时间可能重叠，不能相加；采样数据不外推成完整回合耗时。

## 使用方式

项目根目录、已激活依赖环境下执行：

```bash
python -u launch.py algorithms/mahdrl_oe_ge_apf_score_fast/train.py \
  --task S --resource S --deadline Tight --episodes 600 --model-seed 1 \
  --experiment-protocol configs/f22_split_v1.json \
  --normalization-mode f19_shared_v1 --device cuda \
  --diagnostics-every 100 --profile-mode sampled --profile-every 100

python -u launch.py algorithms/mahdrl_oe_ge_apf_score_fast/evaluate.py \
  --task S --resource S --deadline Tight \
  --train-task S --train-resource S --train-deadline Tight \
  --seeds 700-729 --model-seed 1 \
  --experiment-protocol configs/f22_split_v1.json \
  --normalization-mode f19_shared_v1 --device cuda
```

`--diagnostics-every 1` 恢复逐更新的详细统计；`--profile-mode off` 关闭阶段计时。
这两个选项不改变训练数学、随机序列或更新预算。每回合验证的频率仍保持原设置。
`--dry-run` 可预览配置和输出路径，但不会替代正式运行时的 cache/归一化审计。
需要原先匹配的 FCFS 截止期缓存和 F19 训练源归一化文件。
自定义主机数量评测沿用 AS 的 `--resource-config`、`--deadline-cache` 及
`--train-resource` 用法，`precompute.py` 使用相同的参考生成器。

九场景定义：任务规模固定为 S，资源规模 S/M/L × 截止期 Tight/Medium/Loose。
逐条后台运行指令见同目录 `Linux_CUDA_MAHDRL_AS_Fast_600代训练指令.txt`
与 `Linux_CUDA_MAHDRL_AS_Fast_九场景评测指令.txt`。评测每场景使用 700–729
共 30 个实例，对应源场景的三个 best checkpoint 应来自同一保存组。

训练日志中的 `training_measurements.jsonl` 保留完整回合训练/验证耗时和更新预算；
新增 `stage_timings.jsonl` 解释各阶段的采样耗时；`f11_diagnostics.jsonl`
记录采样范围。测试费用比较应固定设备、线程数、场景和训练回合，避免 GPU 争用。

## 验证与本机性能

```bash
python -X utf8 -B -m unittest common.test_mahdrl_as_fast common.test_mahdrl_oe_ge_apf_score -v
python -X utf8 -B -m algorithms.mahdrl_oe_ge_apf_score_fast.benchmark \
  --device cuda --updates 100 --pack-repeats 150 --episode \
  --out outputs/as_fast_benchmarks/cuda.json
```

测试覆盖随机/并列 APF、同/混合/空候选 batch、只读回放、缓存失效、完整回合
观测/掩码/三层奖励/账本、GE 随机序列、完整训练回合参数一致性、CPU/CUDA
正式宽度参数更新、终止无 bootstrap、保存的后继掩码、诊断关闭时的异常梯度、
CUDA 事件收集、旧权重拒绝与组/源校验，以及训练→保存→加载→评测。
v2 检查 0.92 包含等号的边界、总体成功率聚合、回退、并列保留、旧 99% 规则拒绝、
验证证据复核和三层选中回合一致性测试；通过受控验证结果的完整训练测试检查
测量开关不影响最终训练权重、更新次数与 best 选择。

本机 CPU/CUDA 测量数据与条件见 `MAHDRL_AS_Fast_benchmark_2026-10-26.json`。
Python 3.12.10 / PyTorch 2.5.1+cu121 / NumPy 2.1.2，CPU 单线程，
CUDA 设备为 RTX 4060 Ti。正式宽度、batch=256 的计时结果如下：

| 测量项 | CPU 加速比 | CUDA 加速比 |
| --- | ---: | ---: |
| TA 单次更新 | 1.03× | 1.47× |
| HA 单次更新（6 个候选） | 1.10× | 2.90× |
| VA 单次更新（9 个候选） | 1.13× | 2.68× |
| 短训练回合 | 1.14× | 2.45× |

CUDA 更新计时各 100 次，CPU 各 15 次，均排除前 3 次预热；
短回合每个实现只测一次，因此只作本机运行检查。CPU batch 拼装自身约快
12–15 倍，并不表示整个训练有同样的加速比。原版采用默认逐次详细诊断，
Fast 采用默认每 100 次详细诊断，两版的分析器均关闭。
该性能记录来自 v1 加速实现，未包含 v2 模型选择变更的正式训练费用测量。

短回合测量为 2 个工作流、预填合成经验、warmup=0，使用正式网络和 batch=256，
只测 rollout 与更新，不包含验证、checkpoint 和文件写入；不能直接视为
服务器上 600 回合训练的总加速比。未完成九场景的正式训练与效果实验。
