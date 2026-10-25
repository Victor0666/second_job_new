# MAHDRL-AS：共享候选打分与自适应性能筛选

算法 ID：`mahdrl_oe_ge_apf_score`。显示名称：**MAHDRL-AS**，AS 表示 Adaptive filtering 与 Shared scoring。
算法基线为 MAHDRL-OE-GE-PF；新实现全部放入独立命名空间，不修改既有算法、环境、公共注册表或已有模型和结果。

## 保留与改变

- 保留 OE 的逐分配增量能耗账本、三层奖励公式、权重、缩放和结算边界。
- 保留 TA 的 15 维观测、243 个动作与 `(512,512,256,128)` Dueling 网络。
- HA/VA 改为共享候选编码和逐候选 Dueling Q 打分，删除 VA 的主机 one-hot。
- 固定 1.25 的 PF 改为 APF，包括按时最低增量能耗保留通道。
- 保留 GE：epsilon 探索内一半最快引导、一半最终合法候选均匀随机；验证/测试贪心。
- 保留 F11 实际下一决策转移、F17 权重更新、F19 源训练归一化、F22 划分和原验证能耗选模规则。

相同实例、动作和 phase 边界下，三层奖励与原 OE-GE-PF 相同。新策略会选择不同动作，因此实际训练奖励和物理指标可能变化。

## 网络与观测

| 层 | 上下文 | 单个候选 | 候选数 |
|---|---|---|---|
| HA | 任务/队列22维＋剩余子截止期1维 | 主机原属性5维＋最终APF候选OE摘要5维 | 当前主机数H |
| VA | 任务/队列22维＋所选主机5维＋剩余子截止期1维＋功率2维 | VM原属性5维＋超期/功率变化2维 | 所选主机真实VM数K_h |

候选编码器为 `F→128→128`。对真实候选编码取 mean/max，并追加真实/最终合法候选数量的 log1p。
上下文与汇聚结果进入 `1024→1024→512→512→256` 主干；主干只执行一次。
Value 为 `256→256→256→1`；共享 Advantage 为 `384→256→256→1`。所有隐藏层使用 ReLU。

Q=V+A−合法候选的平均A。存在性掩码用于汇聚，APF动作掩码用于合法均值、探索、贪心选择和TD后继动作。
参数形状不依赖 H 或 K_h；回放按 batch 临时 padding，没有固定资源数量上限。
HA/VA 参数分别共享，不跨层共享。正式 online 参数量为 TA 532596、HA 2571522、VA 2578306。

## APF 的确定规则

原始合法集取原 HA/VA 合法检查的交集，不启用忙碌VM或padding。每次任务分配先建立只读快照：

```text
start_v = max(now, available_v)
finish_v = start_v + 原计算与通信模型的duration_v
T_v = finish_v - now
ΔE_v = OE账本.preview(host_v, start_v, finish_v, pc_v)
g = 原始合法空闲VM数 / (原始合法空闲VM数 + 含当前任务的Ready数)
u = clip(min(T_v) / max(task_subdeadline - now, 1e-9), 0, 1)
r_H = clip(1.5 + 0.5*g - 0.25*u, 1.25, 2.0)
r_V = clip(1.25 + 0.5*g - 0.25*u, 1.25, 2.0)
```

HA 保留最快合法VM时长不超过全局最短时长 r_H 倍的主机；再加入全局原始合法、按时候选中全部并列最低 ΔE 的 VM 所属主机。
VA 在每台主机内保留时长不超过本地最短时长 r_V 倍的 VM；再加入该主机原始合法、按时候选中全部并列最低 ΔE 的 VM。
没有按时候选则该能耗通道为空。原合法集非空时始终保留最快候选。

HA选择后VA复用同一快照，不按所选主机重新计算全局g/u。训练、验证、评测、动作执行检查和TD后继动作都使用相同规则。
时间绝对容差1e-9秒；能耗并列rtol/atol均为1e-9。预览不推进时间、不写账本、不消耗随机数。

OE主机摘要根据最终本地VM候选重算；被HA屏蔽的主机追加5维摘要置零；被VA屏蔽的VM追加2维特征置零。
并列最快取最低能耗，并列最低能耗取最短时长，避免资源编号影响摘要。原基础物理特征保留。

当前公式中r_V实际不超过1.75。两层纯时长通道的总放宽可接近3.5倍全局最快值；能耗通道可额外保留按时候选。
这是筛选启发式，不保证成功率或最终能耗优于原算法。

## 学习参数与版本

HA/VA：lr=0.0003、gamma=0.95、batch=256、buffer=120000、epsilon从1衰减至0.05（80000次成功更新）、tau=0.005。
TA：相同lr/gamma/batch/epsilon，buffer=60000、衰减40000次更新、tau=0.01。
Adam、Huber、梯度裁剪10、warmup=buffer的3%保持原配置。默认600回合，每回合50个工作流。
TA仅在分配任务的phase更新；不把phase结束当作HA/VA episode终止。

结构版`shared_candidate_dueling_v1`，观测版`task23_host10_taskhost30_vm7_v1`，APF版`adaptive_pf_energy_v1`。
检查三层角色、真实参数形状、APF参数、源训练归一化和协议；三层best必须来自同一次保存组。
旧权重不能转换使用，需要从头训练新模型。后续改变资源数量可复用新模型，但调度质量仍需跨规模实验确认。

## 训练与同规模评测

在项目根目录运行。Windows将下列`python -u launch.py`替换为`.\run.cmd`即可。

九种源场景固定task=S，resource=S/M/L与deadline=Tight/Medium/Loose做笛卡尔积。
完整Linux/CUDA后台命令分别见[600回合九场景训练](Linux_CUDA_MAHDRL_AS_600代训练指令.txt)与[九场景评测](Linux_CUDA_MAHDRL_AS_九场景评测指令.txt)。
评测文件明确指定每个源模型，九个同规模目标各测试700–729共30个实例；任务规模泛化改动目标task，源参数保持不变。

```bash
# 预览：不会创建模型或运行训练
python -u launch.py algorithms/mahdrl_oe_ge_apf_score/train.py --task S --resource M --deadline Tight --dry-run

# 正式600回合：需要已准备好的匹配Deadline Cache与F19源尺度
python -u launch.py algorithms/mahdrl_oe_ge_apf_score/train.py --task S --resource M --deadline Tight --episodes 600 --model-seed 1

# 同一个源SM/Tight模型，测试30个实例
python -u launch.py algorithms/mahdrl_oe_ge_apf_score/evaluate.py --task S --resource M --deadline Tight --train-task S --train-resource M --train-deadline Tight --model-seed 1 --seeds 700-729

# 冻结同一个源SM/Tight模型，测试9主机/75VM
python -u launch.py algorithms/mahdrl_oe_ge_apf_score/evaluate.py --task S --resource L --deadline Tight --train-task S --train-resource M --train-deadline Tight --model-seed 1 --seeds 700-729
```

新训练拒绝非空模型/日志目录；评测拒绝非空目标结果目录。50与600回合不要先后写同一训练目录，需先归档自己的短程实验。
`--checkpoint-dir`只选择模型位置，不绕过源尺度与模型版本校验。GPU/CPU通过`--device cuda`或`--device cpu`显式选择。
不同模型种子拥有不同目录。没有在导入时修改其他算法注册表，也不对公共函数做运行时替换。

## 自定义资源数量

已有示例：`configs/mahdrl_as_12hosts.json`（12主机100VM）、`configs/mahdrl_as_18hosts.json`（18主机150VM）。
JSON格式为`{"vms_per_host": [9,9,8,8,8,8]}`，主机数量由列表长度决定。每个值必须为正整数。
首版训练使用已有S/M/L源场景，自定义拓扑用于冻结模型评测及其缓存生成。其内容哈希进入场景名、缓存名和结果目录。

```bash
# 使用原FCFS参考生成器，创建该拓扑匹配的新缓存；不覆盖旧缓存
python -u launch.py algorithms/mahdrl_oe_ge_apf_score/precompute.py --task S --resource-config configs/mahdrl_as_12hosts.json --seeds 0-1000

# 同一个源SM/Tight模型，直接评测12台主机
python -u launch.py algorithms/mahdrl_oe_ge_apf_score/evaluate.py --task S --resource-config configs/mahdrl_as_12hosts.json --deadline Tight --train-task S --train-resource M --train-deadline Tight --model-seed 1 --seeds 700-729
```

每主机VM数也可变为4或12，方法相同。自定义拓扑必须显式指定`--train-resource`，并使用源训练尺度。
显式缓存位置使用`--deadline-cache`；缓存必须通过任务、资源配置、通信/文件模型和协议种子覆盖检查。
不在测试资源上重新拟合归一化。正式F22检查沿用公共审计要求，缓存须覆盖训练、验证和测试名单。

## 输出与验证

模型位于`outputs/ckpts/mahdrl_as_v1_...`，训练日志位于`outputs/logs/mahdrl_as_v1_...`。
评测位于`outputs/run/mahdrl_oe_ge_apf_score/<版本摘要>/from_<源场景及模型种子>/eval_<目标>/`。
版本摘要由完整输出版本的SHA256生成，完整协议和归一化信息仍保存于元数据；缩短目录以兼容当前Windows深层工作区路径。

- `best_manager.pth`、`best_host.pth`、`best_vm.pth`及对应final文件。
- `episodes.jsonl`、`phases.jsonl`、`f11_diagnostics.jsonl`、`training_measurements.jsonl`。
- `training_summary.json`记录参数量、最佳验证能耗、CUDA峰值分配显存。
- `evaluation_mahdrl_as.csv`、`evaluation.jsonl`、`summary.json`。
- `measurements/`复用公共逐工作流指标和平均/P95/P99决策延迟，记录真实checkpoint哈希。
- `apf_episode*.json`、`apf_seed*.json`记录g/u/阈值、候选保留数量、能耗救回/选中次数及各主机使用窗口。

共享候选结构不意味着推理更快或能耗更低，应使用相同硬件、训练预算和计时协议做对照。

```bash
python -X utf8 -B -m unittest common.test_mahdrl_oe_ge_apf_score -v
```

专项覆盖：固定PF动作包含性、能耗通道与并列、预览无副作用、三层奖励/物理一致性、候选排列/padding不变性、
保存后继mask的TD目标、真实梯度更新、严格版本/角色校验、临时真实训练保存加载评测和冻结模型跨主机/VM数迁移。
端到端短程测试使用临时缓存、独立小型协议、2个工作流及小网络；另测试正式HA宽度的真实反向传播。
这些是实现验证，不是正式600回合训练或跨规模性能结论。

2026-10-25 验证：原有214个Python源码SHA256逐一一致；新增内容不修改旧算法注册表。
相关去重回归共69项通过（新算法12项，既有相关算法/场景/F11/计时57项）。
包含CPU/CUDA真实更新、正式HA宽度反向传播、独立协议的端到端短程训练，以及同一源模型到6主机72VM、12主机48VM的冻结迁移。
真实CLI dry-run覆盖6主机50VM训练、9主机75VM评测、12主机100VM评测和18主机150VM缓存入口。
旧计时测试首次因Windows默认GBK读取UTF-8文件失败，启用Python UTF-8模式复测通过，旧测试源码未改动。
没有启动正式600回合训练、30测试实例或正式扩规模性能实验。
