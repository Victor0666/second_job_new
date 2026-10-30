# MAHDRL-AS-Fast-VAware-HScore-FixedVA

独立算法入口：`algorithms/mahdrl_as_fast_hscore/`。模型版本：
`mahdrl_hscore_fixedva_v1`。本版以 VAware 完整三层版为基线，仅 HA 使用可变候选
共享打分，VA 改为固定槽位 Dueling Double DQN。原 VAware 及其 TL、R100、R150
实现和结果不修改；新版不加载旧版权重，需要重新训练。

## 网络与观测

| 层级 | 输入 | 动作与网络 |
|---|---|---|
| TA | 15 维 | 原 243 个排序动作及原网络 |
| HA | context23 / 每主机 candidate14 | 对当前 H 个主机共享打分；关闭 VA 感知时 candidate10 |
| VA | context30 + 9×VM7 = 93 维 | 一个跨主机共享的固定输出网络，输出 9 个本地主机 VM 槽位的 Q |

VA 主干为 `93→1024→1024→512→512→256`；Value 为
`256→256→256→1`，Advantage 为 `256→256→256→9`，隐藏层使用 ReLU。
没有 VM entity encoder、集合 mean/max pooling 或逐 VM 共享 Advantage。
Q 的 Advantage 中心项仅对当前合法动作取均值。真实参数量写入训练汇总，
不声称与集合 VA 参数量相同。

容量固定为 9，不随当前场景重新计算。每主机真实 VM 数可以为 1–9；缺少的槽位
在右侧补零并屏蔽，超过 9 在场景校验阶段报错。填充不会创建物理 VM。沿用
`host_to_vm_indices` 的本地顺序，忙闲变化不重排槽位。VM 的速度、带宽、状态、
预计完成时间及超期/功率变化特征保留。上下文沿用当前 30 维布局，不包含随主机数
变化的 one-hot 或全体主机的拼接特征。

标准 S/M/L 的 3/6/9 台主机、25/50/75 个 VM 均可用。冻结源模型可以评测更多
主机及每主机容量以内的自定义拓扑，使用源训练归一化尺度和目标匹配缓存。
网络维度可复用不等于跨规模调度性能得到保证。

## VAware、动作与学习

每次决策对合法候选主机分别构造 93 维 VA 观测，批量前向得到 `合法主机数×9`
的 Q。沿用原 epsilon/GE 行为概率，计算预期执行时长、边际能耗、超期惩罚和 Q，
提供给 HA；填充和忙碌动作概率为零。HA 选择主机后，VA 从该主机缓存的同一 Q
与概率中选择 VM。预评估不抽样、不修改动作计数、不推进物理环境；网络版本和
epsilon 不一致时拒绝执行。验证/评测 epsilon=0，选首个合法最大 Q。

保留 VAware 的两级 APF 关闭、OE 物理能耗账本、奖励 0.25 能耗/0.75 延迟、
GE 混合、分配完成后更新、真实下一决策经验、HA 默认每 4 条有效经验更新一次及
独立探索时钟。TA 更新节奏和所有原学习率、gamma、batch、buffer、epsilon 衰减、
目标网络设置沿用。`--no-va-preview`、`--ha-update-period` 沿用现有控制参数。

HA 回放仍为可变候选集合；VA 回放为冻结的 93 维状态、9 维掩码及实际后继。
Double DQN 用保存的后继合法掩码选动作，target 网络评估该动作；终止样本不前向
后继网络、不 bootstrap。掩码不排除仅因速度慢、预计超期或能耗较高的物理合法 VM。

选模沿用 `SR >= 0.92` 后最低验证能耗，三层 best 必须属于同一保存组和选中回合。
检查点校验新身份、角色、实际参数形状、固定 9 容量、93 维观测、主机内动作定义、
机制、源归一化、协议与选模证据。旧 VAware 与新模型双向拒绝混用；不提供断点续训。

## 运行

Windows 在项目根目录运行（默认自动选择 CPU/CUDA，也可显式指定）：

```powershell
.\run.cmd algorithms/mahdrl_as_fast_hscore/train.py --task S --resource S --deadline Tight --dry-run
.\run.cmd algorithms/mahdrl_as_fast_hscore/train.py --task S --resource S --deadline Tight --episodes 600 --model-seed 1 --device cuda
.\run.cmd algorithms/mahdrl_as_fast_hscore/evaluate.py --task S --resource S --deadline Tight --train-task S --train-resource S --train-deadline Tight --seeds 700-729 --model-seed 1 --device cuda

# 冻结同一 SS/Tight 源模型，评测 9 台主机 / 75 VM
.\run.cmd algorithms/mahdrl_as_fast_hscore/evaluate.py --task S --resource L --deadline Tight --train-task S --train-resource S --train-deadline Tight --seeds 700-729 --model-seed 1 --device cuda
```

Linux/CUDA 后台函数与原版参数顺序相同，但函数、入口和控制台日志前缀独立：

```bash
source docs/mahdrl_as_fast_hscore_commands.sh
train_mahdrl_as_fast_hscore S Tight
eval_mahdrl_as_fast_hscore_all S Tight
# 可选参数：训练为 preview(on/off)、HA周期、模型种子
train_mahdrl_as_fast_hscore M Medium on 4 2
# 评测为目标任务列表、preview、HA周期、模型种子
eval_mahdrl_as_fast_hscore_all M Medium "S M L" on 4 2
```

上述后台评测函数沿用原版的任务规模泛化：目标 `--task` 变化，资源数保持源配置。
改变主机数时使用上方直接 CLI，明确指定目标 `--resource` 与源 `--train-resource`。

自定义拓扑使用 `{"vms_per_host": [9,8,8,9]}` JSON，通过 `--resource-config`
指定；仍仅用于预计算与评测，评测必须显式指定源 `--train-resource`。例如：

```bash
python -u launch.py algorithms/mahdrl_as_fast_hscore/precompute.py --task S --resource-config configs/mahdrl_as_12hosts.json --seeds 0-1000
python -u launch.py algorithms/mahdrl_as_fast_hscore/evaluate.py --task S --resource-config configs/mahdrl_as_12hosts.json --deadline Tight --train-task S --train-resource S --train-deadline Tight --seeds 700-729 --model-seed 1 --device cuda
```

原物理拓扑的截止期缓存和归一化文件可继续使用，不为网络结构变化重新拟合尺度。
正式协议仍为 600 回合、每回合 50 个工作流；跨规模性能需使用多个独立训练种子验证。

## 输出与验证

训练目录：`outputs/ckpts/mahdrl_hscore/SS_T_m1_cXXXXXXXX/`，日志位于同名
`outputs/logs/mahdrl_hscore/` 子目录。最终 CSV 平铺在 `outputs/run/mahdrl_hscore/`，
包含目标/源场景、机制、模型种子和测试种子，逐分配、计时和能耗审计位于 `_details/`。
已有训练或评测结果拒绝覆盖。CSV 包含 `vm_capacity=9`、`vm_input_dim=93` 和
`vm_selection_scope=host_local_fixed_slots`。

标准场景、模型种子 1 的训练末级目录固定为 17 个字符，例如 `SS_T_m1_cec857181`；
`T/M/L` 分别表示 Tight/Medium/Loose。训练模型和日志均使用这个短场景名。
在当前 Windows 工作区，标准场景训练目录绝对路径最长 97 个字符；相对项目根目录
的路径为 `outputs/ckpts/mahdrl_hscore/SS_T_m1_cec857181`。

最终 CSV 不通过 `SS_T/SS_M/SS_L` 等父目录区分，所有场景均直接写入
`outputs/run/mahdrl_hscore/`。例如同一个 SS/Tight 源模型的三个目标任务场景为：

```text
eval_mahdrl_hscore_noapf_sr092_p1h4_SS_T_from_SS_T_m1_s700-729_cec857181.csv
eval_mahdrl_hscore_noapf_sr092_p1h4_MS_T_from_SS_T_m1_s700-729_cec857181.csv
eval_mahdrl_hscore_noapf_sr092_p1h4_LS_T_from_SS_T_m1_s700-729_cec857181.csv
```

目标截止期变化时，文件名中的目标标记相应变为 `SS_M`、`SS_L`，源标记仍由
`--train-deadline` 决定。`from_` 前面是目标，后面是源模型；直接批量下载根目录
的 `*.csv` 即可按文件名区分。`_details/` 中的子目录仅存逐分配和审计明细。

完整后台函数及九组调用分别见
[600 回合训练指令](Linux_CUDA_MAHDRL_HScore_600代训练指令.txt) 与
[九源场景评测指令](Linux_CUDA_MAHDRL_HScore_九场景评测指令.txt)。

```bash
python -X utf8 -B -m unittest common.test_mahdrl_as_fast_hscore -v
```

专项验证包括忙碌/填充掩码、预评估无副作用与实际观测一致、保存后继掩码的 TD 目标、
终止 bootstrap、固定动作物理/奖励一致性、CPU/CUDA 真实更新、正式宽度网络更新、
短程训练与保存加载、同一源模型到 6/12/18 主机的冻结迁移、旧模型隔离及防覆盖。
短程验证不代表完成 600 回合正式训练或证明跨规模性能。

2026-10-31 验证：专项及 VAware、VAware-TL、R100/R150 相关回归共 52 项全部通过，
0 失败、0 错误、0 跳过，包含 CPU/CUDA 真实更新。三个公开入口 dry-run 通过；
4 组机制配置的 36 个训练路径和 108 个评测 CSV 路径均唯一。原有 578 个文件
SHA256 全部未变。正式 online 参数量为 TA 532596、HA（预评估开启）2572034、
VA 2330378。机器可读结果见 [验证报告](MAHDRL_AS_Fast_HScore_validation.json)。

输出路径复验：11 项专项通过，实际生成同一源模型到 `SS_T`、`SS_M`、`SS_L`
的三个最终 CSV，父目录相同且文件名不同。36 个训练目录、108 个最终 CSV 文件名
均唯一；训练相对路径最长 45 字符。两份独立指令文件通过 Bash 语法检查，后台
函数通过 39 次替身调用及失败停止检查；未启动正式训练或正式 30 实例评测。
