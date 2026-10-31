# MAHDRL-AS-Fast-HScore-TL

独立入口：`algorithms/mahdrl_as_fast_hscore_tl/`，模型版本
`mahdrl_hscore_tl_grouped9_v1`。本版按 HScore 的 TL 消融定义，仅训练 TA、VA，
删除独立 HA 网络、优化器、回放、动作和更新时钟。HScore、VAware 及已有 TL 的
代码与输出目录均不修改。

## 策略与网络

流程：TA 排序取任务 → 同一个 VA 对各主机观测批量前向 → 从全局合法 VM 中
选择动作 → 由 VM 确定主机并执行分配。不存在先随机选择主机或启发式主机筛选。

| 项目 | HScore 完整版 | HScore-TL |
|---|---|---|
| 学习器 | TA / HA / VA | TA / VA |
| VA 网络 | 固定 93→9 | 同一网络类、结构和参数量 |
| VA 的实际动作范围 | HA 选择的主机内 | 所有主机的合法 VM |
| VA 后继估值 | 下一次实际选定主机的局部决策 | 下一次全局 VM 决策 |
| 每主机 VM 容量 | 9 | 9 |
| HA 的预期特征与低频更新 | 开启 | 随 HA 删除 |

每台主机的一行输入仍为 context30 + 9×VM7 = 93 维，包含任务、系统、该主机
物理状态、功率和本地 VM 特征，不包含主机 one-hot。主干和两个分支直接复用
`FixedVMDuelingQNet`，保持 HScore 的合法本地动作 Advantage 均值中心化；无
候选编码器、集合 pooling 或逐 VM 共享输出头。正式参数量：TA 532596、VA 2330378。

一个全局状态保存 `H×93` 观测和 `9H` 平铺掩码，行顺序为当前环境的 `host_ids`。
没有合法 VM 的主机行置零并完全屏蔽，其物理行号仍保留。少于 9 个真实 VM 的
主机在末尾填充槽位；填充不会创建真实资源。每主机超过 9 个 VM 在场景检查时
报错。H 是批次行数，网络参数不依赖 H，不设置固定主机数量上限。

全局动作 `a = 主机行号×9 + 本地槽位`，实际分配时映射到物理 VM；填充、忙碌和
非法槽位禁止选择。并列最高 Q 按主机行、本地槽位的稳定顺序取第一个。

## 全局学习与奖励

训练沿用 epsilon/GE 的解析混合概率，非探索选择全局合法最高 Q；探索的一半在
全部合法 VM 中均匀分配，另一半在全局并列最快 VM 中均匀分配。一次抽样完成动作。
评测 epsilon=0，不改变探索计数、模型参数或优化器。全局推理计时包含 H 行观测
批处理、前向、全局选择与采样。

回放冻结保存完整当前/后继主机批次、掩码和实际全局动作。当前 Q 只对实际动作
所属主机行计算梯度；Double DQN 的 online 网络在后继状态所有主机的合法槽位中
选最大 Q，target 网络对这一主机行与槽位估值。批次中的样本允许有不同 H，彼此
不混用后继动作；终止样本不前向后继网络、不 bootstrap。阶段结束不自动当作终止。

保留完整物理执行、同 VM 通信规则、主机任务窗口能耗账本及 VA 的原奖励：
`0.25*(-0.001*delta_energy) + 0.75*r_delay`，不合并 HA 奖励。阶段尾能耗补记到 VA
一次，终止经验只结算一次。保留真实下一决策转移、分配后更新、TA phase 更新，
以及原学习率、gamma、batch、buffer、warmup、epsilon 衰减和目标网络设置。
两级 APF 关闭，慢、预计超期或能耗较高本身不屏蔽物理合法 VM。

选模仍为 `SR >= .92` 后最低验证能耗。TA/VA 的 best 必须属于同一保存组、同一
选中回合及同一验证证据。即使 VA 的权重形状相同，也拒绝混用 HScore、VAware-TL
或其他策略的检查点；必须重新训练。检查点只保存 `best_manager.pth`、`best_vm.pth`
和对应 final 文件，不创建 HA 占位模型。不提供断点续训。

`--va-preview`、`--ha-update-period` 已删除，传入时明确报错。此处对各主机的批量
前向直接用于全局 VA 决策，不再计算提供给 HA 的四项行为预期特征。

## 运行与输出

Windows 在项目根目录运行：

```powershell
.\run.cmd algorithms/mahdrl_as_fast_hscore_tl/train.py --task S --resource S --deadline Tight --dry-run
.\run.cmd algorithms/mahdrl_as_fast_hscore_tl/train.py --task S --resource S --deadline Tight --episodes 600 --model-seed 1 --device cuda
.\run.cmd algorithms/mahdrl_as_fast_hscore_tl/evaluate.py --task M --resource S --deadline Tight --train-task S --train-resource S --train-deadline Tight --seeds 700-729 --model-seed 1 --device cuda
# 同一个 SS/Tight 源模型，测试 9 台主机 / 75 VM
.\run.cmd algorithms/mahdrl_as_fast_hscore_tl/evaluate.py --task S --resource L --deadline Tight --train-task S --train-resource S --train-deadline Tight --seeds 700-729 --model-seed 1 --device cuda
```

Linux/CUDA 后台命令：

```bash
source docs/mahdrl_as_fast_hscore_tl_commands.sh
# 参数：资源、截止期、可选模型种子
train_mahdrl_as_fast_hscore_tl S Tight
# 参数：源资源、截止期、可选目标任务列表、可选模型种子
eval_mahdrl_as_fast_hscore_tl_all S Tight
train_mahdrl_as_fast_hscore_tl M Medium 2
eval_mahdrl_as_fast_hscore_tl_all M Medium "S M L" 2
```

source 只定义函数；脚本末尾给出全部 9 组源场景训练和评测调用。后台评测函数
改变 task、保持资源规模，跨主机数量评测使用上方直接 CLI。压力负载可以在训练
和评测时均追加 `--workload-profile stress_a1_l006_ref003`。训练协议默认 600 回合、
每回合 50 个工作流，标准到达率 .03。

可直接复制的完整函数与九组调用另见
[600 回合训练指令](Linux_CUDA_MAHDRL_HScore_TL_600代训练指令.txt) 和
[九源场景评测指令](Linux_CUDA_MAHDRL_HScore_TL_九场景评测指令.txt)。

训练末级目录继续采用 17 字符短名：
`outputs/ckpts/mahdrl_hscore_tl/SS_T_m1_cXXXXXXXX/`；日志位于
`outputs/logs/mahdrl_hscore_tl/` 的同名子目录。所有最终 CSV 均直接位于
`outputs/run/mahdrl_hscore_tl/`，例如：

```text
eval_mahdrl_hscore_tl_noapf_sr092_SS_T_from_SS_T_m1_s700-729_cXXXXXXXX.csv
eval_mahdrl_hscore_tl_noapf_sr092_MS_T_from_SS_T_m1_s700-729_cXXXXXXXX.csv
eval_mahdrl_hscore_tl_noapf_sr092_LS_T_from_SS_T_m1_s700-729_cXXXXXXXX.csv
```

目标/源场景、模型种子及测试种子体现在文件名，明细在 `_details/` 下。CSV 包含
`vm_selection_scope=global_grouped_fixed9`、容量9、网络93/9维度；逐分配记录保留
真实 Host/VM ID、全局槽位索引、合法主机/VM 数、VA 版本和真实分配能耗。
已有训练和评测结果拒绝覆盖。

场景标记中前两位分别为任务规模和资源规模，末尾 `T/M/L` 分别表示
`Tight/Medium/Loose`。例如 `SS_M`、`SS_L` 会直接出现在最终 CSV 文件名中，
不会生成以这些场景命名、内部最终 CSV 同名的子目录；`_from_` 后是源模型场景。

自定义拓扑继续用于预计算和冻结评测，通过 `--resource-config` 指定，评测时必须
声明源 `--train-resource`。所有物理拓扑的缓存与源训练归一化规则沿用，不因新网络
流程重新拟合测试尺度。新的物理拓扑需匹配的缓存。

## 验证与消融解释

```bash
python -X utf8 -B -m unittest common.test_mahdrl_as_fast_hscore_tl -v
```

专项覆盖全局最高 Q/GE、后继最优 VM 位于另一台主机、保存掩码、混合主机数
TD、终止跳过后继估值、矩阵观测与本地 VA 一致、填充/忙碌/超期合法性、固定
动作物理及奖励一致性、分配后更新、冻结回放、CPU/CUDA 正式宽度更新、短程训练
保存重载与跨 6/12/18 主机及不等 VM 拓扑的冻结评测、旧权重隔离和输出防覆盖。

这组对照保留主机分组表示与固定 VA 网络，评估独立 HA 及其 VAware 联动机制的
整体收益。它与全局集合打分的 VAware-TL 不同，效果差异仍包含探索范围与学习
目标的改变，不能全部归因于 HA 网络本身。短程实现验证不代表正式训练效果。

2026-10-31 验证：TL 专项 11 项、HScore/VAware/VAware-TL 回归 35 项全部通过，
0 失败、0 错误、0 跳过，包含正式宽度 CPU/CUDA 更新和短程模型生命周期。
三个公开入口 dry-run、39 次后台替身调用及失败停止检查通过。9 个训练目录和
27 个最终 CSV 文件名均唯一，当前工作区训练目录最长 100 字符，末级目录 17 字符。
原有 640 个文件 SHA256 均未变。未启动正式 600 回合训练或正式 30 实例评测。
机器可读记录见 [验证报告](MAHDRL_AS_Fast_HScore_TL_validation.json)。
