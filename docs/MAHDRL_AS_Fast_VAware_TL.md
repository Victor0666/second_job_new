# MAHDRL-AS-Fast-VAware-TL

独立入口为 `algorithms/mahdrl_as_fast_vaaware_tl/`，模型版本为 `mahdrl_as_fast_vaaware_tl_noapf_v1`。它在 VAware 协议下删除 HA，由 TA 排序后交给 VA 从全局合法 VM 中选择，再由 VM 唯一确定 Host。只有 TA、VA 两个学习器；不创建 HA 网络、优化器、经验池或占位模型。

## 实验定义

| 项目 | VAware 默认完整方案 | VAware-TL |
|---|---|---|
| 资源动作 | HA 选择 Host，VA 选择局部 VM | VA 一次选择全局 VM |
| VA 输入 | context30 / candidate7 | context23 / candidate14 |
| HA 感知特征和低频更新 | 开启，周期4 | 随 HA 删除，不适用 |
| 两级 APF | 关闭 | 关闭 |
| 物理执行与 OE 分配边际账本 | 原实现 | 相同 |
| VA 更新 | 补齐真实经验，本次分配之后更新 | 相同 |
| TA 更新 | 原 phase 节奏 | 相同 |
| 选模 | SR >= .92 后最低验证能耗 | 相同 |

全局候选沿用已有 TL 的映射：每个真实 VM 占一行，candidate14 为 VM7、所属 Host5、所属 Host 空闲/功率2。公共 context23 保留任务、系统和剩余子截止期；主机信息移动到对应候选行。忙碌 VM 保留观测行但 mask 为零，填充行和非法动作禁止选择。所有物理合法空闲 VM 均可选，不增加截止期、速度或节能过滤。支持不等 VM 数量的自定义评测拓扑，网络参数形状不依赖 VM 数量。

训练动作与 VAware 使用相同解析 epsilon/GE 混合及一次抽样：非探索概率选首个合法最高 Q；探索概率的一半均匀分给全部合法 VM，另一半分给全局预计最快集合（沿用并列容差）。评测 epsilon=0，不增加探索/动作计数。探索范围的变化属于层级结构变化，不声称与 HA/局部 VA 的最终采样分布或随机数序列相同。

先构造观测并补齐上一条真实 VA 经验；本次动作和分配完成后才执行满足 warmup 条件的更新。VA 每条符合条件的真实经验更新一次；TA 在有分配的 phase 后更新。pending 跨 phase 保留，后继为下一次真实全局 VM 决策；终止/截断结算一次，回放保存冻结历史观测。保持原学习率、batch、buffer、epsilon 衰减、GE 比例、隐藏层及目标网络规则。推理计时包含全局网络前向和采样。

VA 奖励保持 `0.25*(-0.001*delta_energy)+0.75*r_delay`，TA 奖励不变，不合并 HA 奖励。阶段尾未返回的分配能耗仅补到 VA 一次，审计核对物理能耗、分配账本和奖励守恒。未添加 APF、排队、等待动作、额外能耗罚分、预评估增强或短期前瞻。

## 与旧 TL 的区别

已有 `mahdrl_oe_ge_apf_score_fast_tl` 继续可用，但其更新时机和严格 `SR > .92` 选模协议不同。新 TL 采用 VAware 的更新屏障和 `select_sr092_minenergy_v1`；验证50个工作流时46个成功即达门槛。可行模型先最低能耗，再更高成功率；不可行模型先成功率再能耗，完全并列保留较早回合。不能将旧 TL 结果直接替代新消融结果。

TA/VA 必须重新训练。检查点只保存 `best_manager.pth`、`best_vm.pth`、`manager_final.pth`、`vm_final.pth`。严格验证新身份、角色、形状、全局动作协议、源归一化、选模证据与保存组；旧 TL、VAware、跨场景/模型种子、混组以及 final 改名 best 均不能混用。当前不提供断点续训。

## 训练与评测

在项目根目录、已激活 Python 环境的 Bash 中：

```bash
source docs/mahdrl_as_fast_vaaware_tl_commands.sh

# 参数：资源、截止期、可选模型种子（默认1）
train_mahdrl_as_fast_vaaware_tl S Tight

# 参数：资源、截止期、可选目标列表、可选模型种子
eval_mahdrl_as_fast_vaaware_tl_all S Tight
eval_mahdrl_as_fast_vaaware_tl_all S Tight "M L" 1

# 第二个独立模型种子，评测第三个参数仍是目标列表
train_mahdrl_as_fast_vaaware_tl S Tight 2
eval_mahdrl_as_fast_vaaware_tl_all S Tight "S M L" 2
```

函数包含 nohup、独立日志/PID、START/DONE/ALL_DONE 标记和失败停止；source 只定义函数。`GPU=1` 可切换 GPU。脚本末尾列出全部9个源训练和评测调用，每次调用启动一个后台任务，按显存和算力安排并发。

直接运行入口：

```bash
python -u launch.py algorithms/mahdrl_as_fast_vaaware_tl/train.py \
  --task S --resource S --deadline Tight --episodes 600 --model-seed 1 \
  --experiment-protocol configs/f22_split_v1.json \
  --normalization-mode f19_shared_v1 --device cuda

python -u launch.py algorithms/mahdrl_as_fast_vaaware_tl/evaluate.py \
  --task M --resource S --deadline Tight \
  --train-task S --train-resource S --train-deadline Tight \
  --seeds 700-729 --model-seed 1 \
  --experiment-protocol configs/f22_split_v1.json \
  --normalization-mode f19_shared_v1 --device cuda
```

追加 `--dry-run` 查看实际身份、协议和路径。没有 `--va-preview` 或 `--ha-update-period` 参数。压力负载可在训练/评测两侧同时追加 `--workload-profile stress_a1_l006_ref003`；标准负载沿用 .03 到达率。`precompute.py` 复用 FCFS 参考生成器，标准缓存无需重建；自定义拓扑需匹配缓存。

正式实验保持600回合、每回合50个工作流、训练实例0–599、验证600、测试700–729，9个源模型对应27个目标场景；建议至少3个训练种子，按源/目标/模型种子配对统计。30个测试负载种子不能替代独立训练种子。

与完整版比较评估完整 HA 决策机制的收益；关闭感知、HA周期1/4的现有四组控制用于分解机制作用。全局 VA 的输入组织和探索范围随结构改变，不能把完整方案/TL的全部差异归因于 HA 网络本身，也不预设完整方案能耗必然更低。

## 输出与验证

```text
outputs/ckpts/mahdrl_vaaware_tl/SS_T_m1_cXXXXXXXX/
outputs/logs/mahdrl_vaaware_tl/SS_T_m1_cXXXXXXXX/
outputs/run/mahdrl_vaaware_tl/eval_mahdrl_vaaware_tl_noapf_sr092_MS_T_from_SS_T_m1_s700-729_cXXXXXXXX.csv
```

标准场景模型种子1的训练末级目录17字符。最终CSV集中平铺，文件名包含目标/源场景、模型种子、实际测试种子和配置摘要，压力负载另带后缀。CSV行也包含场景信息及 `vm_selection_scope=global`；不带无效的 `p1h4`。明细在 `_details/<摘要>/`，已有训练目录或评测结果拒绝覆盖。

`assignments.jsonl` 记录全局索引、真实 VM/Host ID、合法候选数、VA版本/epsilon和分配能耗；`episodes.jsonl` / `evaluation.jsonl` 记录能耗审计和两角色更新时钟。`stage_timings.jsonl` 含全局观测、全局策略前向和学习计时，公共测量器仅统计 TA/VA。

```bash
python -X utf8 -B -m unittest common.test_mahdrl_as_fast_vaaware_tl -v
```

专项覆盖全局候选与主机信息、忙碌/超期合法性、固定分配物理/奖励等价、全局 GE、分配后更新、历史回放、终止/截断、CPU/CUDA、正式网络宽度更新、训练/保存/重载/冻结评测、身份与防覆盖、短路径和平铺场景CSV及不等VM拓扑。短程验证不代表600回合正式效果。

本次专项12项及相关回归共149项全部通过，0失败、0错误、0跳过，用时168.310秒。训练/评测/预计算三个公开启动入口dry-run通过，后台脚本通过Bash语法及44次替身调用验证，评测失败会停止后续目标。9个训练路径和27个最终CSV路径均唯一；训练末级目录17字符，当前工作区最长训练目录绝对路径101字符、训练测量文件128字符、最终CSV157字符。23个原VAware/旧TL代码文件哈希保持不变，未启动正式600回合训练。机器可读记录见 [验证报告](MAHDRL_AS_Fast_VAware_TL_validation.json)。
