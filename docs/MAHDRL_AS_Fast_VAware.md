# MAHDRL-AS-Fast-VAware

独立入口：`algorithms/mahdrl_as_fast_vaaware/`。模型身份：`mahdrl_as_fast_vaaware_noapf_v1`。基于 `mahdrl_as_fast_noapf_v3`，保留三层学习、OE 账本、原奖励、物理执行、F22 划分、F19 源归一化、工作负载配置和全部物理合法动作。未引入 APF、排队、等待动作、额外能耗罚分或短期前瞻。

## 机制

每次取出任务后，环境只读构造所有合法 Host 的 VA 状态。使用当前 VA online 网络批量前向一次。对于合法 VM 集合 M、预计最快集合 F、首个最高 Q 动作 g，训练行为概率为：

`p(v) = (1-epsilon)*I(v=g) + epsilon*(0.5/|M| + 0.5*I(v in F)/|F|)`。

非法 VM 概率为零。预计最快集合和并列容差沿用 GE。预评估不随机采样、不推进时钟、不增加动作计数、不写能耗账本；确定性验证/测试令 epsilon=0。实际训练 VA 从这个相同分布抽样，评测选择同一缓存 Q 的首个合法最大值。与 V3 的探索分布相同，但随机数调用序列不相同。

HA 原候选10维追加4维：期望完成耗时、期望分配边际能耗、期望超期惩罚和期望Q。耗时除以源 `T_ref`，能耗除以 `100*T_ref`，超期惩罚沿用 `clip(lateness/300,0,1)`。Q特征为 `(1-gamma)*E[Q]/R_ref`，其中 `R_ref=0.25*0.001*100*T_ref+0.75`，只使用源冻结尺度。Q作为输入，不与HA的Q直接相加。

默认HA为23/14，VA仍为30/7，TA仍为15维状态与243动作。关闭增强特征时HA为23/10。环境原始Host观测仍为23/10；运行时预评估适配器产生Agent实际接收的23/14观测。无合法空闲VM的Host不参与预评估，追加特征为零且动作继续屏蔽。

同一次任务的VA预评估、HA选择、VA选择之间不执行梯度更新。先为上一条经验补齐真实后继并入库，确定本次两层动作、执行分配并记录奖励后，才执行资源层更新。TA保持原phase决策和更新。阶段尾能耗补记、终止和截断沿用原规则。回放保存当时增强观测及真实后继增强观测，采样不重新运行当前VA生成历史特征。

默认每4条warmup后符合更新条件的HA经验更新一次，不在回合末补齐跳过的步数。HA epsilon按累计符合更新条件的经验数衰减，目标网络按实际优化器步数更新。VA和TA每条符合条件的真实经验更新一次。HA计数跨回合延续并保存在检查点；当前仍不提供断点续训。

## 四组消融

| 组别 | 参数 | HA候选维度 |
|---|---|---:|
| 新更新顺序控制 | `--no-va-preview --ha-update-period 1` | 10 |
| 仅VA感知 | `--va-preview --ha-update-period 1` | 14 |
| 仅低频HA | `--no-va-preview --ha-update-period 4` | 10 |
| 默认完整变式 | `--va-preview --ha-update-period 4` | 14 |

四组都执行相同的批量VA预评估和动作采样流程，只切换HA输入及HA更新周期。因此新顺序控制组不是原V3的逐随机数等价实现，可用于区分更新顺序与机制收益。原V3及TL入口不变。各组参数进入路径配置摘要和严格检查点身份，不能混合。

## 训练与评测

```bash
python -u launch.py algorithms/mahdrl_as_fast_vaaware/train.py \
  --task S --resource S --deadline Tight --episodes 600 --model-seed 1 \
  --experiment-protocol configs/f22_split_v1.json --normalization-mode f19_shared_v1 \
  --device cuda --va-preview --ha-update-period 4

python -u launch.py algorithms/mahdrl_as_fast_vaaware/evaluate.py \
  --task S --resource S --deadline Tight \
  --train-task S --train-resource S --train-deadline Tight --seeds 700-729 --model-seed 1 \
  --experiment-protocol configs/f22_split_v1.json --normalization-mode f19_shared_v1 \
  --device cuda --va-preview --ha-update-period 4
```

训练和评测必须使用同一机制参数。追加 `--dry-run` 查看路径；压力配置在两条命令中同时追加 `--workload-profile stress_a1_l006_ref003`。`precompute.py` 继续使用公共FCFS参考生成器，支持自定义拓扑。无需因本变式重建标准缓存或重校准。

后台运行函数见 `mahdrl_as_fast_vaaware_commands.sh`，带nohup、独立日志/PID、START/DONE/ALL_DONE标记及失败停止。source只加载函数。默认GPU=0，可用 `GPU=1` 指定设备。默认VA预评估开启、HA更新周期4、模型种子1：

```bash
source docs/mahdrl_as_fast_vaaware_commands.sh
train_mahdrl_as_fast_vaaware S Tight
eval_mahdrl_as_fast_vaaware_all S Tight
# 仅评测M/L任务目标：第三个参数是目标任务列表。
eval_mahdrl_as_fast_vaaware_all S Tight "M L"
# 控制组：训练参数依次为资源/截止期/预评估/HA周期/模型种子。
train_mahdrl_as_fast_vaaware S Tight off 1 1
# 评测参数依次为资源/截止期/目标列表/预评估/HA周期/模型种子。
eval_mahdrl_as_fast_vaaware_all S Tight "S M L" off 1 1
```

脚本末尾提供9个训练与9个源模型评测调用。训练9个S任务源场景；评测对应S/M/L任务目标，共27场景。正式效果比较至少使用多个独立训练种子。默认选模规则与V3一致：验证总体SR >= 0.92后最小平均能耗，再比较更高SR，未达标时先SR后能耗，完全并列保留更早回合。现有TL入口的严格SR>0.92属于不同选模协议，正式机制比较需统一该条件，不能将差异归因于HA。

## 产物与诊断

训练目录使用短标识 `mahdrl_vaaware`，源场景截止期缩写T/M/L分别表示Tight/Medium/Loose。标准场景、模型种子1下，末级目录始终为17字符：

```text
outputs/ckpts/mahdrl_vaaware/SS_L_m1_cXXXXXXXX/
outputs/logs/mahdrl_vaaware/SS_L_m1_cXXXXXXXX/
```

最终CSV统一平铺在 `outputs/run/mahdrl_vaaware/`；场景直接写进CSV文件名：

```text
eval_mahdrl_vaaware_noapf_sr092_p1h4_SS_L_from_SS_L_m1_s700-729_cXXXXXXXX.csv
eval_mahdrl_vaaware_noapf_sr092_p1h4_MS_L_from_SS_L_m1_s700-729_cXXXXXXXX.csv
eval_mahdrl_vaaware_noapf_sr092_p1h4_SS_M_from_SS_M_m1_s700-729_cYYYYYYYY.csv
```

`p1h4`表示预评估开启且HA更新周期4。文件名同时携带目标/源场景、模型种子、实际测试种子和配置摘要；CSV行也记录目标/源场景。非连续种子集合额外带摘要，压力负载另带后缀。明细位于 `_details/<运行摘要>/`，不把最终CSV藏在场景子目录。已有输出拒绝覆盖。

完整科学身份仍为 `mahdrl_as_fast_vaaware_noapf_v1`；只改变路径命名，不改变模型或配置摘要。已有旧长目录中的同版本模型可通过评测参数 `--checkpoint-dir` 显式指定，不移动或覆盖旧目录。

TA/HA/VA成组保存，加载核验模型身份、角色、形状、机制、源归一化、协议、选模证据、回合和保存组。旧V3/TL、不同预评估/更新周期、混合保存组及final冒充best均拒绝加载。

`assignments.jsonl` 记录预评估VA版本、epsilon、4个预期特征、真实能耗、预测误差、是否选择贪心VM，以及主机选择和主机内选择的即时能耗损失。损失仅对所选VM能按当前任务子截止期完成的决策计算，候选参照也必须能按时完成；不把它当长期最优性证明。

`episodes.jsonl` / `evaluation.jsonl` 汇总上述诊断和更新时钟。`stage_timings.jsonl` 的环境计时新增 `va_preview`（沿用原采样计时设置），同时保留完整回合测量；批量预评估的训练及推理费用需要随性能结果报告。

```bash
python -X utf8 -B -m unittest common.test_mahdrl_as_fast_vaaware -v
```

专项覆盖概率与并列、无副作用、合法性、策略版本一致、低频HA与epsilon、V3固定动作物理/奖励等价、真实CPU/CUDA更新、历史回放、终止结算、四组路径、工作负载配置、两回合训练/保存/重载/冻结评测、跨机制/混组/final拒绝和防覆盖。短程验证不代表正式600回合效果，未预设能耗必然优于TL。

本次相关回归共137项全部通过，0失败、0错误、0跳过，用时152.411秒，包含正式网络宽度更新、CPU/CUDA、不等VM数量拓扑以及V3和其他消融的兼容性。四组配置的36个训练目录及108个评测CSV均唯一，三个公开入口dry-run通过。标准场景模型种子1的训练末级目录均为17字符；当前工作区最长训练目录绝对路径98字符、训练测量文件125字符、最终CSV156字符。后台脚本通过Bash语法和44次替身调用检查，覆盖9组默认源场景和自定义控制参数，并验证中途失败停止后续目标；未启动正式训练。24个V3/TL基线文件哈希未变。机器可读记录见 `MAHDRL_AS_Fast_VAware_validation.json`。
