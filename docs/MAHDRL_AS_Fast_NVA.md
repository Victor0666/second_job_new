# MAHDRL-AS-Fast-V2-NVA：主机 APF 保留，删除 VM APF，主机内随机选 VM

独立入口为 `algorithms/mahdrl_oe_ge_apf_score_fast_nva/`，模型身份为
`mahdrl_as_fast_v2_nva_v1`，环境为 `environments/mahdrl_oe_ge_apf_score_fast_nva.py`。
本版本从完整 AS-Fast v2 派生，与完整版、NTA、NHA 的模型和结果独立隔离。

## 实验定义

调度流程为：TA 学习排序 → HA 在主机 APF 候选中学习选主机 → 在所选主机的
全部合法空闲 VM 中等概率随机选一个。VM 的速度、功耗、带宽、预测完成时间
和 Q 值都不参与抽样加权；预测超期的空闲 VM 也允许选择。只有一个候选时
选择该 VM；有效决策若无合法 VM 则明确报错。

| 项目 | AS-Fast v2 | 本 NVA |
| --- | --- | --- |
| TA 排序和学习 | 保留 | 相同 |
| HA 学习/集合打分/GE | 保留 | 相同，23 维上下文 + 10 维候选特征 |
| 主机 APF | 开启 | 保留原速度过滤与能耗候选保留 |
| VM APF | 开启 | 删除，不做 VM 速度或能耗候选筛选 |
| VM 决策 | VA 学习，训练含 GE | 主机内全部合法空闲 VM 等概率随机 |
| 学习器 | TA、HA、VA | TA、HA |
| OE 奖励、账本和物理模型 | 当前实现 | 相同 |
| checkpoint 选择 | SR099 优先，再比较能耗 | 相同 |

这是移除 VA 学习与 VM APF 的联合消融，不能把与完整版的差异全部归因于
VA 网络学习。主机 APF 继续根据原始合法空闲 VM 的全局供需、紧迫度、各
主机最快时长和全局最低边际能耗计算，保留所有符合容差的并列项。

主机内 VM mask 仅保留存在性、所选主机归属和原空闲时间容差（1e-12），
排除忙碌和无效槽位。传给 `vm_assign()` 的是主机内部动作索引，不是全局
VM ID。HA 候选摘要根据其实际可选的全部空闲 VM 重新计算，保留输入维度，
避免沿用 VM APF 过滤后的摘要而与真实动作范围不一致。

## 随机数、训练及奖励

使用独立 `numpy.random.RandomState`，每个训练、验证、测试回合按实例 seed
重建，不消耗环境或学习器全局 RNG。派生公式沿用旧 NVA：

```text
(instance_seed * 1000003 + 20260325) & 0xFFFFFFFF
```

验证 seed600 每回合从相同随机流起点开始。评测 TA/HA 贪心，VM 继续随机。
TA/HA 在本策略下重新训练，不借用完整版或其他消融权重。

正式学习器仅有 TA/HA，没有 VA 经验、优化器更新、梯度或学习权重文件。
初始化短暂按基线顺序构造 TA→HA→VA 后释放 VA，以保持同一 model-seed 下
TA/HA 初始参数及初始化后 RNG 状态一致；临时 VA 不决策、训练、保存或测量。

TA phase 经验与 HA 的真实后继观测/mask 沿用原决策边界。未返回的分配能耗
只补给保留的 HA 轨迹，系数仍为 `(1-alpha_delay_host)*energy_reward_scale`。
保留终止、截断、防重复结算、物理能耗守恒及 HA 奖励平衡。VA 奖励仅作为
随机策略回报统计。

## 选模与模型隔离

直接复用 V2 `select_sr099_minenergy_v1`：验证工作流总体 SR >= 0.99 的
候选优先，再比较最低平均能耗；尚无达标候选时先比较成功率，再比较能耗。
完全并列保留较早回合。默认 50 个验证工作流须全部成功；门槛不保证未知
实例的总体成功概率。

best 仅含 `best_manager.pth`、`best_host.pth`，final 仅含 `manager_final.pth`、
`host_final.pth`，另有 `vm_policy.json`。TA/HA 必须来自同一保存组、选中回合
和验证证据。checkpoint 的 `normalization.mahdrl_as_fast_nva` 保存本变体的
选择记录。加载校验算法身份、角色、形状、训练源、归一化、协议、随机策略及
两级 APF 开关，拒绝完整版/NTA/NHA/旧 NVA 权重混用。final 改名为 best 也
不能通过检查；非空模型、日志和评测目录仍拒绝覆盖，无断点续训入口。

## Linux/CUDA 后台并行指令

在项目根目录、已激活 Python/CUDA 环境下执行，复用匹配的 FCFS 截止期缓存
和 F19 训练源归一化。单条前台命令加 `--dry-run` 可预览参数与路径。

九组训练立即后台并行启动：S 任务 × 资源 S/M/L × 三种截止期，每组 600
回合，每回合 50 个工作流。训练 seed0-599，验证 seed600，模型种子默认1。
`GPU=0` 指定物理 GPU 0，可按实际设备调整；同卡并发显存占用会叠加。

```bash
GPU=0
LOG_DIR="outputs/server_console_logs/nva_train_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
for resource in S M L; do
  for deadline in Tight Medium Loose; do
    CUDA_VISIBLE_DEVICES="$GPU" nohup python -u launch.py algorithms/mahdrl_oe_ge_apf_score_fast_nva/train.py \
      --task S --resource "$resource" --deadline "$deadline" \
      --episodes 600 --model-seed 1 \
      --experiment-protocol configs/f22_split_v1.json \
      --normalization-mode f19_shared_v1 --device cuda \
      > "$LOG_DIR/train_S${resource}_${deadline}.log" 2>&1 < /dev/null &
    echo $! > "$LOG_DIR/train_S${resource}_${deadline}.pid"
  done
done
echo "训练日志：$LOG_DIR"
```

九组训练正常完成后，再并行评测27场景，每场景测试 seed700-729，共810次
测试。加载相同资源、截止期下的 S 任务训练模型；只评测训练规模时，把
`for task in S M L` 改为 `for task in S`，即九场景、270次测试。

```bash
GPU=0
LOG_DIR="outputs/server_console_logs/nva_eval_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
for resource in S M L; do
  for deadline in Tight Medium Loose; do
    for task in S M L; do
      CUDA_VISIBLE_DEVICES="$GPU" nohup python -u launch.py algorithms/mahdrl_oe_ge_apf_score_fast_nva/evaluate.py \
        --task "$task" --resource "$resource" --deadline "$deadline" \
        --train-task S --train-resource "$resource" --train-deadline "$deadline" \
        --seeds 700-729 --model-seed 1 \
        --experiment-protocol configs/f22_split_v1.json \
        --normalization-mode f19_shared_v1 --device cuda \
        > "$LOG_DIR/eval_${task}${resource}_${deadline}.log" 2>&1 < /dev/null &
      echo $! > "$LOG_DIR/eval_${task}${resource}_${deadline}.pid"
    done
  done
done
echo "评测日志：$LOG_DIR"
```

每条 `nohup ... &` 启动后立即继续下一个场景，场景内的测试 seed 沿用原流程。
不要在训练仍写同一保存组时评测该组权重。测量单任务成本和延迟时独占设备。

## 可审计输出和验证

- `episodes.jsonl`、`evaluation.jsonl` 的 `random_va` 记录策略、实例/派生种子、
  随机 VM 决策次数及多个合法候选的决策比例。
- `vms.jsonl` 记录每次训练分配的任务、所选主机、主机内 VM 索引、实际 VM ID
  及合法 VM 数量；逐种子 `apf_seed*.json` 保留评测分配详情。
- APF 记录明确 `host_apf_enabled=true`、`vm_apf_enabled=false`。保留主机
  阈值及能耗补救统计；`r_V`、VM 能耗补救数量和能耗专属 VM 标记为 null。
  每次分配的主机内原始空闲 VM 数等于保留 VM 数。
- 参数、优化器和学习器阶段计时仅统计 TA/HA；评测另计时 `RandomVA`。
- 奖励审计保留物理账本与 HA 奖励平衡，不要求已移除的 VA 学习奖励平衡。

```bash
python -X utf8 -B -m unittest common.test_mahdrl_as_fast_nva common.test_mahdrl_as_fast common.test_mahdrl_as_fast_nta common.test_mahdrl_as_fast_nha -v
```

验证包括主机 APF 与基线一致、被 VM APF 排除的慢 VM 重新可选、HA 摘要覆盖
新增候选、主机内随机选择/忙碌 VM 排除、随机数隔离、正式 TA/HA 初始化一致、
CUDA 实际更新、固定动作物理/奖励/账本一致、尾部能耗只结算 HA、权重防混用、
SR099 选模以及临时目录中的训练→保存→加载→冻结评测。测量开关不改变
训练权重和 best 选择。小规模验证不代表九组600回合的正式效果实验。
