# MAHDRL-AS-Fast-V2-NHA：全空闲候选主机随机选择，关闭主机 APF

独立入口为 `algorithms/mahdrl_oe_ge_apf_score_fast_nha/`，模型身份为
`mahdrl_as_fast_v2_nha_v1`，环境为 `environments/mahdrl_oe_ge_apf_score_fast_nha.py`。
本版本从 AS-Fast v2 派生，与完整版及 NTA 的代码、模型和结果分别隔离。

## 实验定义

调度流程为：TA 学习排序 → 在全部空闲候选主机中等概率随机选择 → VA 在所选
主机内的 VM APF 候选中学习选择 VM。“空闲候选主机”指至少存在一个合法空闲
VM 的主机；该主机允许同时存在其他运行中的 VM。

| 项目 | AS-Fast v2 | 本 NHA |
| --- | --- | --- |
| TA 排序和学习 | 保留 | 相同 |
| 主机候选 | 空闲条件 + 主机 APF | 仅合法空闲 VM 条件 |
| 主机选择 | HA 学习，训练含 GE | 每次决策等概率随机选择 |
| 主机 APF | 开启 | 关闭，不进行主机速度过滤或能耗保留 |
| VM APF | 开启 | 保留所有原规则及全局供需/紧迫度输入定义 |
| VA 集合打分网络与 GE | 保留 | 相同，30 维上下文 + 7 维候选特征 |
| 学习器 | TA、HA、VA | TA、VA |
| OE 奖励/能耗账本/物理模型 | 当前实现 | 相同 |
| checkpoint 选择 | SR099 优先，再比较能耗 | 相同 |

这是移除 HA 学习与主机 APF 的联合消融，不能将与完整版的差异全部归因于
HA 网络学习。VM 级 APF 可以继续排除同一主机内部过慢的 VM。

每台合法候选主机获得相同概率，不按其空闲 VM 数量、速度、功耗或负载加权。
只有一台候选时选择该主机；有效决策若无合法主机则报错。主机候选采用 VM
层相同的 `current_time + 1e-12` 空闲判定，避免选中主机却没有合法 VM。

环境复用 Fast 的观测构建、集合输入、缓存和物理执行；本地 VM APF 快照
返回全部有原始合法空闲 VM 的主机。VM 的局部速度阈值、截止期内最低边际
能耗保留（包含所有并列项）与基线相同。主机和 VM 选择入口按新 mask 校验。

## 随机数、学习与奖励

随机主机控制器使用独立 `numpy.random.RandomState`，每个训练、验证、测试
回合按实例 seed 重建，不消耗环境或学习器全局 RNG。固定公式为：

```text
(instance_seed * 1000003 + 20260325 + 123456 * 97) & 0xFFFFFFFF
```

验证 seed600 每回合从相同随机流起点开始；评测 TA/VA 贪心，HA 继续随机。
TA 和 VA 在这一随机主机策略下重新训练，不借用完整版训练权重。

正式学习器仅有 TA/VA，没有 HA 经验、梯度更新或学习权重文件。为保持同一
model-seed 下 TA/VA 的初始参数及初始化后 RNG 状态与基线一致，构造时短暂
按原顺序初始化 HA，随后立即释放。该临时对象不参与决策、训练、保存和测量。

TA phase 经验和 VA 的真实后继状态/mask 沿用原边界。VA 的后继来自下一次
真实的随机主机选择及其 VA 决策。本版本本地奖励结算仅向保留的 VA 轨迹补记
未返回的分配能耗，系数仍为 `(1-alpha_delay_vm)*energy_reward_scale`；符号、
终止、截断和防重复结算不变。HA 奖励仅作为随机策略回报统计。

## 选模与加载

直接复用 V2 的 `select_sr099_minenergy_v1`：验证工作流总体 SR >= 0.99 的
候选优先，其间选择最低平均能耗；尚无达标候选时先比较成功率，再比较能耗。
完全并列保留较早回合。默认 50 个验证工作流须全部成功，该门槛不保证未知
实例的总体成功概率。

best 文件仅为 `best_manager.pth` 和 `best_vm.pth`，final 文件仅为
`manager_final.pth` 和 `vm_final.pth`，另有 `ha_policy.json`。TA/VA 必须来自
同一保存组和选中回合。checkpoint 的 `normalization.mahdrl_as_fast_nha`
保存本变体选择证据。算法身份、角色、形状、源尺度、协议、归一化、随机主机
策略、主机 APF 关闭和 VM APF 保留设置均校验，拒绝完整版/NTA/旧 NHA 权重
混用，拒绝 final 改名为 best，以及非空模型、日志和评测目录覆盖。

## Linux/CUDA 后台并行命令

在项目根目录、已激活 Python/CUDA 环境下运行，复用匹配的 FCFS 截止期缓存
和 F19 训练源归一化文件。单条前台命令加 `--dry-run` 可先核对配置和目录。
默认训练实例 seed0-599、验证 seed600、测试 seed700-729，每回合 50 个工作流。

九组训练立即后台并行启动：S 任务 × 资源 S/M/L × 三种截止期，每组 600 回合。
`GPU=0` 指定物理 GPU 0，可按服务器设备修改；同卡并发显存占用会叠加。

```bash
GPU=0
LOG_DIR="outputs/server_console_logs/nha_train_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
for resource in S M L; do
  for deadline in Tight Medium Loose; do
    CUDA_VISIBLE_DEVICES="$GPU" nohup python -u launch.py algorithms/mahdrl_oe_ge_apf_score_fast_nha/train.py \
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

九组训练正常完成后，再启动 27 场景并行评测，每场景 30 个实例，共 810 次测试。
每场景加载对应资源、截止期下以 S 任务训练的模型。只评测训练规模时，将
`for task in S M L` 改成 `for task in S`，即九场景、270 次测试。

```bash
GPU=0
LOG_DIR="outputs/server_console_logs/nha_eval_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
for resource in S M L; do
  for deadline in Tight Medium Loose; do
    for task in S M L; do
      CUDA_VISIBLE_DEVICES="$GPU" nohup python -u launch.py algorithms/mahdrl_oe_ge_apf_score_fast_nha/evaluate.py \
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

每条 `nohup ... &` 启动后立即继续下一个场景；场景内的测试 seed 按原评测
流程执行。不要在训练仍写同一保存组时评测该组权重。测量单任务成本和推理
延迟时应独占设备。已有输出仍受覆盖保护，换控制台日志名不会绕过保护。

## 可审计输出与验证

- `episodes.jsonl` 和 `evaluation.jsonl` 的 `random_ha` 记录实例/派生种子、
  主机决策次数、多个候选的决策次数及比例。
- `hosts.jsonl` 记录每次训练分配的任务、主机动作索引、实际主机 ID 和合法
  主机数量；逐种子 `apf_seed*.json` 记录评测分配及 VM APF 详情。
- APF 记录明确 `host_apf_enabled=false`、`vm_apf_enabled=true`，原始与保留
  主机数相同。`r_H`、主机能耗补救数量和主机能耗专属选择标记为 null，避免
  将未启用的主机过滤解释成有效结果。
- 参数、优化器、学习器阶段计时仅统计 TA/VA；评测另外计时 `RandomHA`。
- 能耗审计保留物理账本及 VA 奖励平衡，不要求已移除的 HA 学习奖励平衡。

```bash
python -X utf8 -B -m unittest common.test_mahdrl_as_fast_nha common.test_mahdrl_as_fast common.test_mahdrl_as_fast_nta -v
```

测试覆盖：随机数隔离、等概率合法选择、可变候选数、慢主机重新可选、空闲
时间容差一致、VM APF 候选与基线一致、固定动作的物理/奖励/账本一致、正式
TA/VA 网络初始化一致、CUDA 实际更新、尾部能耗仅结算 VA、模型防混用、
SR099 选模及临时目录的实际训练→保存→加载→冻结评测。测量开关不改变训练
权重和 best 选择。小规模测试不代表正式九组 600 回合效果实验。
