# MAHDRL-AS-Fast-V2：Sync-Task 与 Sync-Phase 更新时钟消融

两个变体都从三层完整版派生，保留 TA、HA、VA、主机/VM 两级 APF、OE 分配能耗
账本、GE、集合打分网络以及原有学习参数。没有修改完整版或 NTA/NHA/NVA/TL。

| 项目 | 原 Async | Sync-Task | Sync-Phase |
| --- | --- | --- | --- |
| TA 决策/排序 | 每阶段 | 每次任务分配前 | 每阶段 |
| HA/VA 决策 | 每次分配 | 每次分配 | 每次分配 |
| 优化器触发 | TA阶段级、HA/VA各自真实经验就绪 | 每次分配末共同检查 | 阶段切换点集中执行 |
| TA 奖励窗口 | 物理阶段 | 相邻真实 TA 决策 | 原物理阶段 |
| HA/VA 更新预算 | 每条达到预热条件的完整经验一次 | 每次分配每个预热完成角色至多一次 | 每条预热完成经验记一次请求，阶段末补齐 |

独立入口：

- `algorithms/mahdrl_oe_ge_apf_score_fast_sync_task/{train,evaluate,precompute}.py`
- `algorithms/mahdrl_oe_ge_apf_score_fast_sync_phase/{train,evaluate,precompute}.py`

模型身份分别为 `mahdrl_as_fast_v2_sync_task_v1`、`mahdrl_as_fast_v2_sync_phase_v1`。
共用实现位于 `algorithms/mahdrl_oe_ge_apf_score_fast_sync/`，环境为
`environments/mahdrl_oe_ge_apf_score_fast_sync.py`。模式由各自场景类固定，不修改全局注册表。

## Sync-Task 的执行边界

每个可实际分配任务的时刻，先读取 TA 状态/合法增量 mask，重新决策并应用权重，
对全部剩余 Ready 任务计算优先级，再取队首交给 HA、VA 分配。
如果仍有 Ready 任务和合法空闲 VM，就在相同仿真时刻重复这一流程。

逐任务权重更新与物理阶段管理分开，不重置阶段分配数、MI累计、能耗或迟到统计。
每次应用权重都失效旧观测/APF缓存；重新排序包括Ready刷新、特征计算、加权评分及排序。
仅当原事件引擎需要推进时才推进；原 `vm_assign` 已推进时间后，必须先结清该物理阶段，
不把新时刻的任务错误地拼进上一阶段。

TA/HA/VA 在一次分配中依次选择动作，期间不运行优化器。分配末共同检查更新条件。
各角色沿用原预热阈值，尚未满足条件时跳过；三层不必在同一个回合首次满足预热条件。
阶段末那次分配的优化屏障延后至物理阶段结算后，以便终止经验先写入，再执行该次唯一屏障。

当前经验没有真实后继时保持 pending。下一次该角色真实决策出现后，才补齐经验并写入 replay。
同一屏障可以从 replay 中采样此前已经完整的经验，不强行构造当前动作的后继。

TA 奖励继续使用原物理能耗和新完成任务迟到增量的加权公式。原引擎在同一时刻连续分配时
不产生这些区间成本，因此中间 TA 短步奖励为零；物理阶段末的原 TA 奖励归入最后一个
尚待结算的 TA 动作，再连接到下一次真实 TA 决策。各短步奖励相加等于原物理区间总奖励。
不使用 VA 边际能耗替换 TA 能耗，不重复分发阶段奖励；终止/截断时结清最后一条经验。

## Sync-Phase 的执行边界与预算

TA仍在阶段开始调权重并排序，阶段内 HA/VA 依次分配。
完整经验写入 replay 时，只检查预热条件并累计该角色的更新请求，不执行梯度更新。
阶段切换点先结算奖励和当时可完成的经验，再按角色执行全部欠下的更新请求。
每次梯度更新都独立采样 batch，并执行原目标网络更新。

因此，一个阶段内可能补做多次 HA/VA 更新，不是每层只更新一次。跨阶段仍无真实后继的
下层 pending 经验保留到下一次实际决策，再计入届时所在阶段的更新请求。
终止经验在最终屏障前关闭，所有请求必须结清。

“预算匹配”指保留每条完整经验产生更新机会的规则，不承诺不同策略轨迹的三角色总步数完全相等。
TA阶段数量、策略轨迹和每回合实际更新数均记录。Sync-Task还改变TA决策频率、反馈/折扣时间尺度
和更新次数；其结果应解释为整体时间尺度设计对照。Sync-Phase保持决策和奖励窗口，用于辅助分析更新时机。
gamma仍按原角色决策步计算，epsilon仍按原优化步退火，未引入新的奖励、折扣或探索设置。

## 数据、选模和严格加载

沿用600回合、每回合50工作流；训练实例0–599、验证600、测试700–729。
F19归一化和FCFS截止期缓存复用；建议各变体使用model seed1–5独立训练。
9源场景和27目标场景的路径对应规则不变，自定义拓扑仍只用于评测/预计算。

保留 `select_sr099_minenergy_v1`：验证总体SR>=.99优先，再比较平均能耗；
未达标时先比较成功率再比较能耗，完全并列保留较早checkpoint。
默认50个验证工作流需要50/50成功，这不保证未知测试实例的总体成功率。

每组best为 `best_manager.pth`、`best_host.pth`、`best_vm.pth`，final对应
`manager_final.pth`、`host_final.pth`、`vm_final.pth`，另有 `synchronization.json`。
三角色必须属于相同更新模式、源配置、保存组和选中回合。拒绝另一同步变体、其他消融、
原版、混合保存组、修改过的时钟元数据、角色错配，以及将final重命名冒充best。
初始三层网络参数与相同model seed的完整版完全一致；两个变体均重新训练。

## 曲线与效率产物

- `learning_curves.csv`：每回合固定验证集SR/能耗、训练SR/能耗、TA决策数、分配数、排序次数/耗时及三角色实际优化步数。
- `episodes.jsonl`：完整回合指标、原始回合总回报、奖励审计、排序报告、更新请求/执行/剩余数量。
- `assignments.jsonl`：任务、主机/VM、TA/HA/VA动作、权重、决策时刻、分配后时刻及该次屏障执行步数。
- `phases.jsonl`：物理阶段成本及本阶段实际执行的优化步数。
- `training_summary.json`：累计优化步数、最后100回合验证波动、首次连续5次SR>=.99的完成回合数、持续达标后的回落事件数；未达标记null。
- `training_measurements.jsonl`：沿用原性能记录器的累计训练/验证耗时，可按episode与学习曲线合并，绘制实际训练时间横轴。
- 评测 `evaluation.jsonl`、`summary.json`：最终质量和排序次数、每千次分配排序次数、排序耗时的均值/标准差。

`rewards` 的单位为回合总回报，不能直接与原版按阶段平均的返回值比较。
波动统计使用未经平滑的固定验证曲线，单训练seed的时间波动不是跨训练seed置信区间。
模型间汇总应按独立model seed配对，测试实例700–729不是30次独立训练。
Async对照需要按相同计时口径采集排序耗时，不能用已有总训练时间替代排序耗时。

## Linux/CUDA 并行启动

在仓库根目录激活环境后执行。以下每项均为 `nohup ... &`，循环只提交作业，不等待上一场景。
默认GPU0，多卡可改为 `GPUS=(0 1 2)`。先以resource=M的三种deadline和5个model seed
训练两个新增变体，共30个并行进程；Async参考的15次训练需使用相同协议另行准备。

```bash
GPUS=(0)
TAG=$(date +%Y%m%d_%H%M%S)
mkdir -p outputs/server_console_logs
i=0
for mode in task phase; do
  for model_seed in 1 2 3 4 5; do
    for deadline in Tight Medium Loose; do
      gpu=${GPUS[$((i % ${#GPUS[@]}))]}
      log="outputs/server_console_logs/sync_${mode}_train_SM_${deadline}_model${model_seed}_${TAG}"
      CUDA_VISIBLE_DEVICES="$gpu" nohup python -u launch.py \
        "algorithms/mahdrl_oe_ge_apf_score_fast_sync_${mode}/train.py" \
        --task S --resource M --deadline "$deadline" --episodes 600 --model-seed "$model_seed" \
        --experiment-protocol configs/f22_split_v1.json --normalization-mode f19_shared_v1 --device cuda \
        > "${log}.log" 2>&1 < /dev/null &
      printf '%s\n' "$!" > "${log}.pid"
      i=$((i + 1))
    done
  done
done
```

上述训练全部结束后启动配套评测。每个源模型迁移到task S/M/L，90个场景进程并行，
各自评测700–729共30个实例。

```bash
GPUS=(0)
TAG=$(date +%Y%m%d_%H%M%S)
mkdir -p outputs/server_console_logs
i=0
for mode in task phase; do
  for model_seed in 1 2 3 4 5; do
    for deadline in Tight Medium Loose; do
      for task in S M L; do
        gpu=${GPUS[$((i % ${#GPUS[@]}))]}
        log="outputs/server_console_logs/sync_${mode}_eval_${task}M_${deadline}_model${model_seed}_${TAG}"
        CUDA_VISIBLE_DEVICES="$gpu" nohup python -u launch.py \
          "algorithms/mahdrl_oe_ge_apf_score_fast_sync_${mode}/evaluate.py" \
          --task "$task" --resource M --deadline "$deadline" \
          --train-task S --train-resource M --train-deadline "$deadline" \
          --model-seed "$model_seed" --seeds 700-729 --experiment-protocol configs/f22_split_v1.json \
          --normalization-mode f19_shared_v1 --device cuda > "${log}.log" 2>&1 < /dev/null &
        printf '%s\n' "$!" > "${log}.pid"
        i=$((i + 1))
      done
    done
  done
done
```

扩展到全部9源场景时，为resource增加S/M/L循环，评测的`--train-resource`必须匹配源模型。
排序/调度耗时比较需保持硬件与并发负载一致；不同负载下的计时不能直接解释为算法差异。

## 自动验证

```bash
python -X utf8 -B -m unittest common.test_mahdrl_as_fast_sync common.test_mahdrl_as_fast common.test_mahdrl_as_fast_nta common.test_mahdrl_as_fast_nha common.test_mahdrl_as_fast_nva common.test_mahdrl_as_fast_tl -v
```

专项测试包含真实短程训练、CUDA更新、best选模、冻结评测、模式/角色/保存组防混用、
相同动作下的原版物理与APF等价、同一时刻连续重排序、迟到奖励分段守恒、真实后继、
截断、终止、更新请求无遗漏/重复、诊断开关不改变参数及不等VM数量拓扑迁移。
自动测试不启动正式600回合实验，不预设稳定性或节能优于Async。
