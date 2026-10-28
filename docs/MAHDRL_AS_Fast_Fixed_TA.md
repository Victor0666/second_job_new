> **2026-10-29 当前版本更新：**本页对应变体现已统一关闭主机和 VM 两级 APF，采用严格 `SR > 0.92` 选模、短训练目录和按场景命名的平铺最终 CSV。当前定义、版本隔离与路径示例见[统一说明](MAHDRL_AS_Fast_消融统一配置_2026-10-29.md)。原命令入口仍有效；下文旧阈值、APF 和路径描述为历史实现记录，以统一说明为准。

# MAHDRL-AS-Fast-V2：ERTF、EDF、CP 固定排序，无主机/VM APF

三个变体以固定规则替换学习式 TA，并取消主机和 VM 两级 PF/APF。HA/VA 继续学习，
保留 GE 引导探索、OE 分配能耗账本、奖励、网络输入形状、学习参数和 Fast 实现。
完整版、NTA、NHA、NVA、TL、Sync 的代码和输出路径不受影响。

HA 的合法候选为所有至少有一个物理合法空闲 VM 的主机，VA 的合法候选为选中主机内
全部物理合法空闲 VM。空闲容差仍为 `current_time + 1e-12`；忙碌 VM、填充槽位不可选。
不执行速度比例筛选、不保留/排除能耗特例候选、不按预计是否按时完成删动作。
主机候选摘要改为基于全部物理合法候选计算，HA 23/10、VA 30/7 的输入形状不变。
GE 的 0.5 引导概率及原 epsilon 规则保持不变；GE 的探索动作偏好不改变合法动作 mask。

这是固定排序与移除两级 APF 的联合设置，相对完整版的差异不能全部归因于 TA 排序。

| 规则 | 独立入口目录（位于 algorithms/） | 原始字段排序键 |
| --- | --- | --- |
| ERTF | mahdrl_oe_ge_apf_score_fast_ertf | `(task_ready_time, global_task_id)` 升序 |
| EDF | mahdrl_oe_ge_apf_score_fast_edf | `(task_baseline_finish, task_ready_time, global_task_id)` 升序 |
| CP | mahdrl_oe_ge_apf_score_fast_cp | `(-task_up_rank, task_ready_time, global_task_id)` 升序 |

各目录均提供 `train.py`、`evaluate.py`、`precompute.py`。共用实现位于
`algorithms/mahdrl_oe_ge_apf_score_fast_fixed_ta/`，固定排序环境位于
`environments/mahdrl_oe_ge_apf_score_fast_fixed_ta.py`。

ERTF 使用环境记录的任务进入 Ready 状态的时间。EDF 使用已有的任务绝对子截止期，
不改为工作流截止期或松弛度；已经逾期的 Ready 任务仍参与排序。CP 使用现有向上秩：
任务上传、计算和下载的估计耗时，加上最大后继向上秩；复用工作流到达时的计算结果。
CP 只引入排序，不引入 HEFT 的资源分配。全部排序只面向已到达工作流的 Ready 任务。

排序仍在原物理阶段准备时执行一次，不逐次分配重排，不额外推进仿真时间。
继续允许同一仿真时刻连续分配任务。直接比较原始字段，避免归一化产生额外并列。
兼容权重固定为 ERTF `[1,0,0,0,0]`、EDF `[0,0,0,0,1]`、CP `[0,0,1,0,0]`，
禁止运行时修改权重。CP 权重对应当前代码实际的第三个属性 `task_up_rank`。

## 训练与模型身份

正式学习器仅为 HA、VA。初始化时沿用现有 NTA 的基线顺序对齐方法，短暂初始化
TA 后释放，使相同模型种子下 HA/VA 初始参数与完整版一致；TA 不参与决策、学习、
保存或性能测量。阶段 TA 奖励仅作为原有统计量记录。

模型身份分别为 `mahdrl_as_fast_v2_ertf_noapf_v2`、`mahdrl_as_fast_v2_edf_noapf_v2`、
`mahdrl_as_fast_v2_cp_noapf_v2`。每个变体独立重训，best 文件为 `best_host.pth`、
`best_vm.pth`，final 为 `host_final.pth`、`vm_final.pth`，另有 `ta_policy.json`。
严格校验规则、两级 APF 关闭状态、角色、源归一化、协议、保存组和验证证据；拒绝
原含 APF 的固定排序 v1 权重、跨规则、其他变体、混合组及 final 重命名冒充 best。
新模型/日志目录与旧版本隔离，原含 APF 的模型不能直接用于本版本评测，需要重训。

与现有主试验一致：模型种子默认且正式命令固定为 **1**；600 回合，每回合 50 个
工作流；训练实例 0–599、每回合验证 600、测试 700–729。复用 F19 和 FCFS 截止期缓存。
选模沿用 `select_sr099_minenergy_v1`，成功率达标后最低能耗优先，未达标时成功率优先。

## 场景直接写入最终 CSV 文件名

同一变体、同一输出版本下，所有目标场景的最终评测 CSV 共用目录：

```text
outputs/run/mahdrl_as_fast_ertf/v2_<配置摘要>/
  mahdrl_as_fast_ertf_noapf_SS_L_from_SS_L_model1_seeds700-729.csv
  mahdrl_as_fast_ertf_noapf_SS_M_from_SS_M_model1_seeds700-729.csv
  mahdrl_as_fast_ertf_noapf_MS_L_from_SS_L_model1_seeds700-729.csv
  details/<运行摘要>/...
```

不创建 `eval_taskS_resS_Loose` 或 `from_taskS_resS_Loose_model1` 等场景目录。
命名中的前两个字母分别代表目标任务规模和资源规模，`T/M/L` 分别代表
Tight/Medium/Loose；`from_...` 表示源模型场景。算法规则、目标场景、源模型、
模型种子和测试种子范围均在文件名中，因此可将三个变体的最终 CSV 直接汇总到
同一个目录。非连续测试种子集合额外带集合摘要，防止相同起止值重名。

CSV 每行对应一个测试实例，除原有质量指标外，包含 `algorithm`、`model_version`、
`host_apf_enabled=False`、`vm_apf_enabled=False`、`ordering_rule`、
`scene`、`task`、`resource`、`deadline`、`source_scene`、`train_task`、`train_resource`、
`train_deadline`、`model_seed`、`seed`、`selected_episode`。详细 JSON 和逐工作流
测量明细放入独立的 `details/<运行摘要>/`，不把场景编码放进目录名。

不同场景可以并发写入公共目录；每个进程独占自己的 CSV 和明细目录。
已存在的 CSV 或明细目录拒绝覆盖。日志中输出最终 CSV 的绝对路径，`--dry-run`
也会输出 `evaluation_csv` 和 `evaluation_details_dir`。

`candidates_episode*.json`、`candidates_seed*.json` 记录原始合法候选数量，明确标记
两级 APF 关闭，`r_H/r_V` 为 null。底层仍复用集合观测的候选快照接口，但不会运行
APF 算法或任何启发式动作筛选。

## Linux/CUDA：每个变体内部九场景并行

在仓库根目录激活环境后执行。一次选择一个规则；完成该规则的训练、评测后，
再将 `RULE` 改为 `edf` 或 `cp` 运行下一变体。不要同时启动三个变体。
`GPUS` 按实际设备修改。每个循环先后台启动九个作业，末尾才等待，不逐场景串行。

```bash
RULE=ertf
GPUS=(0)
TAG=$(date +%Y%m%d_%H%M%S)
mkdir -p outputs/server_console_logs
pids=()
i=0
for resource in S M L; do
  for deadline in Tight Medium Loose; do
    gpu=${GPUS[$((i % ${#GPUS[@]}))]}
    log="outputs/server_console_logs/${RULE}_noapf_train_S${resource}_${deadline}_model1_${TAG}"
    CUDA_VISIBLE_DEVICES="$gpu" nohup python -u launch.py \
      "algorithms/mahdrl_oe_ge_apf_score_fast_${RULE}/train.py" \
      --task S --resource "$resource" --deadline "$deadline" --episodes 600 --model-seed 1 \
      --experiment-protocol configs/f22_split_v1.json --normalization-mode f19_shared_v1 --device cuda \
      > "${log}.log" 2>&1 < /dev/null &
    pids+=("$!")
    printf '%s\n' "$!" > "${log}.pid"
    i=$((i + 1))
  done
done
for pid in "${pids[@]}"; do
  wait "$pid" || printf '训练失败 PID=%s，请检查日志\n' "$pid"
done
```

确认该规则的九个训练均成功完成后，再启动对应九场景评测：

```bash
RULE=ertf
GPUS=(0)
TAG=$(date +%Y%m%d_%H%M%S)
mkdir -p outputs/server_eval_logs
pids=()
i=0
for resource in S M L; do
  for deadline in Tight Medium Loose; do
    gpu=${GPUS[$((i % ${#GPUS[@]}))]}
    log="outputs/server_eval_logs/${RULE}_noapf_eval_S${resource}_${deadline}_model1_${TAG}"
    CUDA_VISIBLE_DEVICES="$gpu" nohup python -u launch.py \
      "algorithms/mahdrl_oe_ge_apf_score_fast_${RULE}/evaluate.py" \
      --task S --resource "$resource" --deadline "$deadline" \
      --train-task S --train-resource "$resource" --train-deadline "$deadline" \
      --model-seed 1 --seeds 700-729 --experiment-protocol configs/f22_split_v1.json \
      --normalization-mode f19_shared_v1 --device cuda > "${log}.log" 2>&1 < /dev/null &
    pids+=("$!")
    printf '%s\n' "$!" > "${log}.pid"
    i=$((i + 1))
  done
done
for pid in "${pids[@]}"; do
  wait "$pid" || printf '评测失败 PID=%s，请检查日志\n' "$pid"
done
```

任务 M/L 泛化评测只改变目标 `--task`，其 `--train-task S`、源资源与源截止期保持
不变；CSV 的目标场景和 `from_...` 随之区分。自定义不等 VM 数量拓扑仍只用于评测
和预计算，不用于训练。

## 验证

```bash
python -X utf8 -B -m unittest common.test_mahdrl_as_fast_fixed_ta common.test_mahdrl_as_fast common.test_mahdrl_as_fast_nta common.test_mahdrl_as_fast_nha common.test_mahdrl_as_fast_nva common.test_mahdrl_as_fast_tl common.test_mahdrl_as_fast_sync -v
```

专项测试涵盖规则方向与并列、CP 通信耗时、慢主机/慢 VM/逾期候选保留、忙碌及填充
动作拒绝、原始 OE 物理/奖励一致性、GE 两种探索分支保留、HA/VA 初始参数对齐、实际
CPU/CUDA 更新、真实短程训练/选模/冻结评测、旧含 APF 模型及规则/保存组防混用、场景
CSV 共存与拒绝覆盖、81 个标准目标组合的文件名唯一性及不等 VM 拓扑迁移。
自动测试不启动正式 600 回合实验，也不预设哪种排序策略更优。
