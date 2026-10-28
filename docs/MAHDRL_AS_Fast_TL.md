> **2026-10-29 当前版本更新：**本页对应变体现已统一关闭主机和 VM 两级 APF，采用严格 `SR > 0.92` 选模、短训练目录和按场景命名的平铺最终 CSV。当前定义、版本隔离与路径示例见[统一说明](MAHDRL_AS_Fast_消融统一配置_2026-10-29.md)。原命令入口仍有效；下文旧阈值、APF 和路径描述为历史实现记录，以统一说明为准。

# MAHDRL-AS-Fast-V2-TL：删除 HA 和主机 APF，VA 全局学习选 VM

独立入口为 `algorithms/mahdrl_oe_ge_apf_score_fast_tl/`，模型身份为
`mahdrl_as_fast_v2_tl_v1`，环境为 `environments/mahdrl_oe_ge_apf_score_fast_tl.py`。
正式学习器只有 TA、VA；完整版、NTA、NHA、NVA 的实现及输出路径保持独立。

## 实验定义

调度流程：TA 学习调整排序权重 → 按排序取当前任务 → 在所有主机分别执行
局部 VM APF → 合并候选 → VA 一次选择全局 VM → 从该 VM 确定所属主机并分配。

| 项目 | AS-Fast v2 | TL |
| --- | --- | --- |
| TA 排序与阶段学习 | 保留 | 相同 |
| HA 资源动作与学习 | 学习选择主机 | 删除，无随机主机策略 |
| 主机 APF | 开启 | 关闭，无主机速度或能耗筛选 |
| VM APF | 各主机内部筛选 | 保留原规则，各主机分别筛选后取并集 |
| VA 动作 | 选定主机内的 VM slot | `vm_ids` 顺序下的全局 VM 索引 |
| VA 输入 | context30 / candidate7 | context23 / candidate14 |
| VA GE、D3QN、Fast replay | 保留 | 相同学习规则，作用于全局合法动作 |
| OE、物理执行、TA 奖励 | 当前实现 | 相同 |
| checkpoint 选择 | SR099 优先，再比较能耗 | 相同 |

“所有主机”指所有至少有一个物理合法空闲 VM 的主机，允许主机同时运行其他任务。
空闲判定沿用 VA 的 `current_time + 1e-12` 容差。VM APF 继续保留本主机内速度
合格的 VM，以及能够按时完成且边际能耗最低的全部并列 VM。rV 的全局供需、
紧迫度输入和局部最快耗时参照不变，不引入新的全局速度阈值。
没有合法动作时保留任务并使用现有阶段推进；截止期不作为额外硬动作过滤条件。

这是删除中层结构与主机 APF 的联合消融，与完整版的差异不能全部归因于 HA。
NHA 使用相同的候选过滤规则，但先随机选主机；TL 由 VA 直接跨主机学习选择。

## 观测、经验和奖励

公共 context23 为任务12维、系统10维、剩余子截止期1维。每个 candidate14 按
以下顺序组织：原 VM7维、所属主机5维、所属主机空闲/当前功率2维。主机信息从
原 VA 上下文移到对应候选行，仍使用共享候选 Dueling 打分，不使用固定数量输出头。

每个真实 VM 占一行，包括暂时忙碌的 VM。`exists_mask` 标记真实行，动作 mask
叠加物理空闲条件与 VM APF；填充行和非法候选不可选择。参数形状与 VM 数量无关。
VA 的真实后继为下一次全局 VA 决策，阶段间保留 pending 轨迹，终止/截断按原规则结算。

VA 奖励保持 `(1-alpha_vm)*(-delta_energy*energy_reward_scale)+alpha_vm*r_delay`，
默认 `alpha_vm=.75`、`energy_reward_scale=.001`。能耗来自现有逐次分配边际账本。
阶段末未返回的分配能耗仅补记到 VA 一次。TA 阶段奖励不变，不合并 HA 奖励。

TA/VA 从头训练，VA 输入形状已改变。无临时 HA 初始化、经验池、优化器或 HA 模型。
沿用原学习率、batch、buffer、epsilon、GE 概率、隐藏层和目标网络更新规则。

## 保存与协议

best 文件只有 `best_manager.pth`、`best_vm.pth`；final 文件只有
`manager_final.pth`、`vm_final.pth`，另有 `tl_policy.json`。模型须来自同一保存组、
同一选中回合和相同验证证据。检查点严格校验 TL 身份、角色、23/14 特征形状、
APF 配置、协议和源训练归一化。拒绝其他变体、混合保存组和 final 冒充 best。

沿用 V2 的 `select_sr099_minenergy_v1`：验证工作流总体 SR >= .99 的模型优先，
其间比较最低平均能耗；未达标时先比较成功率，再比较能耗，完全并列保留较早回合。
默认50个验证工作流需要50/50成功，该阈值不保证测试总体成功率。

正式协议：600回合，每回合50个工作流；训练实例种子0–599，验证600，测试700–729。
训练 taskS × resourceS/M/L × Tight/Medium/Loose 共9个源模型，再评测27个目标场景。
共享 F19 归一化与 FCFS 截止期缓存；自定义拓扑仍仅用于评测和预计算。

## Linux/CUDA 并行运行

以下训练循环每项均以 `nohup ... &` 启动，9个场景并发，循环不等待上一场景完成。
默认使用 GPU 0；多卡时可将 `GPUS=(0)` 改为实际可用设备，例如 `GPUS=(0 1 2)`。
每个作业具有独立日志和 PID 文件；不要重复启动相同场景并写入同一模型目录。

```bash
GPUS=(0)
TAG=$(date +%Y%m%d_%H%M%S)
mkdir -p outputs/server_console_logs
i=0
for resource in S M L; do
  for deadline in Tight Medium Loose; do
    gpu=${GPUS[$((i % ${#GPUS[@]}))]}
    log="outputs/server_console_logs/tl_train_S${resource}_${deadline}_${TAG}"
    CUDA_VISIBLE_DEVICES="$gpu" nohup python -u launch.py algorithms/mahdrl_oe_ge_apf_score_fast_tl/train.py \
      --task S --resource "$resource" --deadline "$deadline" --episodes 600 --model-seed 1 \
      --experiment-protocol configs/f22_split_v1.json --normalization-mode f19_shared_v1 --device cuda \
      > "${log}.log" 2>&1 < /dev/null &
    printf '%s\n' "$!" > "${log}.pid"
    i=$((i + 1))
  done
done
```

训练完成后另行执行以下评测块，27个场景并发；每个进程内部按700–729评测30个种子。

```bash
GPUS=(0)
TAG=$(date +%Y%m%d_%H%M%S)
mkdir -p outputs/server_console_logs
i=0
for resource in S M L; do
  for deadline in Tight Medium Loose; do
    for task in S M L; do
      gpu=${GPUS[$((i % ${#GPUS[@]}))]}
      log="outputs/server_console_logs/tl_eval_${task}${resource}_${deadline}_${TAG}"
      CUDA_VISIBLE_DEVICES="$gpu" nohup python -u launch.py algorithms/mahdrl_oe_ge_apf_score_fast_tl/evaluate.py \
        --task "$task" --resource "$resource" --deadline "$deadline" \
        --train-task S --train-resource "$resource" --train-deadline "$deadline" \
        --seeds 700-729 --model-seed 1 --experiment-protocol configs/f22_split_v1.json \
        --normalization-mode f19_shared_v1 --device cuda \
        > "${log}.log" 2>&1 < /dev/null &
      printf '%s\n' "$!" > "${log}.pid"
      i=$((i + 1))
    done
  done
done
```

## 验证和产物

`assignments.jsonl` 记录训练分配的任务、全局 VM 动作、实际 VM/主机 ID、合法 VM
和主机数量。`apf_episode*.json`、`apf_seed*.json` 明确记录主机 APF 关闭、局部 VM
APF 保留、全局候选数量。性能测量只统计 TA、VA 两个学习策略。

```bash
python -X utf8 -B -m unittest common.test_mahdrl_as_fast_tl common.test_mahdrl_as_fast common.test_mahdrl_as_fast_nta common.test_mahdrl_as_fast_nha common.test_mahdrl_as_fast_nva -v
```

测试覆盖被旧主机 APF 排除的主机重新可选、局部 VM APF、忙碌/非法动作、原版
观测信息与物理/奖励等价、真实训练与 best 选择、CUDA 更新、冻结评测、严格模型
身份、自定义不等 VM 数量拓扑迁移。正式600回合训练不属于自动测试。
