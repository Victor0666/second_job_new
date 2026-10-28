> **2026-10-29 当前版本更新：**本页对应变体现已统一关闭主机和 VM 两级 APF，采用严格 `SR > 0.92` 选模、短训练目录和按场景命名的平铺最终 CSV。当前定义、版本隔离与路径示例见[统一说明](MAHDRL_AS_Fast_消融统一配置_2026-10-29.md)。原命令入口仍有效；下文旧阈值、APF 和路径描述为历史实现记录，以统一说明为准。

# MAHDRL-AS-Fast-V2-NTA

独立入口为 `algorithms/mahdrl_oe_ge_apf_score_fast_nta/`，运行身份为
`mahdrl_as_fast_v2_nta_v1`，基线为 `mahdrl_as_fast_v2`。

本消融仅将 TA 的学习策略替换成随机合法权重增量策略。任务仍按五种属性的
加权分数排序，HA/VA 仍依次选择主机和该主机内的 VM。物理环境直接复用
`environments/mahdrl_oe_ge_apf_score_fast.py`。

## 消融定义

每回合初始权重、五种排序属性、243 个候选动作、动作 mask、非负约束及
权重归一化都沿用基线。上层每次从当前合法动作中等概率抽取一个
`{-0.1, 0, +0.1}^5` 增量，并调用环境原有 `apply_manager_delta()`。
该接口同时使 Fast 决策缓存失效。触发位置沿用基线：回合开始一次，之后每个
非终止 phase 结束后一次。空合法集合直接报错，不把编号 0 当作零增量。

随机策略使用独立 `numpy.random.RandomState`，种子沿用旧 NTA 定义：

```text
(instance_seed * 1000003 + 20260325 + 9176) & 0xFFFFFFFF
```

训练、验证和测试均在每个回合开始时根据实例 seed 重建随机流。验证 seed600
每回合使用同一起点；测试 HA/VA 为贪心策略，上层继续执行可复现的随机动作。
上层不会消耗环境 RNG、Python 全局 RNG 或 NumPy 全局 RNG。

正式学习器仅含 HA、VA。没有 TA 经验、更新、梯度或学习权重文件；phase
级 TA 奖励仍计算并记录为随机策略的回报。为使相同 model-seed 下的 HA/VA
初始参数及初始化后的 RNG 状态与基线一致，构造时短暂按原顺序初始化 TA，
随后立即释放。它不参与决策、训练、保存或测量；测量仍沿用基线排除初始化阶段。

保留 AS-Fast v2 的候选集合打分网络、HA 23/10 与 VA 30/7 输入配置、OE
账本和奖励、GE 探索、APF 规则、D3QN 更新、学习率、batch=256、经验容量、
warmup、epsilon 衰减、目标网络更新、梯度裁剪和 Fast 加速实现。排序变化后，
物理轨迹和后续候选集合可能变化，这属于被测策略的影响。

## 模型选择与隔离

直接复用 V2 的 `select_sr099_minenergy_v1`，每训练回合进行一次验证：

- 按全部验证工作流汇总成功率，SR >= 0.99 的候选优先，再选平均能耗最低者。
- 尚无达标候选时，选成功率最高者，再比较能耗。
- 达标候选能耗相同时优先成功率更高者；完全并列保留较早回合。

默认 seed600 含 50 个工作流，49/50 不达标，须 50/50 成功。该阈值是
验证选择规则，不保证未知实例的总体成功概率。

best 仅包含 `best_host.pth`、`best_vm.pth`；final 仅包含 `host_final.pth`、
`vm_final.pth`。另有 `ta_policy.json` 描述随机策略版本和种子派生规则。
HA/VA 的 best 必须具有相同保存组、选中回合和验证证据。

模型、日志和评测目录独立于完整 AS-Fast v2。加载检查算法身份、角色、参数
形状、训练源、归一化、协议、随机 TA 策略、选择证据和保存组；完整版、旧 NTA、
R090 或其他策略权重不能作为本版本模型使用。final 改名为 best 也不能通过
检查。沿用非空输出拒绝覆盖与无断点续训规则。

## 使用

在项目根目录、已激活依赖环境下执行。现有匹配的 FCFS 截止期缓存和训练源
F19 归一化可以复用。

```bash
python -u launch.py algorithms/mahdrl_oe_ge_apf_score_fast_nta/train.py --task S --resource S --deadline Tight --episodes 600 --model-seed 1 --experiment-protocol configs/f22_split_v1.json --normalization-mode f19_shared_v1 --device cuda

python -u launch.py algorithms/mahdrl_oe_ge_apf_score_fast_nta/evaluate.py --task S --resource S --deadline Tight --train-task S --train-resource S --train-deadline Tight --seeds 700-729 --model-seed 1 --experiment-protocol configs/f22_split_v1.json --normalization-mode f19_shared_v1 --device cuda

python launch.py algorithms/mahdrl_oe_ge_apf_score_fast_nta/train.py --task S --resource M --deadline Medium --dry-run
```

首轮九场景固定 task=S，交叉 resource=S/M/L 与 deadline=Tight/Medium/Loose。
默认 600 回合、每回合 50 个工作流，训练实例 0-599、验证 600、测试 700-729。
M/L 任务泛化使用相同资源与截止期下以 S 任务训练的模型，并保持 train-task=S。
自定义资源评测及预计算参数继承 Fast v2。

Linux Bash 下，以下命令立即后台并行启动九组训练（S 任务 × 三种资源 ×
三种截止期），每组独立控制台日志和 PID。`GPU=0` 指定物理 GPU 0，可按
服务器实际设备调整。同卡并发会叠加显存占用；测量单任务训练成本或推理延迟时
应独占设备。

```bash
GPU=0
LOG_DIR="outputs/server_console_logs/nta_train_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
for resource in S M L; do
  for deadline in Tight Medium Loose; do
    CUDA_VISIBLE_DEVICES="$GPU" nohup python -u launch.py algorithms/mahdrl_oe_ge_apf_score_fast_nta/train.py \
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

九组训练均完成后，再执行下面的后台并行评测：启动 27 个场景进程，每个
场景内部评测 30 个实例，共 810 次测试。每组资源和截止期都加载对应 S 任务
训练的 best 模型；只评测训练规模时，将 `for task in S M L` 改为
`for task in S`，即九场景、270 次测试。评测前须确认训练正常完成，不能在
相同保存组仍写入时加载 best 权重。

```bash
GPU=0
LOG_DIR="outputs/server_console_logs/nta_eval_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
for resource in S M L; do
  for deadline in Tight Medium Loose; do
    for task in S M L; do
      CUDA_VISIBLE_DEVICES="$GPU" nohup python -u launch.py algorithms/mahdrl_oe_ge_apf_score_fast_nta/evaluate.py \
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

每条 `nohup ... &` 都立即启动独立进程，循环不等待前一个场景完成。
日志按启动批次保存在不同目录；后台进程失败需查看对应日志。正式运行前可用
对应单条前台命令加 `--dry-run` 核对目录。

单独启动或重跑某一场景时，可只运行一条命令，例如 SS/Tight 训练：

```bash
mkdir -p outputs/server_console_logs
CUDA_VISIBLE_DEVICES=0 nohup python -u launch.py algorithms/mahdrl_oe_ge_apf_score_fast_nta/train.py --task S --resource S --deadline Tight --episodes 600 --model-seed 1 --experiment-protocol configs/f22_split_v1.json --normalization-mode f19_shared_v1 --device cuda > "outputs/server_console_logs/nta_train_SS_Tight_$(date +%Y%m%d_%H%M%S).log" 2>&1 < /dev/null &
echo "训练 PID：$!"
```

算法仍拒绝覆盖非空模型/日志/评测目录；更换控制台日志名不会改变实验输出
目录。重跑前应自行归档该场景已有输出。

`--diagnostics-every`、`--profile-mode`、`--profile-every` 与基线相同。对比时
固定设备、线程和这些选项，同时报告能耗、工作流成功率、超期及训练/推理费用。

## 可审计输出和验证

- `episodes.jsonl`、`evaluation.jsonl` 的 `random_ta` 记录策略、实例 seed、
  派生 seed 和上层决策次数。
- `phases.jsonl` 记录实际 `ta_action`、合法动作数量与更新后权重。
- 优化器、参数量和学习器阶段计时只统计 HA/VA；TA phase 回报是策略回报，
  不能解释为 TA 学习损失。评测推理计时另包含 `RandomTA` 的抽样调用。
- `validation_selection.jsonl` 与 `training_summary.json` 沿用 V2 的逐回合
  选择证据和 best/final 区分。checkpoint 的 `normalization.mahdrl_as_fast_nta`
  保存本变体选择记录。

```bash
python -X utf8 -B -m unittest common.test_mahdrl_as_fast_nta common.test_mahdrl_as_fast -v
```

NTA 测试包含合法随机采样和 RNG 隔离、真实权重随机游走与缓存失效、正式
网络的 HA/VA 初始化一致性、冻结评测复现、SR099 选择、权重和策略防混用、
保存组/角色/final 校验，以及临时目录中缩小网络和双工作流的实际训练、保存、
加载及评测。测量开关不改变训练权重和 best 选择。冒烟测试不代表九场景的
600 回合正式效果实验。
