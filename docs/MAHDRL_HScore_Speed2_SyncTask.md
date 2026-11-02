# HScore Speed2 的 sync-task 同步变式

入口目录：`algorithms/mahdrl_as_fast_hscore_speed2_sync_task/`。
模型版本：`hscore_speed2_sync_task_srgt092_v1`。

## 同步定义与加速范围

沿用已有 `mahdrl_as_fast_hscore_sync_task` 的任务级同步逻辑：每次实际任务分配前，
TA 根据当前状态选择权重增量并重排可分配任务；随后 HA 选主机、VA 选本地主机 VM。
使用真实后继状态形成经验，分配完成后在统一屏障执行待处理的优化更新。
预览及一次 HA/VA 分配共享冻结的 VA 网络版本。

默认 HA period 仍为 4：同步屏障不代表强制三层在每次分配时各更新一次。
原有 warmup、batch size、更新资格判断、探索时钟、target 更新和终止处理全部复用。
TA 的回报按旧 sync-task 的区间规则累积；保留原训练记录中的 `episode_total` 回报单位。

Speed2 加速包括按原顺序采样的环形回放池、VA batch 打包、已结算能耗事件缓存，以及
直接构造归一化队列特征。同步 rollout 直接调用现有实现，不复制其任务调度逻辑。
同步环境和 Speed2 环境组合继承，以同时保留任务级 TA 方法与 Speed2 的能耗/观测方法。

三层网络、奖励权重 delay=0.75/energy=0.25、GE、preview 默认开启、固定 VA 93/9 维度、
主机内动作掩码、物理合法候选，以及严格 SR > 0.92 后最小验证能耗选模均保持原设置。
使用原 F22 训练 seed 0–599、验证 seed 600、测试 seed 700–729，复用缓存及归一化文件。

## Linux/CUDA 后台训练

在项目根目录、Python 环境激活后加载函数；source 本身不启动训练：

```bash
source docs/mahdrl_as_fast_hscore_speed2_sync_task_commands.sh

# 600 代：第三个参数默认 1
train_mahdrl_as_fast_hscore_speed2_sync_task M Tight

# 6000 代：第三个参数 10，600 个训练种子循环 10 轮
train_mahdrl_as_fast_hscore_speed2_sync_task M Tight 10
```

上述两条是不同预算的实验示例，按需选择。以下为 6000 代的全部九组调用，按需逐条启动；
同一 GPU 上建议每次运行一组：

```bash
# SS 源模型
train_mahdrl_as_fast_hscore_speed2_sync_task S Tight 10
train_mahdrl_as_fast_hscore_speed2_sync_task S Medium 10
train_mahdrl_as_fast_hscore_speed2_sync_task S Loose 10
# SM 源模型
train_mahdrl_as_fast_hscore_speed2_sync_task M Tight 10
train_mahdrl_as_fast_hscore_speed2_sync_task M Medium 10
train_mahdrl_as_fast_hscore_speed2_sync_task M Loose 10
# SL 源模型
train_mahdrl_as_fast_hscore_speed2_sync_task L Tight 10
train_mahdrl_as_fast_hscore_speed2_sync_task L Medium 10
train_mahdrl_as_fast_hscore_speed2_sync_task L Loose 10
```

6000 代在同一进程内持续学习，轮次切换不重置网络、经验池、优化器和探索率。
当前不支持从旧的 600 代模型续训；需要从初始化开始运行。

## 对应评测

训练完成后调用；第三个参数为目标 task 列表，第四个参数为源模型的训练种子轮数：

```bash
source docs/mahdrl_as_fast_hscore_speed2_sync_task_commands.sh

# SS 源模型评测 SS、MS、LS
eval_mahdrl_as_fast_hscore_speed2_sync_task_all S Tight "S M L" 10
eval_mahdrl_as_fast_hscore_speed2_sync_task_all S Medium "S M L" 10
eval_mahdrl_as_fast_hscore_speed2_sync_task_all S Loose "S M L" 10
# SM 源模型评测 SM、MM、LM
eval_mahdrl_as_fast_hscore_speed2_sync_task_all M Tight "S M L" 10
eval_mahdrl_as_fast_hscore_speed2_sync_task_all M Medium "S M L" 10
eval_mahdrl_as_fast_hscore_speed2_sync_task_all M Loose "S M L" 10
# SL 源模型评测 SL、ML、LL
eval_mahdrl_as_fast_hscore_speed2_sync_task_all L Tight "S M L" 10
eval_mahdrl_as_fast_hscore_speed2_sync_task_all L Medium "S M L" 10
eval_mahdrl_as_fast_hscore_speed2_sync_task_all L Loose "S M L" 10
```

若评测 600 代源模型，将末尾 `10` 改成 `1` 或省略。
默认评测整个训练过程按验证规则选出的最佳 checkpoint，冻结学习器。

直接执行或先检查路径的示例：

```bash
python -u launch.py algorithms/mahdrl_as_fast_hscore_speed2_sync_task/train.py \
    --task S --resource M --deadline Tight \
    --episodes 6000 --train-seed-cycles 10 --model-seed 1 \
    --experiment-protocol configs/f22_split_v1.json \
    --normalization-mode f19_shared_v1 --device cuda \
    --va-preview --ha-update-period 4 --dry-run
```

去掉 `--dry-run` 即开始前台训练。直接评测使用同目录 `evaluate.py`，
增加 `--train-task S --train-resource M --train-deadline Tight --seeds 700-729`，
并保留相同的训练预算、模型种子及机制参数。Windows 可通过 `run.cmd` 传入相同参数。

## 输出与检查

独立输出标签为 `hs_s2_sync_task`：

- 模型：`outputs/ckpts/hs_s2_sync_task/SM_T_m1_cXXXXXXXX/`。
- 训练日志：`outputs/logs/hs_s2_sync_task/SM_T_m1_cXXXXXXXX/`。
- 6000 代目录示例：`SM_T_e6000r10_m1_cXXXXXXXX/`。
- 最终 CSV 平铺在 `outputs/run/hs_s2_sync_task/`；文件名含目标场景、源场景、机制、
  模型种子、评测种子和配置哈希，6000 代另含 `_e6000r10.csv`。
- 后台启动日志/PID 分别在 `outputs/server_train_logs/` 和 `outputs/server_eval_logs/`。

新变式与 Speed2、旧 sync-task 的模型身份和输出隔离，严格拒绝混载或覆盖已有结果。

```bash
python -X utf8 -B -m unittest \
    common.test_mahdrl_as_fast_hscore_speed2_sync_task \
    common.test_hscore_seed_cycles \
    common.test_hscore_speed_paths \
    common.test_hscore_speed_commands -v
```

专项验证覆盖旧 sync-task 与新变式的 CPU/CUDA 实际学习，preview 开关，观测、掩码、
分配、奖励、物理记录、经验、随机状态、online/target 参数及 Adam 状态的一致性。
另外检查任务级 TA 计数、更新屏障、HA 更新频率、真实跨轮训练、最终/最佳模型加载、
冻结评测、防覆盖，以及短路径、平铺 CSV 和 600/6000 代后台函数。

完整长程训练和服务器耗时尚未测量。任务级 TA 决策及学习更新的次数更多，
该同步变式的整代耗时需要单独测量。
