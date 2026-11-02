# HScore：600 个训练种子循环 10 轮，训练 6000 代

支持原版 `mahdrl_hscore_fixedva_srgt092_v2`、Speed v1 和 Speed2。
这是一组从初始化开始的 6000 代新实验；当前不支持从旧的 600 代 checkpoint 续训。

## 参数与训练含义

训练和评测都增加：

```bash
--episodes 6000 --train-seed-cycles 10
```

保留 `--experiment-protocol configs/f22_split_v1.json`，不修改协议 JSON，也不向
`train` 数组添加重复条目。该文件继续定义原始的唯一实例划分及单轮 600 代预算。
扩展的总训练预算、轮次和顺序另外记入 `training_schedule` 元数据。

- 第 1–600 代使用 seed 0–599，第 601–1200 代再次使用 0–599，直到第 6000 代。
- 总计 6000 个训练 episode，600 个不同的训练实例；不是 6000 个不同实例。
- 同一训练进程持续维护网络、target 网络、经验池、优化器、探索率及更新计数。
  切换轮次仅重置环境；不能通过启动 10 次独立的 600 代训练代替。
- 仍然每代验证 seed 600；测试仍用 700–729；严格 SR > 0.92 后最小验证能耗选模。
  评测默认读取整个 6000 代过程中选出的最佳模型，最佳代可能早于第 6000 代。
- 网络结构、奖励、学习率、batch size、GE 和各层更新周期保持原值。
  探索率按原有学习时钟继续衰减，不随每轮重新开始，也不自动放慢 10 倍。
- 复用原截止期缓存和归一化文件，无需重新预计算或校准。

直接改成 `--episodes 6000` 仍会报错；必须显式指定 `--train-seed-cycles 10`。
重复训练时两参数必须满足 `总代数 = 协议单轮代数 × 轮数`，评测也必须一致。
不传新参数时仍然是原来的 600 代，旧模型命名和加载规则不变。

## Speed2：Linux/CUDA 后台训练

在项目根目录、已激活 Python 环境的终端运行。先更新到包含本改动的 Git 版本，然后：

```bash
source docs/mahdrl_as_fast_hscore_speed2_commands.sh

# 第三个参数是训练种子轮数；以下每条是一组独立实验，按需逐条启动。
# SS 源模型
train_mahdrl_as_fast_hscore_speed2 S Tight 10
train_mahdrl_as_fast_hscore_speed2 S Medium 10
train_mahdrl_as_fast_hscore_speed2 S Loose 10

# SM 源模型
train_mahdrl_as_fast_hscore_speed2 M Tight 10
train_mahdrl_as_fast_hscore_speed2 M Medium 10
train_mahdrl_as_fast_hscore_speed2 M Loose 10

# SL 源模型
train_mahdrl_as_fast_hscore_speed2 L Tight 10
train_mahdrl_as_fast_hscore_speed2 L Medium 10
train_mahdrl_as_fast_hscore_speed2 L Loose 10
```

函数会立即返回，作业在 GPU 0 后台训练；同一张 GPU 建议每次只启动一组。
日志和 PID 在 `outputs/server_train_logs/`，文件名包含 `e6000r10`。

## Speed2：对应源模型的评测

训练完成后执行；第三个参数是目标 task 列表，第四个参数是源模型的训练种子轮数：

```bash
source docs/mahdrl_as_fast_hscore_speed2_commands.sh

# SS 源模型：SS、MS、LS
eval_mahdrl_as_fast_hscore_speed2_all S Tight "S M L" 10
eval_mahdrl_as_fast_hscore_speed2_all S Medium "S M L" 10
eval_mahdrl_as_fast_hscore_speed2_all S Loose "S M L" 10

# SM 源模型：SM、MM、LM
eval_mahdrl_as_fast_hscore_speed2_all M Tight "S M L" 10
eval_mahdrl_as_fast_hscore_speed2_all M Medium "S M L" 10
eval_mahdrl_as_fast_hscore_speed2_all M Loose "S M L" 10

# SL 源模型：SL、ML、LL
eval_mahdrl_as_fast_hscore_speed2_all L Tight "S M L" 10
eval_mahdrl_as_fast_hscore_speed2_all L Medium "S M L" 10
eval_mahdrl_as_fast_hscore_speed2_all L Loose "S M L" 10
```

不传轮数时函数仍训练或评测旧的 600 代模型。

## 直接命令与其他版本

Speed2 的 SM/Tight 训练及评测示例：

```bash
CUDA_VISIBLE_DEVICES=0 python -u launch.py algorithms/mahdrl_as_fast_hscore_speed2/train.py \
    --task S --resource M --deadline Tight \
    --episodes 6000 --train-seed-cycles 10 --model-seed 1 \
    --experiment-protocol configs/f22_split_v1.json \
    --normalization-mode f19_shared_v1 --device cuda \
    --va-preview --ha-update-period 4 \
    --diagnostics-every 100 --profile-mode sampled --profile-every 100

CUDA_VISIBLE_DEVICES=0 python -u launch.py algorithms/mahdrl_as_fast_hscore_speed2/evaluate.py \
    --task S --resource M --deadline Tight \
    --train-task S --train-resource M --train-deadline Tight \
    --episodes 6000 --train-seed-cycles 10 --model-seed 1 --seeds 700-729 \
    --experiment-protocol configs/f22_split_v1.json \
    --normalization-mode f19_shared_v1 --device cuda \
    --va-preview --ha-update-period 4 \
    --diagnostics-every 100 --profile-mode sampled --profile-every 100
```

Speed v1：加载 `docs/mahdrl_as_fast_hscore_speed_commands.sh`，将函数名中的 `speed2`
替换为 `speed`，参数位置相同。原版 HScore：直接命令的脚本目录替换为
`algorithms/mahdrl_as_fast_hscore/`，其余参数相同。

Windows 可用 `run.cmd` 加同样的脚本路径和参数。先加 `--dry-run` 可检查路径与种子
划分，不启动训练。

## 输出与实验说明

模型目录仍短，例如 `outputs/ckpts/mahdrl_hscore_speed2/SM_T_e6000r10_m1_cXXXXXXXX/`，
与 600 代目录分开。训练日志同样隔离。最终 CSV 仍平铺在
`outputs/run/mahdrl_hscore_speed2/`，名称包含目标场景、源场景以及 `_e6000r10.csv`，
无需依靠父目录区分场景。检查点元数据严格核对训练调度，不能混载不同轮数的模型。

6000 代大致增加到 10 倍训练与逐代验证工作量；重复实例可以继续优化策略，
但不保证泛化更好。实验中应报告“600 个唯一训练种子，循环 10 轮，总计 6000 episode”，
并与同样训练预算的方法比较，测试集只用于最终评测。

验证用小型真实训练跨过种子轮次边界，检查学习状态连续、检查点加载与拒绝混载；
6000 代本身仅检查调度、元数据和 dry-run，不启动完整长程训练。
