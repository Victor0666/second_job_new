# HScore 第二轮实现加速：Speed2

2026-11-03：输出路径已核验并保留原有短目录/平铺 CSV 规则。独立后台函数与九组调用：
[600 代训练](Linux_CUDA_MAHDRL_HScore_speed2_600代训练指令.txt)、
[九源场景评测](Linux_CUDA_MAHDRL_HScore_speed2_九场景评测指令.txt)；
可直接 `source docs/mahdrl_as_fast_hscore_speed2_commands.sh` 加载函数。

标准 model seed=1 的训练末级目录为 17 字符，例如 `SS_T_m1_cXXXXXXXX`，项目内
checkpoint 相对路径最长 52 字符。最终 CSV 直接平铺在 `outputs/run/mahdrl_hscore_speed2/`，
文件名显式包含目标/源场景、截止期、模型种子、测试种子与配置标识，下载该目录的
`*.csv` 即可整理；`_details/` 为分配和审计明细。路径实现无需重命名已有结果。

2026-11-02。新增入口 `algorithms/mahdrl_as_fast_hscore_speed2/`，模型版本为
`mahdrl_hscore_fixedva_srgt092_speed_v2`。原版 HScore、第一轮 Speed v1 的源文件、
检查点和输出均保留，当前实验不会自动切换版本。

## 优化内容

这轮继续使用 Speed v1 的三层网络、回放、优化器和完整训练控制过程，调整两个环境实现：

1. **物理能耗累计**：原版每次推进时间都重建、排序、积分全部历史记录，并再次按主机
   计算能耗。Speed2 缓存已经确定的真实事件区间，只处理新事件和当前尚未结束的区间。
   当前时刻的事件保留为未确定边界，允许继续在该时刻分配任务。全局能耗仍按原来的
   主机和区间顺序累加，使用 float64 `cumsum`，没有改成各主机总量的重新求和。
   原始任务记录、空闲间隔计费、主机首次/末次执行窗口、OE 分配账本和奖励均保留。
2. **归一化队列观测**：原版先计算 legacy 队列特征，再计算归一化队列特征覆盖其中
   8 项。Speed2 直接计算最终使用的特征，保留另外两项、输入校验和原有归一化诊断。
   无 normalizer 时继续走原版路径。

物理记录在正常模拟中只追加，主机功率模型在回合内固定；reset 更换记录列表时建立
新缓存。诊断调用发生时间回退、记录缩短或追加历史区间时，能耗缓存会重建。
这轮没有调整网络、学习率、更新次数、探索时钟、验证预算或选模规则。

完整 50 工作流确定性验证的 cProfile 定位中，原实现耗时约 20.55 s：能耗累计函数
累计约 6.90 s，队列 block2 约 3.90 s；其中约 89.7 万次实际任务时长计算。
这是带 profiler 的定位数据，不是正式每代耗时，也不能与下表直接相减。

## 完整回合实测

与 Speed v1 相同本机 RTX 4060 Ti、PyTorch 2.5.1+cu121、CPU 线程数 1、Tight。
每个场景跑 2 代，训练种子 0/1，每代 50 工作流（1364/1391 个任务节点），每代验证
seed=600，并启用日志、采样 profiler、测量记录器和 checkpoint 保存。
回放池用与当前主机数量匹配的合成冻结经验预填，模型从初始化开始，测量有真实学习
更新的阶段。两版是先后测量，以下是每场景两代平均，不能替代服务器长程测速。

| 场景 | Speed v1 整代 | Speed2 训练 | Speed2 验证 | Speed2 保存 | Speed2 整代 | 耗时减少 |
|---|---:|---:|---:|---:|---:|---:|
| SS / 3 主机 25 VM | 28.74 s | 17.10 s | 5.92 s | 0.14 s | **23.15 s** | **19.5%** |
| SM / 6 主机 50 VM | 35.04 s | 21.03 s | 7.35 s | 0.16 s | **28.54 s** | **18.6%** |
| SL / 9 主机 75 VM | 38.86 s | 23.36 s | 9.74 s | 0.15 s | **33.26 s** | **14.4%** |

Speed2 的两代实际总耗时分别为 SS 23.77/22.54 s、SM 28.43/28.64 s、
SL 32.85/33.66 s。六代均自动对照 Speed v1 保存记录，训练物理指标、验证全部指标、
分配数、各层优化器更新数和学习样本计数精确一致。

原始记录：

- `docs/MAHDRL_HScore_speed2_full_episode_timing_2026-11-02.json`
- `docs/MAHDRL_HScore_speed2_full_episode_timing_SM_2026-11-02.json`
- `docs/MAHDRL_HScore_speed2_full_episode_timing_SL_2026-11-02.json`

## 运行与隔离

任务级同步版本见 [Speed2 Sync-Task](MAHDRL_HScore_Speed2_SyncTask.md)，使用独立模型与输出目录。

6000 代循环训练与配套评测见 [600 个种子循环 10 轮](MAHDRL_HScore_6000_seed_cycles.md)。
默认仍为原来的 600 代；新训练预算使用独立的短目录和带轮次的 CSV 名称。

Linux/CUDA 从项目根目录运行，原版参数继续适用：

```bash
python -u launch.py algorithms/mahdrl_as_fast_hscore_speed2/train.py \
  --task S --resource M --deadline Tight --dry-run

CUDA_VISIBLE_DEVICES=0 python -u launch.py algorithms/mahdrl_as_fast_hscore_speed2/train.py \
  --task S --resource M --deadline Tight --episodes 600 --model-seed 1 \
  --experiment-protocol configs/f22_split_v1.json \
  --normalization-mode f19_shared_v1 --device cuda \
  --va-preview --ha-update-period 4 --diagnostics-every 100 \
  --profile-mode sampled --profile-every 100

CUDA_VISIBLE_DEVICES=0 python -u launch.py algorithms/mahdrl_as_fast_hscore_speed2/evaluate.py \
  --task S --resource M --deadline Tight \
  --train-task S --train-resource M --train-deadline Tight \
  --seeds 700-729 --model-seed 1 \
  --experiment-protocol configs/f22_split_v1.json \
  --normalization-mode f19_shared_v1 --device cuda
```

Windows 使用 `run.cmd` 加相同脚本和参数，例如：

```powershell
.\run.cmd algorithms/mahdrl_as_fast_hscore_speed2/train.py --task S --resource M --deadline Tight --episodes 600 --model-seed 1 --device cuda
```

SM 使用 `--task S --resource M`，SL 使用 `--task S --resource L`。
截止期缓存和归一化文件复用，`precompute.py` 继续使用原参考生成器。

模型、日志、评测分别在 `outputs/ckpts/mahdrl_hscore_speed2/`、
`outputs/logs/mahdrl_hscore_speed2/`、`outputs/run/mahdrl_hscore_speed2/`。
新模型版本独立且严格拒绝混用，沿用原版防覆盖和不支持断点续训的规则。
运行上面的训练命令会开始一组新实验。

## 验证范围

```bash
python -X utf8 -B -m unittest common.test_hscore_prefix_energy common.test_mahdrl_as_fast_hscore_speed2 -v
```

专项检查能耗随机边界、部分执行区间、同刻开始/结束、空闲窗口重开、功率裁剪与小数
负载、诊断时间回退及历史追加、reset、归一化/legacy 观测、preview 开关、CPU/CUDA
真实训练。对照检查逐次观测、掩码、动作、奖励、物理记录、归一化诊断、online/target
参数，以及短程训练、保存、加载、评测、模型身份隔离和防覆盖。
专项 6 项，加上 Speed v1 和原版 HScore 回归 18 项，共 24 项全部通过，包含实际 CUDA。

精确一致是当前机器和测试范围内的证据；尚未测服务器完整 600 代、不同线程配置、
Medium/Loose 截止期或更大任务规模。这轮保留了全部日志写入方式；后续再优化应以
新热点为依据，不能直接把剩余耗时按旧热点占比外推。
