# HScore FixedVA v2 的独立训练加速变式

6000 代循环训练与配套评测见 [600 个种子循环 10 轮](MAHDRL_HScore_6000_seed_cycles.md)。
默认仍为原来的 600 代；新训练预算使用独立的短目录和带轮次的 CSV 名称。

2026-11-03：输出路径已核验并保留原有短目录/平铺 CSV 规则。独立后台函数与九组调用：
[600 代训练](Linux_CUDA_MAHDRL_HScore_speed_600代训练指令.txt)、
[九源场景评测](Linux_CUDA_MAHDRL_HScore_speed_九场景评测指令.txt)；
可直接 `source docs/mahdrl_as_fast_hscore_speed_commands.sh` 加载函数。

标准 model seed=1 的训练末级目录为 17 字符，例如 `SS_T_m1_cXXXXXXXX`，项目内
checkpoint 相对路径最长 51 字符。最终 CSV 直接平铺在 `outputs/run/mahdrl_hscore_speed/`，
文件名显式包含目标/源场景、截止期、模型种子、测试种子与配置标识，下载该目录的
`*.csv` 即可整理；`_details/` 为分配和审计明细。路径实现无需重命名已有结果。

2026-11-02。基线为 `mahdrl_hscore_fixedva_srgt092_v2`，新增入口为
`algorithms/mahdrl_as_fast_hscore_speed/`，模型版本为
`mahdrl_hscore_fixedva_srgt092_speed_v1`。原版源文件、已在运行的训练和已有输出均不修改。

这是实现优化：继续调用原版的环境、调度过程、奖励、更新屏障、日志、验证和选模。
TA/HA/VA 网络、学习率、gamma、batch=256、回放容量、探索时钟、GE、HA 更新周期、
目标网络、每回合 50 个工作流和严格 `SR > 0.92` 后最低验证能耗均沿用原版。

## 哪些循环真正拖慢训练

| 位置 | 原版开销 | 本变式处理 |
|---|---|---|
| `mahdrl_as_fast_hscore/agent.py:149` | VA 每次更新循环 256 条样本，分别校验当前/后继，共调用 512 次 `validate_fixed`，其中有大量小数组 `np.isin` 和归约 | 先 stack，按整个 batch 检查形状、有限值、二值掩码和非终止合法动作；没有取消校验 |
| 三层 `random.sample(self.buffer, 256)` | buffer 为 deque，大容量时随机访问中部元素需要遍历；随机抽样会放大这项开销 | 可直接索引的环形回放，逻辑顺序仍为最老到最新，容量及淘汰顺序相同，继续调用 Python `random.sample` |
| VA 的 `if ongoing.any()` 及 GPU 布尔索引 | 判断 CUDA 张量会等待 GPU；布尔索引也需要确定后继 batch 大小 | 在 CPU 已知的 done 数组上得到非终止行索引，只上传这些后继，仍跳过所有终止后继网络前向 |
| `mahdrl_heft.py:_compute_block2` | Ready 任务 × VM 的时长计算；环境在每个决策处计算特征 | 此次继续复用原版缓存和标量物理计算 |
| `mahdrl_as_fast_vaaware.py:preview_vm_states` / `mahdrl_as_fast_hscore/preview.py` | 逐主机构造观测和行为概率 | 原版 VA 神经网络前向已经按主机批处理；此次保持观测与行为计算 |
| `mahdrl_as_fast_hscore/runtime.py:run_episode` | 顺序执行任务分配、真实后继收集和更新屏障 | 保留：直接并发化这些依赖会改变后继状态和训练语义 |

有不少 `for`/`while`，但是否在高频路径和循环体做了什么，比循环数量更有意义。
初步 cProfile 在 10 次正式宽度 VA CUDA 更新中记录：5120 次 `validate_fixed` 累计
约 0.095 s，总计约 0.163 s。这是带 profiler 的定位数据，性能结论采用下面独立计时。

## 本机实测

Windows、RTX 4060 Ti、Python 3.12.10、PyTorch 2.5.1+cu121、NumPy 2.1.2、CPU 线程数 1。
learner benchmark 使用正式网络、batch=256、满容量回放，512 条不同的冻结经验重复填充；
HA 为 3 台主机，约 1/7 为终止经验。预热 5 次，每轮 30 次更新、交替顺序测 5 轮，
计时前后同步 CUDA，下表为中位数。

| 测量对象 | 原版 | 加速版 | 比值 |
|---|---:|---:|---:|
| TA 一次更新 | 4.44 ms | 4.54 ms | 0.98× |
| HA 一次更新 | 7.03 ms | 6.76 ms | 1.04× |
| VA 一次更新 | 11.04 ms | 4.60 ms | **2.40×** |
| 5 个工作流的真实训练轨迹 | 2.42 s | 1.56 s | **1.55×** |

TA/HA 的整体收益较小，TA 在本轮中略慢；主要收益来自 VA。环形回放本身的抽样
测量分别约快 1.66×、5.14×、3.81×，这只代表抽样耗时，不代表各层整体更新。

最后一行用正式宽度三层网络和预填满的回放池，在相同 seed=0 实例上运行 5 个真实
工作流，交替顺序测 3 轮，每轮 134 次任务分配。最后一轮执行 TA/HA/VA 更新
49/33/134 次。各轮原版/新版逐次分配日志、结果和三层 online 参数完全一致。
计时包含环境交互、观测、动作和学习，不含验证、检查点写盘、磁盘日志和环境 reset。
预填充用于测量已经进入学习阶段的训练，并非正式训练协议。

这不是完整 600 回合加速比。每回合验证、实际回放占用、CPU、GPU、任务和资源规模、
同时运行的实验数量及日志存储均会影响最终收益。原版正式协议每回合还要运行
seed=600 的验证环境；验证仍沿用原版，不会获得 VA 更新的加速。

机器可读结果：

- `docs/MAHDRL_HScore_speed_benchmark_2026-11-02.json`
- `docs/MAHDRL_HScore_speed_rollout_2026-11-02.json`

### 补测：完整一代需要多久

同日本机 RTX 4060 Ti、CPU 线程数 1，使用默认归一化文件和真实截止期缓存，
S/S/Tight、正式三层网络、每代 50 个工作流、每代验证 seed=600，实际运行 2 代。
日志、采样 profiler、测量记录器和 checkpoint 保存均开启。回放先用合成冻结经验
填满，用于测量有学习更新的阶段；模型仍从初始化开始，不代表已收敛模型的速度。
这是独立临时目录中的短程计时，不是正式 600 代实验，未影响正在运行的任务。

| 阶段 | 两代平均 |
|---|---:|
| 训练（含环境、日志及记录范围内 reset） | 20.29 s |
| 验证 | 8.33 s |
| checkpoint 保存 | 0.12 s |
| 完整一代记录时间 | **28.74 s** |

两代分别为 29.66 s、27.83 s；真实任务节点数分别为 1364、1391。
按此短测速度估算 600 代约 4.8 小时，不能代替完整训练实测，其他资源规模和
服务器并发占用也可能改变耗时。原始记录和 sampled stage 数据保存在
`docs/MAHDRL_HScore_speed_full_episode_timing_2026-11-02.json`。

随后按相同设置顺序补测 SM 和 SL，各 2 代。HA 预填回放的候选数分别匹配
6 和 9 台主机；三个场景每代任务节点数均为 1364、1391。

| 场景 | 主机 / VM 数 | 训练平均 | 验证平均 | 保存平均 | 整代平均 |
|---|---:|---:|---:|---:|---:|
| SS / Tight | 3 / 25 | 20.29 s | 8.33 s | 0.12 s | **28.74 s** |
| SM / Tight | 6 / 50 | 25.02 s | 9.91 s | 0.11 s | **35.04 s** |
| SL / Tight | 9 / 75 | 27.34 s | 11.40 s | 0.11 s | **38.86 s** |

SM 两代为 34.49 s、35.59 s；SL 两代为 39.02 s、38.70 s。
对应原始记录为 `docs/MAHDRL_HScore_speed_full_episode_timing_SM_2026-11-02.json`
和 `docs/MAHDRL_HScore_speed_full_episode_timing_SL_2026-11-02.json`。
这些是本机新加速版、预填回放后的短程计时，不能替代服务器实时耗时，也未覆盖
Medium/Loose 截止期或完整 600 代。

下一轮值得优化环境端，这也会缩短约占本次 29% 的验证时间：

1. `_process_finish_events_at_current_time` 查找子任务和父任务时反复遍历全部
   `task_meta`；可维护 `(workflow_id, local_task_id) -> global_task_id` 索引。
2. `_compute_block2` 先计算一组队列统计，归一化时又调用 `queue_features/queue_metrics`
   重算，其中含 Ready 任务 × VM 的循环；可复用统计并省去最终被覆盖的计算。
3. `append_json` 对每条分配/阶段记录反复开关文件；可复用文件句柄，并明确刷新边界。

这些尚未实施，收益需分别测量，并对照观测、动作、物理结果和更新参数的一致性。
此次 checkpoint 只占约 0.4%，减少模型保存不是优先项。CPU sampled stage 的耗时
可能包含等待 CUDA 的时间，不能直接把它当成纯 CPU 算术开销，也不能与 CUDA
计时相加。

## 运行

Linux/CUDA，在项目根目录运行；参数与原版相同，替换入口即可：

```bash
# 先检查身份、输出路径和配置
python -u launch.py algorithms/mahdrl_as_fast_hscore_speed/train.py \
  --task S --resource S --deadline Tight --dry-run

# 独立训练，不影响原来正在跑的版本
CUDA_VISIBLE_DEVICES=0 python -u launch.py algorithms/mahdrl_as_fast_hscore_speed/train.py \
  --task S --resource S --deadline Tight --episodes 600 --model-seed 1 \
  --experiment-protocol configs/f22_split_v1.json \
  --normalization-mode f19_shared_v1 --device cuda \
  --va-preview --ha-update-period 4 --diagnostics-every 100 \
  --profile-mode sampled --profile-every 100

# 评测新版本，目标 task/resource 可独立指定
CUDA_VISIBLE_DEVICES=0 python -u launch.py algorithms/mahdrl_as_fast_hscore_speed/evaluate.py \
  --task S --resource S --deadline Tight \
  --train-task S --train-resource S --train-deadline Tight \
  --seeds 700-729 --model-seed 1 \
  --experiment-protocol configs/f22_split_v1.json \
  --normalization-mode f19_shared_v1 --device cuda

# 在服务器上复测正式宽度 learner 更新；不会启动正式训练
python -u launch.py algorithms/mahdrl_as_fast_hscore_speed/benchmark.py \
  --device cuda --updates 30 --repeats 5 --output tmp/hscore_speed_benchmark.json
```

Windows 可使用：

```powershell
.\run.cmd algorithms/mahdrl_as_fast_hscore_speed/train.py --task S --resource S --deadline Tight --dry-run
.\run.cmd algorithms/mahdrl_as_fast_hscore_speed/train.py --task S --resource S --deadline Tight --episodes 600 --model-seed 1 --device cuda
```

已有物理拓扑的截止期缓存和归一化文件可直接复用。自定义拓扑仍按原版规则仅用于
预计算/评测，入口为新目录的 `precompute.py`，VM 固定容量仍为 9。

检查点、日志、最终评测分别写到独立目录：

```text
outputs/ckpts/mahdrl_hscore_speed/<scene>/
outputs/logs/mahdrl_hscore_speed/<scene>/
outputs/run/mahdrl_hscore_speed/
```

新旧模型身份不同，禁止混用。此变式不提供旧训练断点迁移或断点续训；运行上面的
训练命令将开始一组新实验。已有输出仍拒绝覆盖。

## 等价性验证

```bash
python -X utf8 -B -m unittest common.test_mahdrl_as_fast_hscore_speed -v
```

7 项专项全部通过，含 CPU 和实际 CUDA：环形顺序/淘汰/随机采样及 RNG 状态、
非法输入拒绝、混合终止和全终止 batch、硬/软目标更新、正式 VA 宽度及 batch=256、
逐次损失/online/target/Adam 状态精确一致、真实训练动作/物理记录/结果/三层参数一致、
短程训练/保存/加载/评测、模型身份隔离及防覆盖。
正式网络 learner benchmark 各层结束后也检查了 online 参数精确一致。
原版 HScore、HScore 消融和旧模型评测回归另有 23 项全部通过；训练、评测、
预计算三个新 CLI 的 dry-run 也通过。

上述精确一致是本机测试范围内的证据，不代表不同硬件或 PyTorch 版本保证逐位复现。
尚未完成服务器上的 600 回合全程性能测量和所有 S/M/L 规模的速度比较。

进一步优化应依据服务器 `stage_timings.jsonl` 选择环境特征/事件扫描或日志 I/O；
扩大 HA 更新周期、减少 VA 更新次数、缩小网络、AMP、改变验证频率会影响学习或
选模条件，不能把这些当成相同训练设置下的实现加速。
