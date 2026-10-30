# VAware-R100 / VAware-R150

两个入口分别固定为 12 Host / 100 VM、18 Host / 150 VM。基本单元 `[9,9,8,8,8,8]` 重复两次或三次；没有扩宽网络，也没有增加任务工作量。每回合仍为 50 个工作流、Poisson 到达率 0.03，训练默认 600 回合、S 任务组合。TA/HA/VA、奖励、OE、GE、全部物理合法动作、VA 预评估、HA 更新周期4及 `SR >= .92` 后最低验证能耗选模均沿用原 VAware。

## 固定参考协议

本实验只改变物理资源，两个分支都使用 R50（原 `--resource M`）的 FCFS 截止期参考和对应源训练场景 F19 归一化尺度。相同任务、截止期和实例种子下，R50/R100/R150 的模板、载荷、到达时间及绝对工作流截止期相同。实际调度和能耗计算始终使用各自真实拓扑。任务子截止期仍由原环境根据真实资源计算，没有改动算法。

`fixed_load_r50_reference_v1` 被写入配置摘要和检查点。`physical_source_scene` 记录真实训练场景，`deadline_reference_scene` 记录当前目标任务的 R50 参考，`normalization_reference_scene` 记录源训练任务的冻结尺度。F19 内部兼容字段 `source_train_scene` 仍指归一化参考，不用于冒充物理拓扑。跨任务评测时只切换目标任务参考缓存，继续使用源训练任务归一化。

这不是“各规模自行重算 FCFS 截止期”的协议。缓存按实际 R50 参考拓扑、任务分布、到达率、工作流数量、实例哈希和 F22 覆盖严格核验；不修改公共缓存校验器，不改写原缓存元数据。已有完整 R50 缓存可直接复用。缺失时运行对应 `precompute.py`，它只生成 R50 参考；显式 `--out` 拒绝覆盖，默认完整参考已存在且通过核验时只读复用。

## 入口

在项目根目录、已激活 Python 环境的 Bash 中执行。两个入口不需要 `--resource`；如显式指定，只接受自身 `R100` 或 `R150`。`--resource-config`、其他源资源及 stress 工作负载会报错。保留 `--task`、`--deadline`、模型种子、设备、诊断及 VAware 机制参数。

```bash
# 检查 / 准备 R50 参考。两个分支可复用同一组缓存。
python -u launch.py algorithms/mahdrl_as_fast_vaaware_r100/precompute.py --task S

# R100 训练与 S/M/L 目标评测中的一项。
python -u launch.py algorithms/mahdrl_as_fast_vaaware_r100/train.py \
  --task S --deadline Tight --episodes 600 --model-seed 1 --device cuda
python -u launch.py algorithms/mahdrl_as_fast_vaaware_r100/evaluate.py \
  --task M --deadline Tight --train-task S --train-deadline Tight \
  --seeds 700-729 --model-seed 1 --device cuda

# R150 使用完全相同的训练协议。
python -u launch.py algorithms/mahdrl_as_fast_vaaware_r150/train.py \
  --task S --deadline Tight --episodes 600 --model-seed 1 --device cuda
python -u launch.py algorithms/mahdrl_as_fast_vaaware_r150/evaluate.py \
  --task M --deadline Tight --train-task S --train-deadline Tight \
  --seeds 700-729 --model-seed 1 --device cuda
```

所有入口支持 `--dry-run`；默认 F22 为训练0–599、验证600、测试700–729。默认 F19 配置、R50 参考缓存如覆盖正确，无需重新生成。训练 M/L 源任务时要在评测中明确使用相同的 `--train-task`。关闭预评估或更改 HA 更新周期时，训练与评测必须一致。

## 后台脚本

```bash
source docs/mahdrl_as_fast_vaaware_resources_commands.sh

# 与其他实验相同的资源/截止期调用格式；资源只接受 R100/R150。
train_mahdrl_as_fast_vaaware_resource R100 Tight
eval_mahdrl_as_fast_vaaware_resource_all R100 Tight

# 训练：截止期、可选模型种子、预评估 on/off、HA周期。
train_mahdrl_as_fast_vaaware_r100 Tight
train_mahdrl_as_fast_vaaware_r150 Tight
train_mahdrl_as_fast_vaaware_r100 Medium 2

# 评测：截止期、可选目标列表、模型种子、预评估、HA周期。
eval_mahdrl_as_fast_vaaware_r100_all Tight
eval_mahdrl_as_fast_vaaware_r150_all Tight "M L" 2

# 控制组示例：训练与评测机制必须匹配。
train_mahdrl_as_fast_vaaware_r100 Tight 1 off 1
eval_mahdrl_as_fast_vaaware_r100_all Tight "S M L" 1 off 1
```

source 只定义函数，不启动任务。默认 GPU=0，`GPU=1` 可切换设备。每次调用启动一个 nohup 后台任务，独立日志/PID，包含 START/DONE/ALL_DONE 和失败停止。脚本末尾提供两个规模、三种截止期的全部调用示例。按实际内存、显存和训练吞吐安排并发。

## 输出与检查点

输出按 `mahdrl_va100` / `mahdrl_va150` 分组，训练末级目录统一为 `S_T_m1_cXXXXXXXX` 形式：默认模型种子1时16字符，最大32位模型种子时25字符；资源规模由上一级算法目录及配置摘要标识。示例：

```text
outputs/ckpts/mahdrl_va100/S_T_m1_cXXXXXXXX/
outputs/logs/mahdrl_va100/S_T_m1_cXXXXXXXX/
outputs/ckpts/mahdrl_va150/S_T_m1_cYYYYYYYY/
outputs/logs/mahdrl_va150/S_T_m1_cYYYYYYYY/
```

最终 CSV 直接平铺在 `outputs/run/mahdrl_va100/` 或 `outputs/run/mahdrl_va150/`，源/目标场景、截止期、模型种子、实际测试种子、预评估/HA周期和配置摘要都放进 CSV 文件名。下载后即使放在同一目录，也能区分：

```text
eval_mahdrl_va100_noapf_sr092_p1h4_S100_T_from_S100_T_m1_s700-729_cXXXXXXXX.csv
eval_mahdrl_va100_noapf_sr092_p1h4_M100_T_from_S100_T_m1_s700-729_cXXXXXXXX.csv
eval_mahdrl_va100_noapf_sr092_p1h4_S100_M_from_S100_M_m1_s700-729_cYYYYYYYY.csv
eval_mahdrl_va150_noapf_sr092_p1h4_M150_T_from_S150_T_m1_s700-729_cZZZZZZZZ.csv
```

其中 `S100_T` 表示 S任务/R100资源/Tight截止期，`M100_T` 表示 M任务/R100资源/Tight截止期，`from` 后为训练源场景；T/M/L分别表示Tight/Medium/Loose。详细诊断文件放在 `_details/<运行摘要>/`。物理拓扑、固定参考协议和机制参数继续进入身份摘要。

仅缩短了目录名，科学身份和检查点内容不变。已有上一版 `S100_T_m1_cXXXXXXXX` / `S150_T_m1_cXXXXXXXX` 目录中的模型可以通过评测参数 `--checkpoint-dir` 显式加载，不自动移动或覆盖旧目录。

完整 nohup 函数见 `mahdrl_as_fast_vaaware_resources_commands.sh`，支持日志/PID、START/DONE/ALL_DONE、错误停止及默认 GPU=0。按统一格式运行全部默认场景：

```bash
source docs/mahdrl_as_fast_vaaware_resources_commands.sh

# R100：S任务源模型，三个截止期。
train_mahdrl_as_fast_vaaware_resource R100 Tight
train_mahdrl_as_fast_vaaware_resource R100 Medium
train_mahdrl_as_fast_vaaware_resource R100 Loose

# R150：S任务源模型，三个截止期。
train_mahdrl_as_fast_vaaware_resource R150 Tight
train_mahdrl_as_fast_vaaware_resource R150 Medium
train_mahdrl_as_fast_vaaware_resource R150 Loose

# R100：每个源模型分别评测 S100/M100/L100。
eval_mahdrl_as_fast_vaaware_resource_all R100 Tight
eval_mahdrl_as_fast_vaaware_resource_all R100 Medium
eval_mahdrl_as_fast_vaaware_resource_all R100 Loose

# R150：每个源模型分别评测 S150/M150/L150。
eval_mahdrl_as_fast_vaaware_resource_all R150 Tight
eval_mahdrl_as_fast_vaaware_resource_all R150 Medium
eval_mahdrl_as_fast_vaaware_resource_all R150 Loose
```

训练函数参数为资源、截止期、可选模型种子、预评估开关、HA周期；评测函数参数为资源、截止期、可选目标列表、模型种子、预评估开关、HA周期。原来独立的 r100/r150 函数仍然可以使用。一次调用只启动一个后台任务，按算力安排并发；仅 source 不执行任何训练或评测。

两个分支均从头训练，只加载本分支、同源任务/截止期/模型种子的 TA/HA/VA 成组 best 检查点。原 VAware、R100、R150 默认不能混用，即使参数形状相同。混组、final 改名 best、跨机制及已有输出覆盖会被拒绝。当前不提供断点续训或跨资源泛化加载。

`assignments.jsonl` 中 `legal_vm_count` 为所选 Host 的局部数量，`global_legal_vm_count` 为本次分配前全局合法 VM 数量；回合诊断分别汇总全局、局部和 Host 候选数量。保留完整预评估与推理计时、能耗审计、冻结历史回放及更新时钟。

```bash
python -B -X utf8 -m unittest \
  common.test_mahdrl_as_fast_vaaware_resources \
  common.test_mahdrl_as_fast_vaaware \
  common.test_mahdrl_as_fast_vaaware_tl -v
```

原 VAware 的默认路径/身份及 TL 入口保持兼容。对应的 R100-TL / R150-TL 已补齐，使用相同实际拓扑和固定参考协议，入口为 `mahdrl_as_fast_vaaware_r100_tl` / `mahdrl_as_fast_vaaware_r150_tl`；方案、独立输出和训练/评测命令见 [TL说明](MAHDRL_AS_Fast_VAware_R100_R150_TL.md)。不能直接替代为原TL其他资源规模的结果。短程测试和物理探针不代表正式600回合的性能结论。

## 本次验证记录

首次实现时，资源专项8项、原 VAware 12项和 VAware-TL 12项，共32项全部通过，0失败、0错误、0跳过，完整运行耗时52.402秒。覆盖两个规模的正式网络宽度 CPU/CUDA 梯度更新、短程真实训练、保存重载、冻结评测、防覆盖、跨身份拒绝、固定参考及历史 VAware/TL 回归。

两个分支共六个公开入口 dry-run 通过。R50/R100/R150 各完成一次实际50工作流、到达率0.03的物理探针，每回合1370次分配，50个工作流全部完成，三个工作流载荷/到达时间/绝对截止期集合的哈希完全一致。探针使用未训练的小网络，只用于核验物理执行和协议一致性，不用于算法效果比较。最大分配账本/物理能耗绝对差小于 `7e-10`。

后台脚本通过 Git Bash 语法检查及27次替身调用，核对默认六组训练/18组目标评测、可选模型种子与机制参数，并确认 source 不启动任务、目标评测失败会停止后续目标。未启动正式600回合训练，未保存正式模型或评测结果。机器可读记录见 `MAHDRL_AS_Fast_VAware_resources_validation.json`。

路径缩短后的33项回归全部通过（41.557秒），新增后台命令专项1项通过（2.657秒），合计34项无失败、错误或跳过。路径测试覆盖72个训练配置及216个最终CSV名称，包括三个源任务、三个截止期、两个资源规模、不同机制和最大32位模型种子，文件名在下载合并后仍唯一。六个公开入口 dry-run 通过；默认模型种子1对应六个训练目录、18个平铺CSV。当前工作区绝对路径最长为：检查点目录95字符、日志目录94字符、已核对训练文件122字符、最终CSV156字符。

更新后的 Bash 专项通过35次替身调用，覆盖统一资源/截止期函数、原独立分支函数、模型种子与机制参数、空目标/错误参数拒绝及失败停止。未启动正式实验。路径与命令的机器可读记录见 `MAHDRL_AS_Fast_VAware_resource_paths_validation.json`。
