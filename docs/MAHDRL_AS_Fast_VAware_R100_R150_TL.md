# VAware-R100-TL / VAware-R150-TL

已提供 `algorithms/mahdrl_as_fast_vaaware_r100_tl/`、`algorithms/mahdrl_as_fast_vaaware_r150_tl/` 两个入口，各自包含 train、evaluate、precompute。分别对应完整版 R100、R150 的两层消融：移除 HA，仅学习 TA、VA，由 VA 对所有物理合法空闲 VM 统一评分并选择。没有占位 HA，也不接受 VA 预评估或 HA 更新周期参数。

| 对照 | 物理资源 | 完整版选择 | TL 选择 |
|---|---|---|---|
| R100 / R100-TL | 12 Host / 100 VM | HA 选 Host，VA 选该 Host 内 VM | VA 从全局 100 行候选中选择合法 VM |
| R150 / R150-TL | 18 Host / 150 VM | HA 选 Host，VA 选该 Host 内 VM | VA 从全局 150 行候选中选择合法 VM |

拓扑从对应完整版配置直接导入。两组均保持每回合50工作流、到达率0.03、默认600回合、相同任务模板与载荷、OE奖励、VA GE、无APF、物理合法性、Fast D3QN、冻结历史回放和原TL更新时序。源/目标物理资源固定，不允许自定义拓扑、跨资源加载或stress工作负载。

同种子、任务与截止期下，完整版和TL共用 `fixed_load_r50_reference_v1`：当前目标任务的R50 FCFS参考缓存，以及源训练任务/截止期的R50 F19尺度。工作流到达时间和绝对截止期相同；实际执行始终在各自100/150 VM拓扑上进行。F22仍为训练0–599、验证600、测试700–729。选模仍先满足验证 `SR >= .92`，再最小化验证能耗，回退及并列规则相同。

TL保留现有全局观测 `task23_globalvm14_v1`：上下文23维，每个VM候选14维，包含其Host信息。网络宽度与原TL相同，不随VM数量扩宽。每次决策在当前合法全局集合上进行与VAware相同的解析epsilon/GE混合单次采样，训练回放记录实际观测及合法掩码。VA在分配后更新，TA在阶段结束后更新；没有HA优化器、回放或模型文件。

此比较评估移除Host决策层后的整体效果；HA、预评估机制及全局/局部观测结构随之变化。它不能单独证明预评估或HA更新周期的作用，也不能预设TL在更大规模下一定更差。正式报告应同时给出成功率、能耗、候选数、训练耗时及推理耗时，使用成对实例种子和多个训练种子。

## 训练与评测

在项目根目录的Bash中加载函数；source只定义函数，不启动任务。完整nohup函数见 `docs/mahdrl_as_fast_vaaware_resources_tl_commands.sh`，沿用独立日志/PID、START/DONE/ALL_DONE和目标失败后停止规则。默认GPU=0，可用 `GPU=1` 切换。按硬件安排并发，下面每一行都会启动独立后台任务。

```bash
source docs/mahdrl_as_fast_vaaware_resources_tl_commands.sh

# R100-TL：三个 S任务源模型。
train_mahdrl_as_fast_vaaware_resource_tl R100 Tight
train_mahdrl_as_fast_vaaware_resource_tl R100 Medium
train_mahdrl_as_fast_vaaware_resource_tl R100 Loose

# R150-TL：三个 S任务源模型。
train_mahdrl_as_fast_vaaware_resource_tl R150 Tight
train_mahdrl_as_fast_vaaware_resource_tl R150 Medium
train_mahdrl_as_fast_vaaware_resource_tl R150 Loose

# 训练完成后，每个源模型依次评测三个目标任务 S/M/L。
eval_mahdrl_as_fast_vaaware_resource_tl_all R100 Tight
eval_mahdrl_as_fast_vaaware_resource_tl_all R100 Medium
eval_mahdrl_as_fast_vaaware_resource_tl_all R100 Loose
eval_mahdrl_as_fast_vaaware_resource_tl_all R150 Tight
eval_mahdrl_as_fast_vaaware_resource_tl_all R150 Medium
eval_mahdrl_as_fast_vaaware_resource_tl_all R150 Loose
```

训练函数参数为资源、截止期、可选模型种子（默认1）。评测函数参数为资源、截止期、可选目标列表（默认 `"S M L"`）、可选模型种子（默认1）。例如第二训练种子：

```bash
train_mahdrl_as_fast_vaaware_resource_tl R100 Tight 2
eval_mahdrl_as_fast_vaaware_resource_tl_all R100 Tight "S M L" 2

# 也可使用各规模独立函数。
train_mahdrl_as_fast_vaaware_r150_tl Tight 2
eval_mahdrl_as_fast_vaaware_r150_tl_all Tight "M L" 2
```

直接运行入口亦可；R150将路径中的r100改为r150。训练源任务不是S时，评测须显式指定对应 `--train-task`。

```bash
python -u launch.py algorithms/mahdrl_as_fast_vaaware_r100_tl/train.py \
  --task S --deadline Tight --episodes 600 --model-seed 1 \
  --experiment-protocol configs/f22_split_v1.json \
  --normalization-mode f19_shared_v1 --device cuda \
  --diagnostics-every 100 --profile-mode sampled --profile-every 100

python -u launch.py algorithms/mahdrl_as_fast_vaaware_r100_tl/evaluate.py \
  --task M --deadline Tight --train-task S --train-deadline Tight \
  --seeds 700-729 --model-seed 1 \
  --experiment-protocol configs/f22_split_v1.json \
  --normalization-mode f19_shared_v1 --device cuda \
  --diagnostics-every 100 --profile-mode sampled --profile-every 100

# 已有完整R50参考会只读核验复用；缺失时仅生成R50参考。
python -u launch.py algorithms/mahdrl_as_fast_vaaware_r100_tl/precompute.py --task S
```

全部入口支持 `--dry-run`。预计算行为与完整版完全共用；不为TL重算另一组截止期，不修改已有完整参考缓存。

## 输出和检查点

```text
outputs/ckpts/mahdrl_va100_tl/S_T_m1_cXXXXXXXX/
outputs/logs/mahdrl_va100_tl/S_T_m1_cXXXXXXXX/
outputs/ckpts/mahdrl_va150_tl/S_T_m1_cYYYYYYYY/
outputs/logs/mahdrl_va150_tl/S_T_m1_cYYYYYYYY/

outputs/run/mahdrl_va100_tl/eval_mahdrl_va100_tl_noapf_sr092_M100_T_from_S100_T_m1_s700-729_cXXXXXXXX.csv
outputs/run/mahdrl_va150_tl/eval_mahdrl_va150_tl_noapf_sr092_L150_L_from_S150_L_m1_s700-729_cYYYYYYYY.csv
```

末级训练目录在模型种子1时为16字符；最大32位模型种子时25字符。CSV直接平铺在算法目录中，文件名包含TL身份、规模、源/目标任务、截止期、模型种子、测试种子和配置摘要。详细日志位于 `_details/<运行摘要>/`；CSV列同时记录真实Host/VM数、参考协议及全局选择范围。

仅保存 `best_manager.pth`、`best_vm.pth`、`manager_final.pth`、`vm_final.pth`。每个分支有独立算法和模型版本，拒绝完整版、原TL、另一资源规模及不同源配置的检查点；拒绝混组、不同选模证据、final改名best和覆盖已有输出。需要从头训练；`--checkpoint-dir`只用于显式指定匹配的本分支模型位置，不放宽身份验证。

`assignments.jsonl` 的 `legal_vm_count` 是全局合法VM数量，完整版同名字段是所选Host局部数量。比较时应对齐完整版的 `global_legal_vm_count`。TL回合中 `candidate_diagnostics.global_vm`、`candidate_diagnostics.host` 汇总候选数量；分配日志保留全局动作索引、实际Host/VM、策略版本、epsilon和分配能耗。

## 验证

```bash
python -B -X utf8 -m unittest \
  common.test_mahdrl_as_fast_vaaware_resources_tl \
  common.test_mahdrl_as_fast_vaaware_resources \
  common.test_mahdrl_as_fast_vaaware_tl \
  common.test_mahdrl_as_fast_vaaware -v
```

验证覆盖固定参考与路径、全局物理映射、相同分配的物理/奖励一致性、两个规模CPU及可用CUDA上的正式网络宽度更新、短程真实训练与重载、评测冻结、检查点隔离、防覆盖和Bash替身调用。完整50工作流探针用于核验参考与物理执行，不用于算法优劣比较。未启动正式600回合训练。

最终四套回归共41项全部通过，0失败、0错误、0跳过，耗时84.280秒。本机CUDA可用，两个新规模均完成正式宽度网络的CPU/CUDA梯度更新。六个公开入口dry-run通过；路径测试覆盖36个训练配置、108个平铺且互不重名的CSV文件名。Bash语法与35次替身调用通过，并核验source不启动任务、无效参数被拒绝及目标失败后停止。

R100、R150的完整版与TL各完成一次50工作流成对探针，TL各执行1370次分配；两组对应实例的载荷/到达时间/绝对截止期哈希相同，两种资源规模之间亦相同。最大账本/物理能耗绝对差小于 `7e-10`，评测参数保持冻结。另逐项比对旧版本，原TL与完整版135个配置的模型名称、算法元数据和选模协议保持一致。机器可读记录见 [验证记录](MAHDRL_AS_Fast_VAware_resources_tl_validation.json)。
