# DRL-GNN-Fast-Idle v6

独立算法入口：`drl_gnn_fast_idle`。以 DRL_GNN_FAST v5 为基础新增变式，原 `algorithms/drl_gnn/` 和 `algorithms/drl_gnn_fast/` 无需修改。

## 与 v5 的差异

1. 验证集汇总工作流成功率 **>= 0.92** 时视为达标；达标后优先最低平均总能耗，再比较成功率；未达标时优先最高成功率，再比较能耗；完全相同保留较早检查点。与当前 MAHDRL_AS_FAST_VAAWARE 使用同一选模规则。
2. 所有父任务 **Finished** 后，子任务才能进入 Ready。无父任务的入口任务在工作流到达后进入 Ready。
3. 联合动作仅开放 **Ready 任务 × 当前空闲 VM**。训练采样和确定性评测均使用该掩码。空闲判断为 `vm_available_at <= current_time + 1e-12`；没有合法动作时自动推进到下一完成或到达事件，不产生额外等待动作/PPO转移。

分配后即按现有物理模型从当前时刻执行，通信仍计入任务执行时间。正常运行不会出现等待父任务或忙 VM 的预约队列，`queued_assignments` 应为 0。执行入口仍在任何物理状态修改前校验依赖和空闲条件，防止外部调用绕过掩码；正常调度通过掩码排除忙 VM，不采用报错重试、重采样或替换已选动作。

这些是相对原论文提前预约协议的明确适配；该变式不应标注为完全保留原论文调度协议。

## 保持的设置

直接复用 v5 的 PPO 与网络实现：GIN hidden64、embedding16、3层GIN，任务8维/VM9维输入及其估计方式、完整兼容图、F19源训练尺度、奖励公式、优化器、学习率、折扣、rollout512、minibatch128、4个epoch、4个采样环境、默认每组4图反向传播、bootstrap缓存及CUDA显存策略均不变。训练、验证和评测统一使用新环境。

负载仍为标准负载：到达率0.03、截止期系数2/3。本次未增加压力负载或自定义资源配置。

## 训练与评测

在仓库根目录执行，`python` 应指向已安装仓库依赖的环境。以下以 SS/Tight 源场景为例，实际运行前可追加 `--dry-run` 检查路径及设置。

```bash
python launch.py algorithms/drl_gnn_fast_idle/train.py \
  --task S --resource S --deadline Tight \
  --episodes 600 --model-seed 1 \
  --experiment-protocol configs/f22_split_v1.json \
  --normalization-mode f19_shared_v1 --device cuda

python launch.py algorithms/drl_gnn_fast_idle/evaluate.py \
  --task S --resource S --deadline Tight \
  --train-task S --train-resource S --train-deadline Tight \
  --seeds 700-729 --model-seed 1 \
  --experiment-protocol configs/f22_split_v1.json \
  --normalization-mode f19_shared_v1 --device cuda
```

跨任务规模评测时只调整目标 `--task`，源训练参数应与实际检查点一致。默认训练种子0–599、验证种子600、评测种子700–729。

## 输出隔离

- 模型版本：`drl_gnn_fast_idle_v6`。
- 调度协议：`arrived_parent_finished_idle_vm_v1`。
- 选模版本：`select_sr092_minenergy_v1`。
- 检查点：`outputs/ckpts/drl_idle/SS_T_v6_sr092_m1_c<8位哈希>/best.pth`。
- 训练日志：`outputs/logs/drl_idle/SS_T_v6_sr092_m1_c<8位哈希>/`。
- 所有最终评测CSV平铺在：`outputs/run/drl_idle/`，不再用场景子目录区分同名 `evaluation.csv`。
- CSV示例：`eval_drl_idle_v6_sr092_MS_T_from_SS_T_m1_s700-729_c<8位哈希>.csv`。`MS_T` 为目标场景，`from_SS_T` 为源训练场景；T/M/L分别表示Tight/Medium/Loose。文件名同时包含模型种子、评测种子和配置哈希。
- 详细评测记录：`outputs/run/drl_idle/_details/<16位哈希>/`，其中保留 `summary.json` 和 `measurements/`。

默认模型种子1下，训练目录末级名称为26个字符；9个源场景各有独立短目录，27个目标场景的CSV名称均不重复。配置哈希保留模型/调度/输入版本、数据划分及归一化配置内容的身份信息，不把长配置说明拼入路径。CSV或对应详细记录已存在时拒绝覆盖。

批量后台命令函数见 `docs/drl_gnn_fast_idle_commands.sh`，在仓库根目录先执行：

```bash
source docs/drl_gnn_fast_idle_commands.sh
train_drl_gnn_fast_idle S Tight
# 完成对应训练后，依次评测SS、MS、LS：
eval_drl_gnn_fast_idle_all S Tight
```

函数仅定义任务，source本身不启动训练。每次调用均生成独立的 `.log` 和 `.pid`；每个评测任务内部按 S、M、L 顺序执行。该算法不接收MAHDRL专用的 `--diagnostics-every`、`--profile-mode` 或 `--profile-every` 参数。

旧 DRL_GNN 或 Fast v5 的检查点不能作为此变式的正式检查点，加载器会拒绝版本/调度协议不匹配的权重。需要重新训练；不修改旧输出。

本次仅缩短路径，不改变v6模型或选模规则。如已有同版本v6检查点保存在原长路径，可以通过 `--checkpoint-dir` 显式指定原目录；代码不会自动移动或覆盖它。

## 检查

```bash
python -m unittest common.test_drl_gnn_fast_idle -v
```

测试覆盖动作概率屏蔽、非法动作无状态修改、多父任务释放、全部VM忙时推进、同时到达/完成、物理依赖与无重叠、能耗重新积分、92%选模边界、独立路径、检查点保存加载及CUDA小规模训练评测。

固定合法策略的50工作流/25、50、75 VM物理检查见 `DRL_GNN_FAST_IDLE_physical_smoke_2026-10-30.json`。该检查用于验证调度正确性，不代表训练后的性能或运行速度。未执行600代正式训练。

2026-10-30 验证结果：新增变式10项测试通过（包含CUDA小规模训练、验证、保存加载与评测）；原DRL_GNN环境、显存及Fast v5回归测试共32项通过。三种VM规模共4113次分配，无依赖违规、VM执行重叠或预约排队，能耗重新积分误差均为0。训练/评测CLI的dry-run均通过。修改前后两个原算法目录的全部Python源码SHA-256完全一致。

路径调整后变式测试增加至12项并全部通过，新增27场景文件名唯一性、非连续种子区分、短训练名称、CSV实际写出及重复执行不覆盖检查。算法物理规则未再修改。

后台函数通过Bash语法与调度检查：捕获9条训练、27条评测命令，逐条使用实际CLI进行dry-run，参数、场景映射、输出名称和18组独立日志/PID均通过检查。记录见 `DRL_GNN_FAST_IDLE_commands_validation_2026-10-30.json`；检查过程未启动正式训练。

上传前，从暂存区导出独立源码快照，使用既有归一化校准输入和只读Git版本信息运行完整回归：原DRL_GNN环境、显存、Fast v5及Idle v6共44项测试全部通过（60.080秒）。未使用工作区其余未提交源码。
