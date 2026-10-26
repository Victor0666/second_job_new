# IRWS-R-FE：V3加速版的单成功率门槛选模

日期：2026-10-27。算法ID `irws_r_fe`，显示名 `IRWS-R-FE`，模型版本 `irws_r_v3_fe_v1`。
从当前 `algorithms/irws_r/` 复制后修改；原算法目录、公共文件和PD3QN_DAG_FE保留。
源文件SHA-256记录在 `algorithms/irws_r_fe/source_snapshot_sha256.json`。

## 选择规则

采用当前PD3QN_DAG_FE的 `select_sr099_minenergy_v1`，不是旧FE的ESP规则。
每回合完成训练后只验证一次，使用原固定验证集。总体成功率为各实例成功工作流数之和
除以工作流总数之和；未完成算失败，完成时间不超过截止期（公共容差1e-12）算成功。
按实例平均总能耗排序，不是按任务能耗或每个成功工作流能耗排序。

- 总体成功率至少0.99为可行，可行优先于不可行。
- 可行模型按平均总能耗更低、成功率更高排序。
- 没有可行模型时按成功率更高、平均总能耗更低选备用模型。
- 只在严格更优时替换best，完全平局保留较早回合。
- 不增加成功率奖励，不改变梯度，不提前终止既定训练预算。

当前F22为训练0–599、验证600、测试700–729。验证50个工作流时实际须50/50按时完成；
49/50仅98%。经验可行不构成未知实例的99%概率保证，也不保证训练能达到门槛。

## 与V3的边界

`learner.py`、`networks.py`、`environment.py`、`numerics.py`逐字节复制，学习参数及实现标识不变。
γ=.2、clip=.2、四组学习率、轨迹16、默认rollout64、周期5秒、float32及所有物理计算保持。
新增逻辑位于局部selection、scenario及runner：选模统计、验证隔离、记录和检查点校验。
验证复用公共 `episode_metrics`，不依赖 `EXPERIMENT_METRICS` 开关。
验证不更新参数、优化器或训练计数；恢复Python/NumPy/Torch CPU/CUDA随机状态、网络模式、
RNN隐藏状态、资源/图缓存及数值诊断，异常退出也恢复。原验证时机和实例数保持，统计与排序不再跑一次调度。

公共F22文件、校准文件及hash不变。本变体运行层元数据覆盖选模声明，仍用原始协议文件绑定数据和尺度。
这是新增实验选模机制，不是IRWS论文原有学习机制。

## 记录、加载与续训

输出进入 `outputs/{ckpts,run}/irws_r_fe/irws_r_v3_fe_v1/`，配置目录包含队列、周期、
rollout、精度及选模规则；与V3及所有旧IRWS输出隔离。非空目录继续拒绝覆盖。

- `best_irws_r_fe.pth`：最佳可行模型或尚无可行模型时的最佳备用模型。
- `final_irws_r_fe.pth`：最近完成回合、优化器、RNG及历史best记录，用于同目录续训。
- `validation_selection.jsonl`：逐回合、逐验证实例计数、能耗、候选及历史best、选中标志。
- `training_summary.json`：历史best回合、规则、路径及 `feasible_model_found`。
- 原训练JSONL和评测JSON也记录新规则；评测CSV增加公共工作流成功计数。

加载校验算法/版本、规则、角色、验证种子、可复算指标、历史best记录、模型种子、源校准、
VM维度、队列/周期/rollout/精度及公共目标。正式评测只加载本变体best；续训只从本变体final，
并要求同目录历史best存在且与final记录匹配。旧V3不能直接改标签当成新规则完整训练结果。
只保留旧best/final无法还原被旧规则遗漏的中间候选。

一个源场景选定的best固定用于S/M/L目标任务规模，不按目标测试种子重新选模型。

## Linux/CUDA示例

完整后台函数与9组调用见 [600回合训练指令](Linux_CUDA_IRWS_R_FE_600代训练指令.txt)
及 [27场景共用评测指令](Linux_CUDA_IRWS_R_FE_27场景共用评测指令.txt)。每次调用启动一个进程，按服务器容量分批执行。

在仓库根目录运行，公共截止期缓存与源场景共享校准沿用既有准备流程。
旧IRWS训练/评测入口继续指向旧算法，新变体使用下列独立入口。

```bash
python -u launch.py algorithms/irws_r_fe/train.py \
  --task S --resource M --deadline Tight --model-seed 1 --episodes 600 \
  --experiment-protocol configs/f22_split_v1.json --normalization-mode f19_shared_v1 \
  --queue-capacity 10 --slot-duration 5 --rollout-size 64 --precision float32 --device cuda

for task in S M L; do
  python -u launch.py algorithms/irws_r_fe/evaluate.py \
    --task "$task" --resource M --deadline Tight \
    --train-task S --train-resource M --train-deadline Tight \
    --seeds 700-729 --model-seed 1 \
    --experiment-protocol configs/f22_split_v1.json --normalization-mode f19_shared_v1 \
    --queue-capacity 10 --slot-duration 5 --rollout-size 64 --precision float32 --device cuda
done
```

训练命令和每组三任务评测命令的resource替换为S/M/L、deadline替换为Tight/Medium/Loose，
构成9个源模型和27个目标场景。相同源模型的周期、队列、rollout和精度必须匹配。
两入口均支持 `--dry-run` 查看规则和独立路径。续训加
`--resume-checkpoint <本次输出目录>/final_irws_r_fe.pth`，并保留原场景及配置。

## 验证

```powershell
.\.venv\Scripts\python.exe -X utf8 -B -m unittest common.test_irws_r_fe common.test_irws_r_speed common.test_irws_r common.test_pd3qn_dag_fe -v
```

专项覆盖99%聚合边界、与DAG-FE规则一致、可行优先、备用/能耗/平局排序、V3实际梯度及物理轨迹一致、
CPU/CUDA真实更新、验证状态隔离及异常恢复、诊断关闭下选模、9源/27目标CLI映射、
训练保存加载评测、历史可行best跨续训保留、断点与连续训练权重一致、旧版本和错误配置拒绝、覆盖保护。
短程训练使用临时合成工作流和测试专用协议/尺度，不写公共校准或正式输出。
没有运行正式600回合训练、正式30测试种子评测或服务器多进程性能实验。

本次13项专项检查及39项既有V3/IRWS-R/DAG-FE回归检查通过；专项包含本地CUDA实际更新与验证。
真实launch入口的训练/评测dry-run已用既有共享校准检查，9源/27目标的路径映射另有专项覆盖。
