# IRWS-R-FE_R090_IDLE：方案A，周期调度下只向空闲VM分配

日期：2026-10-29。基线为IRWS-R-FE_R090（超期权重0.9），复制后修改。
算法ID `irws_r_fe_r090_idle`，显示名 `IRWS-R-FE_R090_IDLE`，
模型版本 `irws_r_v3_fe_r090_idle_v1`，调度协议 `idle_only_periodic_v1`。
原R090/FE/V3及其他IRWS变体保留，公共环境、物理计算、实例、校准和实验划分不改。
源文件SHA-256记录在算法目录的 `source_snapshot_sha256.json`。

## 行为

合法VM同时满足没有运行任务和 `vm_available_at <= current_time + 1e-12`。
所有VM等待队列必须为空；发现排队状态会报错，不会暗中恢复队列服务。
分配前重新检查合法性，非法动作不会改变放置、任务状态、负载记录或能耗。
合法分配直接开始服务，取消忙VM入队和自动从队列取下一个任务的分支。

没有空闲VM或没有Ready任务时，推进到后续调度tick，期间准确处理到达/完成事件。
保留默认5秒周期，不改成事件触发分配：12.3秒完成的任务记录仍是12.3秒，
等待任务最早到15秒tick才选择并分配。最后全部任务完成时在真实完成时刻结束，无需等下个tick。
零时长任务在当前tick完成、释放VM和依赖，不产生虚假等待，也不要求未来还有事件才能结束。

Ready任务在全局集合等待，不提前绑定任务或VM；后续真实调度机会重新计算任务选择概率。
等待和物理事件不形成额外训练动作。原rollout内部连接真实后继，非终止边界仍为旧策略期望，
在获得合法后继状态后才估值和更新；等待不是done，最终不足64条经验仍全部更新。

## 保留部分

- 学习器、网络、数值诊断、FE选模模块与基线逐字节复制；γ、clip、学习率、网络宽度等不变。
- 默认跨区间64条批量更新、静态图/资源缓存、拓扑前沿并行、设备张量及float32保持。
- 超期/能耗系数0.9/0.1、能耗尺度0.001、超期尺度300、融合公式与元梯度结构保持。
- 总体工作流成功率>=99%后最低能耗选模；未达标时成功率优先，完全平局保留较早回合。
- 验证状态隔离、历史best续训、固定源模型用于S/M/L目标任务规模、输出覆盖保护保持。
- 输入维度仍为 `5 + 1 + 6*N_VM`，队列长度项恒为0，忙VM剩余时间等原特征继续可见。

`queue-capacity`必须为正（默认10），仅为原观测归一化布局及CLI兼容保留，不表示可以排10个任务。
调度周期参数仍可显式调整，但方案A默认5秒；训练、评测须匹配。

## 文件与兼容

训练目录为 `outputs/ckpts/irws_r_fe_r090_idle/<源场景>_m<模型种子>_c<8位配置指纹>/`，
例如 `SS_Tight_m1_c12345678`（指纹为示意）。取消重复版本目录、长参数拼接和独立model目录。
训练日志为该目录内的 `train.jsonl`，详细计时日志仍在 `logs/`。
检查点名保持 `best_irws_r_fe_r090_idle.pth`、`final_irws_r_fe_r090_idle.pth`。
配置指纹复用已有公共路径工具，覆盖模型版本、调度实现、奖励、选模协议、队列/周期、
rollout/精度、归一化模式及校准内容；源场景与模型种子直接可见，不用文件绝对路径区分配置。

最终CSV及汇总JSON统一直接放在 `outputs/run/irws_r_fe_r090_idle/`，不再放入SS/SM等场景子目录。
文件名示意：

```text
eval_irws_r_fe_r090_idle_SS_Tight_fromSS_Tight_m1_s700-729_c12345678.csv
eval_irws_r_fe_r090_idle_MS_Tight_fromSS_Tight_m1_s700-729_c12345678.csv
eval_irws_r_fe_r090_idle_LS_Tight_fromSS_Tight_m1_s700-729_c12345678.csv
```

`MS`表示中任务/小资源；`fromSS_Tight`表示源训练场景；`m1`为模型种子；`s700-729`为测试实例范围。
同数量但不同测试种子集合会得到不同名称。汇总JSON采用对应的 `summary_...json`。
逐工作流/逐决策明细单独放在 `_details/<16位标识>/measurements/`，对应位置写入汇总JSON。
27种场景的最终CSV共用一个父目录且文件名唯一，下载到同一文件夹仍能区分。
后台控制台日志前缀也同步缩短，固定运行参数仍在指令和运行元数据中记录。
实现标识、检查点、场景元数据、日志和评测摘要均记录新调度协议。
加载校验协议、算法/版本、原有奖励/物理配置/校准/选模指标，排队版和错误协议拒绝加载。
原排队版权重不能直接当成该版本完整训练结果；请从头训练。
本次仅调整输出布局，不改变IDLE模型版本、权重或学习行为。已存在的IDLE长路径结果不自动移动或覆盖；
评测可用 `--checkpoint-dir <原IDLE检查点目录>`，续训可用 `--resume-checkpoint <原目录>/final_irws_r_fe_r090_idle.pth`。
续训继续写原目录，新增训练使用短目录。评测遇到同名CSV/JSON或非空明细目录会拒绝覆盖。

## 运行

在仓库根目录运行，复用已准备的公共缓存和源场景冻结校准。

```bash
python -u launch.py algorithms/irws_r_fe_r090_idle/train.py \
  --task S --resource M --deadline Tight --episodes 600 --model-seed 1 \
  --experiment-protocol configs/f22_split_v1.json --normalization-mode f19_shared_v1 \
  --queue-capacity 10 --slot-duration 5 --rollout-size 64 --precision float32 --device cuda

python -u launch.py algorithms/irws_r_fe_r090_idle/evaluate.py \
  --task L --resource M --deadline Tight \
  --train-task S --train-resource M --train-deadline Tight --seeds 700-729 --model-seed 1 \
  --experiment-protocol configs/f22_split_v1.json --normalization-mode f19_shared_v1 \
  --queue-capacity 10 --slot-duration 5 --rollout-size 64 --precision float32 --device cuda
```

[9源训练函数](Linux_CUDA_IRWS_R_FE_R090_IDLE_600代训练指令.txt)和
[27目标评测函数（9个三合一）](Linux_CUDA_IRWS_R_FE_R090_IDLE_27场景共用评测指令.txt)为专属指令。
后台训练函数 `train_irws_r_fe_r090_idle` 的控制台日志/PID放在 `outputs/server_train_logs/`；
评测函数 `eval_irws_r_fe_r090_idle_all` 放在 `outputs/server_eval_logs/`，第三参数可选目标列表（默认 `S M L`）。
例如 `eval_irws_r_fe_r090_idle_all S Tight "M L"` 只运行MS/LS两个目标；任一场景失败即退出该组。
指令只传IRWS入口实际支持的参数，不包含MAHDRL专用的profile/diagnostics CLI开关。
原R090入口继续允许向忙VM排队。新入口支持 `--dry-run`，打印idle-only协议和独立路径。

## 验证与限制

```powershell
.\.venv\Scripts\python.exe -X utf8 -B -m unittest common.test_irws_r_fe_r090_idle common.test_irws_r_fe_r090 common.test_irws_r_fe common.test_irws_r_speed -v
```

专项覆盖忙VM概率零和非法动作无副作用、全忙等待、12.3秒完成/15秒分配、未来到达、同刻完成、
零时长任务、队列恒空、依赖/服务无重叠、能耗再积分、64/64/32真实分配经验窗口及终止尾批、
CPU/CUDA更新、验证隔离、训练/加载/续训/评测及覆盖保护、9源/27目标CLI映射。
短测使用临时工作流、测试协议与尺度，不改公共正式校准/缓存。

本次14项IDLE专项（含紧凑路径、平铺CSV、精确种子命名及覆盖保护）和26项R090/FE/V3加速回归共40项通过，含本地CUDA实际更新及验证。
9条训练/27条评测命令完成Bash语法、后台展开、实际CLI与源模型路径映射检查；
真实launch入口dry-run通过。82个源IRWS及相关公共文件SHA-256保持，新增Python语法与空白检查通过。

这是收紧动作范围的适配变体，不能称为恢复原论文的等待队列机制；也不等同于MAHDRL的事件触发调度。
未开展正式600回合训练、30测试种子评测或服务器吞吐实验，不能预先保证成功率、能耗或耗时改善。
