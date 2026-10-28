# IRWS-R-FE_IDLE：原0.75 FE增加空闲VM限制

日期：2026-10-29。修正此前将方案A应用于R090分支的范围理解。
本版本从 `algorithms/irws_r_fe/` 复制，奖励权重保持超期0.75、能耗0.25。
原FE、R090、R090_IDLE、其他IRWS以及公共环境/物理计算/校准/数据划分不改。
源FE文件的SHA-256记录在算法目录的 `source_snapshot_sha256.json`。

## 身份与奖励

- 算法ID：`irws_r_fe_idle`；显示名：`IRWS-R-FE_IDLE`。
- 模型版本：`irws_r_v3_fe_idle_v1`；调度协议：`idle_only_periodic_v1`。
- 外部奖励：`0.75 * r_delay + 0.25 * r_energy`。
- 能耗项仍为 `r_energy = -delta_energy_total * 0.001`；超期尺度仍为300。
- 0.75为外部超期/能耗权重，不是内外奖励融合权重；原融合公式和边界约定保持。
- 学习器、网络、数值处理、FE选模文件与原FE逐字节一致。

保留原FE的V3加速：默认跨区间64条真实分配经验更新、图/资源缓存、拓扑前沿批量执行、
设备张量及float32；保留gamma=0.2等学习参数，不改成逐样本更新。
FE选模继续使用验证总体工作流成功率>=99%后选最低能耗；未达标时成功率优先，
再比较能耗，完全相同则保留较早模型。验证、训练、测试划分和验证状态隔离保持。

## 空闲VM方案A

合法VM必须没有运行任务且 `vm_available_at <= current_time + 1e-12`。
所有VM队列恒空，非法忙VM分配在物理修改前报错。
没有空闲VM时，任务在全局Ready集合等待，不提前绑定VM，不形成额外经验。
等待后重新选择任务和合法VM；保留默认5秒调度tick，不改成事件触发分配。
12.3秒完成的物理事件仍记为12.3秒，等待任务最早15秒分配。
最后任务在实际完成时结束；零时长任务在同一tick完成并释放VM和依赖。

输入维度保持 `5 + 1 + 6*N_VM`，队列长度项恒零。
`--queue-capacity 10`仅保留观测布局/CLI兼容，不启用VM排队。
训练和评测的调度周期、rollout和精度必须匹配。

## 输出与兼容

训练目录采用短名称：

```text
outputs/ckpts/irws_r_fe_idle/SS_Tight_m1_c<8位配置指纹>/
```

检查点为 `best_irws_r_fe_idle.pth`、`final_irws_r_fe_idle.pth`，日志为 `train.jsonl`。
最终27场景CSV共用一个父目录 `outputs/run/irws_r_fe_idle/`，例如：

```text
eval_irws_r_fe_idle_SS_Tight_fromSS_Tight_m1_s700-729_c12345678.csv
eval_irws_r_fe_idle_MS_Tight_fromSS_Tight_m1_s700-729_c12345678.csv
eval_irws_r_fe_idle_LS_Tight_fromSS_Tight_m1_s700-729_c12345678.csv
```

配置指纹仅为示意。`MS`是目标中任务/小资源，`fromSS_Tight`是源模型场景。
汇总使用对应 `summary_...json`；明细放在 `_details/<16位标识>/measurements/`。
名称和配置指纹包含源场景、模型种子、准确测试种子集合、奖励与调度/学习配置，
不会和原FE或R090各版本共用输出目录；已有结果拒绝覆盖。
加载校验算法、模型版本、奖励、调度、校准、选模和环境/学习配置，
拒绝原排队版FE、R090及R090_IDLE检查点。本版本需要从头训练。

## 运行与验证

[600回合、9种源场景训练指令](Linux_CUDA_IRWS_R_FE_IDLE_600代训练指令.txt)和
[27目标场景评测指令（9个三合一）](Linux_CUDA_IRWS_R_FE_IDLE_27场景共用评测指令.txt)
使用独立入口和日志前缀。训练函数为 `train_irws_r_fe_idle`，
评测函数为 `eval_irws_r_fe_idle_all`，第三参数可选目标任务列表，默认 `S M L`。
每组评测任何场景失败即退出。

```powershell
.\.venv\Scripts\python.exe -X utf8 -B -m unittest common.test_irws_r_fe_idle common.test_irws_r_fe_r090_idle common.test_irws_r_fe common.test_irws_r_fe_r090 common.test_irws_r_speed -q
```

回归覆盖实际0.75奖励、0.9误配拒绝、忙VM屏蔽与非法动作无副作用、等待及精确完成时刻、
零时长任务、能耗积分、V3的64/64/32窗口及终止尾批、CPU/CUDA更新、
训练/验证/续训/评测、覆盖保护、9源/27目标映射和短路径/平铺CSV。
短测使用临时协议和工作流，不改公共正式缓存及校准。

以上55项回归全部通过，包含本地CUDA实际更新和验证；9组训练、27目标评测指令通过
Bash语法、后台展开、实际CLI及源模型映射检查，可选目标与失败退出行为也已验证。
27个最终CSV名称唯一且共用一个父目录；135个既有IRWS及相关公共文件SHA-256保持不变。

空闲VM限制是适配变体，不等同于原论文等待队列机制。未开展正式600回合训练、
30种子完整评测或服务器吞吐实验，不预先宣称性能改善。
