# IRWS-R-FE_R090_IDLE

从IRWS-R-FE_R090复制的方案A变体：只向空闲VM分配，保留周期调度（默认5秒）。
无空闲VM时，任务在全局Ready集合等待，不绑定VM，不产生额外经验。
物理到达与完成事件保持精确时间；例如12.3秒完成后，下一任务最早在15秒调度tick分配。

算法ID `irws_r_fe_r090_idle`，版本 `irws_r_v3_fe_r090_idle_v1`，
调度协议 `idle_only_periodic_v1`。原R090、FE、V3及公共物理代码保留。
奖励仍为超期0.9、能耗0.1；FE的99%成功率门槛选模、V3批量/缓存/拓扑并行和float32均保留。
观测维度不变，原VM队列长度特征恒零。`--queue-capacity 10`仅保留正数布局兼容值，不开放排队。

入口为 `algorithms/irws_r_fe_r090_idle/train.py`、`evaluate.py`。
检查点为 `best_irws_r_fe_r090_idle.pth` 和 `final_irws_r_fe_r090_idle.pth`，
训练目录缩短为 `outputs/ckpts/irws_r_fe_r090_idle/SS_Tight_m1_c<8位指纹>/`。
全部最终CSV和汇总JSON直接放在 `outputs/run/irws_r_fe_r090_idle/`，文件名包含
目标场景、源场景、模型种子、测试种子和配置指纹；明细存入 `_details/<16位标识>/`。
原排队版检查点仍拒绝加载；旧IDLE检查点可显式指定原路径评测/续训，不自动搬移。
说明和9源/27目标命令见专属文档。
