# IRWS-R-FE_IDLE

从原 `IRWS-R-FE`（超期权重0.75、能耗权重0.25）复制的独立方案A版本。
只向空闲VM分配，保留默认5秒周期；任务在全局Ready集合等待，不进入忙VM队列。
保留V3的64条rollout更新、图/资源缓存、拓扑前沿批量计算、float32和FE选模机制。
原FE、R090、R090_IDLE以及公共环境不改。

算法ID为 `irws_r_fe_idle`，模型版本为 `irws_r_v3_fe_idle_v1`，
调度协议为 `idle_only_periodic_v1`。学习器、网络、数值处理与原FE逐字节一致。
选模门槛按用户要求改为验证集总体工作流成功率>=92%后选最低能耗，
未达标时先比较成功率、再比较能耗；标识为 `select_sr092_minenergy_v1`。
实际奖励和检查点元数据均保留0.75/0.25，不从R090继承0.9权重。

入口为 `algorithms/irws_r_fe_idle/train.py` 和 `evaluate.py`。
检查点为 `best_irws_r_fe_idle.pth`、`final_irws_r_fe_idle.pth`。
训练目录为 `outputs/ckpts/irws_r_fe_idle/SS_Tight_m1_c<8位指纹>/`。
最终评测CSV及汇总JSON平铺在 `outputs/run/irws_r_fe_idle/`，文件名包含目标场景、
源模型场景、模型种子、测试种子和配置指纹；明细在 `_details/<16位标识>/`。
原排队版FE和各R090版本的检查点均不兼容，需要单独训练。
此前99%选模的IDLE检查点也会被拒绝加载；新选模协议自动产生不同配置指纹，保留旧输出。

完整说明见 `docs/IRWS_R_FE_IDLE_implementation.md`，9组训练和27场景评测指令见专属文档。
