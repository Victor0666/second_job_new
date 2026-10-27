# IRWS-R-FE_R090

复制自IRWS-R-FE，唯一奖励参数变化是超期权重由0.75改为0.9，能耗权重自动由0.25变为0.1。
算法ID `irws_r_fe_r090`，显示名 `IRWS-R-FE_R090`，版本 `irws_r_v3_fe_r090_v1`。

学习器、网络、环境、数值诊断和99%成功率选模代码与原FE逐字节一致；
保留V3的64条跨区间批量更新、图/资源缓存、拓扑并行、默认5秒周期及float32。
公共能耗尺度0.001、超期归一化300及奖励融合公式保持。

入口为 `algorithms/irws_r_fe_r090/train.py`、`algorithms/irws_r_fe_r090/evaluate.py`。
独立输出 `outputs/{ckpts,run}/irws_r_fe_r090/irws_r_v3_fe_r090_v1/`。
检查点为 `best_irws_r_fe_r090.pth`、`final_irws_r_fe_r090.pth`，拒绝旧版或0.75奖励设置。
原IRWS-R-FE及所有历史版本保留。

详见 `docs/IRWS_R_FE_R090_implementation.md` 及两份专属Linux/CUDA指令。
