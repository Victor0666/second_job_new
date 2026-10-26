# IRWS-R-FE

复制自 `algorithms/irws_r/` 的加速 V3，模型版本 `irws_r_v3_fe_v1`。
采用 PD3QN_DAG_FE 当前的 `select_sr099_minenergy_v1` 检查点选择规则：
验证集总体工作流成功率至少99%时，优先选择平均总能耗最低的可行模型；
没有可行模型时，优先成功率最高、其次能耗最低的备用模型。完全平局保留较早回合。

`learner.py`、`networks.py`、`environment.py`、`numerics.py`与复制时的V3字节一致。
学习参数、奖励、观测、FIFO队列、跨区间64条批量更新、默认5秒周期和float32保持。
原IRWS、IRWS-R V3、历史版本及PD3QN_DAG_FE不修改。

入口为 `algorithms/irws_r_fe/train.py`、`algorithms/irws_r_fe/evaluate.py`。
输出独立进入 `outputs/{ckpts,run}/irws_r_fe/irws_r_v3_fe_v1/`，配置目录包含选模标识。
最佳检查点为 `best_irws_r_fe.pth`，最终及续训检查点为 `final_irws_r_fe.pth`。
同配置非空输出拒绝覆盖。加载校验算法、版本、角色、规则、验证记录及原有物理和校准条件。

完整设计、运行示例与验证边界见 `docs/IRWS_R_FE_implementation.md`。
