# IRWS restored copy

本目录复制自 `algorithms/irws_paper/` 后独立修改。历史 `algorithms/irws/`、此前新版 `algorithms/irws_paper/` 及公共物理文件保持不变；复制时的18个保护文件SHA-256见 `copied_from_sha256.json`。

模型版本：`irws_restored_queue_v2`。任务Q只输入5维图嵌入；Actor/Critic状态及RNN状态输入已删除任务剩余截止期、工作流剩余截止期和Ready等待时间。环境仍保留公共截止期、能耗—超期奖励和评测。原文未写全处的补全、公式计算图和验证边界见 [实现说明](../../docs/IRWS_restored_implementation.md)。

公共实验参数与MAHDRL一致，包括实例、资源、到达率、每回合工作流数、截止期分布、归一化、训练预算和数据划分；IRWS学习参数保留原文的gamma、batch、学习率和网络宽度。队列容量和策略周期仍是IRWS独有的补设参数。N个VM时，Actor/Critic输入为`5+1+6N`维。

```powershell
# 参数预览，不启动训练
.\run.cmd algorithms/irws_restored/train.py --task S --resource S --deadline Tight --dry-run

# 专项检查
.\.venv\Scripts\python.exe -B -m unittest common.test_irws_restored -v

# 临时缓存、两工作流、一个训练回合的真实DAX闭环
.\run.cmd common/smoke_irws_restored.py --out docs/IRWS_restored_graph_only_smoke.json

# 正式训练：先准备公共新版缓存和有效共享归一化文件
.\run.cmd algorithms/irws_restored/train.py --task S --resource S --deadline Tight --episodes 600 --queue-capacity 10 --slot-duration 1

# 正式评测：队列及区间参数须与训练一致
.\run.cmd algorithms/irws_restored/evaluate.py --task S --resource S --deadline Tight --seeds 700-729 --queue-capacity 10 --slot-duration 1
```

Linux使用 `python launch.py algorithms/irws_restored/train.py ...` 等价调用。`--device cuda` 可用于具有CUDA的PyTorch环境。

模型写入 `outputs/ckpts/irws_restored_queue_v2/`，结果写入 `outputs/run/irws_restored_queue_v2/`。子目录绑定源场景、协议、归一化指纹、队列容量和区间长度。输入维度已改变，旧IRWS、此前新版及含三项观测的`irws_restored_queue_v1`权重均拒绝加载，必须重新训练；旧输出不会被覆盖。本版本续训使用 `--resume-checkpoint` 指向自己的 `final.pth`。
