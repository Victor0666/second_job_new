# IRWS-R

独立对比算法ID：`irws_r`；显示名称：**IRWS-R**；当前模型版本：`irws_r_v2_batch`。修改前的逐样本版本完整保存在`algorithms/irws_r_v1/`，原IRWS、`irws_paper`、`irws_restored`及已有结果保留。历史复制源及公共文件哈希见`copied_from_sha256.json`，v1副本另含`snapshot_sha256.json`。

任务Q只输入5维图嵌入；Actor/Critic及RNN状态输入不含任务剩余截止期、工作流剩余截止期和Ready等待时间。v2采用前驱均值与特征维Softmax、内外层一致编码、区间内批平均更新，以及更新后再选择新区间首任务；Q继续固定，公共奖励、原式融合及FIFO物理环境不变。补全约定和原文差异见[v2机制说明](../../docs/IRWS_R_v2_implementation.md)。

公共实验参数与MAHDRL一致，包括实例、资源、到达率、每回合工作流数、截止期分布、归一化、训练预算和数据划分；IRWS学习参数保留原文的gamma、batch、学习率和网络宽度。队列容量和策略周期仍是IRWS独有的补设参数。N个VM时，Actor/Critic输入为`5+1+6N`维。

```powershell
# 参数预览，不启动训练
.\run.cmd algorithms/irws_r/train.py --task S --resource S --deadline Tight --dry-run

# 专项检查
.\.venv\Scripts\python.exe -B -m unittest common.test_irws_r -v

# 临时缓存、两工作流、一个训练回合的真实DAX闭环
.\run.cmd common/smoke_irws_r.py --out docs/IRWS_R_v2_smoke.json

# 正式训练：先准备公共新版缓存和有效共享归一化文件
.\run.cmd algorithms/irws_r/train.py --task S --resource S --deadline Tight --episodes 600 --model-seed 1 --queue-capacity 10 --slot-duration 1

# 正式评测：队列及区间参数须与训练一致
.\run.cmd algorithms/irws_r/evaluate.py --task S --resource S --deadline Tight --seeds 700-729 --model-seed 1 --queue-capacity 10 --slot-duration 1
```

Linux使用 `python launch.py algorithms/irws_r/train.py ...` 等价调用。`--device cuda` 可用于具有CUDA的PyTorch环境。

模型写入`outputs/ckpts/irws_r/irws_r_v2_batch/<配置标识>/model1/`，文件名为`best_irws_r.pth`、`final_irws_r.pth`；评测写入`outputs/run/irws_r/irws_r_v2_batch/<配置标识>/model1/<目标场景>/`。训练日志与主评测文件以`irws_r`标识。不同`--model-seed`使用独立目录，训练和评测须选择同一模型种子。

新训练和评测入口均拒绝覆盖目标目录已有结果；旧文件不自动迁移或加载。续训使用`--resume-checkpoint`指向本次训练目录的`final_irws_r.pth`。

v2拒绝加载v1模型；需要重新训练。继续使用旧算法或旧模型时，入口为`python launch.py algorithms/irws_r_v1/train.py ...`和`evaluate.py ...`，仍使用原`irws_r_v1`输出目录。
